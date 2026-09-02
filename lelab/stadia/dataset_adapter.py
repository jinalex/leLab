"""Fail-closed LeRobot v0.6.0 recording dataset adapter.

The Stadia recording state machine deliberately depends on a small dataset
port.  This module is the production implementation of that port for the
exact LeRobot revision pinned by LeLab.  It does not import LeRobot at module
import time.

LeRobot's episode save is incremental: it can persist tasks, parquet data,
videos, episode metadata, ``info.json``, and ``stats.json`` before a later
step fails.  Before each attempt this adapter flushes *committed* writers and
resets the pinned writer's file-selection caches.  Consequently, the next
episode uses new data, video, and episode-metadata files.  Shared metadata is
kept as an in-memory byte checkpoint.  A failed save is rolled back and the
adapter is permanently poisoned even when rollback can be positively proven;
finalization and upload are then prohibited.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .recording import AttemptCleanupReport

PINNED_LEROBOT_VERSION = "0.6.0"
PINNED_LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"

# These files contain every private operation used below.  Checking their
# hashes makes a source tree that merely reports the expected package version
# fail closed if the audited private implementation has drifted.
_PINNED_SOURCE_HASHES = {
    "lerobot/datasets/compute_stats.py": "d546a5c690eff716a10e9b13d70cd5d1ef7fed88f2f9c40c99f2f7f50845cf45",
    "lerobot/datasets/dataset_writer.py": "85f040139575884079b49f759a992f42e41d42be1980a6b23542798c1e7f8f5f",
    "lerobot/datasets/dataset_metadata.py": "57d5362571053bb67567ceb4e3fe2f879278fb4215577e9dd2d37b9bebe802c7",
    "lerobot/datasets/feature_utils.py": "928369cff50f0d663a125abfc67683ccbae0e552ed2eb22edde9e6bcff2aef81",
    "lerobot/datasets/image_writer.py": "54cad218c99caad105ea8738507ee87fc8c2858832d31aa5b21d324c97360631",
    "lerobot/datasets/io_utils.py": "af08c7875f34f8469f80c4fd903af12971d6fa96c179be984887b93ef2f6f91c",
    "lerobot/datasets/lerobot_dataset.py": "da387f34291a659d27b5fad0f87372f243259ec4cb7f824d459fe414c293dd28",
    "lerobot/datasets/utils.py": "488b713e643acaaa8c76cca8830ca71cb20584b4ceb3657a9396f81a20b3f83e",
    "lerobot/datasets/video_utils.py": "82d0ced8f3f9adc754d4388122b044d36f2feeadda28d4ddad933cead7cc54d1",
}

_MUTABLE_METADATA_PATHS = (
    Path("meta/info.json"),
    Path("meta/stats.json"),
    Path("meta/tasks.parquet"),
)


class DatasetAdapterError(RuntimeError):
    """Base error for the guarded dataset adapter."""


class DatasetAdapterCompatibilityError(DatasetAdapterError):
    """The dataset or installed LeRobot private API is not the audited one."""


class DatasetAdapterStateError(DatasetAdapterError):
    """The adapter was called out of order or with an invalid checkpoint."""


class DatasetAdapterPoisonedError(DatasetAdapterError):
    """A prior failure makes finalization or upload unsafe."""


@dataclass(frozen=True, slots=True)
class _PathState:
    kind: str
    size: int | None = None
    modified_ns: int | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class _MemoryState:
    info: object
    tasks: object
    episodes: object
    stats: object
    meta_latest_episode: object
    meta_finalized: bool
    writer_latest_episode: object
    writer_current_file_start_frame: object
    writer_recorded_frames: int
    writer_episodes_since_last_encoding: int
    writer_finalized: bool
    dataset_finalized: bool


class _AttemptToken:
    """Opaque identity returned to the state machine for one attempt."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _AttemptCheckpoint:
    token: _AttemptToken
    root: Path
    tree: Mapping[Path, _PathState]
    expected_data_path: Path
    expected_episode_metadata_path: Path
    expected_video_paths: Mapping[str, Path]
    metadata_bytes: Mapping[Path, bytes | None]
    memory: _MemoryState
    total_episodes: int
    total_frames: int
    total_tasks: int
    streaming_closed: bool | None
    streaming_threads: list[object] = field(default_factory=list)
    streaming_capture_errors: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pinned_runtime(dataset: object) -> None:
    try:
        distribution = importlib.metadata.distribution("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise DatasetAdapterCompatibilityError("the pinned lerobot distribution is not installed") from error

    if distribution.version != PINNED_LEROBOT_VERSION:
        raise DatasetAdapterCompatibilityError(
            f"expected lerobot {PINNED_LEROBOT_VERSION}, found {distribution.version}"
        )

    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "")
        commit = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise DatasetAdapterCompatibilityError(
            "lerobot installation has no verifiable direct-url commit receipt"
        ) from error
    if commit != PINNED_LEROBOT_COMMIT:
        raise DatasetAdapterCompatibilityError(
            f"expected lerobot commit {PINNED_LEROBOT_COMMIT}, found {commit}"
        )

    loaded_modules: dict[str, object] = {}
    for relative in _PINNED_SOURCE_HASHES:
        module_name = relative.removesuffix(".py").replace("/", ".")
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise DatasetAdapterCompatibilityError(
                f"could not import audited lerobot source: {module_name}"
            ) from error
        loaded_modules[module_name] = module

    lerobot_dataset_type = getattr(loaded_modules["lerobot.datasets.lerobot_dataset"], "LeRobotDataset", None)
    dataset_writer_type = getattr(loaded_modules["lerobot.datasets.dataset_writer"], "DatasetWriter", None)
    metadata_type = getattr(
        loaded_modules["lerobot.datasets.dataset_metadata"], "LeRobotDatasetMetadata", None
    )
    encoder_type = getattr(loaded_modules["lerobot.datasets.video_utils"], "StreamingVideoEncoder", None)
    expected_types = (
        (dataset, lerobot_dataset_type, "lerobot.datasets.lerobot_dataset.LeRobotDataset"),
        (
            getattr(dataset, "writer", None),
            dataset_writer_type,
            "lerobot.datasets.dataset_writer.DatasetWriter",
        ),
        (
            getattr(dataset, "meta", None),
            metadata_type,
            "lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata",
        ),
    )
    for value, expected_type, label in expected_types:
        if not isinstance(expected_type, type) or type(value) is not expected_type:
            value_type = type(value)
            raise DatasetAdapterCompatibilityError(
                f"expected exact class identity {label}, found {value_type.__module__}.{value_type.__name__}"
            )

    encoder = getattr(getattr(dataset, "writer", None), "_streaming_encoder", None)
    if encoder is not None and (not isinstance(encoder_type, type) or type(encoder) is not encoder_type):
        value_type = type(encoder)
        raise DatasetAdapterCompatibilityError(
            "expected exact class identity lerobot.datasets.video_utils.StreamingVideoEncoder, "
            f"found {value_type.__module__}.{value_type.__name__}"
        )

    for relative, expected_hash in _PINNED_SOURCE_HASHES.items():
        module_name = relative.removesuffix(".py").replace("/", ".")
        module = loaded_modules[module_name]
        loaded_file = getattr(module, "__file__", None)
        if not isinstance(loaded_file, str):
            raise DatasetAdapterCompatibilityError(
                f"audited lerobot module has no source path: {module_name}"
            )
        try:
            loaded_source = Path(loaded_file).resolve(strict=True)
            installed_source = Path(distribution.locate_file(relative)).resolve(strict=True)
            same_source = os.path.samefile(loaded_source, installed_source)
        except OSError as error:
            raise DatasetAdapterCompatibilityError(
                f"could not resolve audited lerobot source: {relative}"
            ) from error
        if not same_source:
            raise DatasetAdapterCompatibilityError(
                f"loaded lerobot module does not match its installed source: {relative}"
            )
        if not loaded_source.is_file() or _sha256(loaded_source) != expected_hash:
            raise DatasetAdapterCompatibilityError(f"audited lerobot source drifted: {relative}")


def _require_attributes(value: object, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not hasattr(value, name)]
    if missing:
        raise DatasetAdapterCompatibilityError(f"{label} is missing private API: {', '.join(missing)}")


def _require_methods(value: object, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not callable(getattr(value, name, None))]
    if missing:
        raise DatasetAdapterCompatibilityError(f"{label} is missing methods: {', '.join(missing)}")


def _lexical_path(value: object, *, label: str) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except TypeError as error:
        raise DatasetAdapterCompatibilityError(f"{label} is not a filesystem path") from error
    try:
        resolved = Path(os.path.realpath(path, strict=True))
    except OSError as error:
        raise DatasetAdapterCompatibilityError(f"{label} does not resolve to an existing path") from error
    if path != resolved:
        raise DatasetAdapterCompatibilityError(f"{label} contains a symlink")
    if not path.is_dir():
        raise DatasetAdapterCompatibilityError(f"{label} is not a directory")
    return path


def _relative_path(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise DatasetAdapterCompatibilityError(f"path escapes the dataset root: {path}") from error


def _scan_tree(root: Path, *, reject_symlinks: bool) -> dict[Path, _PathState]:
    result: dict[Path, _PathState] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = _relative_path(root, path)
                info = entry.stat(follow_symlinks=False)
                mode = info.st_mode
                if stat.S_ISLNK(mode):
                    if reject_symlinks:
                        raise DatasetAdapterCompatibilityError(f"symlink inside dataset root: {relative}")
                    result[relative] = _PathState("symlink")
                elif stat.S_ISDIR(mode):
                    result[relative] = _PathState("directory")
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    is_metadata = bool(relative.parts and relative.parts[0] == "meta")
                    result[relative] = _PathState(
                        "file",
                        size=info.st_size,
                        modified_ns=None if is_metadata else info.st_mtime_ns,
                        digest=_sha256(path) if is_metadata else None,
                    )
                else:
                    if reject_symlinks:
                        raise DatasetAdapterCompatibilityError(
                            f"unsupported filesystem entry inside dataset root: {relative}"
                        )
                    result[relative] = _PathState("special")
    return result


def _namespace(tree: Mapping[Path, _PathState], prefix: tuple[str, ...]) -> dict[Path, _PathState]:
    return {path: state for path, state in tree.items() if path.parts[: len(prefix)] == prefix}


def _safe_relative_path(value: object, *, label: str) -> Path:
    try:
        path = Path(os.fspath(value))
    except TypeError as error:
        raise DatasetAdapterCompatibilityError(f"{label} did not render a filesystem path") from error
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetAdapterCompatibilityError(f"{label} escapes or aliases the dataset root: {path}")
    return path


class LeRobotRecordingDatasetAdapter:
    """Implement ``RecordingDatasetPort`` for the audited LeRobot writer.

    The caller must keep exclusive ownership of ``dataset`` and must invoke
    finalization/upload through this adapter.  ``require_pinned_runtime`` is
    exposed only so dependency-neutral fakes can test rollback; production
    integration must leave it at its default value.
    """

    def __init__(self, dataset: object, *, require_pinned_runtime: bool = True) -> None:
        if require_pinned_runtime:
            _validate_pinned_runtime(dataset)
        self._dataset = dataset
        self._require_private_contract()
        self._root = self._validate_roots()
        _scan_tree(self._root, reject_symlinks=True)
        self._active_checkpoint: _AttemptCheckpoint | None = None
        self._save_attempted = False
        self._poisoned_reason: str | None = None
        self._finalized = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned_reason is not None

    @property
    def poison_reason(self) -> str | None:
        return self._poisoned_reason

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def _require_private_contract(self) -> None:
        dataset = self._dataset
        _require_attributes(dataset, ("root", "writer", "meta", "reader", "_is_finalized"), label="dataset")
        _require_methods(
            dataset,
            ("add_frame", "save_episode", "clear_episode_buffer", "finalize", "push_to_hub"),
            label="dataset",
        )
        writer = dataset.writer
        meta = dataset.meta
        if writer is None:
            raise DatasetAdapterCompatibilityError("dataset is read-only; no writer is attached")
        if dataset.reader is not None:
            raise DatasetAdapterCompatibilityError(
                "dataset must come from LeRobotDataset.create() or resume() in write-only mode"
            )
        _require_attributes(
            writer,
            (
                "_meta",
                "_root",
                "episode_buffer",
                "image_writer",
                "_pq_writer",
                "_latest_episode",
                "_current_file_start_frame",
                "_streaming_encoder",
                "_batch_encoding_size",
                "_episodes_since_last_encoding",
                "_recorded_frames",
                "_finalized",
            ),
            label="dataset writer",
        )
        _require_methods(
            writer,
            (
                "_create_episode_buffer",
                "_wait_image_writer",
                "cancel_pending_videos",
                "clear_episode_buffer",
                "close_writer",
            ),
            label="dataset writer",
        )
        _require_attributes(
            meta,
            (
                "root",
                "info",
                "tasks",
                "episodes",
                "stats",
                "latest_episode",
                "data_path",
                "video_path",
                "video_keys",
                "chunks_size",
                "_pq_writer",
                "_metadata_buffer",
                "_finalized",
                "total_episodes",
                "total_frames",
                "total_tasks",
            ),
            label="dataset metadata",
        )
        _require_methods(meta, ("_close_writer", "_load_metadata"), label="dataset metadata")
        if writer._meta is not meta:
            raise DatasetAdapterCompatibilityError("writer metadata is not the dataset metadata object")
        if isinstance(writer._batch_encoding_size, bool) or writer._batch_encoding_size != 1:
            raise DatasetAdapterCompatibilityError(
                "batch video encoding is unsupported; batch size must be 1"
            )
        if writer._episodes_since_last_encoding != 0:
            raise DatasetAdapterCompatibilityError("pending batched video encoding cannot be checkpointed")

    def _validate_roots(self) -> Path:
        dataset_root = _lexical_path(self._dataset.root, label="dataset root")
        writer_root = _lexical_path(self._dataset.writer._root, label="writer root")
        meta_root = _lexical_path(self._dataset.meta.root, label="metadata root")
        if dataset_root != writer_root or dataset_root != meta_root:
            raise DatasetAdapterCompatibilityError("dataset, writer, and metadata roots differ")
        return dataset_root

    def _ensure_available(self) -> None:
        if self.poisoned:
            raise DatasetAdapterPoisonedError(self._poisoned_reason or "dataset adapter is poisoned")
        if self._finalized or self._dataset._is_finalized:
            raise DatasetAdapterStateError("dataset has already been finalized")

    def _buffer_size(self) -> int:
        buffer = self._dataset.writer.episode_buffer
        if not isinstance(buffer, dict):
            raise DatasetAdapterCompatibilityError("writer episode_buffer is not a dictionary")
        size = buffer.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DatasetAdapterCompatibilityError("writer episode_buffer has an invalid size")
        return size

    def _quiesce_committed_state(self) -> None:
        writer = self._dataset.writer
        meta = self._dataset.meta
        if self._buffer_size() != 0:
            raise DatasetAdapterStateError("begin_attempt must run before adding any episode frames")
        encoder = writer._streaming_encoder
        if encoder is not None:
            _require_attributes(
                encoder,
                (
                    "_episode_active",
                    "_closed",
                    "_video_paths",
                    "_threads",
                    "_frame_queues",
                    "_result_queues",
                    "_stop_events",
                ),
                label="streaming encoder",
            )
            if encoder._episode_active:
                raise DatasetAdapterStateError("streaming encoder already has an active episode")
            if encoder._closed:
                raise DatasetAdapterStateError("streaming encoder is closed")
        writer._wait_image_writer()
        writer.close_writer()
        meta._close_writer()
        # A newly created zero-episode dataset intentionally has only
        # ``meta/info.json``; LeRobot's private loader assumes tasks/stats
        # already exist.  Once at least one committed episode exists, closing
        # the metadata writer above makes all loader inputs complete and a
        # reload is required so path selection sees the flushed episode rows.
        if meta.total_episodes > 0:
            meta._load_metadata()
        if writer._pq_writer is not None or meta._pq_writer is not None:
            raise DatasetAdapterCompatibilityError("committed parquet writers did not close")
        if meta._metadata_buffer:
            raise DatasetAdapterCompatibilityError("committed episode metadata did not flush")
        if writer._recorded_frames != meta.total_frames:
            raise DatasetAdapterCompatibilityError("writer and metadata frame totals disagree")
        episode_count = 0 if meta.episodes is None else len(meta.episodes)
        task_count = 0 if meta.tasks is None else len(meta.tasks)
        if episode_count != meta.total_episodes:
            raise DatasetAdapterCompatibilityError("episode table and metadata total disagree")
        if task_count != meta.total_tasks:
            raise DatasetAdapterCompatibilityError("task table and metadata total disagree")

        # The audited source interprets None as "derive the next path from the
        # committed episode table".  Resetting both caches therefore isolates
        # the next data, video, and episode-metadata writes in new files.
        writer._latest_episode = None
        writer._current_file_start_frame = None
        meta.latest_episode = None

    def _next_file_indices(self, chunk_index: object, file_index: object) -> tuple[int, int]:
        meta = self._dataset.meta
        chunks_size = meta.chunks_size
        if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size < 1:
            raise DatasetAdapterCompatibilityError("metadata chunks_size is not a positive integer")
        try:
            chunk = int(chunk_index)
            file = int(file_index)
        except (TypeError, ValueError) as error:
            raise DatasetAdapterCompatibilityError("committed chunk/file index is invalid") from error
        if chunk < 0 or file < 0 or file >= chunks_size:
            raise DatasetAdapterCompatibilityError("committed chunk/file index is out of range")
        return (chunk + 1, 0) if file == chunks_size - 1 else (chunk, file + 1)

    def _expected_attempt_paths(self, tree: Mapping[Path, _PathState]) -> tuple[Path, Path, dict[str, Path]]:
        meta = self._dataset.meta
        episodes = meta.episodes
        if episodes is None or len(episodes) == 0:
            data_indices = (0, 0)
            metadata_indices = (0, 0)
            video_indices = dict.fromkeys(meta.video_keys, (0, 0))
        else:
            latest = episodes[-1]
            data_indices = self._next_file_indices(latest["data/chunk_index"], latest["data/file_index"])
            metadata_indices = self._next_file_indices(
                latest["meta/episodes/chunk_index"], latest["meta/episodes/file_index"]
            )
            video_indices = {
                key: self._next_file_indices(
                    latest[f"videos/{key}/chunk_index"], latest[f"videos/{key}/file_index"]
                )
                for key in meta.video_keys
            }

        data_path = _safe_relative_path(
            meta.data_path.format(chunk_index=data_indices[0], file_index=data_indices[1]),
            label="data_path",
        )
        metadata_path = Path(
            f"meta/episodes/chunk-{metadata_indices[0]:03d}/file-{metadata_indices[1]:03d}.parquet"
        )
        video_paths: dict[str, Path] = {}
        for key, indices in video_indices.items():
            if meta.video_path is None:
                raise DatasetAdapterCompatibilityError("video keys exist but video_path is missing")
            video_paths[key] = _safe_relative_path(
                meta.video_path.format(video_key=key, chunk_index=indices[0], file_index=indices[1]),
                label=f"video_path[{key}]",
            )
        expected = [data_path, metadata_path, *video_paths.values()]
        if len(set(expected)) != len(expected):
            raise DatasetAdapterCompatibilityError("next attempt paths are not unique")
        collisions = [path for path in expected if path in tree]
        if collisions:
            raise DatasetAdapterCompatibilityError(
                f"next attempt path already exists: {', '.join(str(path) for path in collisions)}"
            )
        return data_path, metadata_path, video_paths

    def _memory_checkpoint(self) -> _MemoryState:
        writer = self._dataset.writer
        meta = self._dataset.meta
        return _MemoryState(
            info=copy.deepcopy(meta.info),
            tasks=copy.deepcopy(meta.tasks),
            episodes=meta.episodes,
            stats=copy.deepcopy(meta.stats),
            meta_latest_episode=copy.deepcopy(meta.latest_episode),
            meta_finalized=bool(meta._finalized),
            writer_latest_episode=copy.deepcopy(writer._latest_episode),
            writer_current_file_start_frame=writer._current_file_start_frame,
            writer_recorded_frames=writer._recorded_frames,
            writer_episodes_since_last_encoding=writer._episodes_since_last_encoding,
            writer_finalized=bool(writer._finalized),
            dataset_finalized=bool(self._dataset._is_finalized),
        )

    def begin_attempt(self) -> object:
        self._ensure_available()
        if self._active_checkpoint is not None:
            raise DatasetAdapterStateError("a recording attempt is already active")
        self._require_private_contract()
        if self._validate_roots() != self._root:
            raise DatasetAdapterCompatibilityError("dataset root changed after adapter construction")
        _scan_tree(self._root, reject_symlinks=True)
        self._quiesce_committed_state()
        tree = _scan_tree(self._root, reject_symlinks=True)
        expected_data, expected_metadata, expected_videos = self._expected_attempt_paths(tree)
        metadata_bytes = {
            relative: (self._root / relative).read_bytes() if (self._root / relative).is_file() else None
            for relative in _MUTABLE_METADATA_PATHS
        }
        meta = self._dataset.meta
        encoder = self._dataset.writer._streaming_encoder
        checkpoint = _AttemptCheckpoint(
            token=_AttemptToken(),
            root=self._root,
            tree=tree,
            expected_data_path=expected_data,
            expected_episode_metadata_path=expected_metadata,
            expected_video_paths=expected_videos,
            metadata_bytes=metadata_bytes,
            memory=self._memory_checkpoint(),
            total_episodes=meta.total_episodes,
            total_frames=meta.total_frames,
            total_tasks=meta.total_tasks,
            streaming_closed=None if encoder is None else bool(encoder._closed),
        )
        self._capture_streaming_threads(checkpoint)
        self._active_checkpoint = checkpoint
        self._save_attempted = False
        return checkpoint.token

    def _active(self) -> _AttemptCheckpoint:
        if self._active_checkpoint is None:
            raise DatasetAdapterStateError("no recording attempt is active")
        return self._active_checkpoint

    def _capture_streaming_threads(self, checkpoint: _AttemptCheckpoint) -> None:
        encoder = self._dataset.writer._streaming_encoder
        if encoder is None:
            return
        try:
            threads = encoder._threads
            if not isinstance(threads, Mapping):
                raise DatasetAdapterCompatibilityError("streaming encoder thread registry is not a mapping")
            known_ids = {id(thread) for thread in checkpoint.streaming_threads}
            for thread in threads.values():
                if id(thread) not in known_ids:
                    checkpoint.streaming_threads.append(thread)
                    known_ids.add(id(thread))
        except Exception as error:
            detail = f"could not capture streaming encoder threads: {type(error).__name__}: {error}"
            if detail not in checkpoint.streaming_capture_errors:
                checkpoint.streaming_capture_errors.append(detail)
            raise

    def _streaming_thread_errors(self, checkpoint: _AttemptCheckpoint) -> list[str]:
        errors = list(checkpoint.streaming_capture_errors)
        for thread in checkpoint.streaming_threads:
            is_alive = getattr(thread, "is_alive", None)
            if not callable(is_alive):
                errors.append("streaming encoder exposed an unverifiable thread handle")
                continue
            try:
                alive = is_alive()
            except Exception as error:
                errors.append(
                    f"streaming encoder thread liveness check failed: {type(error).__name__}: {error}"
                )
            else:
                if alive:
                    errors.append("streaming encoder thread remained alive after cancellation")
        return errors

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        self._ensure_available()
        checkpoint = self._active()
        if not isinstance(frame, Mapping):
            raise TypeError("dataset frame must be a mapping")
        try:
            self._dataset.add_frame(dict(frame))
        finally:
            # Streaming threads are removed from LeRobot's dictionaries by
            # finish/cancel.  Retain their identities before either operation
            # so liveness can still be positively verified afterward.
            self._capture_streaming_threads(checkpoint)

    def _baseline_entries_unchanged_after_commit(
        self, checkpoint: _AttemptCheckpoint, current: Mapping[Path, _PathState]
    ) -> bool:
        mutable = set(_MUTABLE_METADATA_PATHS)
        for path, state in checkpoint.tree.items():
            if path in mutable or state.kind == "directory":
                continue
            if current.get(path) != state:
                return False
        return True

    def _verify_successful_save(self, checkpoint: _AttemptCheckpoint) -> None:
        current = _scan_tree(self._root, reject_symlinks=True)
        meta = self._dataset.meta
        writer = self._dataset.writer
        streaming_errors = self._streaming_thread_errors(checkpoint)
        if streaming_errors:
            raise DatasetAdapterCompatibilityError("; ".join(streaming_errors))
        if not self._streaming_is_clean(checkpoint):
            raise DatasetAdapterCompatibilityError(
                "streaming encoder state was not clean after a successful save"
            )
        if meta.total_episodes != checkpoint.total_episodes + 1:
            raise DatasetAdapterCompatibilityError("save did not add exactly one episode")
        if meta.total_frames <= checkpoint.total_frames:
            raise DatasetAdapterCompatibilityError("save did not add any frames")
        if self._buffer_size() != 0:
            raise DatasetAdapterCompatibilityError("save did not clear the episode buffer")
        if not self._baseline_entries_unchanged_after_commit(checkpoint, current):
            raise DatasetAdapterCompatibilityError(
                "save modified a pre-checkpoint data, video, or episode file"
            )
        allowed_new = {
            checkpoint.expected_data_path,
            checkpoint.expected_episode_metadata_path,
            *checkpoint.expected_video_paths.values(),
            *_MUTABLE_METADATA_PATHS,
        }
        for path in tuple(allowed_new):
            allowed_new.update(path.parents)
        allowed_new.discard(Path("."))
        # The pinned non-streaming image/video path removes per-episode images
        # after embedding/encoding but leaves empty ``images/<camera>`` parent
        # directories.  Empty directories contain no fragment or reference;
        # any file below ``images`` remains unexpected and fails closed.
        allowed_new.update(
            path
            for path, state in current.items()
            if state.kind == "directory" and path.parts[:1] == ("images",)
        )
        unexpected = set(current) - set(checkpoint.tree) - allowed_new
        if unexpected:
            raise DatasetAdapterCompatibilityError(
                f"save left unexpected files or directories: {', '.join(map(str, sorted(unexpected)))}"
            )
        latest = writer._latest_episode
        if not isinstance(latest, dict):
            raise DatasetAdapterCompatibilityError("save did not publish writer episode metadata")
        data_path = Path(
            meta.data_path.format(
                chunk_index=latest["data/chunk_index"], file_index=latest["data/file_index"]
            )
        )
        if data_path != checkpoint.expected_data_path or data_path not in current:
            raise DatasetAdapterCompatibilityError("save did not use the isolated data file")
        episode_metadata = meta.latest_episode
        if not isinstance(episode_metadata, dict):
            raise DatasetAdapterCompatibilityError("save did not publish episode metadata")
        metadata_chunk = episode_metadata["meta/episodes/chunk_index"]
        metadata_file = episode_metadata["meta/episodes/file_index"]
        if isinstance(metadata_chunk, list):
            metadata_chunk = metadata_chunk[0]
        if isinstance(metadata_file, list):
            metadata_file = metadata_file[0]
        metadata_path = Path(
            f"meta/episodes/chunk-{int(metadata_chunk):03d}/file-{int(metadata_file):03d}.parquet"
        )
        if metadata_path != checkpoint.expected_episode_metadata_path:
            raise DatasetAdapterCompatibilityError("save did not use isolated episode metadata")
        for video_key in meta.video_keys:
            video_chunk = episode_metadata[f"videos/{video_key}/chunk_index"]
            video_file = episode_metadata[f"videos/{video_key}/file_index"]
            if isinstance(video_chunk, list):
                video_chunk = video_chunk[0]
            if isinstance(video_file, list):
                video_file = video_file[0]
            video_path = Path(
                meta.video_path.format(
                    video_key=video_key,
                    chunk_index=video_chunk,
                    file_index=video_file,
                )
            )
            if video_path != checkpoint.expected_video_paths[video_key] or video_path not in current:
                raise DatasetAdapterCompatibilityError(f"save did not isolate video file for {video_key}")

    def save_episode(self) -> None:
        self._ensure_available()
        checkpoint = self._active()
        if self._buffer_size() < 1:
            raise DatasetAdapterStateError("cannot save an episode without frames")
        # Once a non-empty save is requested, every subsequent failure may
        # follow an unknown partial mutation and permanently poisons this
        # adapter, including failures in our own preflight or verification.
        self._save_attempted = True
        try:
            self._capture_streaming_threads(checkpoint)
            current = _scan_tree(self._root, reject_symlinks=True)
            expected_paths = {
                checkpoint.expected_data_path,
                checkpoint.expected_episode_metadata_path,
                *checkpoint.expected_video_paths.values(),
            }
            collisions = expected_paths.intersection(current)
            if collisions:
                raise DatasetAdapterCompatibilityError(
                    "isolated save path appeared during the attempt: "
                    f"{', '.join(map(str, sorted(collisions)))}"
                )
            self._dataset.save_episode()
            self._capture_streaming_threads(checkpoint)
            self._verify_successful_save(checkpoint)
        except Exception as error:
            self._poison(f"episode save failed and requires rollback: {type(error).__name__}: {error}")
            raise
        self._active_checkpoint = None
        self._save_attempted = False

    def _remove_post_checkpoint_paths(self, checkpoint: _AttemptCheckpoint) -> list[str]:
        errors: list[str] = []
        try:
            current = _scan_tree(self._root, reject_symlinks=False)
        except Exception as error:
            return [f"could not inspect post-checkpoint paths: {type(error).__name__}: {error}"]

        new_paths = sorted(
            set(current) - set(checkpoint.tree), key=lambda path: (len(path.parts), str(path)), reverse=True
        )
        for relative in new_paths:
            path = self._root / relative
            try:
                _relative_path(self._root, path)
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    path.rmdir()
                else:
                    if stat.S_ISLNK(info.st_mode):
                        errors.append(f"rejected symlink residue: {relative}")
                    elif not stat.S_ISREG(info.st_mode):
                        errors.append(f"rejected special-file residue: {relative}")
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception as error:
                errors.append(f"could not remove {relative}: {type(error).__name__}: {error}")
        return errors

    def _restore_metadata_files(self, checkpoint: _AttemptCheckpoint) -> list[str]:
        errors: list[str] = []
        for relative, original in checkpoint.metadata_bytes.items():
            path = self._root / relative
            try:
                _relative_path(self._root, path)
                parent = path.parent
                if Path(os.path.realpath(parent, strict=True)) != parent:
                    raise DatasetAdapterCompatibilityError(f"metadata parent contains a symlink: {parent}")
                if original is None:
                    if path.exists() or path.is_symlink():
                        path.unlink()
                else:
                    if path.is_symlink():
                        raise DatasetAdapterCompatibilityError(f"metadata path became a symlink: {relative}")
                    path.write_bytes(original)
            except Exception as error:
                errors.append(f"could not restore {relative}: {type(error).__name__}: {error}")
        return errors

    def _restore_memory(self, checkpoint: _AttemptCheckpoint) -> list[str]:
        errors: list[str] = []
        writer = self._dataset.writer
        meta = self._dataset.meta
        state = checkpoint.memory
        try:
            meta.info = copy.deepcopy(state.info)
            meta.tasks = copy.deepcopy(state.tasks)
            meta.episodes = state.episodes
            meta.stats = copy.deepcopy(state.stats)
            meta.latest_episode = copy.deepcopy(state.meta_latest_episode)
            meta._metadata_buffer = []
            meta._pq_writer = None
            meta._finalized = state.meta_finalized
            writer._latest_episode = copy.deepcopy(state.writer_latest_episode)
            writer._current_file_start_frame = state.writer_current_file_start_frame
            writer._recorded_frames = state.writer_recorded_frames
            writer._episodes_since_last_encoding = state.writer_episodes_since_last_encoding
            writer._pq_writer = None
            writer._finalized = state.writer_finalized
            writer.episode_buffer = writer._create_episode_buffer(checkpoint.total_episodes)
            self._dataset._is_finalized = state.dataset_finalized
        except Exception as error:
            errors.append(f"could not restore in-memory state: {type(error).__name__}: {error}")
        return errors

    def _memory_is_clean(self, checkpoint: _AttemptCheckpoint) -> bool:
        writer = self._dataset.writer
        meta = self._dataset.meta
        buffer = writer.episode_buffer
        return bool(
            isinstance(buffer, dict)
            and buffer.get("size") == 0
            and buffer.get("episode_index") == checkpoint.total_episodes
            and writer._pq_writer is None
            and meta._pq_writer is None
            and not meta._metadata_buffer
            and writer._latest_episode == checkpoint.memory.writer_latest_episode
            and writer._current_file_start_frame == checkpoint.memory.writer_current_file_start_frame
            and meta.latest_episode == checkpoint.memory.meta_latest_episode
            and meta.episodes is checkpoint.memory.episodes
            and writer._recorded_frames == checkpoint.total_frames
            and writer._episodes_since_last_encoding == checkpoint.memory.writer_episodes_since_last_encoding
            and bool(writer._finalized) == checkpoint.memory.writer_finalized
            and bool(meta._finalized) == checkpoint.memory.meta_finalized
            and bool(self._dataset._is_finalized) == checkpoint.memory.dataset_finalized
            and meta.total_episodes == checkpoint.total_episodes
            and meta.total_frames == checkpoint.total_frames
            and meta.total_tasks == checkpoint.total_tasks
        )

    def _streaming_is_clean(self, checkpoint: _AttemptCheckpoint) -> bool:
        encoder = self._dataset.writer._streaming_encoder
        if encoder is None:
            return checkpoint.streaming_closed is None
        required = (
            "_episode_active",
            "_closed",
            "_video_paths",
            "_threads",
            "_frame_queues",
            "_result_queues",
            "_stop_events",
        )
        if any(not hasattr(encoder, field) for field in required):
            return False
        return bool(
            not encoder._episode_active
            and bool(encoder._closed) == checkpoint.streaming_closed
            and not encoder._video_paths
            and not encoder._threads
            and not encoder._frame_queues
            and not encoder._result_queues
            and not encoder._stop_events
        )

    def discard_attempt(self, checkpoint: object) -> AttemptCleanupReport:
        active = self._active_checkpoint
        if active is None or checkpoint is not active.token:
            self._poison("attempt cleanup received a stale or foreign checkpoint")
            raise DatasetAdapterStateError("attempt cleanup requires the exact active checkpoint")

        memory_errors: list[str] = []
        streaming_errors: list[str] = []
        writer_errors: list[str] = []
        metadata_errors: list[str] = []
        with contextlib.suppress(Exception):
            self._capture_streaming_threads(active)
        # A capture failure is retained on the checkpoint and makes the
        # streaming cleanup dimension unproven below.

        try:
            self._dataset.clear_episode_buffer(delete_images=True)
        except Exception as error:
            message = f"clear_episode_buffer failed: {type(error).__name__}: {error}"
            memory_errors.append(message)
            streaming_errors.append(message)
        try:
            self._dataset.writer.cancel_pending_videos()
        except Exception as error:
            streaming_errors.append(f"cancel_pending_videos failed: {type(error).__name__}: {error}")
        try:
            self._dataset.writer._wait_image_writer()
        except Exception as error:
            streaming_errors.append(f"image writer wait failed: {type(error).__name__}: {error}")
        with contextlib.suppress(Exception):
            self._capture_streaming_threads(active)
        streaming_errors.extend(self._streaming_thread_errors(active))
        try:
            self._dataset.writer.close_writer()
        except Exception as error:
            writer_errors.append(f"data writer close failed: {type(error).__name__}: {error}")
        try:
            self._dataset.meta._close_writer()
        except Exception as error:
            metadata_errors.append(f"metadata writer close failed: {type(error).__name__}: {error}")

        removal_errors = self._remove_post_checkpoint_paths(active)
        metadata_errors.extend(self._restore_metadata_files(active))
        memory_errors.extend(self._restore_memory(active))

        try:
            current = _scan_tree(self._root, reject_symlinks=True)
        except Exception as error:
            current = {}
            removal_errors.append(f"final tree inspection failed: {type(error).__name__}: {error}")

        tree_equal = current == dict(active.tree)
        memory_clean = not memory_errors and self._memory_is_clean(active)
        streaming_clean = (
            not streaming_errors and not removal_errors and tree_equal and self._streaming_is_clean(active)
        )
        episode_rows_unchanged = bool(
            not writer_errors
            and not removal_errors
            and not metadata_errors
            and _namespace(current, ("data",)) == _namespace(active.tree, ("data",))
            and _namespace(current, ("meta", "episodes")) == _namespace(active.tree, ("meta", "episodes"))
            and self._memory_is_clean(active)
        )
        video_references_unchanged = bool(
            not removal_errors
            and not metadata_errors
            and _namespace(current, ("videos",)) == _namespace(active.tree, ("videos",))
            and _namespace(current, ("meta",)) == _namespace(active.tree, ("meta",))
        )
        metadata_references_unchanged = bool(
            not metadata_errors
            and not removal_errors
            and _namespace(current, ("meta",)) == _namespace(active.tree, ("meta",))
            and self._memory_is_clean(active)
        )
        details = tuple(
            memory_errors
            + streaming_errors
            + writer_errors
            + metadata_errors
            + removal_errors
            + (["adapter remains poisoned after attempted episode save"] if self._save_attempted else [])
        )
        report = AttemptCleanupReport(
            memory_frames_cleared=memory_clean,
            streaming_fragments_cleared=streaming_clean,
            episode_rows_unchanged=episode_rows_unchanged,
            video_references_unchanged=video_references_unchanged,
            metadata_references_unchanged=metadata_references_unchanged,
            details=details,
        )
        self._active_checkpoint = None
        self._save_attempted = False
        if not report.proven_clean:
            self._poison("recording attempt cleanup could not be positively proven")
        return report

    def finalize(self) -> None:
        if self._finalized:
            return
        self._ensure_available()
        if self._active_checkpoint is not None:
            raise DatasetAdapterStateError("discard or save the active attempt before finalizing")
        try:
            self._dataset.finalize()
        except Exception as error:
            self._poison(f"dataset finalization failed: {type(error).__name__}: {error}")
            raise
        self._finalized = True

    def push_to_hub(self, *args: object, **kwargs: object) -> None:
        if self.poisoned:
            raise DatasetAdapterPoisonedError(self._poisoned_reason or "dataset adapter is poisoned")
        if not self._finalized or not self._dataset._is_finalized:
            raise DatasetAdapterStateError("finalize the dataset before upload")
        self._dataset.push_to_hub(*args, **kwargs)
