from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from lelab.stadia.dataset_adapter import (
    _PINNED_SOURCE_HASHES,
    DatasetAdapterCompatibilityError,
    DatasetAdapterPoisonedError,
    DatasetAdapterStateError,
    LeRobotRecordingDatasetAdapter,
)


@dataclass
class _Info:
    total_episodes: int
    total_frames: int
    total_tasks: int
    splits: dict[str, str]


class _Handle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Thread:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive


class _Encoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._episode_active = False
        self._closed = False
        self._video_paths: dict[str, Path] = {}
        self._threads: dict[str, _Thread] = {}
        self._frame_queues: dict[str, object] = {}
        self._result_queues: dict[str, object] = {}
        self._stop_events: dict[str, object] = {}

    def start(self, episode_index: int) -> None:
        temp = self.root / f"streaming-{episode_index:03d}"
        temp.mkdir()
        video = temp / "camera_streaming.mp4"
        video.write_bytes(b"pending-video")
        self._episode_active = True
        self._video_paths = {"camera": video}
        self._threads = {"camera": _Thread()}
        self._frame_queues = {"camera": object()}
        self._result_queues = {"camera": object()}
        self._stop_events = {"camera": object()}

    def cancel_episode(self) -> None:
        for video in self._video_paths.values():
            shutil.rmtree(video.parent, ignore_errors=True)
        for thread in self._threads.values():
            thread.alive = False
        self._episode_active = False
        self._video_paths.clear()
        self._threads.clear()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._stop_events.clear()

    def lose_handles_without_stopping(self) -> None:
        for video in self._video_paths.values():
            shutil.rmtree(video.parent, ignore_errors=True)
        self._episode_active = False
        self._video_paths.clear()
        self._threads.clear()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._stop_events.clear()


class _Meta:
    def __init__(self, root: Path, *, committed: bool) -> None:
        self.root = root
        self.info = _Info(
            total_episodes=1 if committed else 0,
            total_frames=2 if committed else 0,
            total_tasks=1 if committed else 0,
            splits={"train": "0:1"} if committed else {},
        )
        self.tasks = {"committed task": 0} if committed else None
        self.episodes = [self._episode_row(0)] if committed else None
        self.stats = {"action": {"count": 2}} if committed else None
        self.latest_episode = None
        self._pq_writer: _Handle | None = None
        self._metadata_buffer: list[dict[str, object]] = []
        self._finalized = False
        self.data_path = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        self.video_path = "videos/camera/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        self.video_keys = ["camera"]
        self.chunks_size = 1000

    @staticmethod
    def _episode_row(index: int) -> dict[str, int]:
        return {
            "episode_index": index,
            "data/chunk_index": 0,
            "data/file_index": index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": index,
            "videos/camera/chunk_index": 0,
            "videos/camera/file_index": index,
        }

    @property
    def total_episodes(self) -> int:
        return self.info.total_episodes

    @property
    def total_frames(self) -> int:
        return self.info.total_frames

    @property
    def total_tasks(self) -> int:
        return self.info.total_tasks

    def _close_writer(self) -> None:
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None
        self._metadata_buffer.clear()

    def _load_metadata(self) -> None:
        self.episodes = [self._episode_row(index) for index in range(self.info.total_episodes)] or None


class _Writer:
    def __init__(self, meta: _Meta, root: Path, *, streaming: bool) -> None:
        self._meta = meta
        self._root = root
        self.image_writer = None
        self._pq_writer: _Handle | None = None
        self._latest_episode = None
        self._current_file_start_frame = None
        self._streaming_encoder = _Encoder(root) if streaming else None
        self._batch_encoding_size = 1
        self._episodes_since_last_encoding = 0
        self._recorded_frames = meta.total_frames
        self._finalized = False
        self.episode_buffer = self._create_episode_buffer()

    def _create_episode_buffer(self, episode_index: int | None = None) -> dict[str, object]:
        return {
            "size": 0,
            "task": [],
            "frames": [],
            "episode_index": self._meta.total_episodes if episode_index is None else episode_index,
        }

    def _wait_image_writer(self) -> None:
        return None

    def cancel_pending_videos(self) -> None:
        if self._streaming_encoder is not None:
            self._streaming_encoder.cancel_episode()

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        del delete_images
        self.cancel_pending_videos()
        self.episode_buffer = self._create_episode_buffer()

    def close_writer(self) -> None:
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None


class _FakeDataset:
    def __init__(
        self,
        root: Path,
        *,
        committed: bool = False,
        streaming: bool = False,
        fail_stage: str | None = None,
    ) -> None:
        root.mkdir(parents=True)
        (root / "meta").mkdir()
        self.root = root
        self.meta = _Meta(root, committed=committed)
        self.writer = _Writer(self.meta, root, streaming=streaming)
        self.reader = None
        self._is_finalized = False
        self.fail_stage = fail_stage
        self.finalize_calls = 0
        self.push_calls = 0
        self._write_info()
        if committed:
            self._write_committed_files()

    def _write_info(self) -> None:
        (self.root / "meta/info.json").write_text(
            json.dumps(
                {
                    "total_episodes": self.meta.total_episodes,
                    "total_frames": self.meta.total_frames,
                    "total_tasks": self.meta.total_tasks,
                },
                sort_keys=True,
            )
        )

    def _write_committed_files(self) -> None:
        paths = {
            "data/chunk-000/file-000.parquet": b"committed-data",
            "videos/camera/chunk-000/file-000.mp4": b"committed-video",
            "meta/episodes/chunk-000/file-000.parquet": b"committed-episode-row",
            "meta/tasks.parquet": b"committed-task",
            "meta/stats.json": b"committed-stats",
        }
        for relative, content in paths.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def add_frame(self, frame: dict[str, object]) -> None:
        if self.writer.episode_buffer["size"] == 0 and self.writer._streaming_encoder is not None:
            self.writer._streaming_encoder.start(self.meta.total_episodes)
        self.writer.episode_buffer["frames"].append(dict(frame))
        self.writer.episode_buffer["task"].append(frame["task"])
        self.writer.episode_buffer["size"] += 1

    def _raise_at(self, stage: str) -> None:
        if self.fail_stage == stage:
            raise RuntimeError(f"injected failure after {stage}")

    def save_episode(self) -> None:
        index = self.meta.total_episodes
        size = self.writer.episode_buffer["size"]
        tasks = set(self.writer.episode_buffer["task"])

        if self.meta.tasks is None:
            self.meta.tasks = {}
        for task in tasks:
            self.meta.tasks.setdefault(task, len(self.meta.tasks))
        self.meta.info.total_tasks = len(self.meta.tasks)
        (self.root / "meta/tasks.parquet").write_bytes(repr(self.meta.tasks).encode())
        self._raise_at("tasks")

        data_path = self.root / f"data/chunk-000/file-{index:03d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(f"attempt-data-{index}".encode())
        self.writer._pq_writer = _Handle()
        self.writer._latest_episode = {
            "data/chunk_index": 0,
            "data/file_index": index,
            "index": list(range(self.meta.total_frames, self.meta.total_frames + size)),
        }
        self.writer._recorded_frames += size
        self._raise_at("data")

        if self.writer._streaming_encoder is not None:
            if self.fail_stage == "handle_loss":
                self.writer._streaming_encoder.lose_handles_without_stopping()
                self._raise_at("handle_loss")
            self.writer._streaming_encoder.cancel_episode()
        video_path = self.root / f"videos/camera/chunk-000/file-{index:03d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"attempt-video-{index}".encode())
        self._raise_at("video")

        episode_path = self.root / f"meta/episodes/chunk-000/file-{index:03d}.parquet"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.write_bytes(f"attempt-episode-{index}".encode())
        self.meta.latest_episode = {
            "episode_index": [index],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [index],
            "videos/camera/chunk_index": [0],
            "videos/camera/file_index": [index],
        }
        self.meta._metadata_buffer.append({"episode_index": index})
        self.meta._pq_writer = _Handle()
        self._raise_at("episode_metadata")

        self.meta.info.total_episodes += 1
        self.meta.info.total_frames += size
        self.meta.info.splits = {"train": f"0:{self.meta.total_episodes}"}
        self.meta.stats = {"action": {"count": self.meta.total_frames}}
        self._write_info()
        (self.root / "meta/stats.json").write_text(json.dumps(self.meta.stats))
        self._raise_at("metadata")

        self.writer.clear_episode_buffer(delete_images=True)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self.writer.clear_episode_buffer(delete_images)

    def finalize(self) -> None:
        self.finalize_calls += 1
        self.writer.close_writer()
        self.meta._close_writer()
        self.writer._finalized = True
        self.meta._finalized = True
        self._is_finalized = True

    def push_to_hub(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.push_calls += 1


def _adapter(dataset: _FakeDataset) -> LeRobotRecordingDatasetAdapter:
    return LeRobotRecordingDatasetAdapter(dataset, require_pinned_runtime=False)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_module_import_does_not_import_lerobot() -> None:
    code = "import sys; import lelab.stadia.dataset_adapter; print(any(x == 'lerobot' or x.startswith('lerobot.') for x in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_adapter_does_not_expose_raw_dataset_escape_hatch(tmp_path: Path) -> None:
    adapter = _adapter(_FakeDataset(tmp_path / "dataset"))

    assert not hasattr(adapter, "dataset")


def test_pinned_provenance_covers_private_io_and_path_semantics() -> None:
    assert "lerobot/datasets/io_utils.py" in _PINNED_SOURCE_HASHES
    assert "lerobot/datasets/utils.py" in _PINNED_SOURCE_HASHES


def test_normal_discard_proves_all_five_cleanup_dimensions(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})

    report = adapter.discard_attempt(checkpoint)

    assert report.proven_clean
    assert not adapter.poisoned
    assert dataset.writer.episode_buffer["size"] == 0


def test_streaming_discard_removes_temp_fragment_and_thread_state(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", streaming=True)
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})
    assert list(dataset.root.glob("streaming-*"))

    report = adapter.discard_attempt(checkpoint)

    assert report.proven_clean
    assert not list(dataset.root.glob("streaming-*"))
    assert not dataset.writer._streaming_encoder._episode_active
    assert not dataset.writer._streaming_encoder._video_paths


def test_streaming_discard_fails_closed_when_encoder_thread_remains_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", streaming=True)
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})
    encoder = dataset.writer._streaming_encoder
    assert encoder is not None
    monkeypatch.setattr(encoder, "cancel_episode", encoder.lose_handles_without_stopping)
    report = adapter.discard_attempt(checkpoint)

    assert not report.proven_clean
    assert not report.streaming_fragments_cleared
    assert any("remained alive" in detail for detail in report.details)
    assert adapter.poisoned


def test_partial_save_retains_thread_handle_after_writer_loses_tracking(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", streaming=True, fail_stage="handle_loss")
    adapter = _adapter(dataset)
    before = _file_bytes(dataset.root)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})

    with pytest.raises(RuntimeError, match="handle_loss"):
        adapter.save_episode()
    report = adapter.discard_attempt(checkpoint)

    assert not report.proven_clean
    assert not report.streaming_fragments_cleared
    assert any("remained alive" in detail for detail in report.details)
    assert _file_bytes(dataset.root) == before
    assert adapter.poisoned
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.finalize()
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.push_to_hub()


def test_post_save_verification_rejects_lost_live_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", streaming=True)
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})
    encoder = dataset.writer._streaming_encoder
    assert encoder is not None
    monkeypatch.setattr(encoder, "cancel_episode", encoder.lose_handles_without_stopping)

    with pytest.raises(DatasetAdapterCompatibilityError, match="thread remained alive"):
        adapter.save_episode()
    report = adapter.discard_attempt(checkpoint)

    assert not report.proven_clean
    assert not report.streaming_fragments_cleared
    assert adapter.poisoned


@pytest.mark.parametrize("stage", ["tasks", "data", "video", "episode_metadata", "metadata"])
def test_partial_save_rolls_back_every_mutation_stage_but_poison_remains(tmp_path: Path, stage: str) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", committed=True, streaming=True, fail_stage=stage)
    adapter = _adapter(dataset)
    before = _file_bytes(dataset.root)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "new task"})

    with pytest.raises(RuntimeError, match=stage):
        adapter.save_episode()
    report = adapter.discard_attempt(checkpoint)

    assert report.proven_clean, report.details
    assert _file_bytes(dataset.root) == before
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 2
    assert adapter.poisoned
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.finalize()
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.push_to_hub()
    assert dataset.finalize_calls == 0
    assert dataset.push_calls == 0


def test_unproven_path_cleanup_reports_failure_and_blocks_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", fail_stage="data")
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})
    with pytest.raises(RuntimeError):
        adapter.save_episode()
    monkeypatch.setattr(
        adapter,
        "_remove_post_checkpoint_paths",
        lambda unused: ["injected removal failure"],
    )

    report = adapter.discard_attempt(checkpoint)

    assert not report.proven_clean
    assert not report.streaming_fragments_cleared
    assert not report.episode_rows_unchanged
    assert any("injected removal failure" in detail for detail in report.details)
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.finalize()


def test_successful_save_preserves_committed_files_and_allows_next_attempt(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", committed=True, streaming=True)
    adapter = _adapter(dataset)
    committed_data = (dataset.root / "data/chunk-000/file-000.parquet").read_bytes()

    adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "new task"})
    adapter.save_episode()

    assert not adapter.poisoned
    assert dataset.meta.total_episodes == 2
    assert (dataset.root / "data/chunk-000/file-000.parquet").read_bytes() == committed_data
    assert (dataset.root / "data/chunk-000/file-001.parquet").is_file()

    next_checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [3.0], "action": [4.0], "task": "new task"})
    report = adapter.discard_attempt(next_checkpoint)
    assert report.proven_clean
    assert dataset.meta.total_episodes == 2

    adapter.finalize()
    adapter.finalize()
    adapter.push_to_hub(branch="fake")
    assert dataset.finalize_calls == 1
    assert dataset.push_calls == 1


def test_foreign_checkpoint_fails_closed(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    adapter = _adapter(dataset)
    adapter.begin_attempt()

    with pytest.raises(DatasetAdapterStateError):
        adapter.discard_attempt(object())

    assert adapter.poisoned


def test_constructor_rejects_private_api_drift(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    del dataset.writer._latest_episode

    with pytest.raises(DatasetAdapterCompatibilityError, match="_latest_episode"):
        _adapter(dataset)


def test_constructor_rejects_symlinked_dataset_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    dataset = _FakeDataset(real_root)
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    dataset.root = alias
    dataset.meta.root = alias
    dataset.writer._root = alias

    with pytest.raises(DatasetAdapterCompatibilityError, match="symlink"):
        _adapter(dataset)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_path", "../outside-{file_index}.parquet"),
        ("video_path", "../outside-{video_key}-{file_index}.mp4"),
    ],
)
def test_begin_rejects_path_templates_that_escape_root(tmp_path: Path, field: str, value: str) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    setattr(dataset.meta, field, value)
    adapter = _adapter(dataset)

    with pytest.raises(DatasetAdapterCompatibilityError, match="escapes"):
        adapter.begin_attempt()

    assert not (tmp_path / "outside-0.parquet").exists()


def test_begin_rejects_next_path_collision_before_any_attempt_write(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset", committed=True)
    dataset.meta.data_path = "data/chunk-000/file-000.parquet"
    adapter = _adapter(dataset)
    committed = (dataset.root / "data/chunk-000/file-000.parquet").read_bytes()

    with pytest.raises(DatasetAdapterCompatibilityError, match="already exists"):
        adapter.begin_attempt()

    assert (dataset.root / "data/chunk-000/file-000.parquet").read_bytes() == committed


def test_save_path_collision_after_begin_poison_and_blocks_publish(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame({"observation": [1.0], "action": [2.0], "task": "pick"})
    collision = dataset.root / "data/chunk-000/file-000.parquet"
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"appeared after checkpoint")

    with pytest.raises(DatasetAdapterCompatibilityError, match="appeared during the attempt"):
        adapter.save_episode()

    assert adapter.poisoned
    report = adapter.discard_attempt(checkpoint)
    assert report.proven_clean, report.details
    assert not collision.exists()
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.finalize()
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.push_to_hub()
    assert dataset.finalize_calls == 0
    assert dataset.push_calls == 0


def test_symlink_created_during_attempt_is_unlinked_but_cleanup_is_unproven(tmp_path: Path) -> None:
    dataset = _FakeDataset(tmp_path / "dataset")
    adapter = _adapter(dataset)
    checkpoint = adapter.begin_attempt()
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive")
    (dataset.root / "escape").symlink_to(outside)

    report = adapter.discard_attempt(checkpoint)

    assert not report.proven_clean
    assert outside.read_text() == "must survive"
    assert not (dataset.root / "escape").exists()
    assert adapter.poisoned


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="pinned lerobot is unavailable")
def test_pinned_runtime_rejects_relabelled_fake_class_identity(tmp_path: Path) -> None:
    fake_dataset_type = type("LeRobotDataset", (_FakeDataset,), {})
    fake_dataset_type.__module__ = "lerobot.datasets.lerobot_dataset"
    dataset = fake_dataset_type(tmp_path / "dataset")

    with pytest.raises(DatasetAdapterCompatibilityError, match="exact class identity"):
        LeRobotRecordingDatasetAdapter(dataset)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="pinned lerobot is unavailable")
def test_pinned_runtime_rejects_loaded_semantic_module_path_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "real-provenance-dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/stadia-adapter-provenance-test",
        fps=30,
        features=features,
        root=root,
        use_videos=False,
    )
    io_utils = importlib.import_module("lerobot.datasets.io_utils")
    monkeypatch.setattr(io_utils, "__file__", str(Path(__file__).resolve()))

    with pytest.raises(DatasetAdapterCompatibilityError, match="does not match its installed source"):
        LeRobotRecordingDatasetAdapter(dataset)
    dataset.finalize()


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="pinned lerobot is unavailable")
def test_real_pinned_lerobot_no_video_create_save_discard_and_finalize(tmp_path: Path) -> None:
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "real-dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/stadia-adapter-test",
        fps=30,
        features=features,
        root=root,
        use_videos=False,
    )
    adapter = LeRobotRecordingDatasetAdapter(dataset)
    assert not hasattr(adapter, "dataset")

    adapter.begin_attempt()
    adapter.add_frame(
        {
            "observation.state": np.array([1.0], dtype=np.float32),
            "action": np.array([2.0], dtype=np.float32),
            "task": "test returned action",
        }
    )
    adapter.save_episode()
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 1

    adapter.begin_attempt()
    adapter.add_frame(
        {
            "observation.state": np.array([3.0], dtype=np.float32),
            "action": np.array([4.0], dtype=np.float32),
            "task": "test returned action",
        }
    )
    adapter.save_episode()
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 2
    assert len(list((root / "data").rglob("*.parquet"))) == 2

    checkpoint = adapter.begin_attempt()
    adapter.add_frame(
        {
            "observation.state": np.array([5.0], dtype=np.float32),
            "action": np.array([6.0], dtype=np.float32),
            "task": "discard me",
        }
    )
    report = adapter.discard_attempt(checkpoint)
    assert report.proven_clean, report.details
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 2

    adapter.finalize()
    assert dataset._is_finalized

    resumed = LeRobotDataset.resume(repo_id="local/stadia-adapter-test", root=root)
    resumed_adapter = LeRobotRecordingDatasetAdapter(resumed)
    resumed_adapter.begin_attempt()
    resumed_adapter.add_frame(
        {
            "observation.state": np.array([7.0], dtype=np.float32),
            "action": np.array([8.0], dtype=np.float32),
            "task": "resumed episode",
        }
    )
    resumed_adapter.save_episode()
    assert resumed.meta.total_episodes == 3
    assert resumed.meta.total_frames == 3
    assert len(list((root / "data").rglob("*.parquet"))) == 3
    resumed_adapter.finalize()


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="pinned lerobot is unavailable")
def test_real_pinned_partial_save_rolls_back_and_permanently_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "real-partial-dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/stadia-adapter-partial-test",
        fps=30,
        features=features,
        root=root,
        use_videos=False,
    )
    adapter = LeRobotRecordingDatasetAdapter(dataset)
    before = _file_bytes(root)
    checkpoint = adapter.begin_attempt()
    adapter.add_frame(
        {
            "observation.state": np.array([1.0], dtype=np.float32),
            "action": np.array([2.0], dtype=np.float32),
            "task": "fail after parquet",
        }
    )

    def fail_metadata_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected real metadata save failure")

    monkeypatch.setattr(dataset.meta, "save_episode", fail_metadata_save)
    with pytest.raises(RuntimeError, match="real metadata save failure"):
        adapter.save_episode()
    report = adapter.discard_attempt(checkpoint)

    assert report.proven_clean, report.details
    assert _file_bytes(root) == before
    assert dataset.meta.total_episodes == 0
    assert dataset.meta.total_frames == 0
    assert adapter.poisoned
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.finalize()
    with pytest.raises(DatasetAdapterPoisonedError):
        adapter.push_to_hub()


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="pinned lerobot is unavailable")
def test_real_pinned_streaming_threads_are_quiescent_after_discard_and_save(tmp_path: Path) -> None:
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "real-streaming-dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "observation.images.camera": {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/stadia-adapter-streaming-test",
        fps=10,
        features=features,
        root=root,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=1,
    )
    adapter = LeRobotRecordingDatasetAdapter(dataset)

    checkpoint = adapter.begin_attempt()
    for value in (1, 2):
        adapter.add_frame(
            {
                "observation.state": np.array([value], dtype=np.float32),
                "observation.images.camera": np.full((32, 32, 3), value, dtype=np.uint8),
                "action": np.array([value], dtype=np.float32),
                "task": "discard streaming",
            }
        )
    report = adapter.discard_attempt(checkpoint)
    assert report.proven_clean, report.details

    adapter.begin_attempt()
    for value in (3, 4):
        adapter.add_frame(
            {
                "observation.state": np.array([value], dtype=np.float32),
                "observation.images.camera": np.full((32, 32, 3), value, dtype=np.uint8),
                "action": np.array([value], dtype=np.float32),
                "task": "save streaming",
            }
        )
    adapter.save_episode()

    encoder = dataset.writer._streaming_encoder
    assert encoder is not None
    assert not encoder._episode_active
    assert not encoder._threads
    assert not encoder._video_paths
    assert dataset.meta.total_episodes == 1
    assert len(list((root / "videos").rglob("*.mp4"))) == 1
    adapter.finalize()
