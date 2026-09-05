"""Production Stadia recording owner built on the guarded live session.

Importing this module does not import pygame or LeRobot.  The concrete camera,
follower, and dataset classes are resolved only after the controller startup
gate has passed.  The recording state machine itself remains in
``lelab.stadia.recording`` so it can be tested without device dependencies.
"""

from __future__ import annotations

import math
import os
import platform
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from lelab.control_session import ControlOperation, ControlSessionClaim, ControlSessionManager, MotionState

if TYPE_CHECKING:
    from lelab.control_requests import ResolvedControlRequest

from .action import NormalizedAction, validate_returned_action
from .dataset_adapter import LeRobotRecordingDatasetAdapter
from .dataset_safety import DatasetSafetyManifest, write_dataset_safety_manifest
from .mapping import RB_BUTTON, map_stadia_input
from .recording import (
    AttemptCleanupError,
    AttemptCleanupReport,
    ControllerHealth,
    ControlTick,
    FixedRateNoCatchUpPacer,
    FrameAudit,
    MotionBehavior,
    RecordingCounters,
    RecordingEvent,
    RecordingLoopResult,
    RecordingOutcome,
    RecordingPhase,
    SaturationDelta,
    record_stadia_loop,
)
from .session import (
    CONTROL_RATE_HZ,
    MAX_SNAPSHOT_AGE_S,
    FollowerBuildSpec,
    StadiaSessionConfig,
    StadiaSessionRuntimeError,
    StadiaSessionWorker,
    _thermal_failure_status,
    _thermal_status,
)
from .types import ACTION_KEYS, IntegratorCounters

_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_CAMERA_FIELDS = ("type", "camera_index", "width", "height", "fps", "fourcc", "backend")


class RecordingDatasetAdapter(Protocol):
    """The guarded adapter surface retained after raw dataset preparation."""

    @property
    def poisoned(self) -> bool: ...

    @property
    def poison_reason(self) -> str | None: ...

    def begin_attempt(self) -> object: ...

    def add_frame(self, frame: Mapping[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def discard_attempt(self, checkpoint: object) -> AttemptCleanupReport: ...

    def finalize(self) -> None: ...

    def push_to_hub(self, *args: object, **kwargs: object) -> None: ...


class DatasetFrameBuilder(Protocol):
    def __call__(
        self,
        observation: Mapping[str, Any],
        returned_action: NormalizedAction,
        task: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PreparedRecordingDataset:
    """Only guarded operations and an immutable frame-building closure survive."""

    adapter: RecordingDatasetAdapter
    frame_builder: DatasetFrameBuilder


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return numeric


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _freeze_cameras(cameras: Mapping[str, Mapping[str, object]]) -> Mapping[str, Mapping[str, object]]:
    frozen: dict[str, Mapping[str, object]] = {}
    for name, config in cameras.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("camera names must be non-empty trimmed strings")
        if not isinstance(config, Mapping):
            raise ValueError(f"camera {name!r} must be a mapping")
        frozen[name] = MappingProxyType(dict(config))
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class StadiaRecordingConfig:
    """Immutable recording settings already reconciled with the saved robot."""

    dataset_repo_id: str
    single_task: str
    num_episodes: int
    episode_time_s: float
    reset_time_s: float
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    tags: tuple[str, ...] = ()
    private: bool = False
    resume: bool = False
    streaming_encoding: bool = True
    cameras: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_repo_id(self.dataset_repo_id)
        if not isinstance(self.single_task, str) or not self.single_task.strip():
            raise ValueError("single_task must be a non-empty string")
        object.__setattr__(self, "single_task", self.single_task.strip())
        object.__setattr__(self, "num_episodes", _positive_integer(self.num_episodes, "num_episodes"))
        object.__setattr__(
            self,
            "episode_time_s",
            _positive_number(self.episode_time_s, "episode_time_s"),
        )
        object.__setattr__(self, "reset_time_s", _positive_number(self.reset_time_s, "reset_time_s"))
        if isinstance(self.fps, bool) or not isinstance(self.fps, int) or self.fps != int(CONTROL_RATE_HZ):
            raise ValueError("Stadia recording requires exactly 30 fps")
        for label in ("video", "push_to_hub", "private", "resume", "streaming_encoding"):
            if not isinstance(getattr(self, label), bool):
                raise TypeError(f"{label} must be a bool")
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not tag.strip() or tag != tag.strip() for tag in tags):
            raise ValueError("tags must contain only non-empty trimmed strings")
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "cameras", _freeze_cameras(self.cameras))


def _validate_repo_id(repo_id: object) -> str:
    if not isinstance(repo_id, str) or not repo_id or repo_id.strip() != repo_id:
        raise ValueError("dataset_repo_id must be a non-empty trimmed string")
    parts = repo_id.split("/")
    # Pinned LeRobot v0.6.0's sanity_check_dataset_name unconditionally
    # unpacks ``namespace/name``. Reject a bare local name synchronously so
    # coordinator start cannot succeed for a dataset the worker will reject.
    if len(parts) != 2 or any(
        not part or part in {".", ".."} or _REPO_COMPONENT.fullmatch(part) is None for part in parts
    ):
        raise ValueError("dataset_repo_id must use a safe namespace/name")
    if parts[1].startswith("eval_"):
        raise ValueError("dataset name must not start with 'eval_' when recording without a policy")
    return repo_id


def _default_dataset_home() -> Path:
    configured = os.getenv("HF_LEROBOT_HOME")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache / "huggingface" / "lerobot"


def resolve_recording_repo_id(
    requested: str,
    *,
    resume: bool,
    timestamp: datetime | None = None,
    dataset_home: Path | None = None,
) -> str:
    """Resolve the immutable local ID before the coordinator returns start."""

    if not isinstance(requested, str) or not requested or requested.strip() != requested:
        raise ValueError("dataset_repo_id must be a non-empty trimmed string")
    if resume:
        return _validate_repo_id(requested)

    if "/" in requested:
        namespace, name = requested.split("/", 1)
        sanitized = f"{namespace}/{re.sub(r'[^A-Za-z0-9._-]', '_', name)}"
    else:
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", requested)
    _validate_repo_id(sanitized)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"{sanitized}_{stamp}"
    home = dataset_home or _default_dataset_home()
    candidate = base
    suffix = 1
    while (home / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
        if suffix > 10_000:
            raise RuntimeError("could not allocate a collision-free dataset_repo_id")
    return candidate


def _canonical_camera_projection(record_cameras: object) -> dict[str, dict[str, object]]:
    if not isinstance(record_cameras, (list, tuple)):
        raise ValueError("saved robot cameras must be a list")
    projected: dict[str, dict[str, object]] = {}
    for camera in record_cameras:
        name = getattr(camera, "name", None)
        if not isinstance(name, str) or not name or name in projected:
            raise ValueError("saved robot camera names must be unique non-empty strings")
        values = {
            "type": getattr(camera, "type", None),
            "camera_index": getattr(camera, "camera_index", None),
            "width": getattr(camera, "width", None),
            "height": getattr(camera, "height", None),
            "fps": getattr(camera, "fps", None),
        }
        for optional in ("fourcc", "backend"):
            value = getattr(camera, optional, None)
            if value is not None:
                values[optional] = value
        projected[name] = values
    return projected


def _request_camera_projection(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("recording cameras must be a mapping")
    result: dict[str, dict[str, object]] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("recording cameras must map names to settings")
        unknown = set(raw) - set(_CAMERA_FIELDS)
        if unknown:
            raise ValueError(f"camera {name!r} contains unsupported fields: {sorted(unknown)}")
        result[name] = dict(raw)
    return result


def _default_recording_follower_factory(spec: FollowerBuildSpec) -> object:
    """Construct only the follower and its saved OpenCV cameras, lazily."""

    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    default_backends = {
        "Darwin": Cv2Backends.AVFOUNDATION,
        "Linux": Cv2Backends.V4L2,
        "Windows": Cv2Backends.DSHOW,
    }
    default_backend = default_backends.get(platform.system(), Cv2Backends.ANY)
    camera_configs: dict[str, object] = {}
    for name, raw in spec.cameras.items():
        if not isinstance(raw, Mapping) or raw.get("type") != "opencv":
            raise ValueError(f"unsupported saved camera configuration for {name!r}")
        backend_name = raw.get("backend")
        backend = Cv2Backends[str(backend_name)] if backend_name else default_backend
        camera_configs[name] = OpenCVCameraConfig(
            index_or_path=raw["camera_index"],
            backend=backend,
            fps=raw["fps"],
            width=raw["width"],
            height=raw["height"],
            fourcc=raw.get("fourcc") or None,
        )
    config = SO101FollowerConfig(
        port=spec.port,
        id=spec.calibration_id,
        cameras=camera_configs,
        use_degrees=spec.use_degrees,
        max_relative_target=dict(spec.max_relative_target),
    )
    return SO101Follower(config)


def _require_complete_local_resume(root: Path) -> None:
    """Prevent pinned metadata loading from silently falling back to the Hub."""

    meta = root / "meta"
    required = {
        "meta/info.json": meta / "info.json",
        "meta/tasks.parquet": meta / "tasks.parquet",
        "meta/stats.json": meta / "stats.json",
    }
    missing = [label for label, path in required.items() if not path.is_file() or path.is_symlink()]
    episodes = meta / "episodes"
    episode_files = (
        tuple(
            path
            for path in episodes.glob("*/*.parquet")
            if path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
        )
        if episodes.is_dir() and not episodes.is_symlink()
        else ()
    )
    if not episode_files:
        missing.append("meta/episodes/*/*.parquet")
    if root.is_symlink() or not root.is_dir() or meta.is_symlink() or not meta.is_dir():
        missing.append("non-symlinked dataset/meta directories")
    if missing:
        raise FileNotFoundError(
            f"resume requires a complete existing local dataset at {root}; "
            f"missing or unsafe: {', '.join(missing)}"
        )


def _default_dataset_preparer(
    follower: object,
    config: StadiaRecordingConfig,
) -> PreparedRecordingDataset:
    """Create/resume, validate, and immediately hide the raw pinned dataset."""

    from lerobot.common.control_utils import (
        sanity_check_dataset_name,
        sanity_check_dataset_robot_compatibility,
    )
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    action_hw = getattr(follower, "action_features", None)
    observation_hw = getattr(follower, "observation_features", None)
    if not isinstance(action_hw, Mapping) or tuple(action_hw) != ACTION_KEYS:
        raise ValueError("follower action features must be the exact ordered six-key SO-101 action")
    if not isinstance(observation_hw, Mapping):
        raise ValueError("follower observation features must be a mapping")
    action_features = hw_to_dataset_features(dict(action_hw), "action", config.video)
    observation_features = hw_to_dataset_features(dict(observation_hw), "observation", config.video)
    dataset_features = {**action_features, **observation_features}

    sanity_check_dataset_name(config.dataset_repo_id, None)
    root = HF_LEROBOT_HOME / config.dataset_repo_id
    common = {
        "root": root,
        "batch_encoding_size": 1,
        "streaming_encoding": config.streaming_encoding,
        "image_writer_processes": 0,
        # Keep construction synchronous until the guarded adapter has checked
        # the pinned implementation.  Otherwise a rejected raw dataset could
        # leave AsyncImageWriter workers alive with no retained cleanup handle.
        "image_writer_threads": 0,
    }
    if config.resume:
        _require_complete_local_resume(root)
        dataset = LeRobotDataset.resume(config.dataset_repo_id, **common)
    else:
        if root.exists():
            raise FileExistsError(f"dataset path already exists: {root}")
        dataset = LeRobotDataset.create(
            config.dataset_repo_id,
            config.fps,
            features=dataset_features,
            robot_type=getattr(follower, "robot_type", getattr(follower, "name", None)),
            use_videos=config.video,
            **common,
        )
    if getattr(dataset, "repo_id", None) != config.dataset_repo_id:
        raise ValueError("dataset identity does not match dataset_repo_id")
    sanity_check_dataset_robot_compatibility(
        dataset,
        follower,
        config.fps,
        dataset_features,
    )

    # Capture only the validated feature schema.  Once wrapped below, no raw
    # dataset method or object is retained by the worker.
    features = dict(dataset.features)

    def build_frame(
        observation: Mapping[str, Any],
        returned_action: NormalizedAction,
        task: str,
    ) -> Mapping[str, Any]:
        observation_frame = build_dataset_frame(features, dict(observation), "observation")
        action_frame = build_dataset_frame(features, returned_action.as_dict(), "action")
        return {**observation_frame, **action_frame, "task": task}

    adapter = LeRobotRecordingDatasetAdapter(dataset)
    return PreparedRecordingDataset(adapter=adapter, frame_builder=build_frame)


class _TrackedDataset:
    def __init__(self, worker: StadiaRecordingSessionWorker, adapter: RecordingDatasetAdapter) -> None:
        self.worker = worker
        self.adapter = adapter

    def begin_attempt(self) -> object:
        return self.adapter.begin_attempt()

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        self.adapter.add_frame(frame)

    def save_episode(self) -> None:
        self.adapter.save_episode()
        self.worker._episode_saved()

    def discard_attempt(self, checkpoint: object) -> AttemptCleanupReport:
        return self.adapter.discard_attempt(checkpoint)


class _RecordingEventSource:
    def __init__(self, worker: StadiaRecordingSessionWorker) -> None:
        self.worker = worker
        self._last_phase: RecordingPhase | None = None
        self._phase_started_at = worker._now()
        self._timer_fired = False

    def poll(self, phase: RecordingPhase) -> RecordingEvent | None:
        if phase is not self._last_phase:
            self._last_phase = phase
            self._phase_started_at = self.worker._now()
            self._timer_fired = False
            self.worker._observe_recording_phase(phase, self._phase_started_at)

        self.worker.manager.check_lease_expiry()
        if self.worker.claim.stop_requested.is_set() or self.worker.claim.hold_requested.is_set():
            return RecordingEvent.stop_session()
        queued = self.worker._pop_recording_event()
        if queued is not None:
            return queued
        if self._timer_fired:
            return None

        elapsed = self.worker._now() - self._phase_started_at
        if phase is RecordingPhase.RECORDING and elapsed >= self.worker.recording_config.episode_time_s:
            # Preserve LeLab's legacy contract: timeout discards the attempt;
            # only an explicit FINISH commits an episode.
            self._timer_fired = True
            return RecordingEvent.rerecord_episode()
        if phase is RecordingPhase.RESET and elapsed >= self.worker.recording_config.reset_time_s:
            self._timer_fired = True
            return RecordingEvent.finish_episode()
        return None


class _RecordingControl:
    def __init__(self, worker: StadiaRecordingSessionWorker) -> None:
        self.worker = worker
        self._pending_motion = False

    def target_or_hold(
        self,
        observation: Mapping[str, Any],
        *,
        phase: RecordingPhase,
        event: RecordingEvent | None,
    ) -> ControlTick:
        del observation
        integrator = self.worker._integrator
        if integrator is None:
            raise StadiaSessionRuntimeError("recording integrator was not initialized")
        snapshot = self.worker.reader.snapshot()
        now = self.worker._now()
        decision, scheduler_ready, _age, reason = self.worker._evaluate_snapshot(snapshot, now=now)

        if not snapshot.connected:
            health = ControllerHealth.DISCONNECTED
        elif snapshot.read_error:
            health = ControllerHealth.READ_ERROR
        elif (
            isinstance(snapshot.sampled_at, Real)
            and not isinstance(snapshot.sampled_at, bool)
            and math.isfinite(float(snapshot.sampled_at))
            and now - float(snapshot.sampled_at) > MAX_SNAPSHOT_AGE_S
        ):
            health = ControllerHealth.STALE
        elif not decision.profile_valid or not scheduler_ready:
            health = ControllerHealth.READ_ERROR
        else:
            health = ControllerHealth.HEALTHY
        rb_held = snapshot.button(RB_BUTTON)
        guard_ready = bool(
            health is ControllerHealth.HEALTHY
            and decision.profile_valid
            and (
                # Releasing the dead-man always produces a safe, recordable
                # hold in an already-valid attempt.  The neutral gate still
                # has to re-arm before RB-held motion, or before recovery/reset
                # may start a new recording attempt.
                (phase is RecordingPhase.RECORDING and not rb_held)
                or (decision.state.release_seen and decision.state.neutral_armed)
            )
        )
        movement = bool(
            guard_ready
            and phase is not RecordingPhase.RECOVERY
            and event is None
            and decision.motion_enabled
            and not self.worker.claim.hold_requested.is_set()
        )
        before = integrator.counters
        if movement:
            mapped = map_stadia_input(
                snapshot,
                motion_enabled=True,
                deadzone=self.worker.config.deadzone,
                max_step_per_tick=self.worker.config.max_step_per_tick,
                expected_guid=self.worker.config.expected_guid,
            )
            requested = integrator.integrate_one_step(mapped.deltas_dict(), enabled=True)
            action = validate_returned_action(requested.action_dict())
        else:
            action = validate_returned_action(integrator.target)
        after = integrator.counters
        saturations = _counter_delta(before, after)
        self._pending_motion = movement
        self.worker._publish_status(
            snapshot,
            MotionState.ENABLED if movement else MotionState.HOLD,
        )
        return ControlTick(
            requested_action=action,
            motion=MotionBehavior.STEP if movement else MotionBehavior.HOLD,
            controller_health=health,
            guard_ready=guard_ready,
            saturations=saturations,
        )

    def accept_sent_target(self, action: NormalizedAction) -> None:
        integrator = self.worker._integrator
        if integrator is None:
            raise StadiaSessionRuntimeError("recording integrator was not initialized")
        integrator.accept_returned_action(action.as_dict())
        self.worker._commands_sent += 1
        self.worker._movement_steps += int(self._pending_motion)
        self._pending_motion = False


def _counter_delta(before: IntegratorCounters, after: IntegratorCounters) -> SaturationDelta:
    return SaturationDelta(
        step=after.step_saturations - before.step_saturations,
        travel=after.travel_saturations - before.travel_saturations,
        endpoint=after.endpoint_saturations - before.endpoint_saturations,
    )


class _RecordingAudit:
    def __init__(self, worker: StadiaRecordingSessionWorker) -> None:
        self.worker = worker

    def record_frame(self, frame: FrameAudit) -> None:
        self.worker._record_audit(frame)


class _RecordingThermal:
    def __init__(self, worker: StadiaRecordingSessionWorker, guard: object) -> None:
        self.worker = worker
        self.guard = guard

    def check(self) -> object:
        check = getattr(self.guard, "check", None)
        if not callable(check):
            raise TypeError("thermal guard has no check method")
        try:
            snapshot = check()
            self.worker._thermal = _thermal_status(snapshot)
        except Exception as error:
            self.worker._thermal = _thermal_failure_status(
                self.guard,
                error,
                self.worker._thermal,
            )
            raise
        return snapshot


class StadiaRecordingSessionWorker(StadiaSessionWorker):
    """One owner for safe Stadia control, recording transactions, and teardown."""

    def __init__(
        self,
        *,
        manager: ControlSessionManager,
        claim: ControlSessionClaim,
        config: StadiaSessionConfig,
        recording_config: StadiaRecordingConfig,
        dataset_preparer: Callable[[object, StadiaRecordingConfig], PreparedRecordingDataset] = (
            _default_dataset_preparer
        ),
        safety_manifest_writer: Callable[[DatasetSafetyManifest], object] = (write_dataset_safety_manifest),
        pacer_factory: Callable[[Callable[[], float], Callable[[float], None]], object] | None = None,
        **session_dependencies: object,
    ) -> None:
        super().__init__(
            manager=manager,
            claim=claim,
            config=config,
            **session_dependencies,  # type: ignore[arg-type]
        )
        self.recording_config = recording_config
        self._dataset_preparer = dataset_preparer
        self._safety_manifest_writer = safety_manifest_writer
        self._pacer_factory = pacer_factory or (
            lambda clock, sleeper: FixedRateNoCatchUpPacer(clock=clock, sleeper=sleeper)
        )
        self._recording_lock = threading.RLock()
        self._recording_events: deque[RecordingEvent] = deque()
        self._adapter: RecordingDatasetAdapter | None = None
        self._frame_builder: DatasetFrameBuilder | None = None
        self._loop_result: RecordingLoopResult | None = None
        self._recording_active = True
        self._recording_phase = "preparing"
        self._recording_started_at = self._now()
        self._phase_started_at = self._recording_started_at
        self._session_ended_at: float | None = None
        self._saved_episodes = 0
        self._recording_error: str | None = None
        self._dataset_safe = True
        self._dataset_finalized = False
        self._dataset_uploaded = False
        self._latest_recording_counters = RecordingCounters()
        self._persist_dataset_safety()
        self._publish_recording_details()

    @property
    def dataset_repo_id(self) -> str:
        return self.recording_config.dataset_repo_id

    @property
    def recording_result(self) -> RecordingLoopResult | None:
        with self._recording_lock:
            return self._loop_result

    def recording_status(self) -> dict[str, Any]:
        with self._recording_lock:
            now = self._session_ended_at if self._session_ended_at is not None else self._now()
            phase_elapsed = max(0.0, now - self._phase_started_at)
            session_elapsed = max(0.0, now - self._recording_started_at)
            phase_limit: float | None = None
            if self._recording_phase == RecordingPhase.RECORDING.value:
                phase_limit = self.recording_config.episode_time_s
            elif self._recording_phase == RecordingPhase.RESET.value:
                phase_limit = self.recording_config.reset_time_s
            active = self._recording_active
            if self._recording_error is not None:
                message = self._recording_error
            elif not active:
                message = "Recording session completed"
            else:
                message = "Recording status retrieved successfully"
            return {
                "session_id": self.claim.session_id,
                "recording_active": active,
                "current_phase": self._recording_phase,
                "current_episode": min(
                    self.recording_config.num_episodes,
                    self._saved_episodes + 1,
                ),
                "total_episodes": self.recording_config.num_episodes,
                "saved_episodes": self._saved_episodes,
                "phase_elapsed_seconds": phase_elapsed,
                "phase_time_limit_s": phase_limit,
                "session_elapsed_seconds": session_elapsed,
                "session_ended": not active,
                "dataset_repo_id": self.recording_config.dataset_repo_id,
                "dataset_safe": self._dataset_safe,
                "dataset_finalized": self._dataset_finalized,
                "dataset_uploaded": self._dataset_uploaded,
                "upload_available": bool(
                    not active
                    and self._dataset_safe
                    and self._dataset_finalized
                    and not self._dataset_uploaded
                ),
                "error": self._recording_error,
                "message": message,
                "cameras": list(self.recording_config.cameras),
                "camera_feed_available": False,
                "available_controls": {
                    "stop_recording": active,
                    "exit_early": active
                    and self._recording_phase
                    in {
                        RecordingPhase.RECORDING.value,
                        RecordingPhase.RESET.value,
                        RecordingPhase.RECOVERY.value,
                    },
                    "rerecord_episode": active and self._recording_phase == RecordingPhase.RECORDING.value,
                },
            }

    def finish_episode(self) -> dict[str, Any]:
        return self._queue_command(
            RecordingEvent.finish_episode(),
            allowed={
                RecordingPhase.RECORDING.value,
                RecordingPhase.RESET.value,
                RecordingPhase.RECOVERY.value,
            },
            message="Finish episode requested",
        )

    def rerecord_episode(self) -> dict[str, Any]:
        return self._queue_command(
            RecordingEvent.rerecord_episode(),
            allowed={RecordingPhase.RECORDING.value},
            message="Re-record episode requested",
        )

    def request_stop(self, *, reason: str = "recording stop requested") -> dict[str, Any]:
        status = self.recording_status()
        if not status["recording_active"]:
            return self._command_response(False, "No recording session is active")
        try:
            self.manager.request_stop(self.claim.session_id, reason=reason)
        except Exception as error:
            return self._command_response(False, f"Stop request failed: {type(error).__name__}: {error}")
        with self._recording_lock:
            self._recording_events.appendleft(RecordingEvent.stop_session())
            self._recording_phase = "stopping"
            self._phase_started_at = self._now()
        self._publish_recording_details()
        return self._command_response(True, "Recording stop requested")

    def stop_recording(self, *, reason: str = "recording stop requested") -> dict[str, Any]:
        return self.request_stop(reason=reason)

    def _queue_command(
        self,
        event: RecordingEvent,
        *,
        allowed: set[str],
        message: str,
    ) -> dict[str, Any]:
        with self._recording_lock:
            if not self._recording_active:
                return self._command_response(False, "No recording session is active")
            if self._recording_phase not in allowed:
                return self._command_response(
                    False,
                    f"Command is unavailable during {self._recording_phase}",
                )
            self._recording_events.append(event)
        return self._command_response(True, message)

    def _command_response(self, success: bool, message: str) -> dict[str, Any]:
        phase = self.recording_status()["current_phase"]
        return {
            "success": success,
            "message": message,
            "phase": phase,
            "current_phase": phase,
        }

    def _pop_recording_event(self) -> RecordingEvent | None:
        with self._recording_lock:
            return self._recording_events.popleft() if self._recording_events else None

    def _observe_recording_phase(self, phase: RecordingPhase, started_at: float) -> None:
        with self._recording_lock:
            self._recording_phase = phase.value
            self._phase_started_at = started_at
        self._publish_recording_details()

    def _episode_saved(self) -> None:
        with self._recording_lock:
            self._saved_episodes += 1

    def _persist_dataset_safety(self) -> None:
        with self._recording_lock:
            manifest = DatasetSafetyManifest(
                dataset_repo_id=self.recording_config.dataset_repo_id,
                session_id=self.claim.session_id,
                dataset_safe=self._dataset_safe,
                dataset_finalized=self._dataset_finalized,
                dataset_uploaded=self._dataset_uploaded,
                saved_episodes=self._saved_episodes,
                error=self._recording_error,
            )
        self._safety_manifest_writer(manifest)

    def _retain_manifest_failure(self, error: BaseException) -> str:
        detail = f"dataset safety manifest failed: {type(error).__name__}: {error}"
        with self._recording_lock:
            self._dataset_safe = False
            if self._recording_error is None:
                self._recording_error = detail
            elif detail not in self._recording_error:
                self._recording_error = f"{self._recording_error}; {detail}"
        return detail

    def _record_audit(self, frame: FrameAudit) -> None:
        with self._recording_lock:
            self._latest_recording_counters = frame.counters
        self._missed_ticks = frame.counters.missed_ticks
        if self._last_snapshot is not None:
            self._publish_status(
                self._last_snapshot,
                MotionState.ENABLED if frame.motion is MotionBehavior.STEP else MotionState.HOLD,
            )
        self._publish_recording_details()

    def _publish_recording_details(self) -> None:
        status = self.recording_status()
        manager_status = self.manager.status_for(self.claim.session_id, check_expiry=False)
        if manager_status is not None and not manager_status.terminal:
            self.manager.merge_details(self.claim.session_id, {"recording": status})

    def _prepare_dependencies(self, follower: object) -> None:
        try:
            prepared = self._dataset_preparer(follower, self.recording_config)
            if not isinstance(prepared, PreparedRecordingDataset):
                raise TypeError("dataset_preparer must return PreparedRecordingDataset")
            self._adapter = prepared.adapter
            self._frame_builder = prepared.frame_builder
        except Exception as error:
            with self._recording_lock:
                self._dataset_safe = False
                self._recording_error = f"{type(error).__name__}: {error}"
            try:
                self._persist_dataset_safety()
            except Exception as manifest_error:
                self._retain_manifest_failure(manifest_error)
            self._publish_recording_details()
            raise

    def _run_control_loop(self, thermal_guard: object) -> str | None:
        if self._integrator is None or self._robot is None:
            raise StadiaSessionRuntimeError("recording dependencies were not initialized")
        if self._adapter is None or self._frame_builder is None:
            raise StadiaSessionRuntimeError("recording dataset was not prepared")
        events = _RecordingEventSource(self)
        control = _RecordingControl(self)
        audit = _RecordingAudit(self)
        dataset = _TrackedDataset(self, self._adapter)
        thermal = _RecordingThermal(self, thermal_guard)
        initial_action = validate_returned_action(self._integrator.target)
        try:
            result = record_stadia_loop(
                follower=self._robot,  # type: ignore[arg-type]
                control=control,
                dataset=dataset,
                events=events,
                frame_builder=self._frame_builder,
                audit=audit,
                thermal_guard=thermal,
                initial_action=initial_action,
                task=self.recording_config.single_task,
                num_episodes=self.recording_config.num_episodes,
                pacer=self._pacer_factory(self._clock, self._sleeper),  # type: ignore[arg-type]
                thermal_check_interval_ticks=int(CONTROL_RATE_HZ),
            )
        except AttemptCleanupError as error:
            self._mark_recording_error(str(error), unsafe=True)
            raise StadiaSessionRuntimeError(str(error)) from error
        except Exception as error:
            self._mark_recording_error(f"{type(error).__name__}: {error}", unsafe=True)
            raise

        with self._recording_lock:
            self._loop_result = result
            self._saved_episodes = result.saved_episodes
            self._latest_recording_counters = result.counters
        self._missed_ticks = result.counters.missed_ticks
        if result.outcome is RecordingOutcome.ERROR:
            detail = _terminal_event_text(result)
            # The accepted loop has already discarded the current attempt and
            # positively proved all five cleanup dimensions before it can
            # return ERROR.  Earlier committed episodes remain structurally
            # safe when the adapter is not poisoned, but this worker still
            # never finalizes or uploads after a loop error.
            self._mark_recording_error(
                detail,
                unsafe=bool(getattr(self._adapter, "poisoned", False)),
            )
            raise StadiaSessionRuntimeError(detail)
        with self._recording_lock:
            self._recording_phase = "stopping"
            self._phase_started_at = self._now()
        self._publish_recording_details()
        if result.outcome is RecordingOutcome.COMPLETED:
            return f"recording completed after {result.saved_episodes} saved episodes"
        manager_status = self.manager.status_for(self.claim.session_id, check_expiry=False)
        if manager_status is not None and manager_status.stop_reason:
            return manager_status.stop_reason
        return f"recording stopped after {result.saved_episodes} saved episodes"

    def _mark_recording_error(self, error: str, *, unsafe: bool) -> None:
        with self._recording_lock:
            self._recording_error = error
            self._recording_phase = "error"
            self._phase_started_at = self._now()
            if unsafe:
                self._dataset_safe = False
        if unsafe:
            try:
                self._persist_dataset_safety()
            except Exception as manifest_error:
                self._retain_manifest_failure(manifest_error)
        self._publish_recording_details()

    def _on_session_failure(self, error: Exception) -> None:
        # This hook also covers neutral timeout and follower setup failures
        # that happen before the recording loop can return typed evidence.
        if self._recording_error is None:
            self._mark_recording_error(f"{type(error).__name__}: {error}", unsafe=True)

    def _teardown(
        self,
        *,
        bus: object | None,
        bus_connect_attempted: bool,
        bus_connect_succeeded: bool,
        connected_cameras: list[tuple[str, object]],
        reader_start_attempted: bool,
    ):  # type: ignore[no-untyped-def]
        torque, errors = super()._teardown(
            bus=bus,
            bus_connect_attempted=bus_connect_attempted,
            bus_connect_succeeded=bus_connect_succeeded,
            connected_cameras=connected_cameras,
            reader_start_attempted=reader_start_attempted,
        )
        adapter = self._adapter
        result = self.recording_result
        if adapter is not None and bool(getattr(adapter, "poisoned", False)):
            reason = getattr(adapter, "poison_reason", None) or "dataset adapter is poisoned"
            self._mark_recording_error(str(reason), unsafe=True)
        if (
            adapter is not None
            and result is not None
            and result.outcome in {RecordingOutcome.COMPLETED, RecordingOutcome.STOPPED}
            and result.saved_episodes > 0
            and self._dataset_safe
            and not bool(getattr(adapter, "poisoned", False))
        ):
            try:
                adapter.finalize()
                with self._recording_lock:
                    self._dataset_finalized = True
            except Exception as error:
                detail = f"dataset finalize failed: {type(error).__name__}: {error}"
                errors.append(detail)
                self._mark_recording_error(detail, unsafe=True)
            if self.recording_config.push_to_hub and self._dataset_finalized and self._dataset_safe:
                tags = list(self.recording_config.tags)
                if "LeLab" not in tags:
                    tags.append("LeLab")
                try:
                    adapter.push_to_hub(tags=tags, private=self.recording_config.private)
                    with self._recording_lock:
                        self._dataset_uploaded = True
                except Exception as error:
                    detail = f"dataset upload failed: {type(error).__name__}: {error}"
                    errors.append(detail)
                    with self._recording_lock:
                        self._recording_error = detail

        with self._recording_lock:
            self._recording_active = False
            self._session_ended_at = self._now()
            if self._recording_error is None and errors:
                self._recording_error = "; ".join(errors)
            self._recording_phase = "error" if self._recording_error is not None else "completed"
            self._phase_started_at = self._session_ended_at
        try:
            self._persist_dataset_safety()
        except Exception as error:
            errors.append(self._retain_manifest_failure(error))
        try:
            self._publish_recording_details()
        except Exception as error:
            errors.append(f"recording status publication failed: {type(error).__name__}: {error}")
        return torque, errors


def _terminal_event_text(result: RecordingLoopResult) -> str:
    event = result.terminal_event
    reason = event.reason.value if event.reason is not None else event.kind.value
    return f"recording loop {result.outcome.value}: {reason}" + (f": {event.detail}" if event.detail else "")


def build_stadia_recording_worker(
    manager: ControlSessionManager,
    claim: ControlSessionClaim,
    resolved: ResolvedControlRequest,
) -> StadiaRecordingSessionWorker:
    """Build the coordinator-owned production worker without touching devices."""

    operation = getattr(getattr(resolved, "operation", None), "value", getattr(resolved, "operation", None))
    if operation != ControlOperation.STADIA_RECORDING.value:
        raise ValueError("resolved request is not a Stadia recording operation")
    if ControlOperation.STADIA_RECORDING.value not in claim.resource_keys:
        raise ValueError("claim does not own the Stadia recording resource")
    record = resolved.record_model()
    if record.teleoperator_type != "stadia":
        raise ValueError("Stadia recording requires a saved Stadia robot")
    metadata = resolved.metadata_dict()
    if metadata.get("test_mode") is not False:
        raise ValueError("test_mode is unsupported for production Stadia recording")

    canonical_cameras = _canonical_camera_projection(record.cameras)
    requested_cameras = _request_camera_projection(metadata.get("cameras", {}))
    if requested_cameras != canonical_cameras:
        raise ValueError("recording cameras do not exactly match the saved robot configuration")
    final_repo_id = resolve_recording_repo_id(
        metadata["dataset_repo_id"],
        resume=metadata["resume"],
    )
    recording_config = StadiaRecordingConfig(
        dataset_repo_id=final_repo_id,
        single_task=metadata["single_task"],
        num_episodes=metadata["num_episodes"],
        episode_time_s=metadata["episode_time_s"],
        reset_time_s=metadata["reset_time_s"],
        fps=metadata["fps"],
        video=metadata["video"],
        push_to_hub=metadata["push_to_hub"],
        tags=tuple(metadata["tags"]),
        private=metadata["private"],
        resume=metadata["resume"],
        streaming_encoding=metadata["streaming_encoding"],
        cameras=canonical_cameras,
    )
    return StadiaRecordingSessionWorker(
        manager=manager,
        claim=claim,
        config=StadiaSessionConfig(
            follower_port=record.follower.port,
            follower_calibration=record.follower.calibration,
            expected_guid=record.stadia.guid,
            deadzone=record.stadia.deadzone,
            max_step_per_tick=record.stadia.max_step_per_tick,
            cameras=canonical_cameras,
        ),
        recording_config=recording_config,
        follower_factory=_default_recording_follower_factory,
    )


__all__ = [
    "PreparedRecordingDataset",
    "StadiaRecordingConfig",
    "StadiaRecordingSessionWorker",
    "build_stadia_recording_worker",
    "resolve_recording_repo_id",
]
