"""Fake-only integration tests for the production Stadia recording owner."""

from __future__ import annotations

import subprocess
import sys
import types
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lelab.control_session import (
    SO101_MOTOR_NAMES,
    ControlOperation,
    ControlSessionManager,
    ControlState,
)
from lelab.stadia.recording import (
    AttemptCleanupReport,
    HardInvalidationReason,
    RecordingOutcome,
)
from lelab.stadia.recording_session import (
    PreparedRecordingDataset,
    StadiaRecordingConfig,
    StadiaRecordingSessionWorker,
    build_stadia_recording_worker,
    resolve_recording_repo_id,
)
from lelab.stadia.session import StadiaSessionConfig, derive_calibrated_endpoint_bounds
from lelab.stadia.thermal_safety import ConfirmedTemperatureStopError, ThermalSnapshot
from lelab.stadia.types import ACTION_KEYS, STADIA_PRODUCT_NAME, ControllerLayout, StadiaSnapshot


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def snapshot(
    sequence: int,
    *,
    clock: FakeClock,
    rb: bool = False,
    left_x: float = 0.0,
    triggers: tuple[float, float] = (-1.0, -1.0),
    connected: bool = True,
    sampled_at: float | None = None,
    generation: int = 1,
    read_error: str | None = None,
) -> StadiaSnapshot:
    axes = (left_x, 0.0, 0.0, 0.0, *triggers) if connected else ()
    buttons = [False] * 15 if connected else []
    if connected:
        buttons[10] = rb
    return StadiaSnapshot(
        sequence=sequence,
        sampled_at=clock() if sampled_at is None else sampled_at,
        connected=connected,
        product_name=STADIA_PRODUCT_NAME,
        guid="stadia-guid",
        instance_id=7 if connected else None,
        connection_generation=generation,
        axes=axes,
        buttons=tuple(buttons),
        layout=ControllerLayout(len(axes), len(buttons), 0),
        read_error=read_error,
    )


class FakeReader:
    def __init__(
        self,
        trace: list[str],
        clock: FakeClock,
        startup: list[StadiaSnapshot],
        runtime: list[StadiaSnapshot],
    ) -> None:
        self.trace = trace
        self.clock = clock
        self.startup = deque(startup)
        self.runtime = deque(runtime)
        self.latest = snapshot(0, clock=clock, connected=False, read_error="not started")
        self.final_checkpoint_pending = False

    def start(self) -> None:
        self.trace.append("reader.start")

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot:
        if not self.startup:
            self.clock.advance(timeout)
            raise TimeoutError
        item = self.startup.popleft()
        assert item.sequence > after_sequence
        self.clock.advance(0.001)
        self.latest = replace(item, sampled_at=self.clock())
        self.final_checkpoint_pending = not self.startup
        self.trace.append(f"reader.wait.{item.sequence}")
        return self.latest

    def snapshot(self) -> StadiaSnapshot:
        if self.final_checkpoint_pending:
            self.final_checkpoint_pending = False
        elif self.runtime:
            item = self.runtime.popleft()
            if item.sampled_at >= 99.0:
                item = replace(item, sampled_at=self.clock())
            self.latest = item
        self.trace.append(f"reader.snapshot.{self.latest.sequence}")
        return self.latest

    def stop(self, *, timeout: float = 2.0) -> None:
        self.trace.append(f"reader.stop.{timeout:g}")


@dataclass(frozen=True)
class Calibration:
    range_min: int
    range_max: int
    drive_mode: int = 0


class FakeBus:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.is_connected = False
        self.motors = {
            motor: SimpleNamespace(
                model="sts3215",
                norm_mode="range_0_100" if motor == "gripper" else "degrees",
            )
            for motor in SO101_MOTOR_NAMES
        }
        self.calibration = {
            motor: Calibration(0, 4000) if motor == "gripper" else Calibration(1024, 3072)
            for motor in SO101_MOTOR_NAMES
        }
        self.model_resolution_table = {"sts3215": 4096}
        self.apply_drive_mode = True
        self.pose = dict.fromkeys(SO101_MOTOR_NAMES, 2048)
        self.pose["gripper"] = 2000

    def connect(self) -> None:
        self.trace.append("bus.connect")
        self.is_connected = True

    def disable_torque(self, *, num_retry: int = 0) -> None:
        self.trace.append(f"bus.disable.retry{num_retry}")

    def sync_read(
        self,
        data_name: str,
        motors: list[str],
        *,
        normalize: bool,
        num_retry: int,
    ) -> Mapping[str, int]:
        assert motors == list(SO101_MOTOR_NAMES)
        self.trace.append(f"bus.read.{data_name}.normalize{normalize}.retry{num_retry}")
        if data_name == "Present_Position":
            return dict(self.pose)
        if data_name == "Torque_Enable":
            return dict.fromkeys(SO101_MOTOR_NAMES, 0)
        raise AssertionError(data_name)

    def write(
        self,
        data_name: str,
        motor: str,
        value: int,
        *,
        normalize: bool,
        num_retry: int,
    ) -> None:
        del value
        self.trace.append(f"bus.write.{data_name}.{motor}.normalize{normalize}.retry{num_retry}")

    def disconnect(self, *, disable_torque: bool) -> None:
        self.trace.append(f"bus.disconnect.disable{disable_torque}")
        self.is_connected = False

    def write_calibration(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("recording must never write calibration")


class FakeCamera:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.is_connected = False
        self.thread: object | None = None

    def connect(self) -> None:
        self.trace.append("camera.connect")
        self.is_connected = True

    def disconnect(self) -> None:
        self.trace.append("camera.disconnect")
        self.is_connected = False


class FakeFollower:
    name = "so101_follower"
    robot_type = "so101_follower"

    def __init__(self, trace: list[str], bus: FakeBus) -> None:
        self.trace = trace
        self.bus = bus
        self.calibration = bus.calibration
        self.cameras = {"front": FakeCamera(trace)}
        self.sent: list[dict[str, float]] = []
        self.returned: deque[object] = deque()
        self.observation_count = 0

    @property
    def is_calibrated(self) -> bool:
        self.trace.append("follower.is_calibrated")
        return True

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**dict.fromkeys(ACTION_KEYS, float), "front": (8, 8, 3)}

    def configure(self) -> None:
        self.trace.append("follower.configure")

    def get_observation(self) -> Mapping[str, Any]:
        self.trace.append("follower.observe")
        self.observation_count += 1
        return {
            **dict.fromkeys(ACTION_KEYS, float(self.observation_count)),
            "front": object(),
        }

    def send_action(self, action: Mapping[str, float]) -> object:
        requested = dict(action)
        self.sent.append(requested)
        self.trace.append(f"follower.send.{len(self.sent)}")
        if self.returned:
            result = self.returned.popleft()
            if isinstance(result, Exception):
                raise result
            if callable(result):
                return result(requested)
            return result
        return requested


def thermal_snapshot(value: float = 30.0) -> ThermalSnapshot:
    values = dict.fromkeys(SO101_MOTOR_NAMES, value)
    return ThermalSnapshot(
        temperatures=dict(values),
        reported_peaks=dict(values),
        confirmed_peaks=dict(values),
        spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
        warning_motors=(),
    )


class FakeThermal:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.outcomes: deque[object] = deque()
        self.current = thermal_snapshot()

    def check(self) -> object:
        self.trace.append("thermal.check")
        if self.outcomes:
            result = self.outcomes.popleft()
            if isinstance(result, Exception):
                raise result
            self.current = result  # type: ignore[assignment]
        return self.current

    def snapshot(self) -> ThermalSnapshot:
        return self.current


class FakeAdapter:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.current: list[dict[str, Any]] = []
        self.saved: list[list[dict[str, Any]]] = []
        self.poisoned = False
        self.poison_reason: str | None = None
        self.save_error: Exception | None = None
        self.discard_report = AttemptCleanupReport(True, True, True, True, True)
        self.finalize_calls = 0
        self.push_calls: list[dict[str, object]] = []
        self.begin_calls = 0
        self.discard_calls = 0

    def begin_attempt(self) -> object:
        self.trace.append("dataset.begin")
        assert not self.current
        self.begin_calls += 1
        return object()

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        self.trace.append("dataset.add")
        self.current.append(dict(frame))

    def save_episode(self) -> None:
        self.trace.append("dataset.save")
        if self.save_error is not None:
            self.poisoned = True
            self.poison_reason = str(self.save_error)
            raise self.save_error
        self.saved.append(list(self.current))
        self.current.clear()

    def discard_attempt(self, _checkpoint: object) -> AttemptCleanupReport:
        self.trace.append("dataset.discard")
        self.discard_calls += 1
        self.current.clear()
        if not self.discard_report.proven_clean:
            self.poisoned = True
            self.poison_reason = "cleanup not proven"
        return self.discard_report

    def finalize(self) -> None:
        self.trace.append("dataset.finalize")
        if self.poisoned:
            raise RuntimeError("poisoned adapter finalized")
        self.finalize_calls += 1

    def push_to_hub(self, *args: object, **kwargs: object) -> None:
        assert not args
        self.trace.append("dataset.push")
        self.push_calls.append(dict(kwargs))


class ScriptPacer:
    def __init__(
        self,
        trace: list[str],
        clock: FakeClock,
        advances: list[float],
        hooks: Mapping[int, Callable[[], None]],
    ) -> None:
        self.trace = trace
        self.clock = clock
        self.advances = deque(advances)
        self.hooks = dict(hooks)
        self.calls = 0

    def wait_for_next_tick(self) -> int:
        self.calls += 1
        self.trace.append(f"pace.{self.calls}")
        hook = self.hooks.get(self.calls)
        if hook is not None:
            hook()
        if not self.advances:
            raise AssertionError("pacer script exhausted")
        self.clock.advance(self.advances.popleft())
        return 0


@dataclass
class Harness:
    trace: list[str]
    clock: FakeClock
    manager: ControlSessionManager
    claim: object
    reader: FakeReader
    bus: FakeBus
    follower: FakeFollower
    adapter: FakeAdapter
    thermal: FakeThermal
    worker: StadiaRecordingSessionWorker
    pacer_hooks: dict[int, Callable[[], None]]


def make_harness(
    *,
    runtime: list[StadiaSnapshot] | None = None,
    num_episodes: int = 1,
    episode_time_s: float = 10.0,
    reset_time_s: float = 2.0,
    advances: list[float] | None = None,
    push_to_hub: bool = False,
    manifest_sink: list[object] | None = None,
) -> Harness:
    trace: list[str] = []
    clock = FakeClock()
    manager = ControlSessionManager(
        monotonic_clock=clock,
        session_id_factory=lambda: "recording-session",
        lease_ttl_s=1000.0,
    )
    claim = manager.claim(
        ControlOperation.STADIA_RECORDING,
        teleoperator_type="stadia",
        details={"robot_name": "deskbot"},
    )
    startup = [snapshot(index, clock=clock) for index in (1, 2, 3, 4)]
    runtime = runtime or [snapshot(5, clock=clock, rb=True, left_x=1.0), snapshot(6, clock=clock)]
    reader = FakeReader(trace, clock, startup, runtime)
    bus = FakeBus(trace)
    follower = FakeFollower(trace, bus)
    adapter = FakeAdapter(trace)
    thermal = FakeThermal(trace)
    pacer_hooks: dict[int, Callable[[], None]] = {}

    def prepare(selected_follower: object, config: StadiaRecordingConfig) -> PreparedRecordingDataset:
        assert selected_follower is follower
        assert not bus.is_connected
        assert config.dataset_repo_id == "alex/demo_20260902_030405"
        trace.append("dataset.prepare")

        def frame_builder(
            observation: Mapping[str, Any],
            returned_action: object,
            task: str,
        ) -> Mapping[str, Any]:
            trace.append("frame.build")
            return {
                "observation": dict(observation),
                "action": returned_action.as_dict(),  # type: ignore[union-attr]
                "task": task,
            }

        return PreparedRecordingDataset(adapter, frame_builder)

    def pacer_factory(_clock: object, _sleeper: object) -> ScriptPacer:
        return ScriptPacer(
            trace,
            clock,
            advances or [1 / 30, 1 / 30],
            pacer_hooks,
        )

    worker = StadiaRecordingSessionWorker(
        manager=manager,
        claim=claim,  # type: ignore[arg-type]
        config=StadiaSessionConfig(
            follower_port="/fake/follower",
            follower_calibration="follower.json",
            expected_guid="stadia-guid",
            cameras={"front": {"type": "opencv"}},
        ),
        recording_config=StadiaRecordingConfig(
            dataset_repo_id="alex/demo_20260902_030405",
            single_task="Pick up the block",
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            push_to_hub=push_to_hub,
            tags=("demo",),
            cameras={"front": {"type": "opencv"}},
        ),
        reader=reader,
        calibration_resolver=lambda filename: filename.removesuffix(".json"),
        follower_factory=lambda _spec: follower,
        thermal_guard_factory=lambda _bus, _sleeper: thermal,
        dataset_preparer=prepare,
        safety_manifest_writer=(
            (lambda manifest: manifest_sink.append(manifest))
            if manifest_sink is not None
            else (lambda _manifest: None)
        ),
        pacer_factory=pacer_factory,
        clock=clock,
        sleeper=clock.sleep,
    )
    return Harness(
        trace,
        clock,
        manager,
        claim,
        reader,
        bus,
        follower,
        adapter,
        thermal,
        worker,
        pacer_hooks,
    )


def test_recording_persists_initial_and_final_dataset_safety() -> None:
    manifests: list[object] = []
    harness = make_harness(manifest_sink=manifests)
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert len(manifests) >= 2
    first = manifests[0]
    final = manifests[-1]
    assert first.dataset_repo_id == harness.worker.dataset_repo_id  # type: ignore[union-attr]
    assert first.dataset_finalized is False  # type: ignore[union-attr]
    assert first.saved_episodes == 0  # type: ignore[union-attr]
    assert final.dataset_safe is True  # type: ignore[union-attr]
    assert final.dataset_finalized is True  # type: ignore[union-attr]
    assert final.saved_episodes == 1  # type: ignore[union-attr]


def test_module_import_is_pygame_and_lerobot_neutral() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pygame' or name.startswith('pygame.') or name == 'lerobot' or name.startswith('lerobot.'):
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from lelab.stadia.recording_session import build_stadia_recording_worker
assert build_stadia_recording_worker
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_prepares_dataset_before_bus_access_and_publishes_nested_status() -> None:
    harness = make_harness()
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert harness.trace.index("dataset.prepare") < harness.trace.index("bus.connect")
    assert harness.trace.index("dataset.finalize") > harness.trace.index("reader.stop.2")
    status = harness.worker.recording_status()
    assert status["recording_active"] is False
    assert status["current_phase"] == "completed"
    assert status["saved_episodes"] == 1
    assert status["dataset_safe"] is True
    assert status["dataset_finalized"] is True
    assert status["camera_feed_available"] is False
    assert status["message"] == "Recording session completed"
    terminal = harness.manager.status_for("recording-session")
    assert terminal is not None
    assert terminal.details["robot_name"] == "deskbot"
    assert terminal.details["recording"] == status


def test_returned_command_is_recorded_and_adopted_for_finish_hold() -> None:
    harness = make_harness()

    def clipped(requested: dict[str, float]) -> dict[str, float]:
        return {**requested, "shoulder_pan.pos": 0.1}

    harness.follower.returned.append(clipped)
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    recorded = harness.adapter.saved[0][0]
    assert recorded["action"]["shoulder_pan.pos"] == pytest.approx(0.1)
    assert harness.follower.sent[0]["shoulder_pan.pos"] == pytest.approx(-0.35)
    assert harness.follower.sent[1]["shoulder_pan.pos"] == pytest.approx(0.1)
    assert result.relative_clipping_count == 1


@pytest.mark.parametrize(
    ("direction", "left_x", "triggers", "endpoint_index"),
    [
        (1.0, -1.0, (1.0, -1.0), 1),
        (-1.0, 1.0, (-1.0, 1.0), 0),
    ],
)
def test_recording_travels_beyond_former_anchor_envelope_to_calibrated_endpoints(
    direction: float,
    left_x: float,
    triggers: tuple[float, float],
    endpoint_index: int,
) -> None:
    sample_clock = FakeClock()
    runtime = [
        snapshot(
            sequence,
            clock=sample_clock,
            rb=True,
            left_x=left_x,
            triggers=triggers,
        )
        for sequence in range(5, 306)
    ]
    harness = make_harness(
        runtime=runtime,
        episode_time_s=100.0,
        advances=[1 / 30] * len(runtime),
    )
    harness.pacer_hooks[len(runtime)] = lambda: harness.worker.finish_episode()
    endpoint_bounds = derive_calibrated_endpoint_bounds(harness.follower)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert harness.adapter.finalize_calls == 1
    saved_actions = [frame["action"] for frame in harness.adapter.saved[0]]
    assert len(saved_actions) == len(runtime) - 1
    assert any(
        direction * action["shoulder_pan.pos"] > 45.0 and direction * (action["gripper.pos"] - 50.0) > 45.0
        for action in saved_actions
    )
    assert saved_actions[-1]["shoulder_pan.pos"] == (endpoint_bounds["shoulder_pan.pos"][endpoint_index])
    assert saved_actions[-1]["gripper.pos"] == endpoint_bounds["gripper.pos"][endpoint_index]
    for action in saved_actions:
        for key in ("shoulder_pan.pos", "gripper.pos"):
            lower, upper = endpoint_bounds[key]
            assert lower <= action[key] <= upper
    loop = harness.worker.recording_result
    assert loop is not None
    assert loop.counters.travel_saturations == 0
    assert loop.counters.endpoint_saturations > 0


def test_rb_up_with_deflected_stick_is_a_valid_recorded_hold() -> None:
    clock = FakeClock()
    harness = make_harness(
        runtime=[
            snapshot(5, clock=clock, rb=False, left_x=1.0),
            snapshot(6, clock=clock),
        ]
    )
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    loop = harness.worker.recording_result
    assert result.terminal_state is ControlState.STOPPED
    assert loop is not None
    assert loop.outcome is RecordingOutcome.COMPLETED
    assert not any(event.reason is not None for event in loop.event_history)
    assert harness.adapter.discard_calls == 0
    assert len(harness.adapter.saved) == 1
    assert len(harness.adapter.saved[0]) == 1
    expected_hold = dict.fromkeys(ACTION_KEYS, 0.0) | {"gripper.pos": 50.0}
    assert harness.follower.sent[0] == expected_hold
    assert harness.adapter.saved[0][0]["action"] == expected_hold


def test_timed_rerecord_and_reset_use_no_catchup_then_explicit_finish() -> None:
    clock = FakeClock()
    runtime = [snapshot(index, clock=clock) for index in range(5, 10)]
    harness = make_harness(
        runtime=runtime,
        episode_time_s=0.05,
        reset_time_s=0.05,
        advances=[0.01, 0.1, 0.01, 0.1, 0.01, 0.01],
    )
    harness.pacer_hooks[6] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    loop = harness.worker.recording_result
    assert result.terminal_state is ControlState.STOPPED
    assert loop is not None
    assert loop.outcome is RecordingOutcome.COMPLETED
    assert [event.kind.value for event in loop.event_history] == [
        "rerecord_episode",
        "finish_episode",
        "finish_episode",
    ]
    assert harness.adapter.begin_calls == 2
    assert harness.adapter.discard_calls == 1
    assert len(harness.adapter.saved) == 1


def test_explicit_rerecord_command_is_typed_and_phase_gated() -> None:
    harness = make_harness(advances=[0.01, 0.01, 0.01, 0.01, 0.01])
    preparing = harness.worker.rerecord_episode()
    harness.pacer_hooks[2] = lambda: harness.worker.rerecord_episode()
    harness.pacer_hooks[3] = lambda: harness.worker.finish_episode()
    harness.pacer_hooks[5] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    assert preparing == {
        "success": False,
        "message": "Command is unavailable during preparing",
        "phase": "preparing",
        "current_phase": "preparing",
    }
    assert result.terminal_state is ControlState.STOPPED
    loop = harness.worker.recording_result
    assert loop is not None
    assert loop.event_history[0].kind.value == "rerecord_episode"


def test_stale_controller_discards_attempt_and_holds_until_stop() -> None:
    clock = FakeClock()
    stale = snapshot(5, clock=clock, rb=True, left_x=1.0, sampled_at=90.0)
    harness = make_harness(runtime=[stale], advances=[0.01, 0.01])
    harness.pacer_hooks[2] = lambda: harness.worker.stop_recording(reason="test stop")

    result = harness.worker.run()

    loop = harness.worker.recording_result
    assert result.terminal_state is ControlState.STOPPED
    assert loop is not None
    assert loop.outcome is RecordingOutcome.STOPPED
    assert loop.event_history[0].reason is HardInvalidationReason.CONTROLLER_STALE
    assert harness.follower.sent[0] == dict.fromkeys(ACTION_KEYS, 0.0) | {"gripper.pos": 50.0}
    assert not harness.adapter.saved
    assert harness.adapter.discard_calls == 1


@pytest.mark.parametrize(
    ("stop_kind", "expected_reason"),
    [
        ("caller", "operator requested exact stop"),
        ("lease", "control lease expired"),
    ],
)
def test_cooperative_stop_preserves_exact_manager_reason(
    stop_kind: str,
    expected_reason: str,
) -> None:
    harness = make_harness(advances=[0.01, 0.01])
    if stop_kind == "caller":
        harness.pacer_hooks[2] = lambda: harness.worker.stop_recording(reason=expected_reason)
    else:
        harness.pacer_hooks[2] = lambda: harness.clock.advance(1001.0)

    result = harness.worker.run()

    terminal = harness.manager.status_for("recording-session", check_expiry=False)
    assert result.terminal_state is ControlState.STOPPED
    assert result.reason == expected_reason
    assert terminal is not None
    assert terminal.stop_reason == expected_reason


@pytest.mark.parametrize("failure", ["send", "returned", "thermal"])
def test_clean_command_and_thermal_failures_never_finalize(failure: str) -> None:
    harness = make_harness(advances=[0.01])
    if failure == "send":
        harness.follower.returned.append(OSError("send failed"))
    elif failure == "returned":
        harness.follower.returned.append({"shoulder_pan.pos": 1.0})
    else:
        hot = ConfirmedTemperatureStopError(
            {"shoulder_pan": (61.0, 62.0, 30.0)},
            stop_c=60.0,
            required_hot_samples=2,
        )
        # First outcome is the pre-arm check; the loop checks again on tick 1.
        harness.thermal.outcomes.extend([thermal_snapshot(), hot])

    result = harness.worker.run()

    status = harness.worker.recording_status()
    assert result.terminal_state is ControlState.ERROR
    assert status["dataset_safe"] is True
    assert status["dataset_finalized"] is False
    assert status["upload_available"] is False
    assert harness.adapter.finalize_calls == 0
    assert not harness.adapter.push_calls


def test_generic_thermal_failure_publishes_latest_invalid_evidence() -> None:
    harness = make_harness(advances=[0.01])
    invalid_counts = dict.fromkeys(SO101_MOTOR_NAMES, 0)
    invalid_counts["shoulder_pan"] = 3
    latest = replace(thermal_snapshot(), invalid_sample_counts=invalid_counts)
    thermal_error = RuntimeError("temperature sensor invalid after retries")
    harness.thermal.outcomes.extend([thermal_snapshot(), thermal_error])
    harness.pacer_hooks[1] = lambda: setattr(harness.thermal, "current", latest)

    result = harness.worker.run()

    terminal = harness.manager.status_for("recording-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert result.commands_sent == 0
    assert harness.adapter.finalize_calls == 0
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert dict(terminal.thermal_snapshot.invalid_sample_counts)["shoulder_pan"] == 3
    assert terminal.thermal_snapshot.stop_reason == str(thermal_error)


def test_thermal_snapshot_failure_does_not_mask_original_error() -> None:
    harness = make_harness(advances=[0.01])
    thermal_error = RuntimeError("original thermal failure")
    harness.thermal.outcomes.extend([thermal_snapshot(), thermal_error])

    def broken_snapshot() -> ThermalSnapshot:
        raise ValueError("secondary snapshot failure")

    harness.thermal.snapshot = broken_snapshot  # type: ignore[method-assign]

    result = harness.worker.run()

    terminal = harness.manager.status_for("recording-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert "original thermal failure" in result.reason
    assert "secondary snapshot failure" not in result.reason
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert terminal.thermal_snapshot.stop_reason == str(thermal_error)


def test_clean_loop_error_preserves_prior_episode_but_never_finalizes_or_uploads() -> None:
    harness = make_harness(
        num_episodes=2,
        advances=[0.01, 0.01, 0.01, 0.01],
        push_to_hub=True,
    )
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()
    harness.pacer_hooks[3] = lambda: harness.worker.finish_episode()
    harness.follower.returned.extend(
        [
            lambda requested: requested,
            lambda requested: requested,
            lambda requested: requested,
            OSError("second episode send failed"),
        ]
    )

    result = harness.worker.run()

    status = harness.worker.recording_status()
    assert result.terminal_state is ControlState.ERROR
    assert len(harness.adapter.saved) == 1
    assert status["saved_episodes"] == 1
    assert status["dataset_safe"] is True
    assert status["dataset_finalized"] is False
    assert status["upload_available"] is False
    assert harness.adapter.finalize_calls == 0
    assert not harness.adapter.push_calls


def test_preloop_startup_failure_is_published_as_unsafe_recording_error() -> None:
    harness = make_harness()
    harness.reader.startup = deque(snapshot(index, clock=harness.clock, rb=True) for index in (1, 2, 3))

    result = harness.worker.run()

    status = harness.worker.recording_status()
    terminal = harness.manager.status_for("recording-session")
    assert result.terminal_state is ControlState.ERROR
    assert status["current_phase"] == "error"
    assert status["recording_active"] is False
    assert "startup timeout" in status["error"]
    assert status["dataset_safe"] is False
    assert status["dataset_finalized"] is False
    assert status["dataset_uploaded"] is False
    assert terminal is not None
    assert terminal.details["recording"] == status


def test_failed_save_records_no_episode_and_poison_blocks_finalize_and_upload() -> None:
    harness = make_harness(push_to_hub=True)
    harness.adapter.save_error = OSError("partial parquet save")
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    status = harness.worker.recording_status()
    assert result.terminal_state is ControlState.ERROR
    assert not harness.adapter.saved
    assert harness.adapter.discard_calls == 1
    assert harness.adapter.finalize_calls == 0
    assert not harness.adapter.push_calls
    assert status["dataset_safe"] is False
    assert status["dataset_finalized"] is False
    assert status["dataset_uploaded"] is False
    assert "partial parquet save" in status["error"]


def test_unproven_five_part_rollback_is_terminal_and_never_finalizes() -> None:
    harness = make_harness(advances=[0.01])
    harness.follower.returned.append(OSError("send failed"))
    harness.adapter.discard_report = AttemptCleanupReport(
        memory_frames_cleared=True,
        streaming_fragments_cleared=True,
        episode_rows_unchanged=True,
        video_references_unchanged=False,
        metadata_references_unchanged=True,
        details=("video residue",),
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "video_references_unchanged" in result.reason
    assert harness.adapter.finalize_calls == 0
    assert harness.worker.recording_status()["dataset_safe"] is False


def test_successful_requested_upload_happens_only_after_finalize() -> None:
    harness = make_harness(push_to_hub=True)
    harness.pacer_hooks[2] = lambda: harness.worker.finish_episode()

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert harness.trace.index("dataset.finalize") < harness.trace.index("dataset.push")
    assert harness.adapter.push_calls == [{"tags": ["demo", "LeLab"], "private": False}]
    status = harness.worker.recording_status()
    assert status["dataset_uploaded"] is True
    assert status["upload_available"] is False


def test_repo_id_is_sanitized_timestamped_and_collision_avoiding(tmp_path: Path) -> None:
    instant = datetime(2026, 9, 2, 3, 4, 5)
    first = resolve_recording_repo_id(
        "alex/my demo!",
        resume=False,
        timestamp=instant,
        dataset_home=tmp_path,
    )
    (tmp_path / first).mkdir(parents=True)
    second = resolve_recording_repo_id(
        "alex/my demo!",
        resume=False,
        timestamp=instant,
        dataset_home=tmp_path,
    )

    assert first == "alex/my_demo__20260902_030405"
    assert second == "alex/my_demo__20260902_030405_01"
    assert resolve_recording_repo_id("alex/existing_20260902", resume=True) == ("alex/existing_20260902")


@pytest.mark.parametrize("resume", [False, True])
def test_repo_id_requires_pinned_namespace_name_synchronously(resume: bool) -> None:
    with pytest.raises(ValueError, match="namespace/name"):
        resolve_recording_repo_id("demo", resume=resume)

    with pytest.raises(ValueError, match="must not start with 'eval_'"):
        resolve_recording_repo_id("alex/eval_demo", resume=resume)


class FakeResolved:
    operation = ControlOperation.STADIA_RECORDING

    def __init__(self, record: object, metadata: Mapping[str, Any]) -> None:
        self.record = record
        self.metadata = dict(metadata)

    def record_model(self) -> object:
        return self.record

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


def ready_record() -> object:
    return SimpleNamespace(
        name="deskbot",
        teleoperator_type="stadia",
        follower=SimpleNamespace(port="/dev/follower", calibration="follower.json"),
        leader=None,
        stadia=SimpleNamespace(
            guid=None,
            deadzone=0.15,
            max_step_per_tick=0.35,
        ),
        cameras=[
            SimpleNamespace(
                id="camera-1",
                name="front",
                type="opencv",
                camera_index=2,
                device_id="camera-device",
                width=1280,
                height=720,
                fps=30,
                fourcc=None,
                backend=None,
            )
        ],
    )


def metadata() -> dict[str, Any]:
    return {
        "dataset_repo_id": "alex/demo",
        "single_task": "Pick up the block",
        "num_episodes": 2,
        "episode_time_s": 30,
        "reset_time_s": 10,
        "fps": 30,
        "video": True,
        "push_to_hub": False,
        "tags": [],
        "private": False,
        "resume": False,
        "streaming_encoding": True,
        "cameras": {
            "front": {
                "type": "opencv",
                "camera_index": 2,
                "width": 1280,
                "height": 720,
                "fps": 30,
            }
        },
        "test_mode": False,
    }


def test_builder_resolves_final_id_synchronously_without_mutating_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    manager = ControlSessionManager(session_id_factory=lambda: "builder-session")
    claim = manager.claim(ControlOperation.STADIA_RECORDING, teleoperator_type="stadia")
    settings = metadata()
    resolved = FakeResolved(ready_record(), settings)

    worker = build_stadia_recording_worker(manager, claim, resolved)  # type: ignore[arg-type]

    assert worker.dataset_repo_id.startswith("alex/demo_")
    assert settings["dataset_repo_id"] == "alex/demo"
    assert resolved.metadata["dataset_repo_id"] == "alex/demo"
    details = manager.status_for("builder-session").details  # type: ignore[union-attr]
    assert details["recording"]["dataset_repo_id"] == worker.dataset_repo_id


def test_builder_rejects_bare_repo_id_before_worker_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.stadia.recording_session as module

    manager = ControlSessionManager(session_id_factory=lambda: "repo-reject-session")
    claim = manager.claim(ControlOperation.STADIA_RECORDING, teleoperator_type="stadia")
    settings = metadata()
    settings["dataset_repo_id"] = "demo"
    monkeypatch.setattr(
        module,
        "StadiaRecordingSessionWorker",
        lambda **_kwargs: pytest.fail("worker construction must not run"),
    )

    with pytest.raises(ValueError, match="namespace/name"):
        module.build_stadia_recording_worker(  # type: ignore[arg-type]
            manager,
            claim,
            FakeResolved(ready_record(), settings),
        )

    assert "recording" not in manager.status_for("repo-reject-session").details  # type: ignore[union-attr]


@pytest.mark.parametrize("tamper", ["camera", "test_mode"])
def test_builder_rejects_untrusted_camera_or_test_mode_before_device_access(tamper: str) -> None:
    manager = ControlSessionManager(session_id_factory=lambda: "reject-session")
    claim = manager.claim(ControlOperation.STADIA_RECORDING, teleoperator_type="stadia")
    settings = metadata()
    if tamper == "camera":
        settings["cameras"]["front"]["camera_index"] = 99
        match = "exactly match"
    else:
        settings["test_mode"] = True
        match = "test_mode"

    with pytest.raises(ValueError, match=match):
        build_stadia_recording_worker(  # type: ignore[arg-type]
            manager,
            claim,
            FakeResolved(ready_record(), settings),
        )


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch, root: Path, calls: list[object]) -> None:
    modules = {
        name: types.ModuleType(name)
        for name in (
            "lerobot",
            "lerobot.common",
            "lerobot.common.control_utils",
            "lerobot.datasets",
            "lerobot.utils",
            "lerobot.utils.constants",
            "lerobot.utils.feature_utils",
        )
    }

    class RawDataset:
        def __init__(self, repo_id: str, features: Mapping[str, Any]) -> None:
            self.repo_id = repo_id
            self.features = dict(features)

    class LeRobotDataset:
        @classmethod
        def create(cls, repo_id: str, fps: int, **kwargs: object) -> RawDataset:
            calls.append(("create", repo_id, fps, kwargs))
            return RawDataset(repo_id, kwargs["features"])  # type: ignore[arg-type]

        @classmethod
        def resume(cls, repo_id: str, **kwargs: object) -> RawDataset:
            calls.append(("resume", repo_id, kwargs))
            return RawDataset(repo_id, {"action": {}, "observation.state": {}})

    def hw_features(values: Mapping[str, Any], prefix: str, use_video: bool) -> dict[str, Any]:
        calls.append(("features", prefix, use_video, tuple(values)))
        return {prefix if prefix == "action" else "observation.state": {}}

    def build_frame(_features: Mapping[str, Any], values: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        return {prefix: dict(values)}

    modules["lerobot.datasets"].LeRobotDataset = LeRobotDataset  # type: ignore[attr-defined]
    modules["lerobot.utils.constants"].HF_LEROBOT_HOME = root  # type: ignore[attr-defined]
    modules["lerobot.utils.feature_utils"].hw_to_dataset_features = hw_features  # type: ignore[attr-defined]
    modules["lerobot.utils.feature_utils"].build_dataset_frame = build_frame  # type: ignore[attr-defined]
    modules["lerobot.common.control_utils"].sanity_check_dataset_name = (  # type: ignore[attr-defined]
        lambda repo_id, policy: calls.append(("name", repo_id, policy))
    )
    modules["lerobot.common.control_utils"].sanity_check_dataset_robot_compatibility = (  # type: ignore[attr-defined]
        lambda dataset, robot, fps, features: calls.append(
            ("compat", dataset.repo_id, robot.robot_type, fps, dict(features))
        )
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_default_preparer_pins_batch_one_and_checks_resume_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.stadia.recording_session as module

    calls: list[object] = []
    _install_fake_lerobot(monkeypatch, tmp_path, calls)

    class Adapter:
        poisoned = False
        poison_reason = None

        def __init__(self, dataset: object) -> None:
            calls.append(("adapter", dataset))

    monkeypatch.setattr(module, "LeRobotRecordingDatasetAdapter", Adapter)
    follower = SimpleNamespace(
        action_features=dict.fromkeys(ACTION_KEYS, float),
        observation_features={**dict.fromkeys(ACTION_KEYS, float), "front": (8, 8, 3)},
        robot_type="so101_follower",
    )
    create_config = StadiaRecordingConfig(
        dataset_repo_id="alex/new_20260902_030405",
        single_task="task",
        num_episodes=1,
        episode_time_s=1,
        reset_time_s=1,
        cameras={"front": {"type": "opencv"}},
    )

    prepared = module._default_dataset_preparer(follower, create_config)

    create_call = next(call for call in calls if call[0] == "create")
    assert create_call[3]["batch_encoding_size"] == 1
    assert create_call[3]["image_writer_processes"] == 0
    assert create_call[3]["image_writer_threads"] == 0
    assert any(call[0] == "compat" and call[1] == create_config.dataset_repo_id for call in calls)
    assert isinstance(prepared, PreparedRecordingDataset)

    resume_id = "alex/existing"
    resume_meta = tmp_path / resume_id / "meta"
    (resume_meta / "episodes" / "chunk-000").mkdir(parents=True)
    (resume_meta / "info.json").write_text("{}")
    (resume_meta / "tasks.parquet").write_bytes(b"tasks")
    (resume_meta / "stats.json").write_text("{}")
    (resume_meta / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"episodes")
    resume_config = StadiaRecordingConfig(
        dataset_repo_id=resume_id,
        single_task="task",
        num_episodes=1,
        episode_time_s=1,
        reset_time_s=1,
        resume=True,
    )
    module._default_dataset_preparer(follower, resume_config)
    resume_call = next(call for call in calls if call[0] == "resume")
    assert resume_call[2]["batch_encoding_size"] == 1
    assert resume_call[2]["image_writer_processes"] == 0
    assert resume_call[2]["image_writer_threads"] == 0
    assert any(call[0] == "compat" and call[1] == resume_id for call in calls)


def test_resume_requires_local_state_before_adapter_or_bus_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.stadia.recording_session as module

    calls: list[object] = []
    _install_fake_lerobot(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        module,
        "LeRobotRecordingDatasetAdapter",
        lambda dataset: pytest.fail(f"adapter unexpectedly received {dataset}"),
    )
    follower = SimpleNamespace(
        action_features=dict.fromkeys(ACTION_KEYS, float),
        observation_features=dict.fromkeys(ACTION_KEYS, float),
        robot_type="so101_follower",
    )
    config = StadiaRecordingConfig(
        dataset_repo_id="alex/missing",
        single_task="task",
        num_episodes=1,
        episode_time_s=1,
        reset_time_s=1,
        resume=True,
    )

    with pytest.raises(FileNotFoundError, match="existing local dataset"):
        module._default_dataset_preparer(follower, config)
    assert not any(call[0] == "resume" for call in calls)


@pytest.mark.parametrize("missing", ["tasks", "stats", "episodes"])
def test_resume_rejects_partial_metadata_without_hub_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    import lelab.stadia.recording_session as module

    calls: list[object] = []
    _install_fake_lerobot(monkeypatch, tmp_path, calls)
    follower = SimpleNamespace(
        action_features=dict.fromkeys(ACTION_KEYS, float),
        observation_features=dict.fromkeys(ACTION_KEYS, float),
        robot_type="so101_follower",
    )
    repo_id = "alex/partial"
    meta = tmp_path / repo_id / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    if missing != "tasks":
        (meta / "tasks.parquet").write_bytes(b"tasks")
    if missing != "stats":
        (meta / "stats.json").write_text("{}")
    if missing != "episodes":
        (meta / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"episodes")
    config = StadiaRecordingConfig(
        dataset_repo_id=repo_id,
        single_task="task",
        num_episodes=1,
        episode_time_s=1,
        reset_time_s=1,
        resume=True,
    )

    with pytest.raises(FileNotFoundError, match="complete existing local dataset"):
        module._default_dataset_preparer(follower, config)
    assert not any(call[0] == "resume" for call in calls)
