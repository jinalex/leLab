"""Fake-only contract tests for the guarded live Stadia session owner."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from lelab.control_session import (
    SO101_MOTOR_NAMES,
    ControlManagerClosingError,
    ControlOperation,
    ControlSessionManager,
    ControlState,
    MotionState,
    TorqueOutcome,
)
from lelab.stadia.session import (
    MAX_RELATIVE_TARGET,
    FollowerBuildSpec,
    StadiaSessionConfig,
    StadiaSessionWorker,
    derive_calibrated_endpoint_bounds,
)
from lelab.stadia.thermal_safety import (
    ConfirmedTemperatureGuard,
    ConfirmedTemperatureStopError,
    ThermalSnapshot,
)
from lelab.stadia.types import STADIA_PRODUCT_NAME, ControllerLayout, StadiaSnapshot

ACTION_KEYS = tuple(f"{motor}.pos" for motor in SO101_MOTOR_NAMES)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self.sleep_hook: Callable[[float], None] | None = None

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        if self.sleep_hook is not None:
            self.sleep_hook(seconds)


def stadia_snapshot(
    sequence: int,
    *,
    sampled_at: float | None = None,
    generation: int = 1,
    connected: bool = True,
    rb: bool = False,
    left_x: float = 0.0,
    triggers: tuple[float, float] = (-1.0, -1.0),
    guid: str = "stadia-guid",
    instance_id: int | None = 7,
    read_error: str | None = None,
) -> StadiaSnapshot:
    axes = (left_x, 0.0, 0.0, 0.0, *triggers) if connected else ()
    buttons = [False] * 15 if connected else []
    if connected:
        buttons[10] = rb
    layout = ControllerLayout(6, 15, 0)
    return StadiaSnapshot(
        sequence=sequence,
        sampled_at=100.0 + sequence / 1000 if sampled_at is None else sampled_at,
        connected=connected,
        product_name=STADIA_PRODUCT_NAME,
        guid=guid,
        instance_id=instance_id,
        connection_generation=generation,
        axes=axes,
        buttons=tuple(buttons),
        layout=layout,
        read_error=read_error,
    )


class FakeReader:
    def __init__(
        self,
        events: list[str],
        clock: FakeClock,
        startup: list[StadiaSnapshot],
        runtime: list[StadiaSnapshot] | None = None,
        *,
        stop_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.clock = clock
        self.startup = list(startup)
        self.runtime = list(runtime or [])
        self.latest = stadia_snapshot(0, connected=False, read_error="not started", sampled_at=100.0)
        self.stop_error = stop_error
        self.checkpoint_snapshot: StadiaSnapshot | None = None
        self._checkpoint_pending = False

    def start(self) -> None:
        self.events.append("reader.start")

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot:
        if not self.startup:
            self.clock.advance(timeout)
            raise TimeoutError("no scripted startup sample")
        snapshot = self.startup.pop(0)
        assert snapshot.sequence > after_sequence
        self.clock.value = max(self.clock.value, snapshot.sampled_at)
        self.latest = snapshot
        self._checkpoint_pending = not self.startup
        self.events.append(f"reader.wait.{snapshot.sequence}")
        return snapshot

    def snapshot(self) -> StadiaSnapshot:
        if self._checkpoint_pending:
            self._checkpoint_pending = False
            if self.checkpoint_snapshot is not None:
                checkpoint = self.checkpoint_snapshot
                if checkpoint.sampled_at >= 100.0:
                    sampled_at = max(self.clock.value, self.latest.sampled_at + 0.001)
                    self.clock.value = sampled_at
                    checkpoint = replace(checkpoint, sampled_at=sampled_at)
                self.latest = checkpoint
        elif self.runtime:
            scripted = self.runtime.pop(0)
            # Normal scripted runtime publications are produced at the fake
            # reader's current monotonic time. Values below 100 are deliberate
            # stale-sample fixtures and remain unchanged.
            if scripted.sampled_at >= 100.0:
                sampled_at = max(self.clock.value, self.latest.sampled_at + 0.001)
                self.clock.value = sampled_at
                self.latest = replace(scripted, sampled_at=sampled_at)
            else:
                self.latest = scripted
        self.events.append(f"reader.snapshot.{self.latest.sequence}")
        return self.latest

    def stop(self, *, timeout: float = 2.0) -> None:
        self.events.append(f"reader.stop.{timeout:g}")
        if self.stop_error is not None:
            raise self.stop_error


@dataclass(frozen=True)
class FakeCalibration:
    range_min: int
    range_max: int
    drive_mode: int = 0


class FakeBus:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.is_connected = False
        self.motors = {
            motor: SimpleNamespace(
                model="sts3215",
                norm_mode="range_0_100" if motor == "gripper" else "degrees",
            )
            for motor in SO101_MOTOR_NAMES
        }
        self.calibration = {
            motor: FakeCalibration(0 if motor == "gripper" else 1024, 4000 if motor == "gripper" else 3072)
            for motor in SO101_MOTOR_NAMES
        }
        self.apply_drive_mode = True
        self.model_resolution_table = {"sts3215": 4096}
        self.pose: dict[str, object] = dict.fromkeys(SO101_MOTOR_NAMES, 2048)
        self.pose["gripper"] = 2000
        self.torque_readback: object = dict.fromkeys(SO101_MOTOR_NAMES, 0)
        self.disable_errors: list[Exception | None] = []
        self.disconnect_error: Exception | None = None
        self.goal_writes: list[tuple[str, int, bool, int]] = []
        self.goal_write_errors: dict[str, Exception] = {}
        self.temperature_readings: list[dict[str, float]] = []
        self.disable_calls = 0

    def connect(self) -> None:
        self.events.append("bus.connect")
        self.is_connected = True

    def disable_torque(self, *, num_retry: int = 0) -> None:
        self.disable_calls += 1
        self.events.append(f"bus.disable.{self.disable_calls}.retry{num_retry}")
        if self.disable_errors:
            error = self.disable_errors.pop(0)
            if error is not None:
                raise error

    def sync_read(
        self,
        data_name: str,
        motors: list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> object:
        self.events.append(f"bus.read.{data_name}.normalize{normalize}.retry{num_retry}")
        assert motors == list(SO101_MOTOR_NAMES)
        if data_name == "Present_Position":
            return dict(self.pose)
        if data_name == "Torque_Enable":
            if isinstance(self.torque_readback, Exception):
                raise self.torque_readback
            return self.torque_readback
        if data_name == "Present_Temperature":
            if not self.temperature_readings:
                raise AssertionError("no scripted temperature reading")
            return dict(self.temperature_readings.pop(0))
        raise AssertionError(f"unexpected bus read: {data_name}")

    def write(
        self,
        data_name: str,
        motor: str,
        value: int,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        self.events.append(f"bus.write.{data_name}.{motor}.normalize{normalize}.retry{num_retry}")
        assert data_name == "Goal_Position"
        self.goal_writes.append((motor, value, normalize, num_retry))
        if motor in self.goal_write_errors:
            raise self.goal_write_errors[motor]

    def sync_write(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("startup must use acknowledged per-motor Goal Position writes")

    def disconnect(self, *, disable_torque: bool = True) -> None:
        self.events.append(f"bus.disconnect.disable{disable_torque}")
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False

    def write_calibration(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("session must never write calibration")


class FakeCamera:
    def __init__(self, events: list[str], name: str = "camera") -> None:
        self.events = events
        self.name = name
        self.is_connected = False
        self.thread: object | None = None
        self.disconnect_error: Exception | None = None
        self.connect_hook: Callable[[], None] | None = None

    def connect(self) -> None:
        self.events.append(f"{self.name}.connect")
        self.is_connected = True
        if self.connect_hook is not None:
            self.connect_hook()

    def disconnect(self) -> None:
        self.events.append(f"{self.name}.disconnect")
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False


class FakeFollower:
    def __init__(
        self,
        events: list[str],
        bus: FakeBus,
        manager: ControlSessionManager,
        claim_id: str,
    ) -> None:
        self.events = events
        self.bus = bus
        self.calibration = bus.calibration
        self.cameras: dict[str, FakeCamera] = {"opal": FakeCamera(events, "camera.opal")}
        self.manager = manager
        self.claim_id = claim_id
        self.calibrated = True
        self.sent_actions: list[dict[str, float]] = []
        self.return_transforms: list[Callable[[dict[str, float]], object]] = []
        self.send_errors: list[Exception] = []
        self.stop_after_sends: int | None = 1
        self.send_hook: Callable[[int], None] | None = None

    @property
    def is_calibrated(self) -> bool:
        self.events.append("follower.is_calibrated")
        return self.calibrated

    def configure(self) -> None:
        self.events.append("follower.configure.arm")

    def send_action(self, action: Mapping[str, float]) -> object:
        self.events.append(f"follower.send.{len(self.sent_actions) + 1}")
        requested = dict(action)
        self.sent_actions.append(requested)
        if self.send_errors:
            raise self.send_errors.pop(0)
        if self.send_hook is not None:
            self.send_hook(len(self.sent_actions))
        result = self.return_transforms.pop(0)(requested) if self.return_transforms else dict(requested)
        if self.stop_after_sends == len(self.sent_actions):
            self.manager.request_stop(self.claim_id, reason="test complete")
        return result


def normal_thermal(value: float = 30.0) -> ThermalSnapshot:
    temperatures = dict.fromkeys(SO101_MOTOR_NAMES, value)
    return ThermalSnapshot(
        temperatures=dict(temperatures),
        reported_peaks=dict(temperatures),
        confirmed_peaks=dict(temperatures),
        spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
        warning_motors=(),
    )


class FakeThermalGuard:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.current = normal_thermal()
        self.outcomes: list[ThermalSnapshot | Exception] = []
        self.checks = 0

    def check(self) -> ThermalSnapshot:
        self.checks += 1
        self.events.append(f"thermal.check.{self.checks}")
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.current = outcome
        return self.current

    def snapshot(self) -> ThermalSnapshot:
        self.events.append("thermal.snapshot")
        return self.current


class LoggingManager(ControlSessionManager):
    def __init__(self, events: list[str], clock: FakeClock, *, lease_ttl_s: float = 1000.0) -> None:
        super().__init__(
            lease_ttl_s=lease_ttl_s,
            lease_renew_interval_s=min(1.0, lease_ttl_s / 2),
            monotonic_clock=clock,
            utc_clock=lambda: datetime(2026, 9, 2, tzinfo=UTC),
            session_id_factory=lambda: "stadia-session",
        )
        self.events = events

    def mark_running(self, session_id: str):  # type: ignore[no-untyped-def]
        self.events.append("manager.mark_running")
        return super().mark_running(session_id)

    def finish_teardown(self, session_id: str, **kwargs: object):  # type: ignore[no-untyped-def]
        self.events.append("manager.finish_teardown")
        return super().finish_teardown(session_id, **kwargs)  # type: ignore[arg-type]


@dataclass
class Harness:
    events: list[str]
    clock: FakeClock
    manager: LoggingManager
    claim: object
    reader: FakeReader
    bus: FakeBus
    follower: FakeFollower
    guard: FakeThermalGuard
    build_specs: list[FollowerBuildSpec]
    worker: StadiaSessionWorker


def make_harness(
    *,
    startup: list[StadiaSnapshot] | None = None,
    runtime: list[StadiaSnapshot] | None = None,
    reader_stop_error: Exception | None = None,
    lease_ttl_s: float = 1000.0,
    startup_timeout_s: float = 1.0,
    expected_guid: str | None = "stadia-guid",
    joint_broadcaster: Callable[[Mapping[str, object]], None] | None = None,
) -> Harness:
    events: list[str] = []
    clock = FakeClock()
    manager = LoggingManager(events, clock, lease_ttl_s=lease_ttl_s)
    claim = manager.claim(ControlOperation.STADIA_TELEOPERATION, teleoperator_type="stadia")
    startup = startup or [stadia_snapshot(index) for index in (1, 2, 3, 4)]
    runtime = runtime or [stadia_snapshot(5, rb=True, left_x=1.0)]
    reader = FakeReader(
        events,
        clock,
        startup,
        runtime,
        stop_error=reader_stop_error,
    )
    bus = FakeBus(events)
    follower = FakeFollower(events, bus, manager, claim.session_id)
    guard = FakeThermalGuard(events)
    build_specs: list[FollowerBuildSpec] = []

    def resolve_calibration(filename: str) -> str:
        events.append(f"calibration.resolve.{filename}")
        return filename.removesuffix(".json")

    def build_follower(spec: FollowerBuildSpec) -> object:
        events.append("follower.factory")
        build_specs.append(spec)
        return follower

    worker = StadiaSessionWorker(
        manager=manager,
        claim=claim,
        config=StadiaSessionConfig(
            follower_port="/fake/follower",
            follower_calibration="follower.json",
            expected_guid=expected_guid,
            startup_timeout_s=startup_timeout_s,
            cameras={"opal": object()},
        ),
        reader=reader,
        calibration_resolver=resolve_calibration,
        follower_factory=build_follower,
        thermal_guard_factory=lambda selected_bus, _sleeper: (
            guard if selected_bus is bus else pytest.fail("wrong thermal bus")
        ),
        joint_broadcaster=joint_broadcaster,
        clock=clock,
        sleeper=clock.sleep,
    )
    return Harness(
        events,
        clock,
        manager,
        claim,
        reader,
        bus,
        follower,
        guard,
        build_specs,
        worker,
    )


def test_module_import_is_lazy_for_pygame_and_lerobot() -> None:
    script = """
import builtins

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "pygame" or name.startswith("pygame.") or name == "lerobot" or name.startswith("lerobot."):
        raise AssertionError(f"forbidden device import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from lelab.stadia.session import StadiaSessionConfig, StadiaSessionWorker
assert StadiaSessionConfig
assert StadiaSessionWorker
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_startup_waits_for_three_neutral_samples_then_arms_in_safe_order() -> None:
    harness = make_harness()

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert result.torque.outcome is TorqueOutcome.VERIFIED_OFF
    assert [event for event in harness.events if event.startswith("reader.wait")] == [
        "reader.wait.1",
        "reader.wait.2",
        "reader.wait.3",
        "reader.wait.4",
    ]
    construction = harness.events.index("follower.factory")
    assert construction > harness.events.index("reader.wait.3")
    assert harness.events.index("bus.connect") < harness.events.index("bus.disable.1.retry5")
    assert harness.events.index("bus.disable.1.retry5") < harness.events.index("follower.is_calibrated")
    assert "bus.read.Present_Position.normalizeFalse.retry5" in harness.events
    pose_read = "bus.read.Present_Position.normalizeFalse.retry5"
    first_goal_write = "bus.write.Goal_Position.shoulder_pan.normalizeFalse.retry5"
    last_goal_write = "bus.write.Goal_Position.gripper.normalizeFalse.retry5"
    assert harness.events.index("thermal.check.1") < harness.events.index("camera.opal.connect")
    assert harness.events.index("camera.opal.connect") < harness.events.index("reader.wait.4")
    assert harness.events.index("reader.wait.4") < harness.events.index(pose_read)
    assert harness.events.index(pose_read) < harness.events.index(first_goal_write)
    assert harness.events.index(first_goal_write) < harness.events.index(last_goal_write)
    assert harness.events.index(last_goal_write) < harness.events.index("follower.configure.arm")
    assert harness.events.index("follower.configure.arm") < harness.events.index("manager.mark_running")
    spec = harness.build_specs[0]
    assert spec.calibration_id == "follower"
    assert dict(spec.max_relative_target) == dict.fromkeys(SO101_MOTOR_NAMES, MAX_RELATIVE_TARGET)
    assert spec.use_degrees
    assert harness.bus.goal_writes == [
        (motor, 2000 if motor == "gripper" else 2048, False, 5) for motor in SO101_MOTOR_NAMES
    ]
    status = harness.manager.status_for("stadia-session")
    assert status is not None
    assert status.terminal
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_monitoring_active"] is False
    assert status.details["controller_last_observed"]["connected"] is True
    assert len(status.joint_specs) == 6
    assert status.motion_state is MotionState.DISARMED


def test_ble_zero_rest_requires_full_trigger_exercise_before_follower_access() -> None:
    zero_rest = (0.0, 0.0)
    full_press = (1.0, 1.0)
    harness = make_harness(
        startup=[
            stadia_snapshot(1, triggers=zero_rest),
            stadia_snapshot(2, triggers=zero_rest),
            stadia_snapshot(3, triggers=zero_rest),
            stadia_snapshot(4, triggers=full_press),
            *(stadia_snapshot(sequence, triggers=zero_rest) for sequence in range(5, 11)),
        ],
        runtime=[stadia_snapshot(11, triggers=zero_rest)],
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert harness.events.index("follower.factory") > harness.events.index("reader.wait.9")
    assert harness.events.index("camera.opal.connect") < harness.events.index("reader.wait.10")
    assert result.commands_sent == 1
    assert result.movement_steps == 0


def test_camera_setup_pose_drift_is_reread_for_initial_target_and_goal_seed() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, rb=False)])
    camera = harness.follower.cameras["opal"]
    camera.connect_hook = lambda: harness.bus.pose.__setitem__("shoulder_pan", 2500)

    result = harness.worker.run()

    expected_degrees = (2500 - 2048) * 360.0 / 4095
    assert result.terminal_state is ControlState.STOPPED
    assert ("shoulder_pan", 2500, False, 5) in harness.bus.goal_writes
    assert harness.follower.sent_actions[0]["shoulder_pan.pos"] == pytest.approx(expected_degrees)
    assert harness.events.index("camera.opal.connect") < harness.events.index(
        "bus.read.Present_Position.normalizeFalse.retry5"
    )


def test_non_neutral_startup_times_out_without_constructing_or_accessing_follower() -> None:
    harness = make_harness(
        startup=[stadia_snapshot(index, rb=True) for index in (1, 2, 3)],
        startup_timeout_s=0.2,
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "startup timeout" in result.reason
    assert "calibration.resolve" not in " ".join(harness.events)
    assert "follower.factory" not in harness.events
    assert "bus.connect" not in harness.events
    assert result.torque.outcome is TorqueOutcome.NOT_ATTEMPTED


def test_missing_distinct_post_setup_sample_blocks_goal_seed_and_arming() -> None:
    harness = make_harness(
        startup=[stadia_snapshot(index) for index in (1, 2, 3)],
        startup_timeout_s=0.2,
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "safe post-setup sample" in result.reason
    assert "camera.opal.connect" in harness.events
    assert not harness.bus.goal_writes
    assert "follower.configure.arm" not in harness.events


@pytest.mark.parametrize(
    ("post_setup", "status_error", "expected_connected"),
    [
        (
            stadia_snapshot(4, connected=False, read_error="controller unplugged"),
            "disconnected",
            False,
        ),
        (stadia_snapshot(4, generation=2), "generation changed", True),
        (stadia_snapshot(4, sampled_at=90.0), "stale", True),
        (stadia_snapshot(4, read_error="read packet failed"), "read failed", True),
        (stadia_snapshot(4, instance_id=8), "instance identity changed", True),
    ],
)
def test_post_setup_controller_fault_blocks_goal_seed_and_arming(
    post_setup: StadiaSnapshot,
    status_error: str,
    expected_connected: bool,
) -> None:
    harness = make_harness(
        startup=[
            *(stadia_snapshot(index) for index in (1, 2, 3)),
            post_setup,
        ]
    )

    result = harness.worker.run()

    status = harness.manager.status_for("stadia-session")
    assert result.terminal_state is ControlState.ERROR
    assert "camera.opal.connect" in harness.events
    assert not harness.bus.goal_writes
    assert "follower.configure.arm" not in harness.events
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is expected_connected
    assert status_error in status.details["controller_last_observed"]["error"]


@pytest.mark.parametrize(
    ("checkpoint", "status_error", "expected_connected"),
    [
        (
            stadia_snapshot(5, connected=False, read_error="controller unplugged"),
            "disconnected",
            False,
        ),
        (stadia_snapshot(5, generation=2), "generation changed", True),
        (stadia_snapshot(5, sampled_at=90.0), "stale", True),
        (stadia_snapshot(5, read_error="read packet failed"), "read failed", True),
        (stadia_snapshot(5, instance_id=8), "instance identity changed", True),
    ],
)
def test_controller_fault_during_acknowledged_goal_seed_blocks_arming(
    checkpoint: StadiaSnapshot,
    status_error: str,
    expected_connected: bool,
) -> None:
    harness = make_harness()
    harness.reader.checkpoint_snapshot = checkpoint

    result = harness.worker.run()

    status = harness.manager.status_for("stadia-session")
    assert result.terminal_state is ControlState.ERROR
    assert len(harness.bus.goal_writes) == 6
    assert "follower.configure.arm" not in harness.events
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is expected_connected
    assert status_error in status.details["controller_last_observed"]["error"]


def test_rb_must_remain_released_at_the_final_prearm_checkpoint() -> None:
    harness = make_harness()
    harness.reader.checkpoint_snapshot = stadia_snapshot(5, rb=True)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "RB released" in result.reason
    assert len(harness.bus.goal_writes) == 6
    assert "follower.configure.arm" not in harness.events


def test_worker_thread_runs_independently_and_joins_after_owned_teardown() -> None:
    harness = make_harness()

    harness.worker.start()
    result = harness.worker.join(timeout=1.0)

    assert result.terminal_state is ControlState.STOPPED
    assert not harness.worker.is_alive
    assert harness.events[-1] == "manager.finish_teardown"
    with pytest.raises(RuntimeError, match="only be run once"):
        harness.worker.run()


def test_returned_clipping_is_counted_and_adopted_for_the_next_request() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = 2

    def clip_first(requested: dict[str, float]) -> dict[str, float]:
        returned = dict(requested)
        returned["shoulder_pan.pos"] = 0.1
        return returned

    harness.follower.return_transforms = [clip_first, dict]

    result = harness.worker.run()

    assert result.relative_clipping_count == 1
    assert result.movement_steps == 2
    assert harness.follower.sent_actions[0]["shoulder_pan.pos"] == pytest.approx(-0.35)
    assert harness.follower.sent_actions[1]["shoulder_pan.pos"] == pytest.approx(-0.25)


def test_live_speed_change_requires_rb_release_and_updates_exact_joint_caps() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=False),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = 2
    harness.follower.send_hook = lambda count: (
        harness.worker.set_speed_multiplier(2.0) if count == 1 else None
    )

    result = harness.worker.run()

    assert result.movement_steps == 1
    assert harness.follower.sent_actions[1]["shoulder_pan.pos"] == pytest.approx(-0.7)
    status = harness.manager.status_for("stadia-session", check_expiry=False)
    assert status is not None
    assert status.details["stadia_speed_multiplier"] == 2.0
    assert status.details["stadia_effective_max_step_per_tick"] == pytest.approx(0.7)
    assert all(spec.max_step_per_tick == pytest.approx(0.7) for spec in status.joint_specs)


def test_live_speed_change_is_rejected_while_rb_enables_motion() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, rb=True, left_x=1.0)])
    errors: list[str] = []

    def try_speed_change(_count: int) -> None:
        try:
            harness.worker.set_speed_multiplier(2.0)
        except Exception as error:
            errors.append(str(error))

    harness.follower.send_hook = try_speed_change

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert errors == ["release RB before changing Stadia speed"]
    assert harness.follower.sent_actions[0]["shoulder_pan.pos"] == pytest.approx(-0.35)


def test_authoritative_returned_actions_drive_the_urdf_broadcast() -> None:
    broadcasts: list[Mapping[str, object]] = []
    harness = make_harness(
        runtime=[stadia_snapshot(5, rb=True, left_x=1.0)],
        joint_broadcaster=broadcasts.append,
    )

    def clip(requested: dict[str, float]) -> dict[str, float]:
        return {**requested, "shoulder_pan.pos": 0.1}

    harness.follower.return_transforms = [clip]

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert len(broadcasts) == 2
    assert broadcasts[0]["type"] == "joint_update"
    assert broadcasts[0]["joints"] == pytest.approx(
        {
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 50.0 * 3.141592653589793 / 180.0,
        }
    )
    assert broadcasts[1]["joints"]["Rotation"] == pytest.approx(0.1 * 3.141592653589793 / 180.0)


def test_visualizer_broadcast_failure_never_stops_the_device_owner() -> None:
    harness = make_harness(
        joint_broadcaster=lambda _payload: (_ for _ in ()).throw(RuntimeError("viewer gone"))
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert result.commands_sent == 1


def test_rb_up_sends_an_intentional_hold_without_advancing_target() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, rb=False)])

    result = harness.worker.run()

    assert result.commands_sent == 1
    assert result.movement_steps == 0
    assert harness.follower.sent_actions == [
        {
            **dict.fromkeys(ACTION_KEYS, 0.0),
            "gripper.pos": 50.0,
        }
    ]


def test_stale_and_reconnect_states_hold_then_require_fresh_neutral_release() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, sampled_at=90.0, rb=True, left_x=1.0),
            stadia_snapshot(6, generation=2),
            stadia_snapshot(7, generation=2),
            stadia_snapshot(8, generation=2),
            stadia_snapshot(9, generation=2, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = 2

    result = harness.worker.run()

    assert result.movement_steps == 1
    assert [action["shoulder_pan.pos"] for action in harness.follower.sent_actions] == pytest.approx(
        [0.0, -0.35]
    )


@pytest.mark.parametrize(
    ("unsafe_snapshot", "expected_error", "expected_connected"),
    [
        (stadia_snapshot(5, sampled_at=90.0), "stale", True),
        (stadia_snapshot(5, generation=2), "connection changed", True),
        (stadia_snapshot(5, read_error="read packet failed"), "read failed", True),
        (stadia_snapshot(5, guid="different-guid"), "does not match session GUID", True),
        (stadia_snapshot(5, instance_id=8), "instance ID", True),
        (
            stadia_snapshot(5, connected=False, read_error="controller unplugged"),
            "disconnected",
            False,
        ),
    ],
)
def test_runtime_controller_fault_is_published_as_typed_health(
    unsafe_snapshot: StadiaSnapshot,
    expected_error: str,
    expected_connected: bool,
) -> None:
    harness = make_harness(runtime=[unsafe_snapshot])
    harness.follower.stop_after_sends = None
    harness.clock.sleep_hook = lambda _seconds: harness.manager.request_stop(
        harness.claim.session_id,  # type: ignore[union-attr]
        reason="status captured",
    )

    result = harness.worker.run()

    status = harness.manager.status_for("stadia-session")
    assert result.terminal_state is ControlState.STOPPED
    assert result.commands_sent == 0
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is expected_connected
    assert expected_error in status.details["controller_last_observed"]["error"]


def test_session_pins_initial_guid_even_if_an_injected_reader_changes_device() -> None:
    harness = make_harness(
        startup=[stadia_snapshot(index, guid="guid-a") for index in (1, 2, 3, 4)],
        runtime=[stadia_snapshot(5, generation=2, guid="guid-b", rb=True, left_x=1.0)],
        expected_guid=None,
    )
    harness.follower.stop_after_sends = None
    harness.clock.sleep_hook = lambda _seconds: (
        harness.manager.request_stop(harness.claim.session_id, reason="test complete")  # type: ignore[union-attr]
        if not harness.claim.stop_requested.is_set()  # type: ignore[union-attr]
        else None
    )

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert result.commands_sent == 0
    assert not harness.follower.sent_actions
    status = harness.manager.status_for("stadia-session")
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is True
    assert "does not match session GUID" in status.details["controller_last_observed"]["error"]


@pytest.mark.parametrize("failure_kind", ["send", "missing", "nonfinite"])
def test_send_and_return_contract_failures_end_the_session(failure_kind: str) -> None:
    harness = make_harness()
    harness.follower.stop_after_sends = None
    if failure_kind == "send":
        harness.follower.send_errors = [OSError("write failed")]
    elif failure_kind == "missing":
        harness.follower.return_transforms = [
            lambda requested: {key: value for key, value in requested.items() if key != "gripper.pos"}
        ]
    else:
        harness.follower.return_transforms = [lambda requested: {**requested, "gripper.pos": float("nan")}]

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert result.commands_sent == 0
    assert harness.events.index("bus.disable.2.retry5") > harness.events.index("follower.send.1")
    assert harness.manager.status_for("stadia-session").state is ControlState.ERROR  # type: ignore[union-attr]


def test_raising_failure_hook_cannot_orphan_control_ownership() -> None:
    harness = make_harness()
    harness.follower.stop_after_sends = None
    harness.follower.send_errors = [OSError("write failed")]

    def raising_hook(_error: Exception) -> None:
        raise RuntimeError("status backend unavailable")

    harness.worker._on_session_failure = raising_hook  # type: ignore[method-assign]

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "session failure hook failed" in "; ".join(result.teardown_errors)
    assert harness.claim.teardown_completed.is_set()  # type: ignore[union-attr]
    assert harness.manager.active_status(check_expiry=False) is None
    assert harness.manager.status_for("stadia-session", check_expiry=False).terminal  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "pose_update",
    [
        {"gripper": float("nan")},
        {"gripper": 4090},
        {"gripper": 5000},
        {"shoulder_pan": 2048.5},
        {"shoulder_pan": 500.0},
        {"remove": "wrist_roll"},
    ],
)
def test_startup_pose_must_be_exact_finite_and_in_calibrated_bounds(
    pose_update: dict[str, object],
) -> None:
    harness = make_harness()
    if "remove" in pose_update:
        del harness.bus.pose[str(pose_update["remove"])]
    else:
        harness.bus.pose.update(pose_update)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "current pose" in result.reason
    assert "follower.configure.arm" not in harness.events
    assert not harness.follower.sent_actions


def test_raw_pose_uses_pinned_drive_mode_normalization_without_clamping() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, rb=False)])
    harness.bus.calibration["gripper"] = FakeCalibration(0, 4000, drive_mode=1)
    harness.follower.calibration = harness.bus.calibration
    harness.bus.pose["gripper"] = 1000

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert harness.follower.sent_actions[0]["gripper.pos"] == pytest.approx(75.0)
    assert ("gripper", 1000, False, 5) in harness.bus.goal_writes


def test_goal_seed_acknowledgement_failure_blocks_configure() -> None:
    harness = make_harness()
    harness.bus.goal_write_errors["elbow_flex"] = OSError("goal acknowledgement lost")

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "goal acknowledgement lost" in result.reason
    assert [write[0] for write in harness.bus.goal_writes] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
    ]
    assert "follower.configure.arm" not in harness.events


def test_calibration_mismatch_aborts_without_write_or_arming() -> None:
    harness = make_harness()
    harness.follower.calibrated = False

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "separate calibration/apply flow" in result.reason
    assert "follower.configure.arm" not in harness.events
    assert not harness.bus.goal_writes


def test_prearm_and_runtime_confirmed_thermal_stops_end_without_extra_commands() -> None:
    prearm = make_harness()
    hot_error = ConfirmedTemperatureStopError(
        {"shoulder_pan": (61.0, 62.0, 30.0)},
        stop_c=60.0,
        required_hot_samples=2,
    )
    prearm.guard.outcomes = [hot_error]
    prearm.guard.current = normal_thermal(62.0)

    prearm_result = prearm.worker.run()

    assert prearm_result.terminal_state is ControlState.ERROR
    assert prearm.guard.checks == 1
    assert "follower.configure.arm" not in prearm.events

    runtime = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    runtime.follower.stop_after_sends = None
    runtime.guard.outcomes = [normal_thermal(), hot_error]
    runtime.guard.current = normal_thermal(62.0)
    runtime.follower.send_hook = lambda count: runtime.clock.advance(1.1) if count == 1 else None

    runtime_result = runtime.worker.run()

    assert runtime_result.terminal_state is ControlState.ERROR
    assert runtime.guard.checks == 2
    assert len(runtime.follower.sent_actions) == 1


def test_generic_thermal_failure_publishes_latest_invalid_evidence() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = None
    invalid_counts = dict.fromkeys(SO101_MOTOR_NAMES, 0)
    invalid_counts["shoulder_pan"] = 3
    latest = replace(normal_thermal(), invalid_sample_counts=invalid_counts)
    thermal_error = RuntimeError("temperature sensor invalid after retries")
    harness.guard.outcomes = [normal_thermal(), thermal_error]

    def expose_failure_after_first_send(count: int) -> None:
        if count == 1:
            harness.guard.current = latest
            harness.clock.advance(1.1)

    harness.follower.send_hook = expose_failure_after_first_send

    result = harness.worker.run()

    terminal = harness.manager.status_for("stadia-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert result.commands_sent == 1
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert dict(terminal.thermal_snapshot.invalid_sample_counts)["shoulder_pan"] == 3
    assert terminal.thermal_snapshot.stop_reason == str(thermal_error)


def test_prearm_generic_thermal_failure_publishes_latest_invalid_evidence() -> None:
    harness = make_harness()
    invalid_counts = dict.fromkeys(SO101_MOTOR_NAMES, 0)
    invalid_counts["shoulder_pan"] = 3
    harness.guard.current = replace(normal_thermal(), invalid_sample_counts=invalid_counts)
    thermal_error = RuntimeError("temperature sensor missing after retries")
    harness.guard.outcomes = [thermal_error]

    result = harness.worker.run()

    terminal = harness.manager.status_for("stadia-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert result.commands_sent == 0
    assert "follower.configure.arm" not in harness.events
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert dict(terminal.thermal_snapshot.invalid_sample_counts)["shoulder_pan"] == 3
    assert terminal.thermal_snapshot.stop_reason == str(thermal_error)


def test_real_guard_prearm_missing_reading_publishes_nullable_evidence() -> None:
    harness = make_harness()
    harness.bus.temperature_readings = [{motor: 30.0 for motor in SO101_MOTOR_NAMES if motor != "gripper"}]
    harness.worker._thermal_guard_factory = lambda bus, sleeper: ConfirmedTemperatureGuard(
        bus,
        SO101_MOTOR_NAMES,
        sleeper=sleeper,
    )

    result = harness.worker.run()

    terminal = harness.manager.status_for("stadia-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert "follower.configure.arm" not in harness.events
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert dict(terminal.thermal_snapshot.temperatures) == dict.fromkeys(SO101_MOTOR_NAMES)
    assert dict(terminal.thermal_snapshot.invalid_sample_counts) == dict.fromkeys(
        SO101_MOTOR_NAMES,
        0,
    )
    assert terminal.thermal_snapshot.stop_reason == "missing temperature values for: gripper"


def test_real_guard_runtime_nonfinite_failure_keeps_prior_values_and_invalid_counts() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = None
    normal = dict.fromkeys(SO101_MOTOR_NAMES, 30.0)
    invalid = {**normal, "shoulder_pan": float("nan")}
    harness.bus.temperature_readings = [normal, invalid, invalid, invalid]
    harness.worker._thermal_guard_factory = lambda bus, sleeper: ConfirmedTemperatureGuard(
        bus,
        SO101_MOTOR_NAMES,
        sleeper=sleeper,
    )
    harness.follower.send_hook = lambda count: harness.clock.advance(1.1) if count == 1 else None

    result = harness.worker.run()

    terminal = harness.manager.status_for("stadia-session", check_expiry=False)
    assert result.terminal_state is ControlState.ERROR
    assert result.commands_sent == 1
    assert terminal is not None
    assert terminal.thermal_snapshot is not None
    assert dict(terminal.thermal_snapshot.temperatures)["shoulder_pan"] == 30.0
    assert dict(terminal.thermal_snapshot.invalid_sample_counts)["shoulder_pan"] == 3
    assert dict(terminal.thermal_snapshot.last_invalid_values)["shoulder_pan"] is None
    assert "non-finite" in (terminal.thermal_snapshot.stop_reason or "")


def test_thermal_monitoring_continues_while_stale_input_suppresses_commands() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, sampled_at=90.0, rb=True, left_x=1.0)])
    harness.follower.stop_after_sends = None
    hot_error = ConfirmedTemperatureStopError(
        {"shoulder_pan": (61.0, 62.0, 30.0)},
        stop_c=60.0,
        required_hot_samples=2,
    )
    harness.guard.outcomes = [normal_thermal(), hot_error]
    harness.guard.current = normal_thermal(62.0)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert harness.guard.checks == 2
    assert result.commands_sent == 0
    assert not harness.follower.sent_actions


def test_overrun_drops_missed_ticks_and_never_integrates_catch_up_steps() -> None:
    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
        ]
    )
    harness.follower.stop_after_sends = 2
    harness.follower.send_hook = lambda count: harness.clock.advance(0.2) if count == 1 else None

    result = harness.worker.run()

    assert result.commands_sent == 2
    assert result.movement_steps == 2
    assert result.missed_ticks >= 5
    assert [action["shoulder_pan.pos"] for action in harness.follower.sent_actions] == pytest.approx(
        [-0.35, -0.7]
    )


@pytest.mark.parametrize(
    ("direction", "left_x", "triggers", "endpoint_index"),
    [
        (1.0, -1.0, (1.0, -1.0), 1),
        (-1.0, 1.0, (-1.0, 1.0), 0),
    ],
)
def test_session_travels_beyond_former_anchor_envelope_to_calibrated_endpoints(
    direction: float,
    left_x: float,
    triggers: tuple[float, float],
    endpoint_index: int,
) -> None:
    runtime = [
        stadia_snapshot(sequence, rb=True, left_x=left_x, triggers=triggers) for sequence in range(5, 305)
    ]
    harness = make_harness(runtime=runtime)
    harness.follower.stop_after_sends = len(runtime)
    endpoint_bounds = derive_calibrated_endpoint_bounds(harness.follower)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert len(harness.follower.sent_actions) == len(runtime)
    sent = harness.follower.sent_actions
    assert any(
        direction * action["shoulder_pan.pos"] > 45.0 and direction * (action["gripper.pos"] - 50.0) > 45.0
        for action in sent
    )
    assert sent[-1]["shoulder_pan.pos"] == endpoint_bounds["shoulder_pan.pos"][endpoint_index]
    assert sent[-1]["gripper.pos"] == endpoint_bounds["gripper.pos"][endpoint_index]
    for action in sent:
        for key in ("shoulder_pan.pos", "gripper.pos"):
            lower, upper = endpoint_bounds[key]
            assert lower <= action[key] <= upper
    for previous, current in zip(
        [{"shoulder_pan.pos": 0.0, "gripper.pos": 50.0}, *sent[:-1]],
        sent,
        strict=True,
    ):
        assert abs(current["shoulder_pan.pos"] - previous["shoulder_pan.pos"]) <= 0.35 + 1e-12
        assert abs(current["gripper.pos"] - previous["gripper.pos"]) <= 0.35 + 1e-12
    assert dict(harness.build_specs[0].max_relative_target) == dict.fromkeys(
        SO101_MOTOR_NAMES,
        MAX_RELATIVE_TARGET,
    )
    assert result.saturation_count > 0
    terminal = harness.manager.status_for("stadia-session", check_expiry=False)
    assert terminal is not None
    assert all(not hasattr(spec, "startup_min") for spec in terminal.joint_specs)
    by_key = {spec.action_key: spec for spec in terminal.joint_specs}
    assert by_key["shoulder_pan.pos"].calibrated_max == endpoint_bounds["shoulder_pan.pos"][1]
    assert by_key["gripper.pos"].calibrated_min == endpoint_bounds["gripper.pos"][0]


def test_session_surfaces_endpoint_saturation_in_typed_terminal_status() -> None:
    harness = make_harness(runtime=[stadia_snapshot(5, rb=True, left_x=-1.0)])
    harness.bus.pose["shoulder_pan"] = 3072

    result = harness.worker.run()

    status = harness.manager.status_for("stadia-session")
    assert result.saturation_count == 1
    assert status is not None
    assert status.saturation_count == 1
    assert status.relative_clipping_count == 0
    assert status.motion_state is MotionState.DISARMED


def test_teardown_attempts_every_stage_in_order_and_reports_reader_timeout() -> None:
    harness = make_harness(reader_stop_error=TimeoutError("reader join timed out"))

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert result.torque.outcome is TorqueOutcome.VERIFIED_OFF
    assert "reader join timed out" in "; ".join(result.teardown_errors)
    assert harness.manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        harness.manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="must-not-reopen")
    ordered = [
        "follower.send.1",
        "bus.disable.2.retry5",
        "bus.read.Torque_Enable.normalizeFalse.retry5",
        "camera.opal.disconnect",
        "bus.disconnect.disableFalse",
        "reader.stop.2",
        "manager.finish_teardown",
    ]
    assert [harness.events.index(event) for event in ordered] == sorted(
        harness.events.index(event) for event in ordered
    )


@pytest.mark.parametrize("release_target", ["bus", "camera"])
def test_unproven_device_release_quarantines_new_control(release_target: str) -> None:
    harness = make_harness()
    if release_target == "bus":
        harness.bus.disconnect_error = OSError("bus close failed")
    else:
        harness.follower.cameras["opal"].disconnect_error = OSError("camera close failed")

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert harness.manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        harness.manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="must-not-reopen")


@pytest.mark.parametrize("release_target", ["bus", "camera"])
def test_successful_disconnect_return_still_requires_exact_false_postcondition(
    release_target: str,
) -> None:
    harness = make_harness()
    if release_target == "bus":

        def disconnect_bus(*, disable_torque: bool = True) -> None:
            harness.events.append(f"bus.disconnect.disable{disable_torque}")

        harness.bus.disconnect = disconnect_bus  # type: ignore[method-assign]
    else:
        camera = harness.follower.cameras["opal"]

        def disconnect_camera() -> None:
            harness.events.append("camera.opal.disconnect")

        camera.disconnect = disconnect_camera  # type: ignore[method-assign]

    result = harness.worker.run()

    assert result.terminal_state is ControlState.ERROR
    assert "postcondition is not false" in "; ".join(result.teardown_errors)
    assert harness.manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        harness.manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="must-not-reopen")


def test_camera_disconnect_cannot_hide_a_live_pre_disconnect_read_thread() -> None:
    harness = make_harness()
    camera = harness.follower.cameras["opal"]
    joins: list[float | None] = []

    class StuckReadThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            joins.append(timeout)

    camera.thread = StuckReadThread()

    def disconnect_camera() -> None:
        harness.events.append("camera.opal.disconnect")
        camera.thread = None
        camera.is_connected = False

    camera.disconnect = disconnect_camera  # type: ignore[method-assign]

    result = harness.worker.run()

    assert joins == [2.0]
    assert result.terminal_state is ControlState.ERROR
    assert "read thread is still alive" in "; ".join(result.teardown_errors)
    assert harness.manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        harness.manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="must-not-reopen")


def test_teardown_failure_and_unavailable_readback_are_classified_truthfully() -> None:
    failed = make_harness()
    failed.bus.disable_errors = [None, OSError("disable packet lost"), OSError("retry lost")]

    failed_result = failed.worker.run()

    assert failed_result.torque.outcome is TorqueOutcome.FAILED
    assert failed_result.terminal_state is ControlState.ERROR
    assert len([event for event in failed.events if event.startswith("bus.disable")]) == 3

    unknown = make_harness()
    unknown.bus.torque_readback = NotImplementedError("register unavailable")

    unknown_result = unknown.worker.run()

    assert unknown_result.torque.outcome is TorqueOutcome.UNKNOWN
    assert unknown_result.torque.verification_supported is False


def test_lease_expiry_requests_hold_and_teardown() -> None:
    harness = make_harness(lease_ttl_s=0.2)
    harness.follower.stop_after_sends = None
    harness.follower.send_hook = lambda _count: harness.clock.advance(0.3)

    result = harness.worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert result.commands_sent == 1
    assert result.reason == "control lease expired"
    assert harness.claim.hold_requested.is_set()  # type: ignore[union-attr]
    assert harness.claim.stop_requested.is_set()  # type: ignore[union-attr]
