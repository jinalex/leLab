"""Fake-only tests for the controller-only check worker."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lelab.control_session import (
    ControlManagerClosingError,
    ControlOperation,
    ControlSessionManager,
    ControlState,
    TorqueOutcome,
)
from lelab.stadia.controller_check import (
    ControllerCheckConfig,
    ControllerCheckWorker,
)
from lelab.stadia.types import STADIA_PRODUCT_NAME, ControllerLayout, StadiaSnapshot


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def snapshot(
    sequence: int,
    *,
    sampled_at: float | None = None,
    connected: bool = True,
    product_name: str | None = STADIA_PRODUCT_NAME,
    guid: str | None = "stadia-guid",
    generation: int = 1,
    rb: bool = False,
    triggers: tuple[float, float] = (-1.0, -1.0),
    read_error: str | None = None,
) -> StadiaSnapshot:
    axes = (0.0, 0.0, 0.0, 0.0, *triggers) if connected else ()
    buttons = [False] * 15 if connected else []
    if connected:
        buttons[10] = rb
    return StadiaSnapshot(
        sequence=sequence,
        sampled_at=100.0 + sequence / 1000 if sampled_at is None else sampled_at,
        connected=connected,
        product_name=product_name,
        guid=guid,
        instance_id=7 if guid is not None else None,
        connection_generation=generation,
        axes=axes,
        buttons=tuple(buttons),
        layout=ControllerLayout(len(axes), len(buttons), 0),
        read_error=read_error,
    )


class FakeReader:
    def __init__(
        self,
        clock: FakeClock,
        samples: list[StadiaSnapshot],
        *,
        stop_error: Exception | None = None,
        on_publish: Callable[[StadiaSnapshot], None] | None = None,
    ) -> None:
        self.clock = clock
        self.samples = list(samples)
        self.latest = snapshot(0, connected=False, read_error="not started", sampled_at=100.0)
        self.stop_error = stop_error
        self.on_publish = on_publish
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("reader.start")

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot:
        if not self.samples:
            self.clock.advance(timeout)
            raise TimeoutError("no new sample")
        current = self.samples.pop(0)
        assert current.sequence > after_sequence
        self.clock.value = max(self.clock.value, current.sampled_at)
        self.latest = current
        self.events.append(f"reader.sample.{current.sequence}")
        if self.on_publish is not None:
            self.on_publish(current)
        return current

    def snapshot(self) -> StadiaSnapshot:
        self.events.append(f"reader.snapshot.{self.latest.sequence}")
        return self.latest

    def stop(self, *, timeout: float = 2.0) -> None:
        self.events.append(f"reader.stop.{timeout:g}")
        if self.stop_error is not None:
            raise self.stop_error


def manager(clock: FakeClock) -> ControlSessionManager:
    return ControlSessionManager(
        lease_ttl_s=10.0,
        lease_renew_interval_s=1.0,
        monotonic_clock=clock,
        utc_clock=lambda: datetime(2026, 9, 2, tzinfo=UTC),
        session_id_factory=lambda: "controller-check",
    )


def test_three_distinct_neutral_samples_publish_ready_without_robot_access() -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(
        ControlOperation.CONTROLLER_CHECK,
        teleoperator_type="stadia",
        details={"robot_name": "desk-arm"},
    )
    reader = FakeReader(clock, [snapshot(1), snapshot(2), snapshot(3)])
    reader.on_publish = lambda current: (
        sessions.request_stop(claim.session_id, reason="check complete") if current.sequence == 3 else None
    )
    worker = ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(expected_guid="stadia-guid"),
        reader=reader,
        clock=clock,
    )

    result = worker.run()

    assert result.terminal_state is ControlState.STOPPED
    assert result.ready_observed
    assert result.samples_seen == 3
    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.controller_product_name == STADIA_PRODUCT_NAME
    assert status.controller_layout == (6, 15, 0)
    assert status.controls_neutral is None
    assert status.rb_held is None
    assert status.details["controller_monitoring_active"] is False
    assert status.details["controller_last_observed"]["connected"] is True
    assert status.details["controller_last_observed"]["controls_neutral"] is True
    assert status.details["controller_last_observed"]["rb_held"] is False
    assert status.details["controller_ready"] is True
    assert status.details["robot_name"] == "desk-arm"
    assert status.torque.outcome is TorqueOutcome.NOT_ATTEMPTED
    assert reader.events[-1] == "reader.stop.2"


def test_trigger_midpoint_remains_unready_and_explains_required_exercise() -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    reader = FakeReader(
        clock,
        [snapshot(1, triggers=(0.0, 0.0)), snapshot(2, triggers=(0.0, 0.0))],
    )
    reader.on_publish = lambda current: (
        sessions.request_stop(claim.session_id, reason="test complete") if current.sequence == 2 else None
    )
    worker = ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(),
        reader=reader,
        clock=clock,
    )

    result = worker.run()

    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert result.ready_observed is False
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.controls_neutral is None
    assert status.details["controller_last_observed"]["connected"] is True
    assert status.details["controller_last_observed"]["controls_neutral"] is False
    assert status.details["controller_ready"] is False
    assert "exercise and release" in status.details["controller_gate_reason"]


@pytest.mark.parametrize(
    ("bad_sample", "message", "connected"),
    [
        (
            snapshot(1, connected=False, read_error="USB removed"),
            "USB removed",
            False,
        ),
        (
            snapshot(1, product_name="Other Gamepad"),
            "unsupported controller product",
            True,
        ),
        (
            snapshot(1, read_error="read packet failed"),
            "read packet failed",
            True,
        ),
    ],
)
def test_disconnect_wrong_profile_and_read_error_are_typed(
    bad_sample: StadiaSnapshot,
    message: str,
    connected: bool,
) -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    reader = FakeReader(clock, [bad_sample])
    reader.on_publish = lambda _current: sessions.request_stop(
        claim.session_id,
        reason="test complete",
    )

    ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(),
        reader=reader,
        clock=clock,
    ).run()

    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is connected
    assert message in (status.details["controller_last_observed"]["error"] or "")
    assert status.details["controller_ready"] is False


def test_blocked_reader_becomes_stale_while_identity_remains_visible() -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    first = snapshot(1)
    reader = FakeReader(clock, [first])
    timeout_count = 0
    original_snapshot = reader.snapshot

    def stale_snapshot() -> StadiaSnapshot:
        nonlocal timeout_count
        timeout_count += 1
        current = original_snapshot()
        if timeout_count == 2:
            sessions.request_stop(claim.session_id, reason="stale observed")
        return current

    reader.snapshot = stale_snapshot  # type: ignore[method-assign]
    worker = ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(max_snapshot_age_s=0.15),
        reader=reader,
        clock=clock,
    )

    worker.run()

    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert status is not None
    assert status.controller_connected is None
    assert status.controller_error is None
    assert status.details["controller_last_observed"]["connected"] is True
    assert "stale" in (status.details["controller_last_observed"]["error"] or "")
    assert status.controller_guid == "stadia-guid"


def test_generation_change_resets_readiness_until_three_new_samples() -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    samples = [snapshot(1), snapshot(2), snapshot(3)]
    samples.extend(snapshot(index, generation=2) for index in (4, 5, 6))
    reader = FakeReader(clock, samples)
    observed: list[tuple[int, bool]] = []

    def capture(current: StadiaSnapshot) -> None:
        status = sessions.status_for(claim.session_id, check_expiry=False)
        if status is not None:
            observed.append((current.sequence, bool(status.details.get("controller_ready", False))))
        if current.sequence == 6:
            sessions.request_stop(claim.session_id, reason="test complete")

    reader.on_publish = capture
    result = ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(),
        reader=reader,
        clock=clock,
    ).run()

    assert result.ready_observed
    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert status is not None
    assert status.controller_generation == 2
    assert status.details["controller_ready"] is True


def test_reader_cleanup_failure_is_terminal_error_and_never_claims_torque_off() -> None:
    clock = FakeClock()
    sessions = manager(clock)
    claim = sessions.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    reader = FakeReader(clock, [snapshot(1)], stop_error=TimeoutError("join timed out"))
    reader.on_publish = lambda _current: sessions.request_stop(
        claim.session_id,
        reason="test complete",
    )

    result = ControllerCheckWorker(
        manager=sessions,
        claim=claim,
        config=ControllerCheckConfig(),
        reader=reader,
        clock=clock,
    ).run()

    assert result.terminal_state is ControlState.ERROR
    assert "join timed out" in result.reason
    status = sessions.status_for(claim.session_id, check_expiry=False)
    assert status is not None
    assert status.torque.outcome is TorqueOutcome.NOT_ATTEMPTED
    assert sessions.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        sessions.claim(ControlOperation.CONTROLLER_CHECK, session_id="must-not-reopen")


def test_module_import_is_free_of_pygame_lerobot_and_robot_modules() -> None:
    script = r"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    forbidden = ("pygame", "lerobot", "lelab.teleoperate", "lelab.record", "lelab.calibrate")
    if any(name == item or name.startswith(item + ".") for item in forbidden):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from lelab.stadia.controller_check import ControllerCheckConfig, ControllerCheckWorker
assert ControllerCheckConfig and ControllerCheckWorker
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_helper_can_retain_identity_while_disconnected() -> None:
    connected = snapshot(1)
    disconnected = replace(
        connected,
        sequence=2,
        connected=False,
        axes=(),
        buttons=(),
        read_error="removed",
        layout=ControllerLayout(0, 0, 0),
    )
    assert disconnected.guid == connected.guid
    assert disconnected.connected is False
