from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from lelab.control_runtime import ControlRuntime
from lelab.control_session import (
    SO101_MOTOR_NAMES,
    ControlOperation,
    ControlSessionManager,
    ControlState,
    classify_torque_outcome,
)


class FakeClocks:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.monotonic = 100.0
        self.utc = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)

    def monotonic_now(self) -> float:
        with self._lock:
            return self.monotonic

    def utc_now(self) -> datetime:
        with self._lock:
            return self.utc

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.monotonic += seconds
            self.utc += timedelta(seconds=seconds)


def _manager(clocks: FakeClocks) -> ControlSessionManager:
    return ControlSessionManager(
        monotonic_clock=clocks.monotonic_now,
        utc_clock=clocks.utc_now,
    )


def _wait_until(predicate, *, timeout_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def test_watchdog_expires_lease_without_any_api_polling() -> None:
    clocks = FakeClocks()
    manager = _manager(clocks)
    claim = manager.claim(ControlOperation.STADIA_TELEOPERATION, session_id="live")
    manager.mark_running(claim.session_id)
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)
    runtime.start()

    clocks.advance(3.0)

    assert _wait_until(claim.stop_requested.is_set)
    status = manager.active_status(check_expiry=False)
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.stop_reason == "control lease expired"
    assert claim.hold_requested.is_set()
    manager.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.ERROR,
        torque=classify_torque_outcome(disable_attempted=False),
    )
    result = runtime.shutdown(timeout_s=0.5)
    assert result.teardown_complete
    assert result.watchdog_stopped
    assert runtime.watchdog_error is None


def test_shutdown_signals_owner_and_waits_for_real_teardown() -> None:
    clocks = FakeClocks()
    manager = _manager(clocks)
    claim = manager.claim(ControlOperation.LEADER_TELEOPERATION, session_id="leader")
    manager.mark_running(claim.session_id)
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)
    runtime.start()

    def owner() -> None:
        assert claim.stop_requested.wait(0.5)
        manager.finish_teardown(
            claim.session_id,
            terminal_state=ControlState.STOPPED,
            torque=classify_torque_outcome(
                disable_attempted=True,
                readback=dict.fromkeys(SO101_MOTOR_NAMES, 0),
            ),
        )

    owner_thread = threading.Thread(target=owner)
    owner_thread.start()
    result = runtime.shutdown(timeout_s=0.5)
    owner_thread.join(timeout=0.5)

    assert not owner_thread.is_alive()
    assert result.session_id == claim.session_id
    assert result.teardown_complete
    assert result.watchdog_stopped
    terminal = manager.status_for(claim.session_id)
    assert terminal is not None
    assert terminal.state is ControlState.STOPPED
    assert terminal.torque.outcome.value == "verified_off"


def test_shutdown_timeout_never_invents_terminal_state_or_releases_claim() -> None:
    clocks = FakeClocks()
    manager = _manager(clocks)
    claim = manager.claim(ControlOperation.STADIA_RECORDING, session_id="record")
    manager.mark_running(claim.session_id)
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)
    runtime.start()

    result = runtime.shutdown(timeout_s=0.01)

    assert result.session_id == claim.session_id
    assert not result.teardown_complete
    assert result.watchdog_stopped
    status = manager.active_status(check_expiry=False)
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.teardown_completed_at_utc is None
    assert not claim.teardown_completed.is_set()

    manager.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.ERROR,
        torque=classify_torque_outcome(disable_attempted=False),
    )


def test_shutdown_waits_for_owner_returned_by_atomic_begin_shutdown() -> None:
    clocks = FakeClocks()

    class ClaimDuringShutdownManager(ControlSessionManager):
        def begin_shutdown(self, *, reason: str = "server shutdown"):
            if self.active_status(check_expiry=False) is None:
                claim = self.claim(ControlOperation.STADIA_TELEOPERATION, session_id="raced-owner")
                self.mark_running(claim.session_id)
            return super().begin_shutdown(reason=reason)

    manager = ClaimDuringShutdownManager(
        monotonic_clock=clocks.monotonic_now,
        utc_clock=clocks.utc_now,
    )
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)

    result = runtime.shutdown(timeout_s=0)

    assert result.session_id == "raced-owner"
    assert not result.teardown_complete
    status = manager.active_status(check_expiry=False)
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.teardown_completed_at_utc is None
    manager.finish_teardown(
        "raced-owner",
        terminal_state=ControlState.ERROR,
        torque=classify_torque_outcome(disable_attempted=False),
    )


def test_watchdog_failure_is_retained_for_server_diagnostics() -> None:
    clocks = FakeClocks()

    class FailingWatchdogManager(ControlSessionManager):
        def check_lease_expiry(self) -> bool:
            raise RuntimeError("watchdog probe failed")

    manager = FailingWatchdogManager(
        monotonic_clock=clocks.monotonic_now,
        utc_clock=clocks.utc_now,
    )
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)

    runtime.start()

    assert _wait_until(lambda: not runtime.watchdog_alive)
    assert isinstance(runtime.watchdog_error, RuntimeError)
    assert str(runtime.watchdog_error) == "watchdog probe failed"
    result = runtime.shutdown(timeout_s=0.5)
    assert result.teardown_complete
    assert result.watchdog_stopped


def test_watchdog_failure_signals_active_owner_without_inventing_teardown() -> None:
    clocks = FakeClocks()

    class FailingWatchdogManager(ControlSessionManager):
        def check_lease_expiry(self) -> bool:
            raise RuntimeError("watchdog probe failed")

    manager = FailingWatchdogManager(
        monotonic_clock=clocks.monotonic_now,
        utc_clock=clocks.utc_now,
    )
    claim = manager.claim(ControlOperation.STADIA_TELEOPERATION, session_id="live")
    manager.mark_running(claim.session_id)
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)

    runtime.start()

    assert _wait_until(lambda: not runtime.watchdog_alive)
    assert isinstance(runtime.watchdog_error, RuntimeError)
    status = manager.active_status(check_expiry=False)
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.stop_reason == "control lease watchdog failed: RuntimeError: watchdog probe failed"
    assert claim.hold_requested.is_set()
    assert claim.stop_requested.is_set()
    assert not claim.teardown_completed.is_set()
    assert manager.closing

    terminal = manager.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.STOPPED,
        torque=classify_torque_outcome(disable_attempted=False),
        reason=status.stop_reason,
    )
    assert terminal.state is ControlState.ERROR
    assert claim.teardown_completed.is_set()
    result = runtime.shutdown(timeout_s=0.5)
    assert result.teardown_complete
    assert result.watchdog_stopped


def test_runtime_rejects_invalid_intervals_timeouts_and_restart_after_shutdown() -> None:
    clocks = FakeClocks()
    manager = _manager(clocks)
    for value in (True, 0, -1, float("inf"), 3.0):
        with pytest.raises(ValueError):
            ControlRuntime(manager, watchdog_interval_s=value)

    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)
    runtime.start()
    runtime.start()
    for value in (True, -1, float("nan")):
        with pytest.raises(ValueError):
            runtime.shutdown(timeout_s=value)

    result = runtime.shutdown(timeout_s=0.5)
    assert result.session_id is None
    assert result.teardown_complete
    assert result.watchdog_stopped
    with pytest.raises(RuntimeError, match="shutting down"):
        runtime.start()


def test_quarantined_terminal_reports_shutdown_as_incomplete() -> None:
    clocks = FakeClocks()
    manager = _manager(clocks)
    claim = manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="quarantined")
    manager.mark_running(claim.session_id)
    manager.request_failure(claim.session_id, reason="reader close failed")
    manager.quarantine(reason="reader release could not be proven")
    manager.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.ERROR,
        torque=classify_torque_outcome(disable_attempted=False),
    )
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)

    result = runtime.shutdown(timeout_s=0.1)

    assert result.session_id == claim.session_id
    assert result.teardown_complete is False
    assert result.quarantine_reason == "reader release could not be proven"
    assert result.watchdog_stopped
