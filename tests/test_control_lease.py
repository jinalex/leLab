from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from lelab.control_session import (
    ControlResourceKey,
    ControlSessionBusyError,
    ControlSessionManager,
    ControlState,
    InvalidControlTransitionError,
    StaleSessionError,
    classify_torque_outcome,
)


class FakeClocks:
    def __init__(self) -> None:
        self.monotonic = 100.0
        self.utc = datetime(2026, 9, 2, 7, 0, tzinfo=UTC)

    def monotonic_now(self) -> float:
        return self.monotonic

    def utc_now(self) -> datetime:
        return self.utc

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.utc += timedelta(seconds=seconds)


def manager(clocks: FakeClocks) -> ControlSessionManager:
    return ControlSessionManager(
        monotonic_clock=clocks.monotonic_now,
        utc_clock=clocks.utc_now,
    )


def test_claim_starts_three_second_server_lease_and_advertises_one_second_renewal() -> None:
    clocks = FakeClocks()
    sessions = manager(clocks)
    claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="live-1")
    status = sessions.active_status(check_expiry=False)

    assert status is not None
    assert status.session_id == claim.session_id
    assert status.lease_deadline_monotonic == 103.0
    assert status.lease_ttl_s == 3.0
    assert status.lease_renew_interval_s == 1.0


def test_only_exact_starting_or_running_owner_can_renew() -> None:
    clocks = FakeClocks()
    sessions = manager(clocks)
    claim = sessions.claim(ControlResourceKey.STADIA_RECORDING, session_id="record-1")

    clocks.advance(1.0)
    assert sessions.renew_lease(claim.session_id).lease_deadline_monotonic == 104.0
    with pytest.raises(StaleSessionError):
        sessions.renew_lease("stale")

    sessions.mark_running(claim.session_id)
    clocks.advance(1.0)
    assert sessions.renew_lease(claim.session_id).lease_deadline_monotonic == 105.0
    sessions.request_stop(claim.session_id, reason="done")
    with pytest.raises(InvalidControlTransitionError, match="stopping"):
        sessions.renew_lease(claim.session_id)


def test_expiry_requests_hold_and_stop_but_does_not_release_control() -> None:
    clocks = FakeClocks()
    sessions = manager(clocks)
    claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="live-1")
    sessions.mark_running(claim.session_id)

    clocks.advance(3.0)
    assert sessions.check_lease_expiry()
    status = sessions.active_status(check_expiry=False)
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.stop_reason == "control lease expired"
    assert status.hold_requested
    assert status.stop_requested
    assert claim.hold_requested.is_set()
    assert claim.stop_requested.is_set()
    with pytest.raises(ControlSessionBusyError):
        sessions.claim(ControlResourceKey.CONTROLLER_CHECK, session_id="blocked")

    terminal = sessions.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.ERROR,
        torque=classify_torque_outcome(disable_attempted=True, readback=None),
    )
    assert terminal.state is ControlState.ERROR
    assert sessions.active_status() is None


def test_late_renewal_expires_before_it_can_extend_the_lease() -> None:
    clocks = FakeClocks()
    sessions = manager(clocks)
    claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="live-1")
    clocks.advance(3.1)

    with pytest.raises(InvalidControlTransitionError, match="stopping"):
        sessions.renew_lease(claim.session_id)
    assert sessions.active_status(check_expiry=False).state is ControlState.STOPPING  # type: ignore[union-attr]
    assert claim.hold_requested.is_set()
