from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from lelab.control_session import (
    ControlManagerClosingError,
    ControlResourceKey,
    ControlSessionBusyError,
    ControlSessionManager,
    ControlState,
    InvalidControlTransitionError,
    StaleSessionError,
    classify_torque_outcome,
)


def manager(**kwargs: object) -> ControlSessionManager:
    defaults = {
        "monotonic_clock": lambda: 10.0,
        "utc_clock": lambda: datetime(2026, 9, 2, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return ControlSessionManager(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", list(ControlResourceKey))
def test_every_operation_claims_global_control_with_an_audit_key(
    operation: ControlResourceKey,
) -> None:
    sessions = manager()
    claim = sessions.claim(operation, session_id=f"session-{operation.value}")

    assert claim.resource_keys == ("control", operation.value)
    assert sessions.active_status().resource_keys == claim.resource_keys  # type: ignore[union-attr]
    with pytest.raises(ControlSessionBusyError):
        sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="other")


def test_operation_audit_keys_are_the_exact_readiness_operations() -> None:
    assert {operation.value for operation in ControlResourceKey} == {
        "follower_calibration",
        "leader_calibration",
        "leader_teleoperation",
        "stadia_teleoperation",
        "leader_recording",
        "stadia_recording",
        "inference",
        "replay",
        "controller_check",
    }


def test_claim_rejects_unknown_teleoperator_types() -> None:
    sessions = manager()

    with pytest.raises(ValueError, match="teleoperator_type"):
        sessions.claim(
            ControlResourceKey.STADIA_TELEOPERATION,
            session_id="bad-mode",
            teleoperator_type="gamepad",
        )


def test_operation_and_teleoperator_type_cannot_contradict() -> None:
    sessions = manager()

    with pytest.raises(ValueError, match="requires teleoperator_type"):
        sessions.claim(
            ControlResourceKey.STADIA_TELEOPERATION,
            session_id="wrong-mode",
            teleoperator_type="leader_arm",
        )

    inference = sessions.claim(
        ControlResourceKey.INFERENCE,
        session_id="inference-stadia",
        teleoperator_type="stadia",
    )
    assert sessions.status_for(inference.session_id).teleoperator_type == "stadia"  # type: ignore[union-attr]


def test_atomic_claim_allows_exactly_one_concurrent_owner() -> None:
    sessions = manager(session_id_factory=lambda: "generated")
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, str]] = []
    outcomes_lock = threading.Lock()

    def contender(name: str) -> None:
        barrier.wait()
        try:
            claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id=name)
            result = ("claimed", claim.session_id)
        except ControlSessionBusyError:
            result = ("busy", name)
        with outcomes_lock:
            outcomes.append(result)

    threads = [threading.Thread(target=contender, args=(name,)) for name in ("one", "two")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(outcome for outcome, _ in outcomes) == ["busy", "claimed"]
    assert sessions.active_status().session_id in {"one", "two"}  # type: ignore[union-attr]


def test_exact_owner_follows_full_lifecycle_and_only_teardown_releases() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.STADIA_RECORDING, session_id="record-1")

    with pytest.raises(StaleSessionError):
        sessions.mark_running("stale")
    assert sessions.mark_running(claim.session_id).state is ControlState.RUNNING

    stopping = sessions.request_stop(claim.session_id, reason="operator stop")
    assert stopping.state is ControlState.STOPPING
    assert claim.hold_requested.is_set()
    assert claim.stop_requested.is_set()
    with pytest.raises(ControlSessionBusyError):
        sessions.claim(ControlResourceKey.REPLAY, session_id="too-soon")
    with pytest.raises(InvalidControlTransitionError):
        sessions.mark_running(claim.session_id)

    terminal = sessions.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.STOPPED,
        torque=classify_torque_outcome(disable_attempted=False),
    )
    assert terminal.state is ControlState.STOPPED
    assert terminal.stop_reason == "operator stop"
    assert sessions.active_status() is None
    assert sessions.current_status() is terminal
    assert claim.teardown_completed.is_set()
    assert sessions.wait_for_teardown(claim.session_id, timeout=0)

    next_claim = sessions.claim(ControlResourceKey.REPLAY, session_id="replay-1")
    assert sessions.status_for(claim.session_id) is terminal
    with pytest.raises(StaleSessionError):
        sessions.update_details(claim.session_id, {"late": True})
    assert next_claim.session_id == "replay-1"


def test_failure_request_cannot_be_downgraded_by_owner_teardown() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="fatal-1")
    sessions.mark_running(claim.session_id)

    stopping = sessions.request_failure(claim.session_id, reason="lease watchdog failed")
    assert stopping.state is ControlState.STOPPING
    assert stopping.stop_reason == "lease watchdog failed"

    terminal = sessions.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.STOPPED,
        torque=classify_torque_outcome(disable_attempted=False),
        reason="worker stopped cooperatively",
    )

    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "worker stopped cooperatively"


def test_failure_request_replaces_an_older_normal_stop_reason() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.CONTROLLER_CHECK, session_id="fatal-2")
    sessions.mark_running(claim.session_id)
    sessions.request_stop(claim.session_id, reason="operator stop")

    stopping = sessions.request_failure(claim.session_id, reason="monitor failed")

    assert stopping.state is ControlState.STOPPING
    assert stopping.stop_reason == "monitor failed"
    assert stopping.revision > 2


def test_bounded_teardown_wait_tracks_exact_owner_and_never_synthesizes_terminal() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="live-1")
    sessions.mark_running(claim.session_id)
    sessions.request_stop(claim.session_id, reason="operator stop")

    assert not sessions.wait_for_teardown(claim.session_id, timeout=0)
    assert sessions.status_for(claim.session_id).state is ControlState.STOPPING  # type: ignore[union-attr]

    finished = threading.Event()

    def waiter() -> None:
        if sessions.wait_for_teardown(claim.session_id, timeout=1):
            finished.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    sessions.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.STOPPED,
        torque=classify_torque_outcome(disable_attempted=False),
    )
    thread.join(timeout=1)

    assert finished.is_set()
    assert not thread.is_alive()


def test_teardown_cannot_release_before_stopping() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.INFERENCE, session_id="inference-1")

    with pytest.raises(InvalidControlTransitionError, match="only after"):
        sessions.finish_teardown(
            claim.session_id,
            terminal_state=ControlState.ERROR,
            torque=classify_torque_outcome(disable_attempted=False),
        )
    with pytest.raises(ControlSessionBusyError):
        sessions.claim(ControlResourceKey.FOLLOWER_CALIBRATION, session_id="blocked")


def test_shutdown_signals_owner_without_inventing_completion() -> None:
    sessions = manager()
    claim = sessions.claim(ControlResourceKey.FOLLOWER_CALIBRATION, session_id="calibration-1")
    sessions.mark_running(claim.session_id)

    status = sessions.begin_shutdown()

    assert sessions.closing
    assert status is not None
    assert status.state is ControlState.STOPPING
    assert status.stop_reason == "server shutdown"
    assert claim.hold_requested.is_set()
    assert claim.stop_requested.is_set()
    assert sessions.terminal_history() == ()
    with pytest.raises(ControlManagerClosingError):
        sessions.claim(ControlResourceKey.CONTROLLER_CHECK, session_id="check-1")
