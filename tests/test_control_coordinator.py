"""Dependency-neutral server coordination and legacy-adapter tests."""

from __future__ import annotations

import builtins
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lelab.control_coordinator import ControlCoordinator, ControlOwnerStartError
from lelab.control_runtime import ControlRuntime
from lelab.control_session import (
    NOT_ATTEMPTED_TORQUE,
    ControlManagerClosingError,
    ControlOperation,
    ControlSessionBusyError,
    ControlSessionManager,
    ControlState,
    StaleSessionError,
    TorqueOutcome,
)


def wait_for_terminal(
    manager: ControlSessionManager,
    session_id: str,
    *,
    timeout: float = 1.0,
):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status_for(session_id, check_expiry=False)
        if status is not None and status.terminal:
            return status
        time.sleep(0.002)
    pytest.fail(f"session {session_id!r} did not become terminal")


class FakeManagedWorker:
    def __init__(self, manager: ControlSessionManager, claim) -> None:  # type: ignore[no-untyped-def]
        self.manager = manager
        self.claim = claim
        self.thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()

    def join(self, *, timeout: float | None = None) -> object:
        assert self.thread is not None
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("fake worker still alive")
        return object()

    def _run(self) -> None:
        self.manager.mark_running(self.claim.session_id)
        assert self.claim.stop_requested.wait(1.0)
        status = self.manager.status_for(self.claim.session_id, check_expiry=False)
        assert status is not None and status.state is ControlState.STOPPING
        self.manager.finish_teardown(
            self.claim.session_id,
            terminal_state=ControlState.STOPPED,
            torque=NOT_ATTEMPTED_TORQUE,
            reason=status.stop_reason,
        )


def test_managed_worker_owns_terminal_publication_and_exact_stop() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    workers: list[FakeManagedWorker] = []

    def build(claim):  # type: ignore[no-untyped-def]
        worker = FakeManagedWorker(manager, claim)
        workers.append(worker)
        return worker

    claim, _status = coordinator.start_managed_worker(
        ControlOperation.STADIA_TELEOPERATION,
        build,
        teleoperator_type="stadia",
        details={"robot_name": "Desk Arm"},
    )
    with pytest.raises(StaleSessionError):
        coordinator.request_stop("some-other-session")

    stopping = coordinator.request_stop(claim.session_id, reason="UI requested stop")
    assert stopping.state is ControlState.STOPPING
    terminal = wait_for_terminal(manager, claim.session_id)
    assert terminal.state is ControlState.STOPPED
    assert terminal.stop_reason == "UI requested stop"
    assert workers[0].is_alive is False


def test_live_stadia_speed_is_dispatched_only_to_the_exact_managed_owner() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)

    class SpeedWorker(FakeManagedWorker):
        def set_speed_multiplier(self, multiplier: float):  # type: ignore[no-untyped-def]
            return self.manager.merge_details(
                self.claim.session_id,
                {"stadia_speed_multiplier": multiplier},
            )

    worker: SpeedWorker | None = None

    def build(claim):  # type: ignore[no-untyped-def]
        nonlocal worker
        worker = SpeedWorker(manager, claim)
        return worker

    claim, _ = coordinator.start_managed_worker(
        ControlOperation.STADIA_TELEOPERATION,
        build,
        teleoperator_type="stadia",
    )
    deadline = time.monotonic() + 1.0
    while manager.active_status(check_expiry=False).state is ControlState.STARTING:
        assert time.monotonic() < deadline
        time.sleep(0.002)

    updated = coordinator.set_stadia_speed(claim.session_id, 1.5)

    assert updated.details["stadia_speed_multiplier"] == 1.5
    with pytest.raises(StaleSessionError):
        coordinator.set_stadia_speed("another-session", 1.5)
    coordinator.request_stop(claim.session_id)
    wait_for_terminal(manager, claim.session_id)


def test_managed_worker_construction_failure_returns_issued_terminal_status() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager)

    with pytest.raises(ControlOwnerStartError) as caught:
        coordinator.start_managed_worker(
            ControlOperation.CONTROLLER_CHECK,
            lambda _claim: (_ for _ in ()).throw(RuntimeError("factory failed")),
            teleoperator_type="stadia",
        )

    assert caught.value.status.state is ControlState.ERROR
    assert caught.value.status.teardown_completed_at_utc is not None
    assert caught.value.as_result()["session_id"] == caught.value.status.session_id


def test_managed_monitor_start_failure_requests_stop_and_proves_worker_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    workers: list[FakeManagedWorker] = []

    def build(claim):  # type: ignore[no-untyped-def]
        worker = FakeManagedWorker(manager, claim)
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        coordinator,
        "_start_monitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("monitor unavailable")),
    )

    with pytest.raises(ControlOwnerStartError) as caught:
        coordinator.start_managed_worker(
            ControlOperation.STADIA_TELEOPERATION,
            build,
            teleoperator_type="stadia",
        )

    assert caught.value.status.terminal
    assert caught.value.status.state is ControlState.ERROR
    assert workers[0].is_alive is False
    assert manager.active_status(check_expiry=False) is None
    assert coordinator._owner is None


def test_unproven_managed_monitor_failure_retains_exact_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)

    class StubbornWorker:
        is_alive = True

        def start(self) -> None:
            pass

        def join(self, *, timeout: float | None = None) -> None:
            raise TimeoutError(f"still alive after {timeout}")

    worker = StubbornWorker()
    monkeypatch.setattr(
        coordinator,
        "_start_monitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("monitor unavailable")),
    )

    with pytest.raises(ControlOwnerStartError) as caught:
        coordinator.start_managed_worker(
            ControlOperation.STADIA_TELEOPERATION,
            lambda _claim: worker,
            teleoperator_type="stadia",
        )

    assert caught.value.status.state is ControlState.STOPPING
    assert manager.active_status(check_expiry=False) is caught.value.status
    assert coordinator._owner is not None
    assert caught.value.status.teardown_completed_at_utc is None

    worker.is_alive = False
    terminal = wait_for_terminal(manager, caught.value.status.session_id)
    assert terminal.state is ControlState.ERROR
    assert coordinator._owner is None
    assert manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        manager.claim(ControlOperation.CONTROLLER_CHECK, session_id="after-recovery")


def test_delayed_managed_teardown_is_reaped_without_early_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    monkeypatch.setattr("lelab.control_coordinator.MONITOR_START_RECOVERY_TIMEOUT_S", 0.01)
    workers: list[FakeManagedWorker] = []

    class DelayedWorker(FakeManagedWorker):
        def _run(self) -> None:
            self.manager.mark_running(self.claim.session_id)
            assert self.claim.stop_requested.wait(1.0)
            time.sleep(0.03)
            status = self.manager.status_for(self.claim.session_id, check_expiry=False)
            assert status is not None
            self.manager.finish_teardown(
                self.claim.session_id,
                terminal_state=ControlState.STOPPED,
                torque=NOT_ATTEMPTED_TORQUE,
                reason=status.stop_reason,
            )

    def build(claim):  # type: ignore[no-untyped-def]
        worker = DelayedWorker(manager, claim)
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        coordinator,
        "_start_monitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("monitor unavailable")),
    )

    with pytest.raises(ControlOwnerStartError) as caught:
        coordinator.start_managed_worker(
            ControlOperation.STADIA_TELEOPERATION,
            build,
            teleoperator_type="stadia",
        )

    assert caught.value.status.state is ControlState.STOPPING
    terminal = wait_for_terminal(manager, caught.value.status.session_id)
    deadline = time.monotonic() + 0.5
    while coordinator._owner is not None and time.monotonic() < deadline:
        time.sleep(0.002)
    assert terminal.state is ControlState.ERROR
    assert coordinator._owner is None
    assert manager.quarantine_reason is None


def test_stadia_recording_returns_stamped_id_and_exposes_only_exact_active_worker() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    resolved = SimpleNamespace(
        teleoperator_type="stadia",
        robot_name="Desk Arm",
        metadata_dict=lambda: {"dataset_repo_id": "alex/requested"},
    )
    workers: list[FakeManagedWorker] = []

    def build(claim, _resolved):  # type: ignore[no-untyped-def]
        worker = FakeManagedWorker(manager, claim)
        worker.dataset_repo_id = "alex/requested_20260902_041500"
        workers.append(worker)
        return worker

    result = coordinator.start_recording(
        {"robot_name": "Desk Arm"},
        resolver=lambda _request: resolved,
        stadia_worker_factory=build,
    )
    session_id = str(result["session_id"])

    assert result["dataset_id"] == "alex/requested_20260902_041500"
    assert (
        coordinator.active_managed_worker(
            session_id,
            operation=ControlOperation.STADIA_RECORDING,
        )
        is workers[0]
    )
    assert (
        coordinator.active_managed_worker(
            session_id,
            operation=ControlOperation.STADIA_TELEOPERATION,
        )
        is None
    )
    assert coordinator.active_managed_worker("another-session") is None

    coordinator.request_stop(session_id)
    wait_for_terminal(manager, session_id)
    assert coordinator.active_managed_worker(session_id) is None


def test_external_legacy_handler_keeps_its_start_and_stop_functions() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    calls: list[str] = []

    def start() -> dict[str, object]:
        calls.append("legacy.start")
        active.set()
        return {"success": True, "message": "legacy started", "legacy": 1}

    def stop() -> dict[str, object]:
        calls.append("legacy.stop")
        active.clear()
        return {"success": True}

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=start,
        stop=stop,
        is_active=active.is_set,
        details={"robot_name": "legacy"},
    )

    assert result["success"] is True
    assert result["legacy"] == 1
    session_id = str(result["session_id"])
    stopping = coordinator.request_stop(session_id)
    assert stopping.state is ControlState.STOPPING
    terminal = wait_for_terminal(manager, session_id)
    assert calls == ["legacy.start", "legacy.stop"]
    assert terminal.state is ControlState.STOPPED
    assert terminal.torque.outcome is TorqueOutcome.NOT_ATTEMPTED
    assert terminal.torque.outcome is not TorqueOutcome.VERIFIED_OFF


def test_external_monitor_start_failure_stops_and_proves_inactivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    calls: list[str] = []
    monkeypatch.setattr(
        coordinator,
        "_start_monitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("monitor unavailable")),
    )

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=lambda: active.set() or {"success": True},
        stop=lambda: calls.append("stop") or active.clear() or {"success": True},
        is_active=active.is_set,
    )

    assert result["success"] is False
    assert calls == ["stop"]
    assert result["status"]["state"] == "error"
    assert manager.active_status(check_expiry=False) is None
    assert coordinator._owner is None


def test_external_monitor_start_failure_retains_owner_when_inactivity_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002, sleeper=lambda _s: None)
    monkeypatch.setattr("lelab.control_coordinator.MONITOR_START_RECOVERY_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        coordinator,
        "_start_monitor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("monitor unavailable")),
    )

    active = threading.Event()
    active.set()
    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=lambda: {"success": True},
        stop=lambda: {"success": True},
        is_active=active.is_set,
    )

    assert result["success"] is False
    assert result["status"]["state"] == "stopping"
    assert manager.active_status(check_expiry=False) is not None
    assert coordinator._owner is not None

    session_id = str(result["session_id"])
    active.clear()
    terminal = wait_for_terminal(manager, session_id)
    assert terminal.state is ControlState.ERROR
    assert coordinator._owner is None


def test_managed_monitor_internal_failure_signals_and_reaps_owner() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)

    class FlakyWorker:
        def __init__(self, claim) -> None:  # type: ignore[no-untyped-def]
            self.claim = claim
            self.probes = 0

        def start(self) -> None:
            manager.mark_running(self.claim.session_id)

        @property
        def is_alive(self) -> bool:
            self.probes += 1
            if self.probes == 1:
                raise RuntimeError("liveness probe failed")
            return False

        def join(self, *, timeout: float | None = None) -> None:
            return None

    workers: list[FlakyWorker] = []

    def build(claim):  # type: ignore[no-untyped-def]
        worker = FlakyWorker(claim)
        workers.append(worker)
        return worker

    claim, _status = coordinator.start_managed_worker(
        ControlOperation.CONTROLLER_CHECK,
        build,
        teleoperator_type="stadia",
        details={"test": True},
    )

    terminal = wait_for_terminal(manager, claim.session_id)
    assert terminal.state is ControlState.ERROR
    assert "liveness probe failed" in (terminal.stop_reason or "")
    assert workers[0].probes >= 2
    assert coordinator._owner is None


def test_startup_heartbeat_thread_failure_terminalizes_before_legacy_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    start_called = False

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("threads unavailable")

    def legacy_start() -> dict[str, bool]:
        nonlocal start_called
        start_called = True
        return {"success": True}

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=legacy_start,
        stop=lambda: {"success": True},
        is_active=lambda: False,
    )

    assert result["success"] is False
    assert result["status"]["state"] == "error"
    assert not start_called
    assert manager.active_status(check_expiry=False) is None


def test_external_stop_failure_response_is_persisted_as_terminal_error() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()

    def stop() -> dict[str, object]:
        active.clear()
        return {"success": False, "message": "disable acknowledgement missing"}

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=lambda: active.set() or {"success": True},
        stop=stop,
        is_active=active.is_set,
    )

    coordinator.request_stop(str(result["session_id"]))
    terminal = wait_for_terminal(manager, str(result["session_id"]))

    assert terminal.state is ControlState.ERROR
    assert "disable acknowledgement missing" in (terminal.stop_reason or "")
    assert terminal.torque.outcome is TorqueOutcome.NOT_ATTEMPTED


def test_external_natural_failure_uses_retained_terminal_evidence() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    active.set()
    terminal_snapshot: dict[str, object] = {
        "exited": False,
        "outcome": "running",
        "error": None,
        "cleanup_pending": False,
    }

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=lambda: {"success": True},
        stop=lambda: active.clear() or {"success": True},
        is_active=active.is_set,
        terminal_status=lambda: terminal_snapshot,
    )

    terminal_snapshot.update(
        {
            "exited": True,
            "outcome": "failed",
            "error": "leader action read failed",
        }
    )
    active.clear()
    terminal = wait_for_terminal(manager, str(result["session_id"]))

    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "leader action read failed"
    assert manager.quarantine_reason is None


def test_external_unproven_cleanup_quarantines_after_worker_exit() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    active.set()

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_RECORDING,
        teleoperator_type="leader_arm",
        start=lambda: {"success": True},
        stop=lambda: active.clear() or {"success": True},
        is_active=active.is_set,
        terminal_status=lambda: {
            "exited": True,
            "outcome": "failed",
            "error": "serial close failed",
            "cleanup_pending": True,
        },
    )

    active.clear()
    terminal = wait_for_terminal(manager, str(result["session_id"]))

    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "serial close failed"
    assert manager.quarantine_reason == "serial close failed"
    with pytest.raises(ControlManagerClosingError, match="serial close failed"):
        manager.claim(ControlOperation.INFERENCE)


def test_external_failed_stop_is_retried_until_process_exit_is_proven() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    active.set()
    stop_calls = 0

    def stop() -> dict[str, object]:
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            return {"success": False, "message": "process still running", "stop_pending": True}
        active.clear()
        return {"success": True}

    result = coordinator.start_external_operation(
        ControlOperation.INFERENCE,
        teleoperator_type="stadia",
        start=lambda: {"success": True},
        stop=stop,
        is_active=active.is_set,
        terminal_status=lambda: {
            "exited": not active.is_set(),
            "outcome": "stopped" if not active.is_set() else "running",
            "error": None,
        },
    )

    coordinator.request_stop(str(result["session_id"]))
    terminal = wait_for_terminal(manager, str(result["session_id"]))

    assert stop_calls == 2
    assert terminal.state is ControlState.STOPPED
    assert terminal.stop_reason == "stop requested by control UI"


def test_inactive_failed_start_with_unproven_cleanup_never_reopens_manager() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_TELEOPERATION,
        teleoperator_type="leader_arm",
        start=lambda: {"success": False, "message": "leader connect failed"},
        stop=lambda: {"success": True},
        is_active=lambda: False,
        terminal_status=lambda: {
            "exited": True,
            "outcome": "failed",
            "error": "serial close could not be verified",
            "cleanup_pending": True,
        },
    )

    status = manager.status_for(str(result["session_id"]), check_expiry=False)
    assert status is not None and status.state is ControlState.ERROR
    assert "serial close could not be verified" in (status.stop_reason or "")
    assert manager.quarantine_reason is not None
    with pytest.raises(ControlManagerClosingError):
        manager.claim(ControlOperation.INFERENCE)


def test_failed_external_start_releases_only_after_inactive_is_proven() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_RECORDING,
        teleoperator_type="leader_arm",
        start=lambda: {"success": False, "message": "camera unavailable"},
        stop=lambda: {"success": True},
        is_active=lambda: False,
    )

    assert result["success"] is False
    terminal = manager.status_for(str(result["session_id"]), check_expiry=False)
    assert terminal is not None
    assert terminal.state is ControlState.ERROR
    assert "camera unavailable" in (terminal.stop_reason or "")
    replacement = manager.claim(ControlOperation.INFERENCE)
    assert replacement.session_id != result["session_id"]


def test_failed_external_start_that_left_work_active_stays_claimed_until_stop() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    active.set()
    allow_cleanup = threading.Event()

    def stop() -> dict[str, bool]:
        assert allow_cleanup.wait(1.0)
        active.clear()
        return {"success": True}

    result = coordinator.start_external_operation(
        ControlOperation.LEADER_RECORDING,
        teleoperator_type="leader_arm",
        start=lambda: {"success": False, "message": "start response failed"},
        stop=stop,
        is_active=active.is_set,
    )

    assert result["success"] is False
    with pytest.raises(ControlSessionBusyError):
        manager.claim(ControlOperation.INFERENCE)
    allow_cleanup.set()
    terminal = wait_for_terminal(manager, str(result["session_id"]))
    assert terminal.state is ControlState.ERROR
    assert "start response failed" in (terminal.stop_reason or "")


def test_unreadable_external_activity_is_stopped_under_retained_ownership() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = True
    probes = 0
    stopped = threading.Event()

    def is_active() -> bool:
        nonlocal probes
        probes += 1
        if probes == 1:
            raise RuntimeError("activity registry unavailable")
        return active

    def stop() -> dict[str, bool]:
        nonlocal active
        active = False
        stopped.set()
        return {"success": True}

    result = coordinator.start_external_operation(
        ControlOperation.INFERENCE,
        teleoperator_type="stadia",
        start=lambda: {"success": True},
        stop=stop,
        is_active=is_active,
    )

    assert result["success"] is False
    assert "activity registry unavailable" in str(result["message"])
    assert stopped.wait(0.5)
    terminal = wait_for_terminal(manager, str(result["session_id"]))
    assert terminal.state is ControlState.ERROR
    assert "activity registry unavailable" in (terminal.stop_reason or "")


def test_runtime_activity_probe_failure_requests_stop_and_keeps_monitoring() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = True
    probes = 0
    stopped = threading.Event()

    def is_active() -> bool:
        nonlocal probes
        probes += 1
        if probes == 1:
            return active
        if probes == 2:
            raise RuntimeError("process poll failed")
        return active

    def stop() -> dict[str, bool]:
        nonlocal active
        active = False
        stopped.set()
        return {"success": True}

    result = coordinator.start_external_operation(
        ControlOperation.INFERENCE,
        teleoperator_type="stadia",
        start=lambda: {"success": True},
        stop=stop,
        is_active=is_active,
    )

    assert result["success"] is True
    assert stopped.wait(0.5)
    terminal = wait_for_terminal(manager, str(result["session_id"]))
    assert terminal.state is ControlState.ERROR
    assert "process poll failed" in (terminal.stop_reason or "")


def test_synchronous_legacy_start_gets_only_a_bounded_server_heartbeat() -> None:
    manager = ControlSessionManager(lease_ttl_s=0.08, lease_renew_interval_s=0.02)
    runtime = ControlRuntime(manager, watchdog_interval_s=0.005)
    coordinator = ControlCoordinator(
        manager,
        runtime=runtime,
        monitor_interval_s=0.002,
    )
    active = threading.Event()
    runtime.start()
    try:
        result = coordinator.start_external_operation(
            ControlOperation.LEADER_TELEOPERATION,
            teleoperator_type="leader_arm",
            start=lambda: time.sleep(0.14) or active.set() or {"success": True},
            stop=lambda: active.clear() or {"success": True},
            is_active=active.is_set,
        )
        status = manager.status_for(str(result["session_id"]), check_expiry=False)
        assert result["success"] is True
        assert status is not None and status.state is ControlState.RUNNING
        coordinator.request_stop(str(result["session_id"]))
        wait_for_terminal(manager, str(result["session_id"]))
    finally:
        coordinator.shutdown(timeout_s=0.2)


def test_teleoperation_adapter_uses_canonical_record_for_legacy_payload() -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    active = threading.Event()
    received: list[tuple[dict[str, object], object]] = []
    socket_manager = object()
    resolved = SimpleNamespace(
        teleoperator_type="leader_arm",
        robot_name="saved-arm",
        legacy_handler_payload=lambda: {
            "leader_port": "/saved/leader",
            "follower_port": "/saved/follower",
            "leader_config": "saved-leader.json",
            "follower_config": "saved-follower.json",
        },
    )

    def legacy_start(payload, websocket):  # type: ignore[no-untyped-def]
        received.append((dict(payload), websocket))
        active.set()
        return {"success": True}

    result = coordinator.start_teleoperation(
        {"robot_name": "browser-value"},
        websocket_manager=socket_manager,
        resolver=lambda _request: resolved,
        leader_start=legacy_start,
        leader_stop=lambda: active.clear() or {"success": True},
        leader_is_active=active.is_set,
    )

    assert result["success"] is True
    assert received == [
        (
            {
                "leader_port": "/saved/leader",
                "follower_port": "/saved/follower",
                "leader_config": "saved-leader.json",
                "follower_config": "saved-follower.json",
            },
            socket_manager,
        )
    ]
    coordinator.request_stop(str(result["session_id"]))
    wait_for_terminal(manager, str(result["session_id"]))


def test_default_stadia_teleoperation_receives_the_websocket_broadcaster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    socket_manager = SimpleNamespace(broadcast_joint_data_sync=lambda _payload: None)
    record = SimpleNamespace(stadia=SimpleNamespace(max_step_per_tick=0.35))
    resolved = SimpleNamespace(
        teleoperator_type="stadia",
        robot_name="saved-arm",
        record_model=lambda: record,
    )
    received: list[object] = []
    workers: list[FakeManagedWorker] = []

    def build(claim, _resolved, *, websocket_manager=None):  # type: ignore[no-untyped-def]
        received.append(websocket_manager)
        worker = FakeManagedWorker(manager, claim)
        workers.append(worker)
        return worker

    monkeypatch.setattr(coordinator, "_default_stadia_worker", build)

    result = coordinator.start_teleoperation(
        {"robot_name": "saved-arm"},
        websocket_manager=socket_manager,
        resolver=lambda _request: resolved,
    )

    assert result["success"] is True
    assert received == [socket_manager]
    assert result["status"]["details"]["stadia_speed_multiplier"] == 1.0
    assert result["status"]["details"]["stadia_effective_max_step_per_tick"] == 0.35
    coordinator.request_stop(str(result["session_id"]))
    wait_for_terminal(manager, str(result["session_id"]))


def test_default_legacy_adapter_retains_worker_until_thread_really_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab

    class LingeringThread:
        alive = True

        def is_alive(self) -> bool:
            return self.alive

    worker = LingeringThread()
    stop_called = threading.Event()
    legacy = SimpleNamespace(
        teleoperation_active=False,
        teleoperation_thread=None,
        TeleoperateRequest=SimpleNamespace(model_validate=lambda payload: payload),
    )

    def start(_request: object, _websocket: object | None) -> dict[str, bool]:
        legacy.teleoperation_active = True
        legacy.teleoperation_thread = worker
        return {"success": True}

    def stop() -> dict[str, bool]:
        legacy.teleoperation_active = False
        legacy.teleoperation_thread = None
        stop_called.set()
        return {"success": True}

    legacy.handle_start_teleoperation = start
    legacy.handle_stop_teleoperation = stop
    legacy.handle_teleoperation_status = lambda: {
        "exited": not worker.alive,
        "outcome": "stopped" if not worker.alive else "running",
        "error": None,
        "cleanup_pending": False,
    }
    monkeypatch.setitem(sys.modules, "lelab.teleoperate", legacy)
    monkeypatch.setattr(lelab, "teleoperate", legacy, raising=False)

    resolved = SimpleNamespace(
        teleoperator_type="leader_arm",
        robot_name="legacy",
        legacy_handler_payload=lambda: {},
    )
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    result = coordinator.start_teleoperation({}, resolver=lambda _request: resolved)
    session_id = str(result["session_id"])

    coordinator.request_stop(session_id)
    assert stop_called.wait(0.5)
    time.sleep(0.01)
    status = manager.status_for(session_id, check_expiry=False)
    assert status is not None and status.state is ControlState.STOPPING

    worker.alive = False
    terminal = wait_for_terminal(manager, session_id)
    assert terminal.state is ControlState.STOPPED


def test_default_legacy_adapter_maps_retained_worker_failure_to_shared_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab

    class Worker:
        alive = True

        def is_alive(self) -> bool:
            return self.alive

    worker = Worker()
    legacy = SimpleNamespace(
        teleoperation_active=False,
        teleoperation_thread=None,
        TeleoperateRequest=SimpleNamespace(model_validate=lambda payload: payload),
    )
    terminal_snapshot: dict[str, object] = {
        "exited": False,
        "outcome": "running",
        "error": None,
        "cleanup_pending": False,
    }

    def start(_request: object, _websocket: object | None) -> dict[str, bool]:
        legacy.teleoperation_active = True
        legacy.teleoperation_thread = worker
        return {"success": True}

    legacy.handle_start_teleoperation = start
    legacy.handle_stop_teleoperation = lambda: {"success": True}
    legacy.handle_teleoperation_status = lambda: terminal_snapshot
    monkeypatch.setitem(sys.modules, "lelab.teleoperate", legacy)
    monkeypatch.setattr(lelab, "teleoperate", legacy, raising=False)

    resolved = SimpleNamespace(
        teleoperator_type="leader_arm",
        robot_name="legacy",
        legacy_handler_payload=lambda: {},
    )
    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    result = coordinator.start_teleoperation({}, resolver=lambda _request: resolved)

    terminal_snapshot.update(
        {
            "exited": True,
            "outcome": "failed",
            "error": "legacy leader loop failed",
        }
    )
    legacy.teleoperation_active = False
    worker.alive = False
    terminal = wait_for_terminal(manager, str(result["session_id"]))

    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "legacy leader loop failed"


def test_controller_check_distinguishes_malformed_saved_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.utils import config

    robots = tmp_path / "robots"
    robots.mkdir()
    (robots / "broken.json").write_text("{")
    monkeypatch.setattr(config, "ROBOTS_PATH", str(robots))

    result = ControlCoordinator().start_controller_check("broken")

    assert result["success"] is False
    assert "exists but is not valid JSON" in str(result["message"])


def test_control_coordinator_import_has_no_device_dependencies() -> None:
    script = r"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pygame" or name.startswith("pygame.") or name == "lerobot" or name.startswith("lerobot."):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from lelab.control_coordinator import ControlCoordinator
assert ControlCoordinator
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_module_does_not_import_devices_in_current_process(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "pygame" or name.startswith("pygame."):
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert ControlCoordinator
