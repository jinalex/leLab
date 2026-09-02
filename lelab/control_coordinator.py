"""Server-owned coordination for all control-resource operations.

The central :class:`ControlSessionManager` is the only exclusivity authority.
This module imports no robot, camera, controller, or LeRobot classes at import
time.  Device-owning workers and the unchanged legacy handlers are supplied at
the call boundary, which keeps software tests dependency-neutral.
"""

from __future__ import annotations

import contextlib
import inspect
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from .control_requests import (
    ControlRequestError,
    ResolvedControlRequest,
    resolve_recording_request,
    resolve_teleoperate_request,
)
from .control_runtime import ControlRuntime, RuntimeShutdownResult
from .control_session import (
    NOT_ATTEMPTED_TORQUE,
    ControlOperation,
    ControlSessionClaim,
    ControlSessionManager,
    ControlState,
    ControlStatus,
    InvalidControlTransitionError,
    StaleSessionError,
)
from .utils import config as config_module
from .utils.config import (
    RobotOperation,
    RobotRecordValidationError,
    evaluate_robot_readiness,
    get_robot_record_v2,
    is_valid_robot_name,
)

MONITOR_START_RECOVERY_TIMEOUT_S = 1.0


class ManagedWorker(Protocol):
    """One worker that owns cleanup and terminal manager publication."""

    def start(self) -> None: ...

    @property
    def is_alive(self) -> bool: ...

    def join(self, *, timeout: float | None = None) -> object: ...


class ControlOwnerStartError(RuntimeError):
    """Post-start infrastructure failed while an exact owner was retained."""

    def __init__(self, message: str, status: ControlStatus) -> None:
        super().__init__(message)
        self.status = status

    def as_result(self) -> dict[str, Any]:
        return {
            "success": False,
            "message": str(self),
            "session_id": self.status.session_id,
            "status": self.status.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _ManagedOwner:
    claim: ControlSessionClaim
    worker: ManagedWorker


@dataclass(frozen=True, slots=True)
class _ExternalOwner:
    claim: ControlSessionClaim
    is_active: Callable[[], bool]
    stop: Callable[[], Mapping[str, Any]]
    terminal_status: Callable[[], Mapping[str, Any]] | None = None
    startup_error: str | None = None


Owner = _ManagedOwner | _ExternalOwner


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return numeric


def _result_mapping(result: object) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("control start/stop handlers must return a mapping")
    return dict(result)


def _thread_alive(worker: object | None) -> bool:
    if worker is None:
        return False
    probe = getattr(worker, "is_alive", None)
    if not callable(probe):
        raise TypeError("legacy worker must expose is_alive()")
    result = probe()
    if not isinstance(result, bool):
        raise TypeError("legacy worker is_alive() must return a bool")
    return result


def _managed_worker_alive(worker: ManagedWorker) -> bool:
    probe = getattr(worker, "is_alive", None)
    value = probe() if callable(probe) else probe
    if not isinstance(value, bool):
        raise TypeError("managed worker is_alive must be a bool property or bool-returning method")
    return value


class ControlCoordinator:
    """Serialize route-level owners and bridge unchanged legacy handlers.

    Managed Stadia workers finish their own teardown and publish exact torque
    evidence.  Existing leader/calibration/inference handlers keep their device
    constructors and loops; a small monitor only translates their active flag
    into the shared lifecycle.  Because that legacy boundary exposes no
    six-motor readback, it can never claim ``verified_off``.
    """

    def __init__(
        self,
        manager: ControlSessionManager | None = None,
        *,
        runtime: ControlRuntime | None = None,
        monitor_interval_s: float = 0.05,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.manager = manager or ControlSessionManager()
        self.runtime = runtime or ControlRuntime(self.manager)
        self.monitor_interval_s = _finite_positive(monitor_interval_s, "monitor_interval_s")
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._owner: Owner | None = None
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._monitor_errors: dict[str, BaseException] = {}
        self._recovery_sessions: dict[str, str] = {}

    def start_runtime(self) -> None:
        self.runtime.start()

    def shutdown(self, *, timeout_s: float, reason: str = "server shutdown") -> RuntimeShutdownResult:
        self._reap_retained_owner_once()
        return self.runtime.shutdown(timeout_s=timeout_s, reason=reason)

    def status(self, session_id: str | None = None) -> ControlStatus | None:
        self._reap_retained_owner_once()
        if session_id is None:
            return self.manager.current_status()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        return self.manager.status_for(session_id)

    def renew_lease(self, session_id: str) -> ControlStatus:
        self._reap_retained_owner_once()
        return self.manager.renew_lease(session_id)

    def request_stop(
        self,
        session_id: str | None = None,
        *,
        reason: str = "stop requested by control UI",
    ) -> ControlStatus:
        self._reap_retained_owner_once()
        status = self.manager.active_status()
        if status is None:
            if session_id is not None:
                retained = self.manager.status_for(session_id)
                if retained is not None and retained.terminal:
                    return retained
            raise StaleSessionError("no active control session")
        if session_id is not None and session_id != status.session_id:
            raise StaleSessionError(f"session {session_id!r} does not own control")
        return self.manager.request_stop(status.session_id, reason=reason)

    def set_stadia_speed(self, session_id: str, multiplier: float) -> ControlStatus:
        """Change one exact live Stadia teleoperation speed without device access."""

        self._reap_retained_owner_once()
        status = self.manager.active_status()
        if status is None or status.session_id != session_id:
            raise StaleSessionError(f"session {session_id!r} does not own control")
        if status.operation is not ControlOperation.STADIA_TELEOPERATION:
            raise InvalidControlTransitionError("speed controls are available only for Stadia teleoperation")
        with self._lock:
            owner = self._owner
            if not isinstance(owner, _ManagedOwner) or owner.claim.session_id != session_id:
                raise InvalidControlTransitionError("the Stadia worker is not available")
            setter = getattr(owner.worker, "set_speed_multiplier", None)
        if not callable(setter):
            raise InvalidControlTransitionError("the Stadia worker has no speed control")
        updated = setter(multiplier)
        if not isinstance(updated, ControlStatus) or updated.session_id != session_id:
            raise TypeError("the Stadia worker returned invalid speed status")
        return updated

    def start_managed_worker(
        self,
        operation: ControlOperation | str,
        worker_factory: Callable[[ControlSessionClaim], ManagedWorker],
        *,
        teleoperator_type: str | None,
        details: Mapping[str, object] | None = None,
    ) -> tuple[ControlSessionClaim, ControlStatus]:
        """Claim globally, construct one owner, and return its starting status."""

        self._reap_retained_owner_once()
        claim = self.manager.claim(
            operation,
            teleoperator_type=teleoperator_type,
            details=details,
        )
        owner: _ManagedOwner | None = None
        start_attempted = False
        try:
            worker = worker_factory(claim)
            if (
                not callable(getattr(worker, "start", None))
                or inspect.getattr_static(worker, "is_alive", None) is None
                or not callable(getattr(worker, "join", None))
            ):
                raise TypeError("worker_factory must return a ManagedWorker")
            owner = _ManagedOwner(claim=claim, worker=worker)
            self._install_owner(owner)
            start_attempted = True
            worker.start()
            self._start_monitor(claim.session_id, self._monitor_managed, owner)
        except Exception as error:
            if owner is not None and start_attempted:
                status = self._recover_managed_start_infrastructure(owner, error)
                raise ControlOwnerStartError(
                    f"control worker infrastructure failed: {type(error).__name__}: {error}",
                    status,
                ) from error
            self._clear_owner(claim.session_id)
            status = self._finish_failed_start(claim, error)
            raise ControlOwnerStartError(
                f"control worker failed to start: {type(error).__name__}: {error}",
                status,
            ) from error
        status = self.manager.status_for(claim.session_id, check_expiry=False)
        if status is None:
            raise RuntimeError("managed worker start lost its session status")
        return claim, status

    def start_external_operation(
        self,
        operation: ControlOperation | str,
        *,
        teleoperator_type: str | None,
        start: Callable[[], Mapping[str, Any]],
        stop: Callable[[], Mapping[str, Any]],
        is_active: Callable[[], bool],
        terminal_status: Callable[[], Mapping[str, Any]] | None = None,
        details: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Run an unchanged legacy start and monitor its existing active flag.

        A short server-owned heartbeat keeps the three-second UI lease alive
        while a synchronous legacy start is connecting.  It ends before the
        response returns, after which the active UI is the sole renewer.
        """

        self._reap_retained_owner_once()
        claim = self.manager.claim(
            operation,
            teleoperator_type=teleoperator_type,
            details=details,
        )
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._startup_lease_heartbeat,
            args=(claim.session_id, heartbeat_stop),
            name=f"control-start-lease-{claim.session_id}",
            daemon=True,
        )
        try:
            heartbeat.start()
        except Exception as error:
            status = self._finish_failed_start(claim, error)
            return {
                "success": False,
                "message": f"control startup infrastructure failed: {type(error).__name__}: {error}",
                "session_id": claim.session_id,
                "status": status.as_dict(),
            }
        start_error: Exception | None = None
        heartbeat_error: Exception | None = None
        try:
            result = _result_mapping(start())
        except Exception as error:
            start_error = error
            result = {"success": False, "message": f"{type(error).__name__}: {error}"}
        finally:
            heartbeat_stop.set()
            try:
                heartbeat.join(self.manager.lease_renew_interval_s + self.monitor_interval_s)
            except Exception as error:
                heartbeat_error = error

        if heartbeat_error is not None:
            start_error = start_error or heartbeat_error
            result = {
                **result,
                "success": False,
                "message": (
                    f"control startup heartbeat failed: {type(heartbeat_error).__name__}: {heartbeat_error}"
                ),
            }

        success = result.get("success") is True
        try:
            active = self._safe_active(is_active)
        except Exception as error:
            # An owner whose state cannot be inspected must be treated as
            # potentially active. Install its stop monitor and keep the claim
            # rather than leaking an untracked legacy process/thread.
            active = True
            start_error = start_error or error
            success = False
            result = {
                **result,
                "success": False,
                "message": (f"control activity could not be verified: {type(error).__name__}: {error}"),
            }
        if success:
            try:
                status = self.manager.mark_running(claim.session_id)
            except Exception as error:
                start_error = start_error or error
                success = False
                result = {
                    **result,
                    "success": False,
                    "message": f"control lifecycle rejected the start: {type(error).__name__}: {error}",
                }
        else:
            status = self.manager.status_for(claim.session_id, check_expiry=False)

        if success or active:
            owner = _ExternalOwner(
                claim=claim,
                is_active=is_active,
                stop=stop,
                terminal_status=terminal_status,
                startup_error=(
                    f"{type(start_error).__name__}: {start_error}"
                    if start_error is not None
                    else (None if success else str(result.get("message") or "legacy start failed"))
                ),
            )
            self._install_owner(owner)
            if not success:
                self.manager.request_stop(claim.session_id, reason=owner.startup_error or "start failed")
            try:
                self._start_monitor(claim.session_id, self._monitor_external, owner)
            except Exception as error:
                status = self._recover_external_start_infrastructure(owner, error)
                return {
                    **result,
                    "success": False,
                    "message": f"control monitor failed to start: {type(error).__name__}: {error}",
                    "session_id": claim.session_id,
                    "status": status.as_dict(),
                }
        else:
            failure = start_error or RuntimeError(str(result.get("message") or "legacy start failed"))
            terminal_reason, cleanup_pending = self._external_terminal_evidence(terminal_status)
            if terminal_reason is not None:
                failure = RuntimeError(f"{failure}; {terminal_reason}")
            if cleanup_pending:
                self.manager.quarantine(reason=str(failure))
            status = self._finish_failed_start(claim, failure)

        latest = self.manager.status_for(claim.session_id, check_expiry=False) or status
        return {
            **result,
            "success": success,
            "session_id": claim.session_id,
            "status": latest.as_dict() if latest is not None else None,
        }

    def start_teleoperation(
        self,
        request: Mapping[str, Any] | object,
        *,
        websocket_manager: object | None = None,
        resolver: Callable[..., ResolvedControlRequest] = resolve_teleoperate_request,
        leader_start: Callable[[Mapping[str, Any], object | None], Mapping[str, Any]] | None = None,
        leader_stop: Callable[[], Mapping[str, Any]] | None = None,
        leader_is_active: Callable[[], bool] | None = None,
        leader_terminal_status: Callable[[], Mapping[str, Any]] | None = None,
        stadia_worker_factory: (
            Callable[[ControlSessionClaim, ResolvedControlRequest], ManagedWorker] | None
        ) = None,
    ) -> dict[str, Any]:
        """Resolve canonical/compatible selection and start the selected path."""

        try:
            resolved = resolver(request)
        except ControlRequestError as error:
            return {"success": False, "message": error.message, "error": error.as_dict()}

        if resolved.teleoperator_type == "leader_arm":
            if leader_start is None or leader_stop is None or leader_is_active is None:
                from . import teleoperate as legacy

                legacy_worker: object | None = None

                def start_legacy_teleoperation(
                    payload: Mapping[str, Any],
                    socket_manager: object | None,
                ) -> Mapping[str, Any]:
                    nonlocal legacy_worker
                    result = legacy.handle_start_teleoperation(
                        legacy.TeleoperateRequest.model_validate(payload),
                        socket_manager,
                    )
                    legacy_worker = legacy.teleoperation_thread
                    return result

                def stop_legacy_teleoperation() -> Mapping[str, Any]:
                    nonlocal legacy_worker
                    legacy_worker = legacy.teleoperation_thread or legacy_worker
                    return legacy.handle_stop_teleoperation()

                def legacy_teleoperation_is_active() -> bool:
                    return bool(legacy.teleoperation_active or _thread_alive(legacy_worker))

                leader_start = start_legacy_teleoperation
                leader_stop = stop_legacy_teleoperation
                leader_is_active = legacy_teleoperation_is_active
                leader_terminal_status = getattr(legacy, "handle_teleoperation_status", None)
            payload = resolved.legacy_handler_payload()
            return self.start_external_operation(
                ControlOperation.LEADER_TELEOPERATION,
                teleoperator_type="leader_arm",
                start=lambda: leader_start(payload, websocket_manager),
                stop=leader_stop,
                is_active=leader_is_active,
                terminal_status=leader_terminal_status,
                details={"robot_name": resolved.robot_name},
            )

        if stadia_worker_factory is None:

            def build_default_stadia_worker(
                selected_claim: ControlSessionClaim,
                selected: ResolvedControlRequest,
            ) -> ManagedWorker:
                return self._default_stadia_worker(
                    selected_claim,
                    selected,
                    websocket_manager=websocket_manager,
                )

            stadia_worker_factory = build_default_stadia_worker
        record = resolved.record_model()
        try:
            claim, status = self.start_managed_worker(
                ControlOperation.STADIA_TELEOPERATION,
                lambda selected_claim: stadia_worker_factory(selected_claim, resolved),
                teleoperator_type="stadia",
                details={
                    "robot_name": resolved.robot_name,
                    "stadia_speed_multiplier": 1.0,
                    "stadia_effective_max_step_per_tick": record.stadia.max_step_per_tick,
                },
            )
        except ControlOwnerStartError as error:
            return error.as_result()
        return {
            "success": True,
            "message": "Stadia teleoperation is starting",
            "session_id": claim.session_id,
            "status": status.as_dict(),
        }

    def start_recording(
        self,
        request: Mapping[str, Any] | object,
        *,
        resolver: Callable[..., ResolvedControlRequest] = resolve_recording_request,
        leader_start: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        leader_stop: Callable[[], Mapping[str, Any]] | None = None,
        leader_is_active: Callable[[], bool] | None = None,
        leader_terminal_status: Callable[[], Mapping[str, Any]] | None = None,
        stadia_worker_factory: (
            Callable[[ControlSessionClaim, ResolvedControlRequest], ManagedWorker] | None
        ) = None,
    ) -> dict[str, Any]:
        """Start the canonical recording mode without changing the leader loop."""

        try:
            resolved = resolver(request)
        except ControlRequestError as error:
            return {"success": False, "message": error.message, "error": error.as_dict()}

        if resolved.teleoperator_type == "leader_arm":
            if leader_start is None or leader_stop is None or leader_is_active is None:
                from . import record as legacy

                legacy_worker: object | None = None

                def start_legacy_recording(payload: Mapping[str, Any]) -> Mapping[str, Any]:
                    nonlocal legacy_worker
                    result = legacy.handle_start_recording(legacy.RecordingRequest.model_validate(payload))
                    legacy_worker = legacy.recording_thread
                    return result

                def stop_legacy_recording() -> Mapping[str, Any]:
                    nonlocal legacy_worker
                    legacy_worker = legacy.recording_thread or legacy_worker
                    return legacy.handle_stop_recording()

                def legacy_recording_is_active() -> bool:
                    return bool(legacy.recording_active or _thread_alive(legacy_worker))

                leader_start = start_legacy_recording
                leader_stop = stop_legacy_recording
                leader_is_active = legacy_recording_is_active
                leader_terminal_status = getattr(legacy, "handle_recording_status", None)
            payload = resolved.legacy_handler_payload()
            return self.start_external_operation(
                ControlOperation.LEADER_RECORDING,
                teleoperator_type="leader_arm",
                start=lambda: leader_start(payload),
                stop=leader_stop,
                is_active=leader_is_active,
                terminal_status=leader_terminal_status,
                details={"robot_name": resolved.robot_name},
            )

        if stadia_worker_factory is None:
            from .stadia.recording_session import build_stadia_recording_worker

            stadia_worker_factory = lambda selected_claim, selected: build_stadia_recording_worker(  # noqa: E731
                manager=self.manager,
                claim=selected_claim,
                resolved=selected,
            )
        built_worker: ManagedWorker | None = None

        def build_recording_worker(selected_claim: ControlSessionClaim) -> ManagedWorker:
            nonlocal built_worker
            built_worker = stadia_worker_factory(selected_claim, resolved)
            return built_worker

        try:
            claim, status = self.start_managed_worker(
                ControlOperation.STADIA_RECORDING,
                build_recording_worker,
                teleoperator_type="stadia",
                details={"robot_name": resolved.robot_name},
            )
        except ControlOwnerStartError as error:
            return error.as_result()
        stamped_dataset_id = getattr(built_worker, "dataset_repo_id", None)
        if not isinstance(stamped_dataset_id, str) or not stamped_dataset_id.strip():
            stamped_dataset_id = resolved.metadata_dict().get("dataset_repo_id")
        return {
            "success": True,
            "message": "Stadia recording is starting",
            "session_id": claim.session_id,
            "dataset_id": stamped_dataset_id,
            "status": status.as_dict(),
        }

    def start_controller_check(
        self,
        robot_name: str,
        *,
        worker_factory: Callable[[ControlSessionClaim, object], ManagedWorker] | None = None,
    ) -> dict[str, Any]:
        """Start a controller-only worker from one saved Stadia record."""

        if not is_valid_robot_name(robot_name):
            return {"success": False, "message": "robot_name is not a valid saved robot name"}
        try:
            record = get_robot_record_v2(robot_name)
        except RobotRecordValidationError as error:
            return {"success": False, "message": f"saved robot record is invalid: {error}"}
        if record is None:
            path = Path(config_module.ROBOTS_PATH) / f"{robot_name}.json"
            if path.is_file():
                return {
                    "success": False,
                    "message": "saved robot record exists but is not valid JSON",
                }
            return {"success": False, "message": "saved robot record was not found"}
        readiness = evaluate_robot_readiness(record, RobotOperation.CONTROLLER_CHECK)
        if not readiness.ready:
            return {
                "success": False,
                "message": "saved robot is not ready for a controller check",
                "issues": [issue.model_dump(mode="json") for issue in readiness.issues],
            }
        if worker_factory is None:
            worker_factory = self._default_controller_check_worker
        try:
            claim, status = self.start_managed_worker(
                ControlOperation.CONTROLLER_CHECK,
                lambda selected_claim: worker_factory(selected_claim, record),
                teleoperator_type="stadia",
                details={"robot_name": robot_name},
            )
        except ControlOwnerStartError as error:
            return error.as_result()
        return {
            "success": True,
            "message": "Controller check is starting",
            "session_id": claim.session_id,
            "status": status.as_dict(),
        }

    def monitor_error(self, session_id: str) -> BaseException | None:
        with self._lock:
            return self._monitor_errors.get(session_id)

    def active_managed_worker(
        self,
        session_id: str,
        *,
        operation: ControlOperation | str | None = None,
    ) -> ManagedWorker | None:
        """Return only the exact active managed owner selected by its session.

        Route adapters use this narrow lookup for worker-owned typed events.
        Terminal history remains in :class:`ControlSessionManager`; a stale or
        external owner never leaks a device-facing object through this method.
        """

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        expected = ControlOperation(operation) if operation is not None else None
        status = self.manager.active_status()
        if status is None or status.session_id != session_id:
            return None
        if expected is not None and status.operation is not expected:
            return None
        with self._lock:
            owner = self._owner
            if not isinstance(owner, _ManagedOwner):
                return None
            if owner.claim.session_id != session_id:
                return None
            return owner.worker

    def _default_stadia_worker(
        self,
        claim: ControlSessionClaim,
        resolved: ResolvedControlRequest,
        *,
        websocket_manager: object | None = None,
    ) -> ManagedWorker:
        from .stadia.session import StadiaSessionConfig, StadiaSessionWorker

        record = resolved.record_model()
        broadcaster = (
            getattr(websocket_manager, "broadcast_joint_data_sync", None)
            if websocket_manager is not None
            else None
        )
        if broadcaster is not None and not callable(broadcaster):
            raise TypeError("websocket manager broadcaster must be callable")
        return StadiaSessionWorker(
            manager=self.manager,
            claim=claim,
            config=StadiaSessionConfig(
                follower_port=record.follower.port,
                follower_calibration=record.follower.calibration,
                expected_guid=record.stadia.guid,
                deadzone=record.stadia.deadzone,
                max_step_per_tick=record.stadia.max_step_per_tick,
                arm_startup_travel_degrees=record.stadia.arm_startup_travel_degrees,
                gripper_startup_travel_percentage_points=(
                    record.stadia.gripper_startup_travel_percentage_points
                ),
                # Live teleoperation has never opened recording cameras.  Keep
                # that legacy resource boundary; the Stadia recorder supplies
                # its own lazily constructed camera configs.
                cameras={},
            ),
            joint_broadcaster=broadcaster,
        )

    def _default_controller_check_worker(
        self,
        claim: ControlSessionClaim,
        record: object,
    ) -> ManagedWorker:
        from .stadia.controller_check import (
            ControllerCheckConfig,
            ControllerCheckWorker,
        )
        from .utils.config import RobotRecordV2

        canonical = RobotRecordV2.model_validate(record)
        return ControllerCheckWorker(
            manager=self.manager,
            claim=claim,
            config=ControllerCheckConfig(
                expected_guid=canonical.stadia.guid,
                deadzone=canonical.stadia.deadzone,
            ),
        )

    def _install_owner(self, owner: Owner) -> None:
        with self._lock:
            if self._owner is not None:
                raise RuntimeError("coordinator owner registry is already occupied")
            self._owner = owner

    def _clear_owner(self, session_id: str) -> None:
        with self._lock:
            if self._owner is not None and self._owner.claim.session_id == session_id:
                self._owner = None
            self._monitor_threads.pop(session_id, None)
            self._recovery_sessions.pop(session_id, None)

    def _start_monitor(
        self,
        session_id: str,
        target: Callable[[Owner], None],
        owner: Owner,
    ) -> None:
        thread = threading.Thread(
            target=self._monitor_entry,
            args=(session_id, target, owner),
            name=f"control-owner-monitor-{session_id}",
            daemon=True,
        )
        with self._lock:
            self._monitor_threads[session_id] = thread
        try:
            thread.start()
        except BaseException:
            with self._lock:
                self._monitor_threads.pop(session_id, None)
            raise

    def _monitor_entry(
        self,
        session_id: str,
        target: Callable[[Owner], None],
        owner: Owner,
    ) -> None:
        try:
            target(owner)
        except BaseException as error:
            with self._lock:
                self._monitor_errors[session_id] = error
                self._recovery_sessions[session_id] = (
                    f"control owner monitor failed: {type(error).__name__}: {error}"
                )
            try:
                status = self.manager.status_for(session_id, check_expiry=False)
                if status is not None and not status.terminal:
                    self.manager.request_failure(
                        session_id,
                        reason=self._recovery_sessions[session_id],
                    )
            except BaseException as signal_error:
                with contextlib.suppress(BaseException):
                    error.add_note(
                        "additionally failed to signal owner teardown: "
                        f"{type(signal_error).__name__}: {signal_error}"
                    )
            self._start_recovery_monitor(session_id, owner)
        finally:
            status = self.manager.status_for(session_id, check_expiry=False)
            if status is not None and status.terminal:
                self._clear_owner(session_id)

    def _monitor_managed(self, owner: Owner) -> None:
        if not isinstance(owner, _ManagedOwner):
            raise TypeError("managed monitor received an external owner")
        while _managed_worker_alive(owner.worker):
            self._sleeper(self.monitor_interval_s)
        owner.worker.join(timeout=0.0)

    def _recover_managed_start_infrastructure(
        self,
        owner: _ManagedOwner,
        error: Exception,
    ) -> ControlStatus:
        """Fail closed after a managed worker may already have started."""

        reason = f"control worker infrastructure failed: {type(error).__name__}: {error}"
        with self._lock:
            self._monitor_errors[owner.claim.session_id] = error
            self._recovery_sessions[owner.claim.session_id] = reason
        status = self.manager.status_for(owner.claim.session_id, check_expiry=False)
        if status is None:
            raise StaleSessionError(
                f"session {owner.claim.session_id!r} disappeared after worker start"
            ) from error
        if not status.terminal:
            status = self.manager.request_failure(owner.claim.session_id, reason=reason)
        try:
            owner.worker.join(timeout=MONITOR_START_RECOVERY_TIMEOUT_S)
        except Exception as join_error:
            with contextlib.suppress(BaseException):
                error.add_note(
                    f"managed worker teardown could not be proven: {type(join_error).__name__}: {join_error}"
                )
        latest = self.manager.status_for(owner.claim.session_id, check_expiry=False) or status
        if latest.terminal:
            try:
                alive = _managed_worker_alive(owner.worker)
            except Exception:
                alive = True
            if not alive:
                self._clear_owner(owner.claim.session_id)
                return latest
        self._start_recovery_monitor(owner.claim.session_id, owner)
        return latest

    def _recover_external_start_infrastructure(
        self,
        owner: _ExternalOwner,
        error: Exception,
    ) -> ControlStatus:
        """Stop a started legacy owner and release only after proven inactivity."""

        reason = f"control monitor failed to start: {type(error).__name__}: {error}"
        with self._lock:
            self._monitor_errors[owner.claim.session_id] = error
            self._recovery_sessions[owner.claim.session_id] = reason
        status = self.manager.status_for(owner.claim.session_id, check_expiry=False)
        if status is None:
            raise StaleSessionError(
                f"session {owner.claim.session_id!r} disappeared after legacy start"
            ) from error
        if not status.terminal:
            status = self.manager.request_failure(owner.claim.session_id, reason=reason)

        stop_error: Exception | None = None
        try:
            stop_result = _result_mapping(owner.stop())
            if stop_result.get("success") is not True:
                raise RuntimeError(str(stop_result.get("message") or "legacy stop reported failure"))
        except Exception as caught:
            stop_error = caught

        active: bool | None = None
        deadline = time.monotonic() + MONITOR_START_RECOVERY_TIMEOUT_S
        while True:
            try:
                active = self._safe_active(owner.is_active)
            except Exception as activity_error:
                with contextlib.suppress(BaseException):
                    error.add_note(
                        "legacy inactivity could not be proven: "
                        f"{type(activity_error).__name__}: {activity_error}"
                    )
                active = None
                break
            if not active or time.monotonic() >= deadline:
                break
            self._sleeper(min(self.monitor_interval_s, max(0.0, deadline - time.monotonic())))

        if active is False:
            final_reason = reason
            if stop_error is not None:
                final_reason += f"; legacy stop failed: {type(stop_error).__name__}: {stop_error}"
            terminal_reason, cleanup_pending = self._external_terminal_evidence(owner.terminal_status)
            final_reason = self._join_reasons(final_reason, terminal_reason) or reason
            if cleanup_pending:
                self.manager.quarantine(reason=final_reason)
            latest = self.manager.status_for(owner.claim.session_id, check_expiry=False)
            if latest is not None and not latest.terminal:
                latest = self.manager.finish_teardown(
                    owner.claim.session_id,
                    terminal_state=ControlState.ERROR,
                    torque=NOT_ATTEMPTED_TORQUE,
                    reason=final_reason,
                )
            self._clear_owner(owner.claim.session_id)
            return latest or status

        if stop_error is not None:
            with contextlib.suppress(BaseException):
                error.add_note(f"legacy stop failed: {type(stop_error).__name__}: {stop_error}")
        self._start_recovery_monitor(owner.claim.session_id, owner)
        return self.manager.status_for(owner.claim.session_id, check_expiry=False) or status

    def _start_recovery_monitor(self, session_id: str, owner: Owner) -> None:
        """Best-effort reaper for an owner whose primary monitor was lost."""

        with self._lock:
            current = self._monitor_threads.get(session_id)
            if current is not None and current is not threading.current_thread() and current.is_alive():
                return
            thread = threading.Thread(
                target=self._recovery_monitor_entry,
                args=(session_id,),
                name=f"control-owner-recovery-{session_id}",
                daemon=True,
            )
            self._monitor_threads[session_id] = thread
        try:
            thread.start()
        except BaseException as error:
            with self._lock:
                if self._monitor_threads.get(session_id) is thread:
                    self._monitor_threads.pop(session_id, None)
                self._monitor_errors[session_id] = error

    def _recovery_monitor_entry(self, session_id: str) -> None:
        while True:
            try:
                if self._reap_retained_owner_once(session_id):
                    return
            except BaseException as error:
                with self._lock:
                    self._monitor_errors[session_id] = error
            time.sleep(self.monitor_interval_s)

    def _reap_retained_owner_once(self, session_id: str | None = None) -> bool:
        """Advance only owners explicitly quarantined by monitor failure."""

        with self._lock:
            owner = self._owner
            if owner is None:
                return True
            identity = owner.claim.session_id
            if session_id is not None and identity != session_id:
                return True
            reason = self._recovery_sessions.get(identity)
        if reason is None:
            return False

        status = self.manager.status_for(identity, check_expiry=False)
        if isinstance(owner, _ManagedOwner):
            try:
                alive = _managed_worker_alive(owner.worker)
            except BaseException as error:
                with self._lock:
                    self._monitor_errors[identity] = error
                return False
            if alive:
                return False
            join_error: BaseException | None = None
            try:
                owner.worker.join(timeout=0.0)
            except BaseException as error:
                join_error = error
                with self._lock:
                    self._monitor_errors[identity] = error
            if status is not None and not status.terminal:
                if status.state is not ControlState.STOPPING:
                    self.manager.request_failure(identity, reason=reason)
                proof_reason = (
                    f"{reason}; managed worker join failed: {type(join_error).__name__}: {join_error}"
                    if join_error is not None
                    else f"{reason}; worker exited without terminal teardown evidence"
                )
                self.manager.quarantine(reason=proof_reason)
                self.manager.finish_teardown(
                    identity,
                    terminal_state=ControlState.ERROR,
                    torque=NOT_ATTEMPTED_TORQUE,
                    reason=proof_reason,
                )
            self._clear_owner(identity)
            return True

        try:
            if owner.claim.stop_requested.is_set():
                owner.stop()
            active = self._safe_active(owner.is_active)
        except BaseException as error:
            with self._lock:
                self._monitor_errors[identity] = error
            return False
        if active:
            return False
        if status is not None and not status.terminal:
            terminal_reason, cleanup_pending = self._external_terminal_evidence(owner.terminal_status)
            final_reason = self._join_reasons(reason, terminal_reason) or reason
            if cleanup_pending:
                self.manager.quarantine(reason=final_reason)
            if status.state is not ControlState.STOPPING:
                self.manager.request_failure(identity, reason=final_reason)
            self.manager.finish_teardown(
                identity,
                terminal_state=ControlState.ERROR,
                torque=NOT_ATTEMPTED_TORQUE,
                reason=final_reason,
            )
        self._clear_owner(identity)
        return True

    def _monitor_external(self, owner: Owner) -> None:
        if not isinstance(owner, _ExternalOwner):
            raise TypeError("external monitor received a managed owner")
        stop_called = False
        stop_error: Exception | None = None
        activity_error: Exception | None = None
        while True:
            if owner.claim.stop_requested.is_set() and not stop_called:
                try:
                    stop_result = _result_mapping(owner.stop())
                    if stop_result.get("success") is not True:
                        message = stop_result.get("message")
                        detail = str(message).strip() if message is not None else ""
                        raise RuntimeError(detail or "legacy stop handler reported failure")
                    stop_called = True
                    stop_error = None
                except Exception as error:
                    if stop_error is None:
                        stop_error = error
            try:
                active = self._safe_active(owner.is_active)
            except Exception as error:
                activity_error = error
                status = self.manager.status_for(owner.claim.session_id, check_expiry=False)
                if status is not None and status.state is not ControlState.STOPPING:
                    self.manager.request_failure(
                        owner.claim.session_id,
                        reason=(f"legacy activity check failed: {type(error).__name__}: {error}"),
                    )
                self._sleeper(self.monitor_interval_s)
                continue
            if not active:
                break
            self._sleeper(self.monitor_interval_s)

        status = self.manager.status_for(owner.claim.session_id, check_expiry=False)
        if status is None or status.terminal:
            return
        terminal_reason, cleanup_pending = self._external_terminal_evidence(owner.terminal_status)
        reasons = [owner.startup_error]
        if stop_error is not None:
            reasons.append(f"legacy stop failed: {type(stop_error).__name__}: {stop_error}")
        if activity_error is not None:
            reasons.append(f"legacy activity check failed: {type(activity_error).__name__}: {activity_error}")
        reasons.append(terminal_reason)
        reason = self._join_reasons(*reasons)
        if cleanup_pending:
            self.manager.quarantine(reason=reason or "legacy operation ended with unproven resource cleanup")
        if status.state is not ControlState.STOPPING:
            if reason is not None:
                status = self.manager.request_failure(owner.claim.session_id, reason=reason)
            else:
                status = self.manager.request_stop(
                    owner.claim.session_id,
                    reason="legacy operation completed",
                )
        terminal = ControlState.ERROR if reason is not None else ControlState.STOPPED
        self.manager.finish_teardown(
            owner.claim.session_id,
            terminal_state=terminal,
            torque=NOT_ATTEMPTED_TORQUE,
            reason=reason or status.stop_reason or "legacy operation completed",
        )

    @staticmethod
    def _join_reasons(*values: str | None) -> str | None:
        parts: list[str] = []
        for value in values:
            if value is None:
                continue
            normalized = value.strip()
            if normalized and normalized not in parts:
                parts.append(normalized)
        return "; ".join(parts) if parts else None

    @staticmethod
    def _external_terminal_evidence(
        terminal_status: Callable[[], Mapping[str, Any]] | None,
    ) -> tuple[str | None, bool]:
        """Classify retained legacy terminal evidence after liveness is false.

        Legacy handlers keep their existing worker loops. Their status snapshot
        supplies the missing distinction between natural success, a worker
        exception, and cleanup that could not be proven. A broken default
        snapshot is itself cleanup-unknown and therefore closes the manager.
        """

        if terminal_status is None:
            return None, False
        try:
            snapshot = _result_mapping(terminal_status())
        except Exception as error:
            return (
                f"legacy terminal status failed: {type(error).__name__}: {error}",
                True,
            )

        cleanup_pending = snapshot.get("cleanup_pending", False)
        if not isinstance(cleanup_pending, bool):
            return "legacy terminal status has invalid cleanup_pending evidence", True

        outcome = snapshot.get("outcome")
        if outcome is not None and not isinstance(outcome, str):
            return "legacy terminal status has an invalid outcome", True
        lifecycle = snapshot.get("status")
        if lifecycle is not None and not isinstance(lifecycle, str):
            return "legacy terminal status has an invalid status", True
        error_value = snapshot.get("error")
        if error_value is not None and not isinstance(error_value, str):
            return "legacy terminal status has invalid error evidence", True

        error_detail = error_value.strip() if isinstance(error_value, str) else ""
        failed = outcome == "failed" or lifecycle == "error"
        if cleanup_pending:
            detail = error_detail or "legacy operation ended with unproven resource cleanup"
            return detail, True
        if failed:
            return error_detail or "legacy operation reported a terminal failure", False
        if outcome == "running" or lifecycle in {"connecting", "recording", "stopping"}:
            return "legacy terminal status contradicts proven worker inactivity", True
        return None, False

    def _startup_lease_heartbeat(self, session_id: str, stop: threading.Event) -> None:
        interval = self.manager.lease_renew_interval_s
        while not stop.wait(interval):
            try:
                self.manager.renew_lease(session_id)
            except (InvalidControlTransitionError, StaleSessionError):
                return

    @staticmethod
    def _safe_active(predicate: Callable[[], bool]) -> bool:
        value = predicate()
        if not isinstance(value, bool):
            raise TypeError("legacy active predicate must return a bool")
        return value

    def _finish_failed_start(
        self,
        claim: ControlSessionClaim,
        error: Exception,
    ) -> ControlStatus:
        reason = f"start failed: {type(error).__name__}: {error}"
        status = self.manager.status_for(claim.session_id, check_expiry=False)
        if status is None:
            raise StaleSessionError(f"session {claim.session_id!r} disappeared during start failure")
        if status.state is not ControlState.STOPPING:
            self.manager.request_stop(claim.session_id, reason=reason)
        return self.manager.finish_teardown(
            claim.session_id,
            terminal_state=ControlState.ERROR,
            torque=NOT_ATTEMPTED_TORQUE,
            reason=reason,
        )


__all__ = ["ControlCoordinator", "ManagedWorker"]
