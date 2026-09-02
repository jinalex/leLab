"""Controller-only identity, layout, freshness, and neutralization worker.

This path deliberately owns only the pygame reader.  It imports no LeRobot
robot classes and cannot open a serial port, read calibration, enable torque,
or produce an action.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from lelab.control_session import (
    NOT_ATTEMPTED_TORQUE,
    ControlSessionClaim,
    ControlSessionManager,
    ControlState,
    MotionState,
)

from .mapping import NeutralReleaseGate
from .types import NeutralGateDecision, NeutralGateState, StadiaSnapshot


class ControllerCheckError(RuntimeError):
    """The controller-only worker could not continue truthfully."""


class _StopRequestedError(Exception):
    pass


class ControllerReader(Protocol):
    def start(self) -> None: ...

    def snapshot(self) -> StadiaSnapshot: ...

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot: ...

    def stop(self, *, timeout: float = 2.0) -> None: ...


@dataclass(frozen=True, slots=True)
class ControllerCheckConfig:
    expected_guid: str | None = None
    deadzone: float = 0.15
    max_snapshot_age_s: float = 0.15
    reader_join_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if self.expected_guid is not None and (
            not isinstance(self.expected_guid, str)
            or not self.expected_guid
            or self.expected_guid.strip() != self.expected_guid
            or "\x00" in self.expected_guid
        ):
            raise ValueError("expected_guid must be a non-empty trimmed string or None")
        for label, value, allow_zero in (
            ("deadzone", self.deadzone, True),
            ("max_snapshot_age_s", self.max_snapshot_age_s, False),
            ("reader_join_timeout_s", self.reader_join_timeout_s, False),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{label} must be finite")
            numeric = float(value)
            minimum_ok = numeric >= 0 if allow_zero else numeric > 0
            if not math.isfinite(numeric) or not minimum_ok:
                raise ValueError(f"{label} is outside its valid range")
        if self.deadzone >= 1:
            raise ValueError("deadzone must be below 1")


@dataclass(frozen=True, slots=True)
class ControllerCheckResult:
    terminal_state: ControlState
    reason: str
    samples_seen: int
    ready_observed: bool
    cleanup_errors: tuple[str, ...]


def _default_reader(expected_guid: str | None) -> ControllerReader:
    from .device_reader import StadiaDeviceReader

    return StadiaDeviceReader(expected_guid=expected_guid)


class ControllerCheckWorker:
    """Sole owner of one controller reader and its terminal cleanup."""

    def __init__(
        self,
        *,
        manager: ControlSessionManager,
        claim: ControlSessionClaim,
        config: ControllerCheckConfig,
        reader: ControllerReader | None = None,
        reader_factory: Callable[[str | None], ControllerReader] = _default_reader,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manager = manager
        self.claim = claim
        self.config = config
        self.reader = reader if reader is not None else reader_factory(config.expected_guid)
        self._clock = clock
        self._gate = NeutralReleaseGate(
            expected_guid=config.expected_guid,
            deadzone=config.deadzone,
            stable_samples_required=3,
        )
        self._gate_state = NeutralGateState()
        self._highest_sequence = -1
        self._latest_sampled_at = -float("inf")
        self._samples_seen = 0
        self._ready_observed = False
        self._lifecycle_lock = threading.Lock()
        self._started = False
        self._thread: threading.Thread | None = None
        self._result: ControllerCheckResult | None = None
        self._error: BaseException | None = None

    @property
    def is_alive(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("controller-check workers may only be started once")
            self._started = True
            self._thread = threading.Thread(
                target=self._thread_main,
                name=f"lelab-controller-check-{self.claim.session_id}",
                daemon=False,
            )
            self._thread.start()

    def run(self) -> ControllerCheckResult:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("controller-check workers may only be run once")
            self._started = True
        return self._run_owned()

    def join(self, *, timeout: float | None = None) -> ControllerCheckResult:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or float(timeout) < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")
        with self._lifecycle_lock:
            thread = self._thread
        if thread is None:
            raise RuntimeError("controller-check worker was not started asynchronously")
        if thread is threading.current_thread():
            raise RuntimeError("controller-check worker cannot join itself")
        thread.join(None if timeout is None else float(timeout))
        if thread.is_alive():
            raise TimeoutError("controller-check worker did not finish before the join timeout")
        with self._lifecycle_lock:
            error = self._error
            result = self._result
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("controller-check worker exited without terminal evidence")
        return result

    def _thread_main(self) -> None:
        try:
            result = self._run_owned()
        except BaseException as error:
            with self._lifecycle_lock:
                self._error = error
            return
        with self._lifecycle_lock:
            self._result = result

    def _run_owned(self) -> ControllerCheckResult:
        reader_started = False
        failure: Exception | None = None
        reason = "controller check stopped"
        cleanup_errors: list[str] = []
        try:
            reader_started = True
            self.reader.start()
            self.manager.mark_running(self.claim.session_id)
            self._sample_loop()
        except _StopRequestedError as error:
            reason = str(error)
        except Exception as error:
            failure = error
            reason = f"{type(error).__name__}: {error}"
        finally:
            status = self.manager.status_for(self.claim.session_id, check_expiry=False)
            if status is not None and status.state is not ControlState.STOPPING:
                try:
                    self.manager.request_stop(self.claim.session_id, reason=reason)
                except Exception as error:
                    failure = failure or error
                    cleanup_errors.append(f"manager stop transition failed: {type(error).__name__}: {error}")
            if reader_started:
                try:
                    self.reader.stop(timeout=self.config.reader_join_timeout_s)
                except Exception as error:
                    cleanup_errors.append(f"Stadia reader stop failed: {type(error).__name__}: {error}")
                    self.manager.quarantine(
                        reason=(
                            f"controller reader release could not be proven: {type(error).__name__}: {error}"
                        )
                    )
                else:
                    try:
                        self.manager.mark_controller_monitoring_inactive(self.claim.session_id)
                    except Exception as error:
                        cleanup_errors.append(
                            f"controller monitoring status failed: {type(error).__name__}: {error}"
                        )

        terminal_state = ControlState.ERROR if failure is not None or cleanup_errors else ControlState.STOPPED
        if cleanup_errors:
            reason = f"{reason}; teardown failed: {'; '.join(cleanup_errors)}"
        terminal = self.manager.finish_teardown(
            self.claim.session_id,
            terminal_state=terminal_state,
            torque=NOT_ATTEMPTED_TORQUE,
            reason=reason,
        )
        terminal_state = terminal.state
        reason = terminal.stop_reason or reason
        return ControllerCheckResult(
            terminal_state=terminal_state,
            reason=reason,
            samples_seen=self._samples_seen,
            ready_observed=self._ready_observed,
            cleanup_errors=tuple(cleanup_errors),
        )

    def _sample_loop(self) -> None:
        waited_after = -1
        while True:
            self._raise_if_stop_requested()
            try:
                snapshot = self.reader.wait_for_snapshot(
                    after_sequence=waited_after,
                    timeout=0.1,
                )
                if isinstance(snapshot.sequence, int) and not isinstance(snapshot.sequence, bool):
                    waited_after = max(waited_after, snapshot.sequence)
            except TimeoutError:
                snapshot = self.reader.snapshot()
            decision, age, sample_error = self._evaluate(snapshot)
            self._publish(snapshot, decision, age, sample_error)

    def _evaluate(
        self,
        snapshot: StadiaSnapshot,
    ) -> tuple[NeutralGateDecision, float | None, str | None]:
        now = self._now()
        sample_error: str | None = None
        age: float | None = None
        if (
            isinstance(snapshot.sequence, bool)
            or not isinstance(snapshot.sequence, int)
            or snapshot.sequence < 0
        ):
            sample_error = "controller sample sequence is invalid"
        elif (
            isinstance(snapshot.connection_generation, bool)
            or not isinstance(snapshot.connection_generation, int)
            or snapshot.connection_generation < 0
        ):
            sample_error = "controller connection generation is invalid"
        elif (
            isinstance(snapshot.sampled_at, bool)
            or not isinstance(snapshot.sampled_at, Real)
            or not math.isfinite(float(snapshot.sampled_at))
        ):
            sample_error = "controller sample timestamp is invalid"
        elif snapshot.sequence < self._highest_sequence:
            sample_error = "controller sample sequence regressed"
        elif snapshot.sequence > self._highest_sequence and snapshot.sampled_at <= self._latest_sampled_at:
            sample_error = "controller sample timestamp did not advance"
        elif not snapshot.connected and snapshot.read_error:
            sample_error = f"Stadia read failed: {snapshot.read_error}"
        else:
            age = now - float(snapshot.sampled_at)
            if age < 0:
                sample_error = "controller sample timestamp is in the future"
            elif age > self.config.max_snapshot_age_s:
                sample_error = f"controller sample is stale ({age:.3f}s old)"

        if sample_error is None:
            decision = self._gate.evaluate(snapshot, self._gate_state)
            self._gate_state = decision.state
            if not decision.profile_valid:
                sample_error = decision.reason or "controller profile is invalid"
        else:
            generation = (
                snapshot.connection_generation
                if isinstance(snapshot.connection_generation, int)
                and not isinstance(snapshot.connection_generation, bool)
                and snapshot.connection_generation >= 0
                else None
            )
            self._gate_state = NeutralGateState(connection_generation=generation)
            decision = NeutralGateDecision(
                state=self._gate_state,
                profile_valid=False,
                controls_neutral=False,
                motion_enabled=False,
                reason=sample_error,
            )

        if snapshot.sequence > self._highest_sequence and sample_error is None:
            self._highest_sequence = snapshot.sequence
            self._latest_sampled_at = float(snapshot.sampled_at)
            self._samples_seen += 1
        ready = (
            sample_error is None
            and decision.profile_valid
            and decision.controls_neutral
            and decision.state.neutral_armed
            and not snapshot.button(10)
        )
        self._ready_observed = self._ready_observed or ready
        return decision, age, sample_error

    def _publish(
        self,
        snapshot: StadiaSnapshot,
        decision: NeutralGateDecision,
        age: float | None,
        sample_error: str | None,
    ) -> None:
        instance_id = (
            snapshot.instance_id
            if isinstance(snapshot.instance_id, int)
            and not isinstance(snapshot.instance_id, bool)
            and snapshot.instance_id >= 0
            else None
        )
        generation = (
            snapshot.connection_generation
            if isinstance(snapshot.connection_generation, int)
            and not isinstance(snapshot.connection_generation, bool)
            and snapshot.connection_generation >= 0
            else None
        )
        sequence = (
            snapshot.sequence
            if isinstance(snapshot.sequence, int)
            and not isinstance(snapshot.sequence, bool)
            and snapshot.sequence >= 0
            else None
        )
        self.manager.update_runtime_status(
            self.claim.session_id,
            controller_connected=snapshot.connected,
            controller_error=(sample_error if sample_error is not None else snapshot.read_error),
            controller_product_name=snapshot.product_name,
            controller_guid=snapshot.guid,
            controller_instance_id=instance_id,
            controller_generation=generation,
            controller_layout=(snapshot.layout.axes, snapshot.layout.buttons, snapshot.layout.hats),
            sample_sequence=sequence,
            sample_age_s=age,
            rb_held=snapshot.button(10) if len(snapshot.buttons) > 10 else False,
            release_observed=decision.state.release_seen,
            controls_neutral=decision.controls_neutral,
            motion_state=MotionState.DISARMED,
            joint_specs=(),
            saturation_count=0,
            relative_clipping_count=0,
            thermal_snapshot=None,
            details_patch={
                "controller_gate_reason": decision.reason,
                "controller_ready": (
                    sample_error is None
                    and decision.profile_valid
                    and decision.controls_neutral
                    and decision.state.neutral_armed
                    and not snapshot.button(10)
                ),
            },
        )

    def _raise_if_stop_requested(self) -> None:
        self.manager.check_lease_expiry()
        if self.claim.stop_requested.is_set() or self.claim.hold_requested.is_set():
            status = self.manager.status_for(self.claim.session_id, check_expiry=False)
            reason = status.stop_reason if status is not None and status.stop_reason else "stop requested"
            raise _StopRequestedError(reason)

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ControllerCheckError("monotonic clock returned a non-finite value")
        return value


__all__ = [
    "ControllerCheckConfig",
    "ControllerCheckError",
    "ControllerCheckResult",
    "ControllerCheckWorker",
]
