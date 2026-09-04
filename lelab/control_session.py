"""Dependency-neutral ownership, lease, and status primitives for control sessions.

This module deliberately owns no devices and starts no background work.  Callers
keep the returned signals and perform teardown themselves; the manager retains
the global claim until that owner explicitly reports teardown completion.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Real
from typing import Any

GLOBAL_CONTROL_RESOURCE = "control"
DEFAULT_LEASE_RENEW_INTERVAL_S = 1.0
DEFAULT_LEASE_TTL_S = 3.0
SO101_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_ACTION_KEYS = tuple(f"{motor}.pos" for motor in SO101_MOTOR_NAMES)


class ControlOperation(StrEnum):
    """Exact operation audit label paired with the global control resource."""

    FOLLOWER_CALIBRATION = "follower_calibration"
    LEADER_CALIBRATION = "leader_calibration"
    LEADER_TELEOPERATION = "leader_teleoperation"
    STADIA_TELEOPERATION = "stadia_teleoperation"
    LEADER_RECORDING = "leader_recording"
    STADIA_RECORDING = "stadia_recording"
    INFERENCE = "inference"
    REPLAY = "replay"
    CONTROLLER_CHECK = "controller_check"


# Compatibility name for the original foundation slice. Resource ownership is
# global; this enum value is the exact operation audit key.
ControlResourceKey = ControlOperation


class ControlState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TorqueOutcome(StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    VERIFIED_OFF = "verified_off"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MotionState(StrEnum):
    DISARMED = "disarmed"
    HOLD = "hold"
    ENABLED = "enabled"


class JointStatusUnit(StrEnum):
    DEGREES = "degrees"
    GRIPPER_PERCENTAGE_POINTS = "gripper_percentage_points"


@dataclass(frozen=True)
class JointStatusSpec:
    """One exact action key's configured units and safety bounds."""

    action_key: str
    unit: JointStatusUnit
    max_step_per_tick: float
    max_relative_target: float
    startup_min: float
    startup_max: float
    calibrated_min: float
    calibrated_max: float

    def __post_init__(self) -> None:
        if self.action_key not in SO101_ACTION_KEYS:
            raise ValueError(f"unknown SO-101 action key: {self.action_key!r}")
        selected_unit = JointStatusUnit(self.unit)
        expected_unit = (
            JointStatusUnit.GRIPPER_PERCENTAGE_POINTS
            if self.action_key == "gripper.pos"
            else JointStatusUnit.DEGREES
        )
        if selected_unit is not expected_unit:
            raise ValueError(f"{self.action_key} must use {expected_unit.value}")
        object.__setattr__(self, "unit", selected_unit)

        numeric_fields = (
            "max_step_per_tick",
            "max_relative_target",
            "startup_min",
            "startup_max",
            "calibrated_min",
            "calibrated_max",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be a finite number")
            object.__setattr__(self, field_name, float(value))
        if self.max_step_per_tick <= 0 or self.max_relative_target <= 0:
            raise ValueError("per-joint step and relative-target limits must be positive")
        if self.startup_min > self.startup_max:
            raise ValueError("startup bounds must be ordered")
        if self.calibrated_min > self.calibrated_max:
            raise ValueError("calibrated bounds must be ordered")

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "unit": self.unit.value,
            "max_step_per_tick": self.max_step_per_tick,
            "max_relative_target": self.max_relative_target,
            "startup_min": self.startup_min,
            "startup_max": self.startup_max,
            "calibrated_min": self.calibrated_min,
            "calibrated_max": self.calibrated_max,
        }


def _ordered_motor_floats(values: Mapping[str, object], label: str) -> tuple[tuple[str, float], ...]:
    if set(values) != set(SO101_MOTOR_NAMES):
        raise ValueError(f"{label} must contain exactly the six SO-101 motors")
    result = []
    for motor in SO101_MOTOR_NAMES:
        value = values[motor]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{label}.{motor} must be a finite number")
        result.append((motor, float(value)))
    return tuple(result)


def _ordered_motor_optional_floats(
    values: Mapping[str, object],
    label: str,
) -> tuple[tuple[str, float | None], ...]:
    if set(values) != set(SO101_MOTOR_NAMES):
        raise ValueError(f"{label} must contain exactly the six SO-101 motors")
    result = []
    for motor in SO101_MOTOR_NAMES:
        value = values[motor]
        if value is None:
            result.append((motor, None))
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{label}.{motor} must be a finite number or null")
        result.append((motor, float(value)))
    return tuple(result)


def _ordered_motor_counts(values: Mapping[str, object], label: str) -> tuple[tuple[str, int], ...]:
    if set(values) != set(SO101_MOTOR_NAMES):
        raise ValueError(f"{label} must contain exactly the six SO-101 motors")
    result = []
    for motor in SO101_MOTOR_NAMES:
        value = values[motor]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{motor} must be a non-negative integer")
        result.append((motor, value))
    return tuple(result)


@dataclass(frozen=True)
class ThermalStatus:
    """Finite-or-null, immutable thermal evidence safe for persistent status.

    A missing or non-finite first sensor packet has no truthful numeric value.
    Exact-six nullable maps retain that absence without inventing a temperature
    or dropping the fail-closed evidence entirely.
    """

    temperatures: tuple[tuple[str, float | None], ...]
    reported_peaks: tuple[tuple[str, float | None], ...]
    confirmed_peaks: tuple[tuple[str, float | None], ...]
    spike_counts: tuple[tuple[str, int], ...]
    invalid_sample_counts: tuple[tuple[str, int], ...]
    last_invalid_values: tuple[tuple[str, float | None], ...]
    warning_motors: tuple[str, ...] = ()
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "temperatures",
            "reported_peaks",
            "confirmed_peaks",
            "last_invalid_values",
        ):
            raw = getattr(self, field_name)
            if not isinstance(raw, tuple) or len(raw) != len(SO101_MOTOR_NAMES):
                raise ValueError(f"{field_name} must contain exactly the six SO-101 motors")
            object.__setattr__(
                self,
                field_name,
                _ordered_motor_optional_floats(dict(raw), field_name),
            )
        for field_name in ("spike_counts", "invalid_sample_counts"):
            raw = getattr(self, field_name)
            if not isinstance(raw, tuple) or len(raw) != len(SO101_MOTOR_NAMES):
                raise ValueError(f"{field_name} must contain exactly the six SO-101 motors")
            object.__setattr__(
                self,
                field_name,
                _ordered_motor_counts(dict(raw), field_name),
            )
        warnings = tuple(self.warning_motors)
        if len(set(warnings)) != len(warnings) or any(motor not in SO101_MOTOR_NAMES for motor in warnings):
            raise ValueError("warning_motors must be unique SO-101 motor names")
        object.__setattr__(self, "warning_motors", warnings)
        object.__setattr__(self, "stop_reason", _optional_text(self.stop_reason, "thermal stop_reason"))

    @classmethod
    def from_mappings(
        cls,
        *,
        temperatures: Mapping[str, object],
        reported_peaks: Mapping[str, object],
        confirmed_peaks: Mapping[str, object],
        spike_counts: Mapping[str, object],
        invalid_sample_counts: Mapping[str, object],
        last_invalid_values: Mapping[str, object],
        warning_motors: tuple[str, ...] = (),
        stop_reason: str | None = None,
    ) -> ThermalStatus:
        return cls(
            temperatures=_ordered_motor_optional_floats(temperatures, "temperatures"),
            reported_peaks=_ordered_motor_optional_floats(reported_peaks, "reported_peaks"),
            confirmed_peaks=_ordered_motor_optional_floats(confirmed_peaks, "confirmed_peaks"),
            spike_counts=_ordered_motor_counts(spike_counts, "spike_counts"),
            invalid_sample_counts=_ordered_motor_counts(invalid_sample_counts, "invalid_sample_counts"),
            last_invalid_values=_ordered_motor_optional_floats(
                last_invalid_values,
                "last_invalid_values",
            ),
            warning_motors=warning_motors,
            stop_reason=stop_reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperatures": dict(self.temperatures),
            "reported_peaks": dict(self.reported_peaks),
            "confirmed_peaks": dict(self.confirmed_peaks),
            "spike_counts": dict(self.spike_counts),
            "invalid_sample_counts": dict(self.invalid_sample_counts),
            "last_invalid_values": dict(self.last_invalid_values),
            "warning_motors": list(self.warning_motors),
            "stop_reason": self.stop_reason,
        }


class ControlSessionError(RuntimeError):
    """Base class for ownership and lifecycle errors."""


class ControlSessionBusyError(ControlSessionError):
    """The global control resource is still owned by another session."""


class ControlManagerClosingError(ControlSessionError):
    """The manager is closing and cannot accept another owner."""


class StaleSessionError(ControlSessionError):
    """An update did not come from the exact active session ID."""


class InvalidControlTransitionError(ControlSessionError):
    """A session transition would skip required lifecycle states."""


@dataclass(frozen=True)
class TorqueEvidence:
    """JSON-safe evidence supporting one conservative torque classification."""

    outcome: TorqueOutcome
    disable_attempted: bool
    verification_supported: bool
    readback: tuple[tuple[str, bool | None], ...]
    missing_motors: tuple[str, ...]
    invalid_motors: tuple[str, ...]
    unexpected_motors: tuple[str, ...]
    disable_errors: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "disable_attempted": self.disable_attempted,
            "verification_supported": self.verification_supported,
            "readback": dict(self.readback),
            "missing_motors": list(self.missing_motors),
            "invalid_motors": list(self.invalid_motors),
            "unexpected_motors": list(self.unexpected_motors),
            "disable_errors": dict(self.disable_errors),
        }


def _torque_bit(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    return None


def classify_torque_outcome(
    *,
    disable_attempted: bool,
    readback: Mapping[str, object] | None = None,
    disable_errors: Mapping[str, object] | None = None,
    verification_supported: bool = True,
) -> TorqueEvidence:
    """Classify torque truth from all six expected SO-101 motors.

    Any disable error or known-enabled motor is a failure.  Only a complete,
    supported six-motor disabled readback is verified off.  Incomplete or
    unavailable evidence with no known-enabled motor remains unknown.
    """

    values = dict(readback or {})
    errors = tuple(sorted((str(key), str(value)) for key, value in (disable_errors or {}).items()))
    normalized = tuple((motor, _torque_bit(values.get(motor))) for motor in SO101_MOTOR_NAMES)
    missing = tuple(motor for motor in SO101_MOTOR_NAMES if motor not in values)
    invalid = tuple(motor for motor, value in normalized if motor in values and value is None)
    unexpected = tuple(sorted(str(key) for key in set(values) - set(SO101_MOTOR_NAMES)))
    any_enabled = any(value is True for _, value in normalized)
    complete_off = (
        verification_supported
        and not missing
        and not invalid
        and all(value is False for _, value in normalized)
    )

    if errors or any_enabled:
        outcome = TorqueOutcome.FAILED
    elif complete_off:
        outcome = TorqueOutcome.VERIFIED_OFF
    elif not disable_attempted and not values:
        outcome = TorqueOutcome.NOT_ATTEMPTED
    else:
        outcome = TorqueOutcome.UNKNOWN

    return TorqueEvidence(
        outcome=outcome,
        disable_attempted=bool(disable_attempted),
        verification_supported=bool(verification_supported),
        readback=normalized,
        missing_motors=missing,
        invalid_motors=invalid,
        unexpected_motors=unexpected,
        disable_errors=errors,
    )


NOT_ATTEMPTED_TORQUE = classify_torque_outcome(disable_attempted=False)


def _utc_text(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat()


def _finite_monotonic(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise ValueError("monotonic clock must return a finite value")
    return value


def _validated_details_json(details: Mapping[str, object] | None) -> str:
    if details is None:
        return "{}"
    if not isinstance(details, Mapping):
        raise TypeError("status details must be a mapping")
    if any(not isinstance(key, str) for key in details):
        raise TypeError("status detail keys must be strings")
    try:
        return json.dumps(
            dict(details),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("status details must contain finite JSON-safe values") from error


def _optional_text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string or None")
    return value


def _optional_nonnegative_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or None")
    return value


def _optional_nonnegative_float(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a non-negative finite number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be a non-negative finite number or None")
    return numeric


def _optional_bool(value: bool | None, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean or None")
    return value


def _normalized_layout(layout: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    if layout is None:
        return None
    if not isinstance(layout, tuple) or len(layout) != 3:
        raise TypeError("controller_layout must be an (axes, buttons, hats) tuple or None")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in layout):
        raise ValueError("controller layout counts must be non-negative integers")
    return layout


def _normalized_joint_specs(
    specs: tuple[JointStatusSpec, ...],
) -> tuple[JointStatusSpec, ...]:
    if not isinstance(specs, tuple) or any(not isinstance(spec, JointStatusSpec) for spec in specs):
        raise TypeError("joint_specs must be a tuple of JointStatusSpec values")
    if not specs:
        return ()
    by_key = {spec.action_key: spec for spec in specs}
    if len(by_key) != len(specs) or set(by_key) != set(SO101_ACTION_KEYS):
        raise ValueError("joint_specs must be empty or contain each SO-101 action key exactly once")
    return tuple(by_key[key] for key in SO101_ACTION_KEYS)


def _nonnegative_count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ControlStatus:
    session_id: str
    state: ControlState
    operation: ControlOperation
    resource_keys: tuple[str, str]
    teleoperator_type: str | None
    claimed_at_utc: str
    updated_at_utc: str
    lease_deadline_monotonic: float
    lease_ttl_s: float
    lease_renew_interval_s: float
    controller_connected: bool | None = None
    controller_error: str | None = None
    controller_product_name: str | None = None
    controller_guid: str | None = None
    controller_instance_id: int | None = None
    controller_generation: int | None = None
    controller_layout: tuple[int, int, int] | None = None
    sample_sequence: int | None = None
    sample_age_s: float | None = None
    rb_held: bool | None = None
    release_observed: bool = False
    controls_neutral: bool | None = None
    motion_state: MotionState = MotionState.DISARMED
    joint_specs: tuple[JointStatusSpec, ...] = ()
    saturation_count: int = 0
    relative_clipping_count: int = 0
    thermal_snapshot: ThermalStatus | None = None
    stop_reason: str | None = None
    hold_requested: bool = False
    stop_requested: bool = False
    torque: TorqueEvidence = NOT_ATTEMPTED_TORQUE
    teardown_completed_at_utc: str | None = None
    revision: int = 0
    _details_json: str = "{}"

    @property
    def terminal(self) -> bool:
        return self.state in (ControlState.STOPPED, ControlState.ERROR)

    @property
    def details(self) -> dict[str, Any]:
        return json.loads(self._details_json)

    def as_dict(self) -> dict[str, Any]:
        """Return a finite JSON-safe copy suitable for an API response."""

        result = {
            "session_id": self.session_id,
            "state": self.state.value,
            "operation": self.operation.value,
            "resource_keys": list(self.resource_keys),
            "teleoperator_type": self.teleoperator_type,
            "claimed_at_utc": self.claimed_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "lease_deadline_monotonic": self.lease_deadline_monotonic,
            "lease_ttl_s": self.lease_ttl_s,
            "lease_renew_interval_s": self.lease_renew_interval_s,
            "controller_connected": self.controller_connected,
            "controller_error": self.controller_error,
            "controller_product_name": self.controller_product_name,
            "controller_guid": self.controller_guid,
            "controller_instance_id": self.controller_instance_id,
            "controller_generation": self.controller_generation,
            "controller_layout": (
                {
                    "axes": self.controller_layout[0],
                    "buttons": self.controller_layout[1],
                    "hats": self.controller_layout[2],
                }
                if self.controller_layout is not None
                else None
            ),
            "sample_sequence": self.sample_sequence,
            "sample_age_s": self.sample_age_s,
            "rb_held": self.rb_held,
            "release_observed": self.release_observed,
            "controls_neutral": self.controls_neutral,
            "motion_state": self.motion_state.value,
            "joint_units": {spec.action_key: spec.unit.value for spec in self.joint_specs},
            "joint_limits": {
                spec.action_key: {
                    "max_step_per_tick": spec.max_step_per_tick,
                    "max_relative_target": spec.max_relative_target,
                    "startup_min": spec.startup_min,
                    "startup_max": spec.startup_max,
                    "calibrated_min": spec.calibrated_min,
                    "calibrated_max": spec.calibrated_max,
                }
                for spec in self.joint_specs
            },
            "joint_specs": [spec.as_dict() for spec in self.joint_specs],
            "saturation_count": self.saturation_count,
            "relative_clipping_count": self.relative_clipping_count,
            "stop_reason": self.stop_reason,
            "hold_requested": self.hold_requested,
            "stop_requested": self.stop_requested,
            "torque": self.torque.as_dict(),
            "torque_outcome": self.torque.outcome.value,
            "thermal_snapshot": (
                self.thermal_snapshot.as_dict() if self.thermal_snapshot is not None else None
            ),
            "teardown_completed_at_utc": self.teardown_completed_at_utc,
            "revision": self.revision,
            "details": self.details,
        }
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class ControlSessionClaim:
    """Owner-held identity and cooperative teardown signals."""

    session_id: str
    resource_keys: tuple[str, str]
    hold_requested: threading.Event
    stop_requested: threading.Event
    teardown_completed: threading.Event


class ControlSessionManager:
    """Serialize all control owners behind one process-local atomic claim."""

    def __init__(
        self,
        *,
        history_limit: int = 16,
        lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
        lease_renew_interval_s: float = DEFAULT_LEASE_RENEW_INTERVAL_S,
        monotonic_clock: Callable[[], float] | None = None,
        utc_clock: Callable[[], datetime] | None = None,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(history_limit, int) or isinstance(history_limit, bool) or history_limit <= 0:
            raise ValueError("history_limit must be a positive integer")
        for label, value in (
            ("lease_ttl_s", lease_ttl_s),
            ("lease_renew_interval_s", lease_renew_interval_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if lease_renew_interval_s >= lease_ttl_s:
            raise ValueError("lease renew interval must be shorter than the server TTL")

        self.lease_ttl_s = float(lease_ttl_s)
        self.lease_renew_interval_s = float(lease_renew_interval_s)
        self._monotonic_clock = monotonic_clock or __import__("time").monotonic
        self._utc_clock = utc_clock or (lambda: datetime.now(UTC))
        self._session_id_factory = session_id_factory or (lambda: str(uuid.uuid4()))
        self._lock = threading.Lock()
        self._active_status: ControlStatus | None = None
        self._active_claim: ControlSessionClaim | None = None
        self._latest_terminal: ControlStatus | None = None
        self._terminal_history: deque[ControlStatus] = deque(maxlen=history_limit)
        # Fatal infrastructure failures are requested cooperatively, just like
        # ordinary stops, but the eventual owner-reported teardown may not
        # downgrade them to a successful STOPPED terminal.
        self._terminal_error_requested: set[str] = set()
        # Never recycle an identity: a delayed update from an old owner must
        # not become valid again after its terminal snapshot ages out.
        self._issued_session_ids: set[str] = set()
        self._closing = False
        self._quarantine_reason: str | None = None
        self._recyclable_after_shutdown = False

    @property
    def closing(self) -> bool:
        with self._lock:
            return self._closing

    @property
    def quarantine_reason(self) -> str | None:
        with self._lock:
            return self._quarantine_reason

    @property
    def recyclable_after_shutdown(self) -> bool:
        with self._lock:
            return self._recyclable_after_shutdown

    def quarantine(self, *, reason: str) -> None:
        """Block future claims after resource release could not be proven."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quarantine reason must be a non-empty string")
        with self._lock:
            self._closing = True
            self._recyclable_after_shutdown = False
            self._quarantine_reason = reason
            self._signal_owner_locked()

    def claim(
        self,
        operation: ControlOperation | str,
        *,
        teleoperator_type: str | None = None,
        session_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ControlSessionClaim:
        operation = ControlOperation(operation)
        if teleoperator_type not in (None, "leader_arm", "stadia"):
            raise ValueError("teleoperator_type must be 'leader_arm', 'stadia', or None")
        operation_type = None
        if operation in {
            ControlOperation.STADIA_TELEOPERATION,
            ControlOperation.STADIA_RECORDING,
            ControlOperation.CONTROLLER_CHECK,
        }:
            operation_type = "stadia"
        elif operation in {
            ControlOperation.LEADER_CALIBRATION,
            ControlOperation.LEADER_TELEOPERATION,
            ControlOperation.LEADER_RECORDING,
        }:
            operation_type = "leader_arm"
        if operation_type is not None:
            if teleoperator_type is not None and teleoperator_type != operation_type:
                raise ValueError(f"{operation.value} requires teleoperator_type={operation_type!r}")
            teleoperator_type = operation_type
        details_json = _validated_details_json(details)
        with self._lock:
            if self._closing:
                detail = f": {self._quarantine_reason}" if self._quarantine_reason is not None else ""
                raise ControlManagerClosingError(f"control manager is closing{detail}")
            if self._active_status is not None:
                raise ControlSessionBusyError(
                    f"control resource is owned by session {self._active_status.session_id}"
                )
            identity = session_id if session_id is not None else self._session_id_factory()
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("session_id must be a non-empty string")
            if identity in self._issued_session_ids:
                raise ValueError("session_id must never be reused")

            now = _finite_monotonic(self._monotonic_clock)
            utc_now = _utc_text(self._utc_clock)
            resource_keys = (GLOBAL_CONTROL_RESOURCE, operation.value)
            claim = ControlSessionClaim(
                session_id=identity,
                resource_keys=resource_keys,
                hold_requested=threading.Event(),
                stop_requested=threading.Event(),
                teardown_completed=threading.Event(),
            )
            self._active_claim = claim
            self._issued_session_ids.add(identity)
            self._active_status = ControlStatus(
                session_id=identity,
                state=ControlState.STARTING,
                operation=operation,
                resource_keys=resource_keys,
                teleoperator_type=teleoperator_type,
                claimed_at_utc=utc_now,
                updated_at_utc=utc_now,
                lease_deadline_monotonic=now + self.lease_ttl_s,
                lease_ttl_s=self.lease_ttl_s,
                lease_renew_interval_s=self.lease_renew_interval_s,
                _details_json=details_json,
            )
            return claim

    def current_status(self, *, check_expiry: bool = True) -> ControlStatus | None:
        with self._lock:
            if check_expiry:
                self._expire_if_due_locked(_finite_monotonic(self._monotonic_clock))
            return self._active_status or self._latest_terminal

    def active_status(self, *, check_expiry: bool = True) -> ControlStatus | None:
        with self._lock:
            if check_expiry:
                self._expire_if_due_locked(_finite_monotonic(self._monotonic_clock))
            return self._active_status

    def terminal_history(self) -> tuple[ControlStatus, ...]:
        with self._lock:
            return tuple(self._terminal_history)

    def status_for(self, session_id: str, *, check_expiry: bool = True) -> ControlStatus | None:
        """Return an exact active or retained terminal status by session ID."""

        with self._lock:
            if check_expiry:
                self._expire_if_due_locked(_finite_monotonic(self._monotonic_clock))
            return self._status_for_locked(session_id)

    def wait_for_teardown(self, session_id: str, *, timeout: float) -> bool:
        """Wait boundedly for the exact owner's cleanup without inventing completion."""

        if not isinstance(timeout, Real) or isinstance(timeout, bool):
            raise TypeError("timeout must be a finite non-negative number")
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value < 0:
            raise ValueError("timeout must be a finite non-negative number")

        with self._lock:
            retained = self._status_for_locked(session_id)
            if retained is not None and retained.terminal:
                return True
            if self._active_status is None or self._active_status.session_id != session_id:
                if session_id not in self._issued_session_ids:
                    raise StaleSessionError(f"unknown session {session_id!r}")
                # A terminal status can age out of bounded history. Never claim
                # success without retained evidence.
                return False
            if self._active_claim is None:
                raise StaleSessionError(f"session {session_id!r} has no active owner")
            completed = self._active_claim.teardown_completed

        return completed.wait(timeout_value)

    def mark_running(self, session_id: str) -> ControlStatus:
        with self._lock:
            now = _finite_monotonic(self._monotonic_clock)
            self._expire_if_due_locked(now)
            status = self._owned_status_locked(session_id)
            if status.state is not ControlState.STARTING:
                raise InvalidControlTransitionError(f"cannot mark {status.state.value} session running")
            return self._replace_active_locked(state=ControlState.RUNNING)

    def update_details(self, session_id: str, details: Mapping[str, object]) -> ControlStatus:
        details_json = _validated_details_json(details)
        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.terminal:
                raise InvalidControlTransitionError("cannot update terminal status")
            return self._replace_active_locked(_details_json=details_json)

    def merge_details(self, session_id: str, details: Mapping[str, object]) -> ControlStatus:
        """Merge JSON-safe operational details without losing claim identity."""

        patch = json.loads(_validated_details_json(details))
        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.terminal:
                raise InvalidControlTransitionError("cannot update terminal status")
            merged = status.details
            merged.update(patch)
            return self._replace_active_locked(
                _details_json=_validated_details_json(merged),
            )

    def update_joint_specs_and_details(
        self,
        session_id: str,
        *,
        joint_specs: tuple[JointStatusSpec, ...],
        details_patch: Mapping[str, object],
    ) -> ControlStatus:
        """Atomically publish a live joint-cap change and its audit details."""

        specs = _normalized_joint_specs(joint_specs)
        patch = json.loads(_validated_details_json(details_patch))
        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state is not ControlState.RUNNING:
                raise InvalidControlTransitionError(
                    "joint control settings can change only while a session is running"
                )
            details = status.details
            details.update(patch)
            return self._replace_active_locked(
                joint_specs=specs,
                _details_json=_validated_details_json(details),
            )

    def update_runtime_status(
        self,
        session_id: str,
        *,
        controller_product_name: str | None,
        controller_guid: str | None,
        controller_instance_id: int | None,
        controller_generation: int | None,
        controller_layout: tuple[int, int, int] | None,
        sample_sequence: int | None,
        sample_age_s: float | None,
        rb_held: bool | None,
        release_observed: bool,
        controls_neutral: bool | None,
        motion_state: MotionState | str,
        joint_specs: tuple[JointStatusSpec, ...],
        saturation_count: int,
        relative_clipping_count: int,
        thermal_snapshot: ThermalStatus | None,
        controller_connected: bool | None = None,
        controller_error: str | None = None,
        details_patch: Mapping[str, object] | None = None,
    ) -> ControlStatus:
        """Atomically publish every plan-owned live status field."""

        connected = _optional_bool(controller_connected, "controller_connected")
        error_text = _optional_text(controller_error, "controller_error")
        if connected is False and error_text is None:
            raise ValueError("a disconnected controller must publish a controller_error")
        if connected is None and error_text is not None:
            raise ValueError("controller_error requires an explicit disconnected state")
        product_name = _optional_text(controller_product_name, "controller_product_name")
        guid = _optional_text(controller_guid, "controller_guid")
        instance_id = _optional_nonnegative_int(controller_instance_id, "controller_instance_id")
        generation = _optional_nonnegative_int(controller_generation, "controller_generation")
        layout = _normalized_layout(controller_layout)
        sequence = _optional_nonnegative_int(sample_sequence, "sample_sequence")
        age = _optional_nonnegative_float(sample_age_s, "sample_age_s")
        rb_value = _optional_bool(rb_held, "rb_held")
        if not isinstance(release_observed, bool):
            raise TypeError("release_observed must be a boolean")
        neutral_value = _optional_bool(controls_neutral, "controls_neutral")
        selected_motion_state = MotionState(motion_state)
        specs = _normalized_joint_specs(joint_specs)
        saturation = _nonnegative_count(saturation_count, "saturation_count")
        clipping = _nonnegative_count(relative_clipping_count, "relative_clipping_count")
        if thermal_snapshot is not None and not isinstance(thermal_snapshot, ThermalStatus):
            raise TypeError("thermal_snapshot must be ThermalStatus or None")
        normalized_details_patch = (
            {} if details_patch is None else json.loads(_validated_details_json(details_patch))
        )

        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state not in (ControlState.STARTING, ControlState.RUNNING, ControlState.STOPPING):
                raise InvalidControlTransitionError("cannot update terminal runtime status")
            details = status.details
            details.update(normalized_details_patch)
            details["controller_monitoring_active"] = True
            return self._replace_active_locked(
                controller_connected=connected,
                controller_error=error_text,
                controller_product_name=product_name,
                controller_guid=guid,
                controller_instance_id=instance_id,
                controller_generation=generation,
                controller_layout=layout,
                sample_sequence=sequence,
                sample_age_s=age,
                rb_held=rb_value,
                release_observed=release_observed,
                controls_neutral=neutral_value,
                motion_state=selected_motion_state,
                joint_specs=specs,
                saturation_count=saturation,
                relative_clipping_count=clipping,
                thermal_snapshot=thermal_snapshot,
                _details_json=_validated_details_json(details),
            )

    def mark_controller_monitoring_inactive(self, session_id: str) -> ControlStatus:
        """Clear live-only controller fields after its reader is proven stopped."""

        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state not in (
                ControlState.STARTING,
                ControlState.RUNNING,
                ControlState.STOPPING,
            ):
                raise InvalidControlTransitionError(
                    "controller monitoring can stop only before terminal publication"
                )
            details = status.details
            details["controller_monitoring_active"] = False
            details["controller_last_observed"] = {
                "connected": status.controller_connected,
                "error": status.controller_error,
                "sample_age_s": status.sample_age_s,
                "rb_held": status.rb_held,
                "controls_neutral": status.controls_neutral,
                "motion_state": status.motion_state.value,
            }
            return self._replace_active_locked(
                controller_connected=None,
                controller_error=None,
                sample_age_s=None,
                rb_held=None,
                controls_neutral=None,
                motion_state=MotionState.DISARMED,
                _details_json=_validated_details_json(details),
            )

    def renew_lease(self, session_id: str) -> ControlStatus:
        with self._lock:
            now = _finite_monotonic(self._monotonic_clock)
            self._expire_if_due_locked(now)
            status = self._owned_status_locked(session_id)
            if status.state not in (ControlState.STARTING, ControlState.RUNNING):
                raise InvalidControlTransitionError(
                    f"cannot renew a lease while session is {status.state.value}"
                )
            return self._replace_active_locked(lease_deadline_monotonic=now + self.lease_ttl_s)

    def check_lease_expiry(self) -> bool:
        with self._lock:
            return self._expire_if_due_locked(_finite_monotonic(self._monotonic_clock))

    def request_stop(self, session_id: str, *, reason: str) -> ControlStatus:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stop reason must be a non-empty string")
        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state not in (
                ControlState.STARTING,
                ControlState.RUNNING,
                ControlState.STOPPING,
            ):
                raise InvalidControlTransitionError(f"cannot stop a session while it is {status.state.value}")
            if status.state is ControlState.STOPPING:
                self._signal_owner_locked()
                return status
            return self._transition_to_stopping_locked(reason)

    def request_failure(self, session_id: str, *, reason: str) -> ControlStatus:
        """Signal fail-closed teardown and preserve an eventual ERROR terminal."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state not in (
                ControlState.STARTING,
                ControlState.RUNNING,
                ControlState.STOPPING,
            ):
                raise InvalidControlTransitionError(f"cannot fail a session while it is {status.state.value}")
            self._terminal_error_requested.add(session_id)
            if status.state is ControlState.STOPPING:
                self._signal_owner_locked()
                if status.stop_reason == reason:
                    return status
                return self._replace_active_locked(stop_reason=reason)
            return self._transition_to_stopping_locked(reason)

    def begin_shutdown(
        self,
        *,
        reason: str = "server shutdown",
        terminal_error: bool = False,
    ) -> ControlStatus | None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("shutdown reason must be a non-empty string")
        if not isinstance(terminal_error, bool):
            raise TypeError("terminal_error must be a bool")
        with self._lock:
            was_closing = self._closing
            self._closing = True
            if terminal_error:
                self._recyclable_after_shutdown = False
            elif not was_closing:
                self._recyclable_after_shutdown = True
            if self._active_status is None:
                return self._latest_terminal
            if terminal_error:
                self._terminal_error_requested.add(self._active_status.session_id)
            if self._active_status.state is ControlState.STOPPING:
                self._signal_owner_locked()
                if self._active_status.stop_reason == reason:
                    return self._active_status
                return self._replace_active_locked(stop_reason=reason)
            return self._transition_to_stopping_locked(reason)

    def finish_teardown(
        self,
        session_id: str,
        *,
        terminal_state: ControlState | str,
        torque: TorqueEvidence,
        reason: str | None = None,
    ) -> ControlStatus:
        terminal_state = ControlState(terminal_state)
        if terminal_state not in (ControlState.STOPPED, ControlState.ERROR):
            raise ValueError("teardown must finish as stopped or error")
        if not isinstance(torque, TorqueEvidence):
            raise TypeError("torque must be TorqueEvidence")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("terminal reason must be a non-empty string or None")

        with self._lock:
            status = self._owned_status_locked(session_id)
            if status.state is not ControlState.STOPPING:
                raise InvalidControlTransitionError(
                    "teardown can finish only after the session enters stopping"
                )
            if session_id in self._terminal_error_requested:
                terminal_state = ControlState.ERROR
            utc_now = _utc_text(self._utc_clock)
            terminal = replace(
                status,
                state=terminal_state,
                stop_reason=reason or status.stop_reason,
                torque=torque,
                teardown_completed_at_utc=utc_now,
                updated_at_utc=utc_now,
                revision=status.revision + 1,
            )
            terminal.as_dict()
            claim = self._active_claim
            self._terminal_history.append(terminal)
            self._latest_terminal = terminal
            self._active_status = None
            self._active_claim = None
            self._terminal_error_requested.discard(session_id)
            if claim is not None:
                claim.teardown_completed.set()
            return terminal

    def _status_for_locked(self, session_id: str) -> ControlStatus | None:
        if self._active_status is not None and self._active_status.session_id == session_id:
            return self._active_status
        for status in reversed(self._terminal_history):
            if status.session_id == session_id:
                return status
        return None

    def _owned_status_locked(self, session_id: str) -> ControlStatus:
        status = self._active_status
        if status is None or status.session_id != session_id:
            raise StaleSessionError(f"session {session_id!r} does not own control")
        return status

    def _replace_active_locked(self, **changes: object) -> ControlStatus:
        if self._active_status is None:
            raise StaleSessionError("no active control session")
        status = replace(
            self._active_status,
            updated_at_utc=_utc_text(self._utc_clock),
            revision=self._active_status.revision + 1,
            **changes,
        )
        status.as_dict()
        self._active_status = status
        return status

    def _signal_owner_locked(self) -> None:
        if self._active_claim is not None:
            self._active_claim.hold_requested.set()
            self._active_claim.stop_requested.set()

    def _transition_to_stopping_locked(self, reason: str) -> ControlStatus:
        self._signal_owner_locked()
        return self._replace_active_locked(
            state=ControlState.STOPPING,
            stop_reason=reason,
            hold_requested=True,
            stop_requested=True,
        )

    def _expire_if_due_locked(self, now: float) -> bool:
        status = self._active_status
        if status is None or status.state not in (ControlState.STARTING, ControlState.RUNNING):
            return False
        if now < status.lease_deadline_monotonic:
            return False
        self._transition_to_stopping_locked("control lease expired")
        return True


__all__ = [
    "DEFAULT_LEASE_RENEW_INTERVAL_S",
    "DEFAULT_LEASE_TTL_S",
    "GLOBAL_CONTROL_RESOURCE",
    "SO101_ACTION_KEYS",
    "SO101_MOTOR_NAMES",
    "ControlManagerClosingError",
    "ControlOperation",
    "ControlResourceKey",
    "ControlSessionBusyError",
    "ControlSessionClaim",
    "ControlSessionError",
    "ControlSessionManager",
    "ControlState",
    "ControlStatus",
    "InvalidControlTransitionError",
    "JointStatusSpec",
    "JointStatusUnit",
    "MotionState",
    "StaleSessionError",
    "ThermalStatus",
    "TorqueEvidence",
    "TorqueOutcome",
    "classify_torque_outcome",
]
