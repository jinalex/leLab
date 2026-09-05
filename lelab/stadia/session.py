"""Guarded live Stadia-to-SO-101 session ownership.

This module is safe to import without pygame or LeRobot installed.  Device
libraries are resolved only after the controller has supplied the required
fresh, neutral, RB-up samples.  Tests can replace every device-facing seam.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from lelab.control_session import (
    SO101_MOTOR_NAMES,
    ControlSessionClaim,
    ControlSessionManager,
    ControlState,
    ControlStatus,
    JointStatusSpec,
    JointStatusUnit,
    MotionState,
    ThermalStatus,
    TorqueEvidence,
    classify_torque_outcome,
)

from .action import ActionValidationError, compare_requested_returned, validate_returned_action
from .integrator import BoundedStadiaIntegrator
from .mapping import NeutralReleaseGate, map_stadia_input
from .thermal_safety import ConfirmedTemperatureGuard
from .timing import NoCatchUpScheduler
from .types import (
    ACTION_KEYS,
    DEFAULT_JOINT_SPECS,
    NeutralGateDecision,
    NeutralGateState,
    StadiaSnapshot,
)

CONTROL_RATE_HZ = 30.0
MAX_SNAPSHOT_AGE_S = 0.15
THERMAL_INTERVAL_S = 1.0
FOLLOWER_READ_RETRIES = 5
MAX_RELATIVE_TARGET = 5.0
MIN_SPEED_MULTIPLIER = 0.25
MAX_SPEED_MULTIPLIER = 2.0

_ACTION_TO_URDF_JOINT = {
    "shoulder_pan.pos": "Rotation",
    "shoulder_lift.pos": "Pitch",
    "elbow_flex.pos": "Elbow",
    "wrist_flex.pos": "Wrist_Pitch",
    "wrist_roll.pos": "Wrist_Roll",
    "gripper.pos": "Jaw",
}


class StadiaSessionError(RuntimeError):
    """The live worker cannot continue while preserving its safety contract."""


class StadiaSessionStartupError(StadiaSessionError):
    """The worker failed before entering its 30 Hz runtime."""


class StadiaSessionRuntimeError(StadiaSessionError):
    """The worker stopped because a live command-path operation failed."""


class _StopRequestedError(Exception):
    """Internal cooperative stop signal that is not an operational failure."""


class StadiaReader(Protocol):
    """Reader operations used by the session owner."""

    def start(self) -> None: ...

    def snapshot(self) -> StadiaSnapshot: ...

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot: ...

    def stop(self, *, timeout: float = 2.0) -> None: ...


@dataclass(frozen=True, slots=True)
class StadiaSessionConfig:
    """Validated follower and controller settings for one live session."""

    follower_port: str
    follower_calibration: str
    expected_guid: str | None = None
    deadzone: float = 0.15
    max_step_per_tick: float = 0.35
    startup_timeout_s: float = 5.0
    reader_join_timeout_s: float = 2.0
    torque_disable_attempts: int = 2
    cameras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.follower_port, str) or not self.follower_port.strip():
            raise ValueError("follower_port must be a non-empty string")
        calibration = self.follower_calibration
        if (
            not isinstance(calibration, str)
            or not calibration
            or calibration.strip() != calibration
            or not calibration.endswith(".json")
            or Path(calibration).name != calibration
            or "/" in calibration
            or "\\" in calibration
            or ".." in calibration
        ):
            raise ValueError("follower_calibration must be a safe .json basename")
        if self.expected_guid is not None and (
            not isinstance(self.expected_guid, str)
            or not self.expected_guid
            or self.expected_guid.strip() != self.expected_guid
            or "\x00" in self.expected_guid
        ):
            raise ValueError("expected_guid must be a non-empty trimmed string when provided")

        self._finite_range("deadzone", self.deadzone, lower=0.0, upper=1.0, upper_inclusive=False)
        self._finite_range("max_step_per_tick", self.max_step_per_tick, lower=0.0, upper=0.35)
        self._finite_range("startup_timeout_s", self.startup_timeout_s, lower=0.0)
        self._finite_range("reader_join_timeout_s", self.reader_join_timeout_s, lower=0.0)
        if (
            isinstance(self.torque_disable_attempts, bool)
            or not isinstance(self.torque_disable_attempts, int)
            or not 1 <= self.torque_disable_attempts <= 4
        ):
            raise ValueError("torque_disable_attempts must be between 1 and 4")
        if not isinstance(self.cameras, Mapping) or any(
            not isinstance(key, str) or not key for key in self.cameras
        ):
            raise ValueError("cameras must map non-empty string names to camera configurations")
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))

    @staticmethod
    def _finite_range(
        label: str,
        value: object,
        *,
        lower: float,
        upper: float | None = None,
        upper_inclusive: bool = True,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{label} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= lower:
            if label == "deadzone" and numeric == lower:
                pass
            else:
                raise ValueError(f"{label} is outside its safe range")
        if upper is not None and (numeric > upper if upper_inclusive else numeric >= upper):
            raise ValueError(f"{label} is outside its safe range")


@dataclass(frozen=True, slots=True)
class FollowerBuildSpec:
    """Exact lazy-construction request for the Stadia-only follower."""

    port: str
    calibration_id: str
    max_relative_target: Mapping[str, float]
    cameras: Mapping[str, object]
    use_degrees: bool = True

    def __post_init__(self) -> None:
        relative = dict(self.max_relative_target)
        if set(relative) != set(SO101_MOTOR_NAMES) or any(
            isinstance(value, bool) or not isinstance(value, Real) or float(value) != MAX_RELATIVE_TARGET
            for value in relative.values()
        ):
            raise ValueError("max_relative_target must contain all six motors at 5.0")
        object.__setattr__(self, "max_relative_target", MappingProxyType(relative))
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))


@dataclass(frozen=True, slots=True)
class StadiaSessionResult:
    """Immutable evidence returned after terminal manager publication."""

    terminal_state: ControlState
    reason: str
    torque: TorqueEvidence
    commands_sent: int
    movement_steps: int
    missed_ticks: int
    saturation_count: int
    relative_clipping_count: int
    teardown_errors: tuple[str, ...]


def _default_reader(expected_guid: str | None) -> StadiaReader:
    from .device_reader import StadiaDeviceReader

    return StadiaDeviceReader(expected_guid=expected_guid)


def _default_calibration_resolver(filename: str) -> str:
    # Importing LeLab's config module is intentionally delayed until controller
    # neutralization has completed.
    from lelab.utils.config import setup_follower_calibration_file

    return str(setup_follower_calibration_file(filename))


def _default_follower_factory(spec: FollowerBuildSpec) -> object:
    # LeRobot remains absent from the import path until the controller gate has
    # proved a safe startup state.
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        port=spec.port,
        id=spec.calibration_id,
        cameras=dict(spec.cameras),
        use_degrees=spec.use_degrees,
        max_relative_target=dict(spec.max_relative_target),
    )
    return SO101Follower(config)


def _default_thermal_guard(bus: object, sleeper: Callable[[float], None]) -> object:
    return ConfirmedTemperatureGuard(bus, SO101_MOTOR_NAMES, sleeper=sleeper)  # type: ignore[arg-type]


def _mode_value(mode: object) -> str:
    value = getattr(mode, "value", mode)
    return str(value).casefold()


@dataclass(frozen=True, slots=True)
class _MotorPositionScale:
    """Pinned LeRobot v0.6.0 raw-to-normalized position contract."""

    mode: str
    native_max: int
    calibration_min: int
    calibration_max: int
    reverse_range: bool

    def normalize(self, raw_position: int) -> float:
        if self.mode == "degrees":
            midpoint = (self.calibration_min + self.calibration_max) / 2.0
            return (raw_position - midpoint) * 360.0 / self.native_max
        normalized = (
            (raw_position - self.calibration_min) / (self.calibration_max - self.calibration_min) * 100.0
        )
        return 100.0 - normalized if self.reverse_range else normalized


def _native_integer(value: object, label: str) -> int:
    numeric = _finite_number(value, label)
    if not numeric.is_integer():
        raise StadiaSessionStartupError(f"{label} must be an integer native encoder value")
    return int(numeric)


def _position_scales(follower: object) -> dict[str, _MotorPositionScale]:
    """Validate loaded calibration against each motor's native resolution."""

    bus = getattr(follower, "bus", None)
    if bus is None:
        raise StadiaSessionStartupError("follower has no motor bus")
    motors = getattr(bus, "motors", None)
    calibration = getattr(bus, "calibration", None)
    if not isinstance(motors, Mapping) or set(motors) != set(SO101_MOTOR_NAMES):
        raise StadiaSessionStartupError("follower bus must contain exactly the six SO-101 motors")
    if not isinstance(calibration, Mapping) or set(calibration) != set(SO101_MOTOR_NAMES):
        raise StadiaSessionStartupError("loaded calibration must contain exactly the six SO-101 motors")

    resolution_table = getattr(bus, "model_resolution_table", None)
    if not isinstance(resolution_table, Mapping):
        raise StadiaSessionStartupError("follower bus does not expose motor resolution data")
    apply_drive_mode = getattr(bus, "apply_drive_mode", False)
    if not isinstance(apply_drive_mode, bool):
        raise StadiaSessionStartupError("follower bus drive-mode policy must be boolean")

    scales: dict[str, _MotorPositionScale] = {}
    for motor in SO101_MOTOR_NAMES:
        definition = motors[motor]
        entry = calibration[motor]
        mode = _mode_value(getattr(definition, "norm_mode", ""))
        if mode not in {"degrees", "range_0_100"}:
            raise StadiaSessionStartupError(f"unsupported normalization mode for {motor}: {mode!r}")

        model = getattr(definition, "model", None)
        try:
            resolution = _native_integer(resolution_table[model], f"resolution.{motor}")
        except (KeyError, TypeError) as error:
            raise StadiaSessionStartupError(f"resolution is missing for {motor}") from error
        if resolution <= 1:
            raise StadiaSessionStartupError(f"resolution for {motor} must exceed one")
        native_max = resolution - 1
        range_min = _native_integer(
            getattr(entry, "range_min", None),
            f"calibration.{motor}.range_min",
        )
        range_max = _native_integer(
            getattr(entry, "range_max", None),
            f"calibration.{motor}.range_max",
        )
        if not 0 <= range_min < range_max <= native_max:
            raise StadiaSessionStartupError(
                f"calibration range for {motor} must be increasing within native encoder range "
                f"[0, {native_max}]"
            )
        drive_mode = getattr(entry, "drive_mode", 0)
        if isinstance(drive_mode, bool) or not isinstance(drive_mode, int) or drive_mode not in (0, 1):
            raise StadiaSessionStartupError(f"calibration drive_mode for {motor} must be 0 or 1")
        scales[motor] = _MotorPositionScale(
            mode=mode,
            native_max=native_max,
            calibration_min=range_min,
            calibration_max=range_max,
            reverse_range=bool(apply_drive_mode and drive_mode) if mode == "range_0_100" else False,
        )
    return scales


def _endpoint_bounds_from_scales(
    scales: Mapping[str, _MotorPositionScale],
) -> dict[str, tuple[float, float]]:
    if set(scales) != set(SO101_MOTOR_NAMES):
        raise StadiaSessionStartupError("position scales must contain exactly the six SO-101 motors")
    bounds: dict[str, tuple[float, float]] = {}
    for motor in SO101_MOTOR_NAMES:
        scale = scales[motor]
        first = scale.normalize(scale.calibration_min)
        second = scale.normalize(scale.calibration_max)
        lower, upper = sorted((first, second))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise StadiaSessionStartupError(f"derived endpoint bounds for {motor} are invalid")
        bounds[f"{motor}.pos"] = (lower, upper)
    return bounds


def derive_calibrated_endpoint_bounds(follower: object) -> dict[str, tuple[float, float]]:
    """Derive normalized endpoints from the loaded six-motor calibration."""

    return _endpoint_bounds_from_scales(_position_scales(follower))


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StadiaSessionStartupError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise StadiaSessionStartupError(f"{label} must be a finite number")
    return numeric


def _normalize_raw_pose(
    values: object,
    scales: Mapping[str, _MotorPositionScale],
) -> tuple[dict[str, int], dict[str, float]]:
    """Reject unsafe raw positions before applying LeRobot's normalization."""

    if not isinstance(values, Mapping) or set(values) != set(SO101_MOTOR_NAMES):
        raise StadiaSessionStartupError("raw current pose must contain exactly the six SO-101 motors")
    if set(scales) != set(SO101_MOTOR_NAMES):
        raise StadiaSessionStartupError("position scales must contain exactly the six SO-101 motors")
    raw_pose: dict[str, int] = {}
    pose: dict[str, float] = {}
    for motor in SO101_MOTOR_NAMES:
        scale = scales[motor]
        raw = _native_integer(values[motor], f"raw current pose {motor}")
        if not 0 <= raw <= scale.native_max:
            raise StadiaSessionStartupError(
                f"raw current pose {motor}={raw} is outside native encoder range [0, {scale.native_max}]"
            )
        if not scale.calibration_min <= raw <= scale.calibration_max:
            raise StadiaSessionStartupError(
                f"raw current pose {motor}={raw} is outside loaded calibration range "
                f"[{scale.calibration_min}, {scale.calibration_max}]"
            )
        key = f"{motor}.pos"
        normalized = scale.normalize(raw)
        if not math.isfinite(normalized):
            raise StadiaSessionStartupError(f"normalized current pose {motor} is not finite")
        raw_pose[motor] = raw
        pose[key] = normalized
    return raw_pose, pose


def _thermal_status(snapshot: Any, *, stop_reason: str | None = None) -> ThermalStatus:
    def finite_or_none(values: Mapping[str, object]) -> dict[str, float | None]:
        return {
            motor: (
                float(value)
                if not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
                else None
            )
            for motor, value in values.items()
        }

    return ThermalStatus.from_mappings(
        temperatures=finite_or_none(snapshot.temperatures),
        reported_peaks=finite_or_none(snapshot.reported_peaks),
        confirmed_peaks=finite_or_none(snapshot.confirmed_peaks),
        spike_counts=snapshot.spike_counts,
        invalid_sample_counts=snapshot.invalid_sample_counts,
        last_invalid_values=finite_or_none(snapshot.last_invalid_values),
        warning_motors=tuple(snapshot.warning_motors),
        stop_reason=stop_reason,
    )


def _thermal_failure_status(
    guard: object,
    error: Exception,
    previous: ThermalStatus | None,
) -> ThermalStatus | None:
    """Capture latest fail-closed evidence without replacing the triggering error."""

    reason = str(error).strip() or f"{type(error).__name__} during thermal safety check"
    try:
        snapshot_method = getattr(guard, "snapshot", None)
        if callable(snapshot_method):
            return _thermal_status(snapshot_method(), stop_reason=reason)
    except Exception:
        # The original thermal failure is authoritative. Snapshot conversion is
        # best effort and must never mask it during teardown.
        pass
    return replace(previous, stop_reason=reason) if previous is not None else None


class StadiaSessionWorker:
    """Sole owner of one reader, follower, runtime loop, and teardown."""

    def __init__(
        self,
        *,
        manager: ControlSessionManager,
        claim: ControlSessionClaim,
        config: StadiaSessionConfig,
        reader: StadiaReader | None = None,
        reader_factory: Callable[[str | None], StadiaReader] = _default_reader,
        calibration_resolver: Callable[[str], str] = _default_calibration_resolver,
        follower_factory: Callable[[FollowerBuildSpec], object] = _default_follower_factory,
        thermal_guard_factory: Callable[[object, Callable[[float], None]], object] = (_default_thermal_guard),
        joint_broadcaster: Callable[[Mapping[str, object]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.manager = manager
        self.claim = claim
        self.config = config
        self.reader = reader if reader is not None else reader_factory(config.expected_guid)
        self._calibration_resolver = calibration_resolver
        self._follower_factory = follower_factory
        self._thermal_guard_factory = thermal_guard_factory
        self._joint_broadcaster = joint_broadcaster
        self._clock = clock
        self._sleeper = sleeper

        self._gate = NeutralReleaseGate(
            expected_guid=config.expected_guid,
            deadzone=config.deadzone,
            stable_samples_required=3,
        )
        self._gate_state = NeutralGateState()
        self._gate_decision: NeutralGateDecision | None = None
        self._last_snapshot: StadiaSnapshot | None = None
        self._joint_status_specs: tuple[JointStatusSpec, ...] = ()
        self._thermal: ThermalStatus | None = None
        self._integrator: BoundedStadiaIntegrator | None = None
        self._robot: object | None = None
        self._commands_sent = 0
        self._movement_steps = 0
        self._missed_ticks = 0
        self._configure_started = False
        self._session_guid = config.expected_guid
        self._session_instance_id: int | None = None
        self._session_instance_generation: int | None = None
        self._highest_sequence = -1
        self._latest_sampled_at = -float("inf")
        self._controller_error: str | None = None
        self._controller_monitoring_active = False
        self._resource_release_unproven = False
        self._speed_lock = threading.Lock()
        self._speed_multiplier = 1.0
        self._movement_enabled = False
        self._lifecycle_lock = threading.Lock()
        self._run_started = False
        self._thread: threading.Thread | None = None
        self._thread_result: StadiaSessionResult | None = None
        self._thread_error: BaseException | None = None

    def run(self) -> StadiaSessionResult:
        """Run synchronously until stop or failure, then publish terminal truth."""

        with self._lifecycle_lock:
            if self._run_started:
                raise RuntimeError("Stadia session workers may only be run once")
            self._run_started = True
        return self._run_owned()

    def start(self) -> None:
        """Start this owner on its independent, non-daemon worker thread."""

        with self._lifecycle_lock:
            if self._run_started:
                raise RuntimeError("Stadia session workers may only be started once")
            self._run_started = True
            self._thread = threading.Thread(
                target=self._thread_main,
                name=f"lelab-stadia-session-{self.claim.session_id}",
                daemon=False,
            )
            self._thread.start()

    @property
    def is_alive(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def join(self, *, timeout: float | None = None) -> StadiaSessionResult:
        """Wait boundedly for worker-owned cleanup and return its evidence."""

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
            raise RuntimeError("Stadia session worker was not started asynchronously")
        if thread is threading.current_thread():
            raise RuntimeError("Stadia session worker cannot join itself")
        thread.join(None if timeout is None else float(timeout))
        if thread.is_alive():
            raise TimeoutError("Stadia session worker did not finish before the join timeout")
        with self._lifecycle_lock:
            error = self._thread_error
            result = self._thread_result
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("Stadia session worker exited without terminal evidence")
        return result

    def request_stop(self, *, reason: str = "stop requested") -> None:
        """Signal this exact manager-owned session to hold and tear down."""

        self.manager.request_stop(self.claim.session_id, reason=reason)

    def set_speed_multiplier(self, multiplier: float) -> ControlStatus:
        """Apply a bounded live Stadia speed while the dead-man is released."""

        if isinstance(multiplier, bool) or not isinstance(multiplier, Real):
            raise ValueError("speed multiplier must be a finite number")
        numeric = float(multiplier)
        if not math.isfinite(numeric) or not MIN_SPEED_MULTIPLIER <= numeric <= MAX_SPEED_MULTIPLIER:
            raise ValueError(
                f"speed multiplier must be between {MIN_SPEED_MULTIPLIER:g}x and {MAX_SPEED_MULTIPLIER:g}x"
            )
        status = self.manager.status_for(self.claim.session_id, check_expiry=False)
        if status is None or status.state is not ControlState.RUNNING:
            raise StadiaSessionRuntimeError("Stadia speed can change only while teleoperation is running")
        with self._speed_lock:
            if self._movement_enabled:
                raise StadiaSessionRuntimeError("release RB before changing Stadia speed")
            if self._integrator is None or not self._joint_status_specs:
                raise StadiaSessionRuntimeError("Stadia speed controls are not ready")
            effective_step = self.config.max_step_per_tick * numeric
            self._integrator.set_max_step_per_tick(effective_step)
            self._joint_status_specs = tuple(
                replace(spec, max_step_per_tick=effective_step) for spec in self._joint_status_specs
            )
            self._speed_multiplier = numeric
            return self.manager.update_joint_specs_and_details(
                self.claim.session_id,
                joint_specs=self._joint_status_specs,
                details_patch=self._speed_details(),
            )

    def _thread_main(self) -> None:
        try:
            result = self._run_owned()
        except BaseException as error:
            with self._lifecycle_lock:
                self._thread_error = error
            return
        with self._lifecycle_lock:
            self._thread_result = result

    def _run_owned(self) -> StadiaSessionResult:
        """Execute after either synchronous or asynchronous ownership is claimed."""

        follower: object | None = None
        bus: object | None = None
        reader_start_attempted = False
        bus_connect_attempted = False
        bus_connect_succeeded = False
        connected_cameras: list[tuple[str, object]] = []
        lifecycle_errors: list[str] = []
        failure: Exception | None = None
        stop_reason = "stop requested"

        try:
            reader_start_attempted = True
            self.reader.start()
            self._controller_monitoring_active = True
            startup_snapshot = self._wait_for_neutral_startup()
            self._raise_if_stop_requested()

            calibration_id = self._calibration_resolver(self.config.follower_calibration)
            if (
                not calibration_id
                or Path(calibration_id).name != calibration_id
                or calibration_id in {".", ".."}
                or "/" in calibration_id
                or "\\" in calibration_id
            ):
                raise StadiaSessionStartupError("follower calibration resolver returned an invalid ID")
            build_spec = FollowerBuildSpec(
                port=self.config.follower_port,
                calibration_id=calibration_id,
                max_relative_target=dict.fromkeys(SO101_MOTOR_NAMES, MAX_RELATIVE_TARGET),
                cameras=self.config.cameras,
            )
            follower = self._follower_factory(build_spec)
            self._robot = follower
            bus = getattr(follower, "bus", None)
            if bus is None:
                raise StadiaSessionStartupError("constructed follower has no motor bus")

            # Recording subclasses prepare and validate non-device resources
            # (notably the dataset transaction adapter) here.  This runs only
            # after the follower object exists, but before its bus or cameras
            # are touched.
            self._prepare_dependencies(follower)
            self._raise_if_stop_requested()
            bus_connect_attempted = True
            bus.connect()
            bus_connect_succeeded = True
            bus.disable_torque(num_retry=FOLLOWER_READ_RETRIES)
            self._raise_if_stop_requested()
            self._validate_loaded_calibration(follower)
            self._raise_if_stop_requested()

            thermal_guard = self._thermal_guard_factory(bus, self._sleeper)
            try:
                self._thermal = _thermal_status(thermal_guard.check())
            except Exception as error:
                self._thermal = _thermal_failure_status(thermal_guard, error, self._thermal)
                self._publish_status(startup_snapshot, MotionState.DISARMED)
                raise

            self._raise_if_stop_requested()
            for name, camera in getattr(follower, "cameras", {}).items():
                connected_cameras.append((str(name), camera))
                camera.connect()
                self._raise_if_stop_requested()

            # Controller state may change while serial, calibration, thermal,
            # or camera work blocks. Require a distinct independent publication
            # after all of that work and before writing any goal or arming.
            prearm_snapshot = self._wait_for_prearm_confirmation(startup_snapshot)
            self._raise_if_stop_requested()

            # Read the initial target only after every potentially blocking setup step.
            # Raw validation must precede LeRobot's normalization so an invalid
            # RANGE_0_100 value cannot be hidden by its endpoint clamp.
            raw_pose = bus.sync_read(
                "Present_Position",
                list(SO101_MOTOR_NAMES),
                normalize=False,
                num_retry=FOLLOWER_READ_RETRIES,
            )
            position_scales = _position_scales(follower)
            endpoint_bounds = _endpoint_bounds_from_scales(position_scales)
            validated_raw_pose, pose = _normalize_raw_pose(raw_pose, position_scales)
            self._integrator = BoundedStadiaIntegrator(
                initial_action=pose,
                endpoint_bounds=endpoint_bounds,
                specs=self._integrator_specs(),
            )
            self._joint_status_specs = self._status_specs(endpoint_bounds)
            self._raise_if_stop_requested()

            for motor in SO101_MOTOR_NAMES:
                # Pinned LeRobot v0.6.0 documents single-motor write() as
                # acknowledged (writeTxRx), unlike unacknowledged sync_write().
                bus.write(
                    "Goal_Position",
                    motor,
                    validated_raw_pose[motor],
                    normalize=False,
                    num_retry=FOLLOWER_READ_RETRIES,
                )
            self._require_prearm_snapshot_still_safe(prearm_snapshot)

            # SOFollower.configure() uses a torque-disabled context and its exit
            # is the actual arming operation in pinned LeRobot v0.6.0.
            self._configure_started = True
            follower.configure()
            self._raise_if_stop_requested()
            self.manager.mark_running(self.claim.session_id)
            self._broadcast_joint_action(pose)
            loop_reason = self._run_control_loop(thermal_guard)
            if loop_reason is not None:
                if not isinstance(loop_reason, str) or not loop_reason.strip():
                    raise StadiaSessionRuntimeError(
                        "control loop terminal reason must be a non-empty string or None"
                    )
                stop_reason = loop_reason.strip()
        except _StopRequestedError as error:
            stop_reason = str(error)
        except Exception as error:
            failure = error
            stop_reason = f"{type(error).__name__}: {error}"
            try:
                self._on_session_failure(error)
            except Exception as hook_error:
                lifecycle_errors.append(
                    f"session failure hook failed: {type(hook_error).__name__}: {hook_error}"
                )
        finally:
            try:
                self._enter_stopping(stop_reason)
            except Exception as error:
                if failure is None:
                    failure = error
                stop_reason = (
                    f"{stop_reason}; manager stopping transition failed: {type(error).__name__}: {error}"
                )
            torque, teardown_errors = self._teardown(
                bus=bus,
                bus_connect_attempted=bus_connect_attempted,
                bus_connect_succeeded=bus_connect_succeeded,
                connected_cameras=connected_cameras,
                reader_start_attempted=reader_start_attempted,
            )
            teardown_errors.extend(lifecycle_errors)

        if not self._controller_monitoring_active:
            try:
                self.manager.mark_controller_monitoring_inactive(self.claim.session_id)
            except Exception as error:
                teardown_errors.append(
                    f"controller monitoring status failed: {type(error).__name__}: {error}"
                )
        if self._resource_release_unproven:
            self.manager.quarantine(
                reason="Stadia session resource release could not be proven; restart required"
            )
        terminal_state = (
            ControlState.ERROR if failure is not None or teardown_errors else ControlState.STOPPED
        )
        if teardown_errors:
            cleanup_reason = "teardown failed: " + "; ".join(teardown_errors)
            stop_reason = f"{stop_reason}; {cleanup_reason}" if stop_reason else cleanup_reason
        terminal = self.manager.finish_teardown(
            self.claim.session_id,
            terminal_state=terminal_state,
            torque=torque,
            reason=stop_reason,
        )
        terminal_state = terminal.state
        stop_reason = terminal.stop_reason or stop_reason

        saturation_count, clipping_count = self._counter_values()
        return StadiaSessionResult(
            terminal_state=terminal_state,
            reason=stop_reason,
            torque=torque,
            commands_sent=self._commands_sent,
            movement_steps=self._movement_steps,
            missed_ticks=self._missed_ticks,
            saturation_count=saturation_count,
            relative_clipping_count=clipping_count,
            teardown_errors=tuple(teardown_errors),
        )

    def _wait_for_neutral_startup(self) -> StadiaSnapshot:
        deadline = self._now() + self.config.startup_timeout_s
        last_sequence = -1
        last_reason = "waiting for a fresh Stadia sample"
        while True:
            self._raise_if_stop_requested()
            remaining = deadline - self._now()
            if remaining <= 0:
                raise StadiaSessionStartupError(
                    f"controller did not become safely neutral before startup timeout: {last_reason}"
                )
            try:
                snapshot = self.reader.wait_for_snapshot(
                    after_sequence=last_sequence,
                    timeout=min(remaining, 0.1),
                )
            except TimeoutError:
                continue
            last_sequence = max(last_sequence, snapshot.sequence)
            decision, healthy, _age, reason = self._evaluate_snapshot(snapshot)
            last_reason = reason or "controller startup gate is not armed"
            self._publish_status(snapshot, MotionState.DISARMED)
            if healthy and decision.controls_neutral and decision.state.neutral_armed:
                return snapshot

    def _wait_for_prearm_confirmation(self, startup_snapshot: StadiaSnapshot) -> StadiaSnapshot:
        """Require a new safe publication after all blocking follower setup."""

        deadline = self._now() + self.config.startup_timeout_s
        last_sequence = max(self._highest_sequence, startup_snapshot.sequence)
        expected_generation = startup_snapshot.connection_generation
        expected_instance_id = startup_snapshot.instance_id
        last_reason = "waiting for a newly published neutral controller sample"
        while True:
            self._raise_if_stop_requested()
            remaining = deadline - self._now()
            if remaining <= 0:
                raise StadiaSessionStartupError(
                    "controller did not provide a safe post-setup sample before startup timeout: "
                    f"{last_reason}"
                )
            try:
                snapshot = self.reader.wait_for_snapshot(
                    after_sequence=last_sequence,
                    timeout=min(remaining, 0.1),
                )
            except TimeoutError:
                continue

            decision, healthy, _age, reason = self._evaluate_snapshot(snapshot)
            if (
                isinstance(snapshot.sequence, bool)
                or not isinstance(snapshot.sequence, int)
                or snapshot.sequence <= last_sequence
            ):
                self._publish_forced_controller_error(
                    snapshot,
                    "controller reader returned a non-advancing post-setup sequence",
                )
                raise StadiaSessionStartupError(
                    "controller reader returned a non-advancing post-setup sequence"
                )
            last_sequence = snapshot.sequence
            if snapshot.connection_generation != expected_generation:
                self._publish_forced_controller_error(
                    snapshot,
                    "controller connection generation changed during follower setup",
                )
                raise StadiaSessionStartupError(
                    "controller connection generation changed during follower setup"
                )
            if snapshot.instance_id != expected_instance_id:
                self._publish_forced_controller_error(
                    snapshot,
                    "controller instance identity changed without a new connection generation",
                )
                raise StadiaSessionStartupError(
                    "controller instance identity changed without a new connection generation"
                )
            if self._controller_error is not None or not decision.profile_valid or not healthy:
                self._publish_status(snapshot, MotionState.DISARMED)
                raise StadiaSessionStartupError(
                    f"controller became unsafe during follower setup: "
                    f"{self._controller_error or reason or 'unknown controller fault'}"
                )
            self._publish_status(snapshot, MotionState.DISARMED)
            if (
                decision.controls_neutral
                and decision.state.neutral_armed
                and not snapshot.button(10)
                and snapshot.sampled_at > startup_snapshot.sampled_at
            ):
                return snapshot
            last_reason = reason or "release RB and neutralize all controls"

    def _require_prearm_snapshot_still_safe(
        self,
        expected_snapshot: StadiaSnapshot,
    ) -> None:
        """Recheck current reader truth after acknowledged Goal writes."""

        self._raise_if_stop_requested()
        snapshot = self.reader.snapshot()
        decision, healthy, _age, reason = self._evaluate_snapshot(snapshot)
        if snapshot.connection_generation != expected_snapshot.connection_generation:
            self._publish_forced_controller_error(
                snapshot,
                "controller connection generation changed before follower arming",
            )
            raise StadiaSessionStartupError("controller connection generation changed before follower arming")
        if snapshot.instance_id != expected_snapshot.instance_id:
            self._publish_forced_controller_error(
                snapshot,
                "controller instance identity changed before follower arming",
            )
            raise StadiaSessionStartupError("controller instance identity changed before follower arming")
        if snapshot.sequence < expected_snapshot.sequence:
            self._publish_forced_controller_error(
                snapshot,
                "controller sample sequence regressed before follower arming",
            )
            raise StadiaSessionStartupError("controller sample sequence regressed before follower arming")
        if self._controller_error is not None or not decision.profile_valid or not healthy:
            self._publish_status(snapshot, MotionState.DISARMED)
            raise StadiaSessionStartupError(
                f"controller became unsafe before follower arming: "
                f"{self._controller_error or reason or 'unknown controller fault'}"
            )
        if not (decision.controls_neutral and decision.state.neutral_armed and not snapshot.button(10)):
            self._publish_status(snapshot, MotionState.DISARMED)
            raise StadiaSessionStartupError(
                "controller must remain neutral with RB released before follower arming"
            )
        self._publish_status(snapshot, MotionState.DISARMED)

    def _prepare_dependencies(self, follower: object) -> None:
        """Prepare non-device resources before any follower device access."""

    def _on_session_failure(self, error: Exception) -> None:
        """Allow specializations to publish a failure before terminal teardown."""

    def _run_control_loop(self, thermal_guard: object) -> str | None:
        if self._integrator is None:
            raise StadiaSessionRuntimeError("integrator was not initialized")
        scheduler = NoCatchUpScheduler(rate_hz=CONTROL_RATE_HZ, start_time=self._now())
        next_thermal_at = self._now() + THERMAL_INTERVAL_S

        while True:
            self._raise_if_stop_requested()
            snapshot = self.reader.snapshot()
            # Sample the session clock after acquiring the immutable reader
            # publication so a concurrently published sample cannot appear to
            # come from the future.
            now = self._now()
            decision, healthy, _age, _reason = self._evaluate_snapshot(snapshot, now=now)

            # Torque remains armed while the accepted goal is held, so thermal
            # monitoring must continue even while controller input is stale or
            # disconnected and the scheduler is suppressing commands.
            if now >= next_thermal_at:
                try:
                    self._thermal = _thermal_status(thermal_guard.check())
                except Exception as error:
                    self._thermal = _thermal_failure_status(thermal_guard, error, self._thermal)
                    self._publish_status(snapshot, MotionState.HOLD)
                    raise
                next_thermal_at = self._now() + THERMAL_INTERVAL_S
                # A confirmation sequence may consume a meaningful fraction of
                # the freshness window. Recheck the same immutable publication
                # and let the scheduler drop elapsed deadlines before sending.
                now = self._now()
                decision, healthy, _age, _reason = self._evaluate_snapshot(snapshot, now=now)

            movement_enabled = healthy and decision.motion_enabled and not self.claim.hold_requested.is_set()
            with self._speed_lock:
                self._movement_enabled = movement_enabled
                effective_step = self.config.max_step_per_tick * self._speed_multiplier
            motion_state = MotionState.ENABLED if movement_enabled else MotionState.HOLD
            tick = scheduler.poll(now, ready=healthy)
            self._missed_ticks += tick.missed_ticks
            if tick.should_step:
                try:
                    mapped = map_stadia_input(
                        snapshot,
                        motion_enabled=movement_enabled,
                        deadzone=self.config.deadzone,
                        max_step_per_tick=effective_step,
                        expected_guid=self.config.expected_guid,
                    )
                    with self._speed_lock:
                        integrated = self._integrator.integrate_one_step(
                            mapped.deltas_dict(),
                            enabled=movement_enabled,
                        )
                    requested = integrated.action_dict()
                    if self._robot is None:
                        raise StadiaSessionRuntimeError("follower was not initialized")
                    returned_raw = self._robot.send_action(requested)
                    returned = validate_returned_action(returned_raw)
                    comparison = compare_requested_returned(requested, returned)
                    with self._speed_lock:
                        self._integrator.accept_returned_action(
                            comparison.adopted_action,
                            requested_action=comparison.requested,
                            tolerance=comparison.tolerance,
                        )
                except ActionValidationError:
                    raise
                except Exception as error:
                    raise StadiaSessionRuntimeError(
                        f"follower command failed: {type(error).__name__}: {error}"
                    ) from error
                self._commands_sent += 1
                self._movement_steps += int(movement_enabled)
                self._broadcast_joint_action(comparison.adopted_action.as_dict())

            self._publish_status(snapshot, motion_state)
            sleep_for = max(0.0, scheduler.next_deadline - self._now())
            if sleep_for > 0:
                self._sleeper(sleep_for)

    def _evaluate_snapshot(
        self,
        snapshot: StadiaSnapshot,
        *,
        now: float | None = None,
    ) -> tuple[NeutralGateDecision, bool, float | None, str | None]:
        self._last_snapshot = snapshot
        current = self._now() if now is None else now
        previous_generation = self._gate_state.connection_generation
        reason: str | None = None
        controller_error: str | None = None
        age: float | None = None
        if (
            isinstance(snapshot.sequence, bool)
            or not isinstance(snapshot.sequence, int)
            or snapshot.sequence < 0
        ):
            reason = "controller sample sequence is invalid"
        elif (
            isinstance(snapshot.connection_generation, bool)
            or not isinstance(snapshot.connection_generation, int)
            or snapshot.connection_generation < 1
        ):
            reason = "controller connection generation is invalid"
        elif (
            isinstance(snapshot.sampled_at, bool)
            or not isinstance(snapshot.sampled_at, Real)
            or not math.isfinite(float(snapshot.sampled_at))
        ):
            reason = "controller sample timestamp is invalid"
        elif snapshot.connected and (
            isinstance(snapshot.instance_id, bool)
            or not isinstance(snapshot.instance_id, int)
            or snapshot.instance_id < 0
        ):
            reason = "controller instance ID is invalid"
        elif snapshot.sequence < self._highest_sequence:
            reason = "controller sample sequence regressed"
        elif (
            snapshot.sequence > self._highest_sequence
            and float(snapshot.sampled_at) <= self._latest_sampled_at
        ):
            reason = "controller sample timestamp did not advance"
        elif snapshot.connected and self._session_guid is not None and snapshot.guid != self._session_guid:
            reason = f"controller GUID {snapshot.guid!r} does not match session GUID {self._session_guid!r}"
        elif (
            snapshot.connected
            and self._session_instance_generation == snapshot.connection_generation
            and self._session_instance_id is not None
            and snapshot.instance_id != self._session_instance_id
        ):
            reason = (
                f"controller instance ID {snapshot.instance_id!r} does not match current identity "
                f"{self._session_instance_id!r} without a new connection generation"
            )
        else:
            calculated_age = current - float(snapshot.sampled_at)
            if calculated_age < 0:
                reason = "controller sample timestamp is in the future"
            else:
                age = calculated_age
                if age > MAX_SNAPSHOT_AGE_S:
                    reason = f"controller sample is stale ({age:.3f}s old)"

        if reason is not None:
            controller_error = reason
            generation = (
                snapshot.connection_generation
                if isinstance(snapshot.connection_generation, int)
                and not isinstance(snapshot.connection_generation, bool)
                and snapshot.connection_generation >= 0
                else None
            )
            self._gate_state = NeutralGateState(connection_generation=generation)
            decision = NeutralGateDecision(self._gate_state, False, False, False, reason)
            scheduler_ready = False
        else:
            decision = self._gate.evaluate(snapshot, self._gate_state)
            self._gate_state = decision.state
            reason = decision.reason
            if decision.profile_valid and self._session_guid is None:
                self._session_guid = snapshot.guid
            generation_changed = previous_generation != snapshot.connection_generation
            if decision.profile_valid and self._session_instance_generation != snapshot.connection_generation:
                self._session_instance_generation = snapshot.connection_generation
                self._session_instance_id = snapshot.instance_id
            scheduler_ready = decision.profile_valid and not generation_changed
            if not decision.profile_valid or generation_changed:
                controller_error = reason or "controller profile is not ready"
        if (reason is None or decision.profile_valid) and snapshot.sequence > self._highest_sequence:
            self._highest_sequence = snapshot.sequence
            self._latest_sampled_at = float(snapshot.sampled_at)
        self._controller_error = controller_error
        self._gate_decision = decision
        return decision, scheduler_ready, age, reason

    def _validate_loaded_calibration(self, follower: Any) -> None:
        bus = follower.bus
        file_calibration = getattr(follower, "calibration", None)
        bus_calibration = getattr(bus, "calibration", None)
        if not isinstance(file_calibration, Mapping) or set(file_calibration) != set(SO101_MOTOR_NAMES):
            raise StadiaSessionStartupError(
                "configured follower calibration did not load all six SO-101 motors"
            )
        if not isinstance(bus_calibration, Mapping) or dict(bus_calibration) != dict(file_calibration):
            raise StadiaSessionStartupError("follower bus is not using the configured calibration")
        calibrated = getattr(follower, "is_calibrated", False)
        if callable(calibrated):
            calibrated = calibrated()
        if calibrated is not True:
            raise StadiaSessionStartupError(
                "follower calibration does not match the device; use the separate calibration/apply flow"
            )

    def _integrator_specs(self) -> tuple[object, ...]:
        with self._speed_lock:
            effective_step = self.config.max_step_per_tick * self._speed_multiplier
        specs = []
        for spec in DEFAULT_JOINT_SPECS:
            specs.append(
                type(spec)(
                    spec.action_key,
                    spec.unit,
                    effective_step,
                )
            )
        return tuple(specs)

    def _status_specs(
        self,
        endpoint_bounds: Mapping[str, tuple[float, float]],
    ) -> tuple[JointStatusSpec, ...]:
        with self._speed_lock:
            effective_step = self.config.max_step_per_tick * self._speed_multiplier
        result = []
        for key in ACTION_KEYS:
            calibrated_lower, calibrated_upper = endpoint_bounds[key]
            result.append(
                JointStatusSpec(
                    action_key=key,
                    unit=(
                        JointStatusUnit.GRIPPER_PERCENTAGE_POINTS
                        if key == "gripper.pos"
                        else JointStatusUnit.DEGREES
                    ),
                    max_step_per_tick=effective_step,
                    max_relative_target=MAX_RELATIVE_TARGET,
                    calibrated_min=calibrated_lower,
                    calibrated_max=calibrated_upper,
                )
            )
        return tuple(result)

    def _speed_details(self) -> dict[str, float]:
        return {
            "stadia_speed_multiplier": self._speed_multiplier,
            "stadia_effective_max_step_per_tick": (self.config.max_step_per_tick * self._speed_multiplier),
        }

    def _broadcast_joint_action(self, action: Mapping[str, float]) -> None:
        if self._joint_broadcaster is None:
            return
        joints = {
            urdf_name: float(action[action_key]) * math.pi / 180.0
            for action_key, urdf_name in _ACTION_TO_URDF_JOINT.items()
        }
        try:
            self._joint_broadcaster(
                {
                    "type": "joint_update",
                    "joints": joints,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            # Visualization is advisory and must never perturb the device owner.
            return

    def _publish_status(self, snapshot: StadiaSnapshot, motion_state: MotionState) -> None:
        decision = self._gate_decision
        age = None
        # Connection reports physical reader truth. Health faults are carried
        # independently in controller_error and must not rewrite a connected
        # controller as disconnected.
        controller_connected = bool(snapshot.connected)
        controller_error = self._controller_error
        if (
            not isinstance(snapshot.sampled_at, bool)
            and isinstance(snapshot.sampled_at, Real)
            and math.isfinite(float(snapshot.sampled_at))
        ):
            calculated = self._now() - float(snapshot.sampled_at)
            if calculated >= 0 and math.isfinite(calculated):
                age = calculated
                if age > MAX_SNAPSHOT_AGE_S:
                    controller_error = f"controller sample is stale ({age:.3f}s old)"
        if not controller_connected and controller_error is None:
            controller_error = snapshot.read_error or "controller is disconnected or unsafe"
        saturation, clipping = self._counter_values()
        self.manager.update_runtime_status(
            self.claim.session_id,
            controller_connected=controller_connected,
            controller_error=controller_error,
            controller_product_name=snapshot.product_name,
            controller_guid=snapshot.guid,
            controller_instance_id=(
                snapshot.instance_id
                if isinstance(snapshot.instance_id, int)
                and not isinstance(snapshot.instance_id, bool)
                and snapshot.instance_id >= 0
                else None
            ),
            controller_generation=(
                snapshot.connection_generation
                if isinstance(snapshot.connection_generation, int)
                and not isinstance(snapshot.connection_generation, bool)
                and snapshot.connection_generation >= 0
                else None
            ),
            controller_layout=(snapshot.layout.axes, snapshot.layout.buttons, snapshot.layout.hats),
            sample_sequence=(
                snapshot.sequence
                if isinstance(snapshot.sequence, int)
                and not isinstance(snapshot.sequence, bool)
                and snapshot.sequence >= 0
                else None
            ),
            sample_age_s=age,
            rb_held=snapshot.button(10) if len(snapshot.buttons) > 10 else False,
            release_observed=decision.state.release_seen if decision is not None else False,
            controls_neutral=decision.controls_neutral if decision is not None else False,
            motion_state=motion_state,
            joint_specs=self._joint_status_specs,
            saturation_count=saturation,
            relative_clipping_count=clipping,
            thermal_snapshot=self._thermal,
            details_patch=self._speed_details(),
        )

    def _publish_forced_controller_error(
        self,
        snapshot: StadiaSnapshot,
        error: str,
    ) -> None:
        self._controller_error = error
        self._publish_status(snapshot, MotionState.DISARMED)

    def _counter_values(self) -> tuple[int, int]:
        if self._integrator is None:
            return 0, 0
        counters = self._integrator.counters
        return (
            counters.step_saturations + counters.travel_saturations + counters.endpoint_saturations,
            counters.returned_clippings,
        )

    def _raise_if_stop_requested(self) -> None:
        self.manager.check_lease_expiry()
        if self.claim.stop_requested.is_set() or self.claim.hold_requested.is_set():
            status = self.manager.status_for(self.claim.session_id, check_expiry=False)
            reason = status.stop_reason if status is not None and status.stop_reason else "stop requested"
            raise _StopRequestedError(reason)

    def _enter_stopping(self, reason: str) -> None:
        status = self.manager.status_for(self.claim.session_id, check_expiry=False)
        if status is None or status.state is not ControlState.STOPPING:
            self.manager.request_stop(self.claim.session_id, reason=reason)
        if self._last_snapshot is not None:
            self._publish_status(
                self._last_snapshot,
                MotionState.HOLD if self._configure_started else MotionState.DISARMED,
            )

    def _teardown(
        self,
        *,
        bus: object | None,
        bus_connect_attempted: bool,
        bus_connect_succeeded: bool,
        connected_cameras: list[tuple[str, object]],
        reader_start_attempted: bool,
    ) -> tuple[TorqueEvidence, list[str]]:
        errors: list[str] = []
        disable_errors: dict[str, str] = {}
        readback: Mapping[str, object] | None = None
        verification_supported = True
        disable_attempted = False
        bus_connected = bus_connect_succeeded
        if bus is not None and bus_connect_attempted and not bus_connected:
            try:
                bus_connected = bool(bus.is_connected)
            except Exception as error:
                errors.append(f"follower bus connectivity check failed: {type(error).__name__}: {error}")
                # A failed connect may still have opened the serial endpoint.
                # Attempt bounded safety cleanup rather than assuming it did not.
                bus_connected = True

        if bus_connected and bus is not None:
            for attempt in range(1, self.config.torque_disable_attempts + 1):
                disable_attempted = True
                try:
                    bus.disable_torque(num_retry=FOLLOWER_READ_RETRIES)
                    break
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}"
                    disable_errors[f"attempt_{attempt}"] = detail
                    errors.append(f"torque disable attempt {attempt}: {detail}")
            try:
                raw_readback = bus.sync_read(
                    "Torque_Enable",
                    list(SO101_MOTOR_NAMES),
                    normalize=False,
                    num_retry=FOLLOWER_READ_RETRIES,
                )
                if not isinstance(raw_readback, Mapping):
                    errors.append("torque readback failed: result was not a mapping")
                else:
                    readback = raw_readback
            except (AttributeError, NotImplementedError) as error:
                verification_supported = False
                errors.append(f"torque readback unsupported: {type(error).__name__}: {error}")
            except Exception as error:
                errors.append(f"torque readback failed: {type(error).__name__}: {error}")

        torque = classify_torque_outcome(
            disable_attempted=disable_attempted,
            readback=readback,
            disable_errors=disable_errors,
            verification_supported=verification_supported,
        )

        for name, camera in connected_cameras:
            camera_thread: object | None = None
            camera_release_proven = True
            try:
                camera_thread = getattr(camera, "thread", None)
            except Exception as error:
                camera_release_proven = False
                errors.append(f"camera {name} read-thread capture failed: {type(error).__name__}: {error}")
            try:
                camera.disconnect()
            except Exception as error:
                errors.append(f"camera {name} disconnect failed: {type(error).__name__}: {error}")
            if camera_thread is not None:
                is_alive = getattr(camera_thread, "is_alive", None)
                join = getattr(camera_thread, "join", None)
                if not callable(is_alive) or not callable(join):
                    camera_release_proven = False
                    errors.append(f"camera {name} read-thread join contract is unavailable")
                else:
                    try:
                        if is_alive():
                            join(timeout=self.config.reader_join_timeout_s)
                        if is_alive():
                            camera_release_proven = False
                            errors.append(f"camera {name} read thread is still alive")
                    except Exception as error:
                        camera_release_proven = False
                        errors.append(
                            f"camera {name} read-thread proof failed: {type(error).__name__}: {error}"
                        )
            try:
                camera_connected = camera.is_connected
            except Exception as error:
                camera_release_proven = False
                errors.append(
                    f"camera {name} disconnect postcondition failed: {type(error).__name__}: {error}"
                )
            else:
                if camera_connected is not False:
                    camera_release_proven = False
                    errors.append(
                        f"camera {name} disconnect postcondition is not false: {camera_connected!r}"
                    )
            if not camera_release_proven:
                self._resource_release_unproven = True

        if bus_connected and bus is not None:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as error:
                errors.append(f"follower bus disconnect failed: {type(error).__name__}: {error}")
            try:
                bus_still_connected = bus.is_connected
            except Exception as error:
                self._resource_release_unproven = True
                errors.append(
                    f"follower bus disconnect postcondition failed: {type(error).__name__}: {error}"
                )
            else:
                if bus_still_connected is not False:
                    self._resource_release_unproven = True
                    errors.append(
                        f"follower bus disconnect postcondition is not false: {bus_still_connected!r}"
                    )

        if reader_start_attempted:
            try:
                self.reader.stop(timeout=self.config.reader_join_timeout_s)
                self._controller_monitoring_active = False
            except Exception as error:
                self._resource_release_unproven = True
                errors.append(f"Stadia reader stop failed: {type(error).__name__}: {error}")

        return torque, errors

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise StadiaSessionError("monotonic clock returned a non-finite value")
        return now


def run_stadia_session(
    *,
    manager: ControlSessionManager,
    claim: ControlSessionClaim,
    config: StadiaSessionConfig,
    **worker_dependencies: object,
) -> StadiaSessionResult:
    """Convenience entry point for a server-owned worker thread."""

    return StadiaSessionWorker(
        manager=manager,
        claim=claim,
        config=config,
        **worker_dependencies,  # type: ignore[arg-type]
    ).run()


__all__ = [
    "CONTROL_RATE_HZ",
    "FOLLOWER_READ_RETRIES",
    "MAX_RELATIVE_TARGET",
    "MAX_SNAPSHOT_AGE_S",
    "FollowerBuildSpec",
    "StadiaSessionConfig",
    "StadiaSessionError",
    "StadiaSessionResult",
    "StadiaSessionRuntimeError",
    "StadiaSessionStartupError",
    "StadiaSessionWorker",
    "derive_calibrated_endpoint_bounds",
    "run_stadia_session",
]
