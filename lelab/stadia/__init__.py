"""Dependency-neutral control primitives for LeLab's Stadia teleoperator."""

from .action import (
    ActionComparison,
    ActionValidationError,
    NormalizedAction,
    compare_requested_returned,
    validate_returned_action,
)
from .device_reader import StadiaDeviceReader, StadiaReaderError
from .integrator import BoundedStadiaIntegrator
from .mapping import NeutralReleaseGate, map_stadia_input, validate_stadia_profile
from .session import (
    FollowerBuildSpec,
    StadiaSessionConfig,
    StadiaSessionError,
    StadiaSessionResult,
    StadiaSessionRuntimeError,
    StadiaSessionStartupError,
    StadiaSessionWorker,
    derive_calibrated_endpoint_bounds,
    run_stadia_session,
)
from .thermal_safety import ConfirmedTemperatureGuard, ConfirmedTemperatureStopError
from .timing import NoCatchUpScheduler
from .types import (
    ACTION_KEYS,
    DEFAULT_JOINT_SPECS,
    STADIA_LAYOUT,
    STADIA_PRODUCT_NAME,
    ControllerLayout,
    JointControlSpec,
    PositionUnit,
    StadiaSnapshot,
)

__all__ = [
    "ACTION_KEYS",
    "DEFAULT_JOINT_SPECS",
    "STADIA_LAYOUT",
    "STADIA_PRODUCT_NAME",
    "ActionComparison",
    "ActionValidationError",
    "BoundedStadiaIntegrator",
    "ConfirmedTemperatureGuard",
    "ConfirmedTemperatureStopError",
    "ControllerLayout",
    "FollowerBuildSpec",
    "JointControlSpec",
    "NeutralReleaseGate",
    "NoCatchUpScheduler",
    "NormalizedAction",
    "PositionUnit",
    "StadiaDeviceReader",
    "StadiaReaderError",
    "StadiaSessionConfig",
    "StadiaSessionError",
    "StadiaSessionResult",
    "StadiaSessionRuntimeError",
    "StadiaSessionStartupError",
    "StadiaSessionWorker",
    "StadiaSnapshot",
    "compare_requested_returned",
    "derive_calibrated_endpoint_bounds",
    "map_stadia_input",
    "run_stadia_session",
    "validate_returned_action",
    "validate_stadia_profile",
]
