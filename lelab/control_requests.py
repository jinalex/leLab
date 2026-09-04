"""Dependency-neutral canonical start-request resolution.

This module deliberately depends only on typed configuration. Feature handlers
adapt the immutable result into their existing request types after resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import config as config_module
from .utils.config import (
    RobotOperation,
    RobotRecordV2,
    RobotRecordValidationError,
    evaluate_robot_readiness,
    get_robot_record_v2,
    is_valid_robot_name,
)

_LEGACY_SELECTOR_FIELDS = frozenset(
    {
        "leader_port",
        "follower_port",
        "leader_config",
        "follower_config",
    }
)
_CANONICAL_SELECTOR_FIELDS = frozenset({"robot_name"})


class SelectorKind(StrEnum):
    CANONICAL = "canonical"
    LEGACY = "legacy"


class ControlRequestErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_FIELDS = "unknown_fields"
    MISSING_SELECTOR = "missing_selector"
    MISSING_LEGACY_FIELDS = "missing_legacy_fields"
    MIXED_SELECTORS = "mixed_selectors"
    INVALID_ROBOT_NAME = "invalid_robot_name"
    ROBOT_NOT_FOUND = "robot_not_found"
    ROBOT_INVALID = "robot_invalid"
    ROBOT_NOT_READY = "robot_not_ready"
    LEGACY_NO_MATCH = "legacy_no_match"
    LEGACY_AMBIGUOUS = "legacy_ambiguous"
    LEGACY_MODE_UNSUPPORTED = "legacy_mode_unsupported"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class ControlRequestError(ValueError):
    """Typed, route-adaptable failure to resolve a start request."""

    def __init__(
        self,
        code: ControlRequestErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: Mapping[str, Any] = _freeze(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": _thaw(self.details),
        }


class CanonicalTeleoperateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_name: str


class LegacyTeleoperateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str


class RecordingSessionSettings(BaseModel):
    """Existing recording settings, excluding robot identity fields."""

    model_config = ConfigDict(extra="forbid")

    dataset_repo_id: str
    single_task: str
    num_episodes: int = 5
    episode_time_s: int = 30
    reset_time_s: int = 10
    fps: int = 30
    video: bool = True
    push_to_hub: bool = False
    tags: list[str] = Field(default_factory=list)
    private: bool = False
    resume: bool = False
    streaming_encoding: bool = True
    cameras: dict[str, Any] = Field(default_factory=dict)
    test_mode: bool = False


class CanonicalRecordingRequest(RecordingSessionSettings):
    robot_name: str


class LegacyRecordingRequest(RecordingSessionSettings):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str


@dataclass(frozen=True, slots=True)
class ResolvedControlRequest:
    """Immutable canonical record and operation metadata for a route adapter."""

    operation: RobotOperation
    selector_kind: SelectorKind
    canonical_record: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @property
    def robot_name(self) -> str:
        return str(self.canonical_record["name"])

    @property
    def teleoperator_type(self) -> Literal["leader_arm", "stadia"]:
        return self.canonical_record["teleoperator_type"]

    def record_dict(self) -> dict[str, Any]:
        """Return a fresh canonical dictionary without exposing stored state."""
        return _thaw(self.canonical_record)

    def record_model(self) -> RobotRecordV2:
        """Return a fresh validated model for a worker or feature adapter."""
        return RobotRecordV2.model_validate(self.record_dict())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh operation-settings dictionary."""
        return _thaw(self.metadata)

    def legacy_handler_payload(self) -> dict[str, Any]:
        """Build a fresh legacy leader payload from canonical configuration.

        Browser selector fields are never copied into this result.
        """
        if self.teleoperator_type != "leader_arm":
            raise ValueError("legacy handler payloads are only valid for leader_arm records")
        leader = self.canonical_record["leader"]
        if not isinstance(leader, Mapping):
            raise ValueError("leader_arm record has no leader configuration")
        payload = self.metadata_dict()
        payload.update(
            {
                "leader_port": leader["port"],
                "follower_port": self.canonical_record["follower"]["port"],
                "leader_config": leader["calibration"],
                "follower_config": self.canonical_record["follower"]["calibration"],
            }
        )
        return payload


RecordLoader = Callable[[str], RobotRecordV2 | None]
RecordLister = Callable[[], list[RobotRecordV2]]


def _default_record_loader(name: str) -> RobotRecordV2 | None:
    """Preserve the distinction between a missing file and an unreadable record."""
    path = Path(config_module.ROBOTS_PATH) / f"{name}.json"
    existed = path.is_file()
    record = get_robot_record_v2(name)
    if record is None and existed:
        raise RobotRecordValidationError(
            f"saved robot record {name!r} exists but could not be read as valid JSON"
        )
    return record


def _default_record_lister() -> list[RobotRecordV2]:
    """Load every saved record strictly for legacy uniqueness checks.

    A malformed sibling cannot be silently skipped: until every saved record
    can be compared, the server cannot prove that a legacy four-field selector
    identifies exactly one canonical record.
    """

    root = Path(config_module.ROBOTS_PATH)
    if not root.exists():
        return []
    try:
        paths = sorted(path for path in root.iterdir() if path.suffix == ".json")
    except OSError as error:
        raise RobotRecordValidationError(f"could not list saved robot records: {error}") from error

    records: list[RobotRecordV2] = []
    for path in paths:
        record = _default_record_loader(path.stem)
        if record is None:
            raise RobotRecordValidationError(
                f"saved robot record {path.stem!r} could not be loaded for legacy resolution"
            )
        records.append(record)
    return records


def _request_mapping(request: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(request, BaseModel):
        return request.model_dump(mode="python")
    if not isinstance(request, Mapping):
        raise ControlRequestError(
            ControlRequestErrorCode.INVALID_REQUEST,
            "Start request must be a JSON object.",
        )
    return dict(request)


def _parse_request(
    request: Mapping[str, Any] | BaseModel,
    *,
    canonical_model: type[BaseModel],
    legacy_model: type[BaseModel],
    settings_fields: frozenset[str],
) -> tuple[SelectorKind, BaseModel]:
    data = _request_mapping(request)
    fields = set(data)
    canonical_present = bool(fields & _CANONICAL_SELECTOR_FIELDS)
    legacy_present = bool(fields & _LEGACY_SELECTOR_FIELDS)
    known = settings_fields | _CANONICAL_SELECTOR_FIELDS | _LEGACY_SELECTOR_FIELDS
    unknown = fields - known
    if unknown:
        raise ControlRequestError(
            ControlRequestErrorCode.UNKNOWN_FIELDS,
            "Start request contains unknown fields.",
            details={"fields": sorted(unknown)},
        )
    if canonical_present and legacy_present:
        raise ControlRequestError(
            ControlRequestErrorCode.MIXED_SELECTORS,
            "Do not mix robot_name with legacy port/calibration selector fields.",
        )
    if not canonical_present and not legacy_present:
        raise ControlRequestError(
            ControlRequestErrorCode.MISSING_SELECTOR,
            "Start request must provide robot_name or all four legacy selector fields.",
        )
    if legacy_present:
        missing = _LEGACY_SELECTOR_FIELDS - fields
        if missing:
            raise ControlRequestError(
                ControlRequestErrorCode.MISSING_LEGACY_FIELDS,
                "Legacy start request must provide all four robot identity fields.",
                details={"fields": sorted(missing)},
            )
        model = legacy_model
        kind = SelectorKind.LEGACY
    else:
        model = canonical_model
        kind = SelectorKind.CANONICAL

    try:
        return kind, model.model_validate(data)
    except ValidationError as error:
        raise ControlRequestError(
            ControlRequestErrorCode.INVALID_REQUEST,
            "Start request fields are invalid.",
            details={"errors": error.errors(include_url=False)},
        ) from error


def _load_canonical_record(name: str, loader: RecordLoader) -> RobotRecordV2:
    if not is_valid_robot_name(name):
        raise ControlRequestError(
            ControlRequestErrorCode.INVALID_ROBOT_NAME,
            "robot_name is not a valid saved robot name.",
            details={"robot_name": name},
        )
    try:
        record = loader(name)
    except RobotRecordValidationError as error:
        raise ControlRequestError(
            ControlRequestErrorCode.ROBOT_INVALID,
            "The saved robot record is invalid and must be repaired before starting.",
            details={"robot_name": name, "error": str(error)},
        ) from error
    if record is None:
        raise ControlRequestError(
            ControlRequestErrorCode.ROBOT_NOT_FOUND,
            "No saved robot record exists with that name.",
            details={"robot_name": name},
        )
    return record


def _legacy_identity(request: BaseModel) -> tuple[str, str, str, str]:
    return (
        request.leader_port,
        request.follower_port,
        request.leader_config,
        request.follower_config,
    )


def _record_identity(record: RobotRecordV2) -> tuple[str, str, str, str] | None:
    if record.leader is None:
        return None
    return (
        record.leader.port,
        record.follower.port,
        record.leader.calibration,
        record.follower.calibration,
    )


def _resolve_legacy_record(request: BaseModel, lister: RecordLister) -> RobotRecordV2:
    requested_identity = _legacy_identity(request)
    try:
        matching = [record for record in lister() if _record_identity(record) == requested_identity]
    except RobotRecordValidationError as error:
        raise ControlRequestError(
            ControlRequestErrorCode.ROBOT_INVALID,
            "A saved robot record is invalid; legacy identity cannot be resolved safely.",
            details={"error": str(error)},
        ) from error
    if not matching:
        raise ControlRequestError(
            ControlRequestErrorCode.LEGACY_NO_MATCH,
            "Legacy robot identity does not exactly match any saved canonical record.",
        )
    if len(matching) != 1:
        raise ControlRequestError(
            ControlRequestErrorCode.LEGACY_AMBIGUOUS,
            "Legacy robot identity matches more than one saved record; use robot_name.",
            details={"robot_names": sorted(record.name for record in matching)},
        )
    record = matching[0]
    if record.teleoperator_type != "leader_arm":
        raise ControlRequestError(
            ControlRequestErrorCode.LEGACY_MODE_UNSUPPORTED,
            "Legacy port/calibration selectors are supported only for leader_arm records.",
            details={"robot_name": record.name},
        )
    return record


def _operation_for(record: RobotRecordV2, *, recording: bool) -> RobotOperation:
    if record.teleoperator_type == "stadia":
        return RobotOperation.STADIA_RECORDING if recording else RobotOperation.STADIA_TELEOPERATION
    return RobotOperation.LEADER_RECORDING if recording else RobotOperation.LEADER_TELEOPERATION


def _require_ready(record: RobotRecordV2, operation: RobotOperation) -> None:
    readiness = evaluate_robot_readiness(record, operation)
    if readiness.ready:
        return
    raise ControlRequestError(
        ControlRequestErrorCode.ROBOT_NOT_READY,
        "Saved robot configuration is not ready for the requested operation.",
        details={
            "robot_name": record.name,
            "operation": operation.value,
            "issues": [issue.model_dump(mode="json") for issue in readiness.issues],
        },
    )


def _resolved(
    *,
    record: RobotRecordV2,
    operation: RobotOperation,
    selector_kind: SelectorKind,
    metadata: Mapping[str, Any],
) -> ResolvedControlRequest:
    return ResolvedControlRequest(
        operation=operation,
        selector_kind=selector_kind,
        canonical_record=_freeze(record.model_dump(mode="json")),
        metadata=_freeze(metadata),
    )


def resolve_teleoperate_request(
    request: Mapping[str, Any] | BaseModel,
    *,
    record_loader: RecordLoader | None = None,
    record_lister: RecordLister | None = None,
) -> ResolvedControlRequest:
    """Resolve canonical or temporary legacy teleoperation selection."""
    selector_kind, parsed = _parse_request(
        request,
        canonical_model=CanonicalTeleoperateRequest,
        legacy_model=LegacyTeleoperateRequest,
        settings_fields=frozenset(),
    )
    if selector_kind == SelectorKind.CANONICAL:
        record = _load_canonical_record(parsed.robot_name, record_loader or _default_record_loader)
    else:
        record = _resolve_legacy_record(parsed, record_lister or _default_record_lister)
    operation = _operation_for(record, recording=False)
    _require_ready(record, operation)
    return _resolved(
        record=record,
        operation=operation,
        selector_kind=selector_kind,
        metadata={},
    )


_RECORDING_SETTINGS_FIELDS = frozenset(RecordingSessionSettings.model_fields)


def resolve_recording_request(
    request: Mapping[str, Any] | BaseModel,
    *,
    record_loader: RecordLoader | None = None,
    record_lister: RecordLister | None = None,
) -> ResolvedControlRequest:
    """Resolve robot selection while preserving recording session settings."""
    selector_kind, parsed = _parse_request(
        request,
        canonical_model=CanonicalRecordingRequest,
        legacy_model=LegacyRecordingRequest,
        settings_fields=_RECORDING_SETTINGS_FIELDS,
    )
    if selector_kind == SelectorKind.CANONICAL:
        record = _load_canonical_record(parsed.robot_name, record_loader or _default_record_loader)
    else:
        record = _resolve_legacy_record(parsed, record_lister or _default_record_lister)
    operation = _operation_for(record, recording=True)
    _require_ready(record, operation)
    metadata = {field: getattr(parsed, field) for field in RecordingSessionSettings.model_fields}
    return _resolved(
        record=record,
        operation=operation,
        selector_kind=selector_kind,
        metadata=metadata,
    )


resolve_teleoperation_start = resolve_teleoperate_request
resolve_recording_start = resolve_recording_request
