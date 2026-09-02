# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import platform
import shutil
import time
from collections.abc import Mapping
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

RobotSide = Literal["leader", "follower"]

# Define the calibration config paths (shared between features)
CALIBRATION_BASE_PATH_TELEOP = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/teleoperators")
CALIBRATION_BASE_PATH_ROBOTS = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots")
LEADER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_TELEOP, "so_leader")
FOLLOWER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "so_follower")

# Define port storage path
PORT_CONFIG_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/ports")
LEADER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "leader_port.txt")
FOLLOWER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "follower_port.txt")

# Define configuration storage path
CONFIG_STORAGE_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/saved_configs")
LEADER_CONFIG_FILE = os.path.join(CONFIG_STORAGE_PATH, "leader_config.txt")
FOLLOWER_CONFIG_FILE = os.path.join(CONFIG_STORAGE_PATH, "follower_config.txt")

# Robot config records (per-robot JSON metadata)
ROBOTS_PATH = os.path.expanduser("~/.cache/huggingface/lerobot/robots")

# Tag stamped on every dataset pushed to the Hub from LeLab, so we can later
# query the Hub for LeLab-produced datasets and compute usage metrics.
LELAB_TAG = "LeLab"


def with_lelab_tag(tags: list[str] | None) -> list[str]:
    """Return `tags` with LELAB_TAG appended (deduped, order preserved)."""
    out = list(tags or [])
    if LELAB_TAG not in out:
        out.append(LELAB_TAG)
    return out


def _atomic_write_text(path: str, content: str) -> None:
    """Write to <path>.tmp then os.replace, so a crash mid-write never leaves
    a half-written file on disk."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def _port_file_for(robot_type: RobotSide) -> str:
    if robot_type == "leader":
        return LEADER_PORT_FILE
    if robot_type == "follower":
        return FOLLOWER_PORT_FILE
    raise ValueError(f"robot_type must be 'leader' or 'follower', got {robot_type!r}")


def _config_file_for(robot_type: RobotSide) -> str:
    rt = robot_type.lower() if isinstance(robot_type, str) else robot_type
    if rt == "leader":
        return LEADER_CONFIG_FILE
    if rt == "follower":
        return FOLLOWER_CONFIG_FILE
    raise ValueError(f"robot_type must be 'leader' or 'follower', got {robot_type!r}")


def setup_calibration_files(leader_config: str, follower_config: str):
    """Setup calibration files in the correct locations for teleoperation and recording"""
    # Extract config names from file paths (remove .json extension)
    leader_config_name = os.path.splitext(leader_config)[0]
    follower_config_name = os.path.splitext(follower_config)[0]

    # Log the full paths to check if files exist
    leader_config_full_path = os.path.join(LEADER_CONFIG_PATH, leader_config)
    follower_config_full_path = os.path.join(FOLLOWER_CONFIG_PATH, follower_config)

    logger.info("Checking calibration files:")
    logger.info(f"Leader config path: {leader_config_full_path}")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Leader config exists: {os.path.exists(leader_config_full_path)}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directories if they don't exist
    leader_calibration_dir = LEADER_CONFIG_PATH
    follower_calibration_dir = FOLLOWER_CONFIG_PATH
    os.makedirs(leader_calibration_dir, exist_ok=True)
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration files to the correct locations if they're not already there
    leader_target_path = os.path.join(leader_calibration_dir, f"{leader_config_name}.json")
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(leader_target_path):
        if os.path.exists(leader_config_full_path):
            shutil.copy2(leader_config_full_path, leader_target_path)
            logger.info(f"Copied leader calibration to {leader_target_path}")
        else:
            raise FileNotFoundError(f"Leader calibration file not found: {leader_config_full_path}")
    else:
        logger.info(f"Leader calibration already exists at {leader_target_path}")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return leader_config_name, follower_config_name


def setup_follower_calibration_file(follower_config: str):
    """Setup follower calibration file in the correct location for replay functionality"""
    # Extract config name from file path (remove .json extension)
    follower_config_name = os.path.splitext(follower_config)[0]

    # Log the full path to check if file exists
    follower_config_full_path = os.path.join(FOLLOWER_CONFIG_PATH, follower_config)

    logger.info("Checking follower calibration file:")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directory if it doesn't exist
    follower_calibration_dir = FOLLOWER_CONFIG_PATH
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration file to the correct location if it's not already there
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return follower_config_name


def find_available_ports():
    """Find all available serial ports on the system"""
    try:
        from serial.tools import list_ports  # Part of pyserial library
    except ImportError as exc:
        raise ImportError("pyserial library is required. Install it with: pip install pyserial") from exc

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return sorted(ports)


def find_robot_port(robot_type="robot"):
    """
    Find the port for a robot by detecting the difference when disconnecting/reconnecting

    Args:
        robot_type (str): Type of robot ("leader" or "follower" or generic "robot")

    Returns:
        str: The detected port
    """
    logger.info(f"Finding port for {robot_type}")

    # Get initial ports
    ports_before = find_available_ports()
    logger.info(f"Ports before disconnecting: {ports_before}")

    # This function returns the port detection logic, but the actual user interaction
    # should be handled by the frontend
    return {"ports_before": ports_before, "robot_type": robot_type}


def detect_port_after_disconnect(ports_before, timeout_s: float = 15.0, poll_interval_s: float = 0.3):
    """
    Wait for the user to unplug the robot and detect which port disappeared.

    Polls the available ports until exactly one entry from ``ports_before`` vanishes,
    or until ``timeout_s`` elapses. Polling avoids racing the user — they may need
    several seconds to physically pull the USB cable.

    Args:
        ports_before (list): List of ports before disconnection
        timeout_s (float): Maximum seconds to wait for a port to disappear
        poll_interval_s (float): Seconds between checks

    Returns:
        str: The detected port

    Raises:
        OSError: If the timeout elapses with no change, or more than one port disappears.
    """
    before_set = set(ports_before)
    deadline = time.monotonic() + timeout_s
    last_diff: list = []

    while time.monotonic() < deadline:
        ports_after = find_available_ports()
        ports_diff = list(before_set - set(ports_after))
        last_diff = ports_diff

        if len(ports_diff) == 1:
            port = ports_diff[0]
            logger.info(f"Detected port: {port}")
            return port
        if len(ports_diff) > 1:
            raise OSError(f"Could not detect the port. More than one port disappeared: {ports_diff}.")

        time.sleep(poll_interval_s)

    logger.info(f"Timed out waiting for unplug. Final diff: {last_diff}")
    raise OSError(
        "Timed out waiting for the robot to be unplugged. Please try again and unplug the USB cable when prompted."
    )


def save_robot_port(robot_type: RobotSide, port: str) -> None:
    """Persist the robot port for `robot_type` ('leader' or 'follower')."""
    port_file = _port_file_for(robot_type)
    _atomic_write_text(port_file, port)
    logger.info(f"Saved {robot_type} port: {port}")


def get_saved_robot_port(robot_type: RobotSide) -> str | None:
    """Return the saved port for `robot_type`, or None if no file exists."""
    port_file = _port_file_for(robot_type)
    if not os.path.exists(port_file):
        logger.info(f"No saved port found for {robot_type}")
        return None
    with open(port_file) as f:
        port = f.read().strip()
    logger.info(f"Retrieved saved {robot_type} port: {port}")
    return port


def get_default_robot_port(robot_type: RobotSide) -> str:
    """Saved port if present, else a platform-typical default."""
    saved_port = get_saved_robot_port(robot_type)
    if saved_port:
        return saved_port
    if platform.system() == "Windows":
        return "COM3"
    return "/dev/ttyUSB0"


def save_robot_config(robot_type: RobotSide, config_name: str) -> bool:
    try:
        config_file_path = _config_file_for(robot_type)
    except ValueError as e:
        logger.error(str(e))
        return False
    try:
        _atomic_write_text(config_file_path, config_name.strip())
    except Exception as e:
        logger.error(f"Error saving {robot_type} configuration: {e}")
        return False
    logger.info(f"Saved {robot_type} configuration: {config_name}")
    return True


def get_saved_robot_config(robot_type: RobotSide) -> str | None:
    try:
        config_file_path = _config_file_for(robot_type)
    except ValueError as e:
        logger.error(str(e))
        return None
    if not os.path.exists(config_file_path):
        logger.info(f"No saved {robot_type} configuration found")
        return None
    try:
        with open(config_file_path) as f:
            config_name = f.read().strip()
    except OSError as e:
        logger.error(f"Error reading saved {robot_type} configuration: {e}")
        return None
    if not config_name:
        return None
    logger.info(f"Found saved {robot_type} configuration: {config_name}")
    return config_name


def get_default_robot_config(robot_type: str, available_configs: list):
    """Get the default configuration for a robot, checking saved configs first"""
    saved_config = get_saved_robot_config(robot_type)
    if saved_config and saved_config in available_configs:
        return saved_config

    # Return first available config as fallback
    if available_configs:
        return available_configs[0]

    return None


# ---------------------------------------------------------------------------
# Robot record helpers
# ---------------------------------------------------------------------------

# Characters disallowed in a robot name (filesystem safety)
_INVALID_NAME_CHARS = ("/", "\\", "..")
_ROBOT_STRING_FIELDS = ("leader_port", "follower_port", "leader_config", "follower_config")
_ROBOT_LIST_FIELDS = ("cameras",)

_LEGACY_RECORD_FIELDS = {
    "name",
    *_ROBOT_STRING_FIELDS,
    *_ROBOT_LIST_FIELDS,
}
_V2_RECORD_FIELDS = {
    "schema_version",
    "name",
    "teleoperator_type",
    "follower",
    "leader",
    "stadia",
    "cameras",
}
_LEGACY_ONLY_RECORD_FIELDS = set(_ROBOT_STRING_FIELDS)
_V2_ONLY_RECORD_FIELDS = {
    "schema_version",
    "teleoperator_type",
    "follower",
    "leader",
    "stadia",
}


class RobotRecordValidationError(ValueError):
    """A saved robot record or robot-record patch violates its schema."""


class DeviceRecord(BaseModel):
    """One serial device and its saved calibration filename.

    Blank values are valid while a robot record is being configured. Readiness,
    rather than persistence, decides whether an operation can start.
    """

    model_config = ConfigDict(extra="forbid")

    port: str = ""
    calibration: str = ""

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("port cannot contain a NUL byte")
        return value

    @field_validator("calibration")
    @classmethod
    def _validate_calibration(cls, value: str) -> str:
        if not value:
            return value
        if value.strip() != value:
            raise ValueError("calibration filename cannot have surrounding whitespace")
        if not value.endswith(".json"):
            raise ValueError("calibration filename must end with .json")
        if Path(value).name != value or "/" in value or "\\" in value or ".." in value:
            raise ValueError("calibration filename must be a safe basename")
        return value


class CameraRecord(BaseModel):
    """The OpenCV camera fields already persisted by LeLab's frontend."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: Literal["opencv"] = "opencv"
    # The existing recorder has always defaulted a missing legacy index to 0.
    # Keep that normalization behavior while still persisting a concrete V2
    # value on the next explicit save.
    camera_index: int = Field(default=0, ge=0)
    device_id: str = ""
    width: int = Field(default=640, ge=1, le=8192)
    height: int = Field(default=480, ge=1, le=8192)
    fps: int | None = Field(default=30, ge=1, le=240)
    fourcc: str | None = Field(default=None, min_length=4, max_length=4)
    backend: (
        Literal[
            "ANY",
            "V4L2",
            "DSHOW",
            "PVAPI",
            "ANDROID",
            "AVFOUNDATION",
            "MSMF",
        ]
        | None
    ) = None

    @field_validator("id", "name")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("camera id and name must be non-empty and trimmed")
        return value

    @field_validator("device_id")
    @classmethod
    def _validate_device_id(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("camera device_id cannot contain a NUL byte")
        return value


class StadiaConfig(BaseModel):
    """Persisted, user-selectable Stadia settings with plan-owned safety caps."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    guid: str | None = None
    deadzone: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_step_per_tick: float = Field(default=0.35, gt=0.0, le=0.35)
    arm_startup_travel_degrees: float = Field(default=45.0, gt=0.0, le=45.0)
    gripper_startup_travel_percentage_points: float = Field(default=45.0, gt=0.0, le=45.0)

    @field_validator(
        "deadzone",
        "max_step_per_tick",
        "arm_startup_travel_degrees",
        "gripper_startup_travel_percentage_points",
        mode="before",
    )
    @classmethod
    def _validate_numeric_type(cls, value: object) -> float:
        # Pydantic's normal float parsing accepts strings and bools. Robot
        # safety settings must arrive as actual JSON numbers.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Stadia settings must be numeric values")
        return float(value)

    @field_validator("guid")
    @classmethod
    def _validate_guid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.strip() != value or "\x00" in value:
            raise ValueError("guid must be non-empty, trimmed, and contain no NUL byte")
        return value


class RobotRecordV1(BaseModel):
    """Strict schema for the legacy flat records already stored by LeLab."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    leader_port: str = ""
    follower_port: str = ""
    leader_config: str = ""
    follower_config: str = ""
    cameras: list[CameraRecord] = Field(default_factory=list)


class RobotRecordV2(BaseModel):
    """Canonical server-owned robot configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    name: str
    teleoperator_type: Literal["leader_arm", "stadia"] = "leader_arm"
    follower: DeviceRecord = Field(default_factory=DeviceRecord)
    leader: DeviceRecord | None = Field(default_factory=DeviceRecord)
    stadia: StadiaConfig = Field(default_factory=StadiaConfig)
    cameras: list[CameraRecord] = Field(default_factory=list)


class RobotOperation(StrEnum):
    FOLLOWER_CALIBRATION = "follower_calibration"
    LEADER_CALIBRATION = "leader_calibration"
    LEADER_TELEOPERATION = "leader_teleoperation"
    STADIA_TELEOPERATION = "stadia_teleoperation"
    LEADER_RECORDING = "leader_recording"
    STADIA_RECORDING = "stadia_recording"
    INFERENCE = "inference"
    REPLAY = "replay"
    CONTROLLER_CHECK = "controller_check"


class ReadinessIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    field: str | None
    message: str


class ReadinessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: RobotOperation
    ready: bool
    issues: tuple[ReadinessIssue, ...] = ()


def _robot_record_path(name: str) -> str:
    return os.path.join(ROBOTS_PATH, f"{name}.json")


def is_valid_robot_name(name: str) -> bool:
    """Check that a robot name is safe to use as a filename."""
    if not name or not isinstance(name, str):
        return False
    if name.strip() != name:
        return False
    return not any(bad in name for bad in _INVALID_NAME_CHARS)


def _empty_record(name: str) -> dict:
    record: dict = {"name": name}
    for field in _ROBOT_STRING_FIELDS:
        record[field] = ""
    for field in _ROBOT_LIST_FIELDS:
        record[field] = []
    return record


def _validation_error(message: str, error: Exception | None = None) -> RobotRecordValidationError:
    if error is None:
        return RobotRecordValidationError(message)
    return RobotRecordValidationError(f"{message}: {error}")


def normalize_robot_record(name: str, raw: Mapping[str, Any]) -> RobotRecordV2:
    """Normalize a strict legacy or V2 record without writing it.

    The filename is authoritative for ``name``. Legacy records are interpreted
    as leader-arm records and receive default Stadia settings in memory.
    """
    if not is_valid_robot_name(name):
        raise _validation_error(f"invalid robot name {name!r}")
    if not isinstance(raw, Mapping):
        raise _validation_error("robot record must be a JSON object")

    data = dict(raw)
    keys = set(data)
    is_v2 = bool(keys & _V2_ONLY_RECORD_FIELDS)
    if is_v2 and keys & _LEGACY_ONLY_RECORD_FIELDS:
        raise _validation_error("robot record cannot mix legacy flat fields with V2 fields")

    try:
        if is_v2:
            unknown = keys - _V2_RECORD_FIELDS
            if unknown:
                raise _validation_error(f"unknown V2 robot-record fields: {sorted(unknown)}")
            data["name"] = name
            return RobotRecordV2.model_validate(data)

        unknown = keys - _LEGACY_RECORD_FIELDS
        if unknown:
            raise _validation_error(f"unknown legacy robot-record fields: {sorted(unknown)}")
        data["name"] = name
        legacy = RobotRecordV1.model_validate(data)
        return RobotRecordV2(
            name=name,
            teleoperator_type="leader_arm",
            follower=DeviceRecord(
                port=legacy.follower_port,
                calibration=legacy.follower_config,
            ),
            leader=DeviceRecord(
                port=legacy.leader_port,
                calibration=legacy.leader_config,
            ),
            cameras=legacy.cameras,
        )
    except RobotRecordValidationError:
        raise
    except ValidationError as error:
        raise _validation_error(f"invalid robot record {name!r}", error) from error


def _read_robot_record_data(name: str) -> dict[str, Any] | None:
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as error:
        logger.error(f"Failed to read robot record {name}: {error}")
        return None
    if not isinstance(data, dict):
        raise _validation_error(f"robot record {name!r} must contain a JSON object")
    return data


def get_robot_record_v2(name: str) -> RobotRecordV2 | None:
    """Load and normalize a record without mutating its file."""
    raw = _read_robot_record_data(name)
    if raw is None:
        return None
    return normalize_robot_record(name, raw)


load_robot_record = get_robot_record_v2


def _legacy_record_projection(record: RobotRecordV2) -> dict[str, Any]:
    """Project V2 into the flat shape consumed by existing routes/features."""
    leader = record.leader or DeviceRecord()
    return {
        "name": record.name,
        "leader_port": leader.port,
        "follower_port": record.follower.port,
        "leader_config": leader.calibration,
        "follower_config": record.follower.calibration,
        "cameras": [camera.model_dump(mode="json", exclude_none=True) for camera in record.cameras],
    }


def get_robot_record(name: str) -> dict | None:
    """Return the legacy flat projection by name, or None if missing/invalid."""
    try:
        record = get_robot_record_v2(name)
    except RobotRecordValidationError as error:
        logger.error(f"Failed to validate robot record {name}: {error}")
        return None
    return _legacy_record_projection(record) if record is not None else None


def list_robot_records() -> list[dict]:
    """Return all robot records on disk."""
    if not os.path.exists(ROBOTS_PATH):
        return []
    records = []
    for filename in sorted(os.listdir(ROBOTS_PATH)):
        if not filename.endswith(".json"):
            continue
        name = os.path.splitext(filename)[0]
        record = get_robot_record(name)
        if record is not None:
            records.append(record)
    return records


def list_robot_records_v2() -> list[RobotRecordV2]:
    """Return all valid canonical robot records without modifying disk state."""
    if not os.path.exists(ROBOTS_PATH):
        return []
    records = []
    for filename in sorted(os.listdir(ROBOTS_PATH)):
        if not filename.endswith(".json"):
            continue
        name = os.path.splitext(filename)[0]
        record = get_robot_record_v2(name)
        if record is not None:
            records.append(record)
    return records


def _apply_legacy_patch(base: dict[str, Any], patch: Mapping[str, Any]) -> None:
    unknown = set(patch) - _LEGACY_RECORD_FIELDS
    if unknown:
        raise _validation_error(f"unknown legacy robot-record fields: {sorted(unknown)}")
    for field in _ROBOT_STRING_FIELDS:
        if field not in patch:
            continue
        value = patch[field]
        if not isinstance(value, str):
            raise _validation_error(f"legacy field {field!r} must be a string")
        if field == "leader_port":
            leader = base.get("leader") or {}
            leader["port"] = value
            base["leader"] = leader
        elif field == "follower_port":
            base["follower"]["port"] = value
        elif field == "leader_config":
            leader = base.get("leader") or {}
            leader["calibration"] = value
            base["leader"] = leader
        elif field == "follower_config":
            base["follower"]["calibration"] = value
    if "cameras" in patch:
        base["cameras"] = patch["cameras"]


def _apply_v2_patch(name: str, base: dict[str, Any], patch: Mapping[str, Any]) -> None:
    unknown = set(patch) - _V2_RECORD_FIELDS
    if unknown:
        raise _validation_error(f"unknown V2 robot-record fields: {sorted(unknown)}")
    if set(patch) & _LEGACY_ONLY_RECORD_FIELDS:
        raise _validation_error("robot-record patch cannot mix legacy flat fields with V2 fields")
    if "name" in patch and patch["name"] != name:
        raise _validation_error("robot-record body name must match the path name")
    if "schema_version" in patch and patch["schema_version"] != 2:
        raise _validation_error("schema_version must be 2")

    for field in ("teleoperator_type", "cameras"):
        if field in patch:
            base[field] = patch[field]
    for field in ("follower", "stadia"):
        if field not in patch:
            continue
        value = patch[field]
        if not isinstance(value, Mapping):
            raise _validation_error(f"V2 field {field!r} must be an object")
        base[field].update(value)
    if "leader" in patch:
        value = patch["leader"]
        if value is None:
            base["leader"] = None
        elif isinstance(value, Mapping):
            leader = base.get("leader") or DeviceRecord().model_dump(mode="json")
            leader.update(value)
            base["leader"] = leader
        else:
            raise _validation_error("V2 field 'leader' must be an object or null")


def save_robot_record_v2(
    name: str,
    data: Mapping[str, Any],
    allow_create: bool = True,
) -> RobotRecordV2 | None:
    """Strictly merge and atomically persist a canonical V2 robot record."""
    if not is_valid_robot_name(name):
        raise _validation_error(f"invalid robot name {name!r}")
    if not isinstance(data, Mapping):
        raise _validation_error("robot-record patch must be a JSON object")

    existing = get_robot_record_v2(name)
    if existing is None and not allow_create:
        return None
    record = existing or RobotRecordV2(name=name)
    merged = record.model_dump(mode="json")
    patch = dict(data)
    has_legacy = bool(set(patch) & _LEGACY_ONLY_RECORD_FIELDS)
    has_v2 = bool(set(patch) & _V2_ONLY_RECORD_FIELDS)
    if has_legacy and has_v2:
        raise _validation_error("robot-record patch cannot mix legacy flat fields with V2 fields")

    if has_legacy:
        _apply_legacy_patch(merged, patch)
    else:
        _apply_v2_patch(name, merged, patch)
    merged["name"] = name

    try:
        normalized = RobotRecordV2.model_validate(merged)
    except ValidationError as error:
        raise _validation_error(f"invalid robot-record patch for {name!r}", error) from error

    os.makedirs(ROBOTS_PATH, exist_ok=True)
    path = _robot_record_path(name)
    _atomic_write_text(path, json.dumps(normalized.model_dump(mode="json"), indent=2))
    logger.info(f"Saved V2 robot record {name}: {normalized.model_dump(mode='json')}")
    return normalized


def save_robot_record(name: str, data: dict, allow_create: bool = True) -> bool:
    """
    Upsert a robot record. Merges `data` into the existing record, preserving
    fields not provided. Returns True if a write occurred, False if no-oped.

    - If the record exists: merge and write.
    - If the record does not exist and `allow_create` is True: create with empty
      fields then merge.
    - If the record does not exist and `allow_create` is False: log and no-op.
    """
    if not is_valid_robot_name(name):
        logger.error(f"Invalid robot name: {name!r}")
        return False

    record = save_robot_record_v2(name, data, allow_create=allow_create)
    if record is None:
        logger.info(f"save_robot_record no-op: {name} does not exist (allow_create=False)")
        return False
    return True


def delete_robot_record(name: str) -> bool:
    """Delete a robot record. Returns True if a file was removed."""
    if not is_valid_robot_name(name):
        return False
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return False
    os.remove(path)
    logger.info(f"Deleted robot record {name}")
    return True


def is_robot_record_clean(record: RobotRecordV2 | Mapping[str, Any]) -> bool:
    """
    A record is 'clean' when all four operational fields are populated AND both
    referenced calibration files exist on disk. Cameras are optional and don't
    affect cleanliness.
    """
    if isinstance(record, RobotRecordV2) or (
        isinstance(record, Mapping) and set(record) & _V2_ONLY_RECORD_FIELDS
    ):
        try:
            normalized = (
                record
                if isinstance(record, RobotRecordV2)
                else normalize_robot_record(str(record.get("name", "")), record)
            )
        except RobotRecordValidationError:
            return False
        operation = (
            RobotOperation.LEADER_TELEOPERATION
            if normalized.teleoperator_type == "leader_arm"
            else RobotOperation.STADIA_TELEOPERATION
        )
        return evaluate_robot_readiness(normalized, operation).ready

    if not record:
        return False
    for field in _ROBOT_STRING_FIELDS:
        value = record.get(field, "")
        if not isinstance(value, str) or not value.strip():
            return False
    leader_path = os.path.join(LEADER_CONFIG_PATH, record["leader_config"])
    follower_path = os.path.join(FOLLOWER_CONFIG_PATH, record["follower_config"])
    return os.path.exists(leader_path) and os.path.exists(follower_path)


def _missing_issue(side: str, field: str) -> ReadinessIssue:
    label = side.capitalize()
    return ReadinessIssue(
        code=f"{side}_{field}_missing",
        field=f"{side}.{'calibration' if field == 'calibration' else 'port'}",
        message=f"{label} {field} is required.",
    )


def _device_issues(
    side: Literal["leader", "follower"],
    device: DeviceRecord | None,
    *,
    require_existing_calibration: bool,
) -> list[ReadinessIssue]:
    if device is None:
        return [
            ReadinessIssue(
                code=f"{side}_configuration_missing",
                field=side,
                message=f"{side.capitalize()} configuration is required.",
            )
        ]

    issues = []
    if not device.port.strip():
        issues.append(_missing_issue(side, "port"))
    if not device.calibration.strip():
        issues.append(_missing_issue(side, "calibration"))
    elif require_existing_calibration:
        directory = LEADER_CONFIG_PATH if side == "leader" else FOLLOWER_CONFIG_PATH
        if not os.path.isfile(os.path.join(directory, device.calibration)):
            issues.append(
                ReadinessIssue(
                    code=f"{side}_calibration_not_found",
                    field=f"{side}.calibration",
                    message=f"{side.capitalize()} calibration file was not found.",
                )
            )
    return issues


def evaluate_robot_readiness(
    record: RobotRecordV2 | Mapping[str, Any],
    operation: RobotOperation | str,
) -> ReadinessResult:
    """Evaluate static, operation-specific readiness without device access."""
    if not isinstance(record, RobotRecordV2):
        record = normalize_robot_record(str(record.get("name", "")), record)
    try:
        selected_operation = RobotOperation(operation)
    except ValueError as error:
        raise ValueError(f"unknown robot operation: {operation!r}") from error

    required_type: str | None = None
    if selected_operation in {
        RobotOperation.LEADER_CALIBRATION,
        RobotOperation.LEADER_TELEOPERATION,
        RobotOperation.LEADER_RECORDING,
    }:
        required_type = "leader_arm"
    elif selected_operation in {
        RobotOperation.STADIA_TELEOPERATION,
        RobotOperation.STADIA_RECORDING,
        RobotOperation.CONTROLLER_CHECK,
    }:
        required_type = "stadia"

    if required_type is not None and record.teleoperator_type != required_type:
        issue = ReadinessIssue(
            code="wrong_teleoperator_type",
            field="teleoperator_type",
            message=f"Operation requires teleoperator_type={required_type!r}.",
        )
        return ReadinessResult(operation=selected_operation, ready=False, issues=(issue,))

    issues: list[ReadinessIssue] = []
    if selected_operation == RobotOperation.FOLLOWER_CALIBRATION:
        issues.extend(_device_issues("follower", record.follower, require_existing_calibration=False))
    elif selected_operation == RobotOperation.LEADER_CALIBRATION:
        issues.extend(_device_issues("leader", record.leader, require_existing_calibration=False))
    elif selected_operation in {
        RobotOperation.LEADER_TELEOPERATION,
        RobotOperation.LEADER_RECORDING,
    }:
        issues.extend(_device_issues("leader", record.leader, require_existing_calibration=True))
        issues.extend(_device_issues("follower", record.follower, require_existing_calibration=True))
    elif selected_operation in {
        RobotOperation.STADIA_TELEOPERATION,
        RobotOperation.STADIA_RECORDING,
        RobotOperation.INFERENCE,
        RobotOperation.REPLAY,
    }:
        issues.extend(_device_issues("follower", record.follower, require_existing_calibration=True))
    elif selected_operation == RobotOperation.CONTROLLER_CHECK:
        # Pydantic has already validated Stadia settings. Runtime exclusivity is
        # added later by the central session manager, never by touching a robot.
        pass

    return ReadinessResult(
        operation=selected_operation,
        ready=not issues,
        issues=tuple(issues),
    )


def evaluate_all_robot_readiness(
    record: RobotRecordV2 | Mapping[str, Any],
) -> dict[str, ReadinessResult]:
    """Return the normalized static readiness result for every operation."""
    return {operation.value: evaluate_robot_readiness(record, operation) for operation in RobotOperation}
