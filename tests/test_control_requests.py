"""Adversarial canonical and legacy start-request resolution tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from lelab.control_requests import (
    ControlRequestError,
    ControlRequestErrorCode,
    SelectorKind,
    resolve_recording_request,
    resolve_teleoperate_request,
)
from lelab.utils import config as cfg


@pytest.fixture(autouse=True)
def _isolated_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    robots = tmp_path / "robots"
    leaders = tmp_path / "leaders"
    followers = tmp_path / "followers"
    for path in (robots, leaders, followers):
        path.mkdir()
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))
    monkeypatch.setattr(cfg, "LEADER_CONFIG_PATH", str(leaders))
    monkeypatch.setattr(cfg, "FOLLOWER_CONFIG_PATH", str(followers))
    return {"robots": robots, "leaders": leaders, "followers": followers}


def _save_ready_record(
    paths: dict[str, Path],
    name: str,
    *,
    mode: str = "leader_arm",
    leader_port: str = "/dev/leader",
    follower_port: str = "/dev/follower",
) -> None:
    cfg.save_robot_record_v2(
        name,
        {
            "schema_version": 2,
            "teleoperator_type": mode,
            "leader": {
                "port": leader_port,
                "calibration": "leader.json",
            },
            "follower": {
                "port": follower_port,
                "calibration": "follower.json",
            },
        },
    )
    (paths["followers"] / "follower.json").write_text("{}")
    if mode == "leader_arm":
        (paths["leaders"] / "leader.json").write_text("{}")


def _legacy_selector() -> dict[str, str]:
    return {
        "leader_port": "/dev/leader",
        "follower_port": "/dev/follower",
        "leader_config": "leader.json",
        "follower_config": "follower.json",
    }


def _recording_settings() -> dict:
    return {
        "dataset_repo_id": "alex/demo",
        "single_task": "Pick up the block",
        "num_episodes": 7,
        "episode_time_s": 45,
        "reset_time_s": 12,
        "fps": 30,
        "video": True,
        "push_to_hub": False,
        "tags": ["demo", "stadia"],
        "private": True,
        "resume": False,
        "streaming_encoding": False,
        "cameras": {
            "front": {
                "type": "opencv",
                "camera_index": 2,
                "width": 1280,
                "height": 720,
                "fps": 30,
            }
        },
        "test_mode": False,
    }


@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("leader_arm", cfg.RobotOperation.LEADER_TELEOPERATION),
        ("stadia", cfg.RobotOperation.STADIA_TELEOPERATION),
    ],
)
def test_canonical_teleoperation_selects_saved_mode_and_configuration(
    mode: str,
    operation: cfg.RobotOperation,
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot", mode=mode)

    resolved = resolve_teleoperate_request({"robot_name": "robot"})

    assert resolved.selector_kind == SelectorKind.CANONICAL
    assert resolved.operation == operation
    assert resolved.robot_name == "robot"
    assert resolved.teleoperator_type == mode
    assert resolved.record_dict()["follower"]["port"] == "/dev/follower"


@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("leader_arm", cfg.RobotOperation.LEADER_RECORDING),
        ("stadia", cfg.RobotOperation.STADIA_RECORDING),
    ],
)
def test_canonical_recording_selects_saved_mode(
    mode: str,
    operation: cfg.RobotOperation,
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot", mode=mode)

    resolved = resolve_recording_request({"robot_name": "robot", **_recording_settings()})

    assert resolved.selector_kind == SelectorKind.CANONICAL
    assert resolved.operation == operation


def test_resolution_enforces_mode_specific_readiness(_isolated_records: dict[str, Path]) -> None:
    _save_ready_record(_isolated_records, "leader", mode="leader_arm")
    (_isolated_records["leaders"] / "leader.json").unlink()

    with pytest.raises(ControlRequestError) as leader_error:
        resolve_teleoperate_request({"robot_name": "leader"})
    assert leader_error.value.code == ControlRequestErrorCode.ROBOT_NOT_READY
    assert leader_error.value.details["operation"] == "leader_teleoperation"
    assert leader_error.value.details["issues"][0]["code"] == "leader_calibration_not_found"

    _save_ready_record(_isolated_records, "stadia", mode="stadia")
    (_isolated_records["followers"] / "follower.json").unlink()
    with pytest.raises(ControlRequestError) as stadia_error:
        resolve_recording_request({"robot_name": "stadia", **_recording_settings()})
    assert stadia_error.value.code == ControlRequestErrorCode.ROBOT_NOT_READY
    assert stadia_error.value.details["operation"] == "stadia_recording"
    assert stadia_error.value.details["issues"][0]["code"] == "follower_calibration_not_found"


def test_canonical_resolution_does_not_migrate_legacy_disk_record(
    _isolated_records: dict[str, Path],
) -> None:
    legacy = {"name": "old", **_legacy_selector(), "cameras": []}
    path = _isolated_records["robots"] / "old.json"
    original = json.dumps(legacy, indent=2)
    path.write_text(original)
    (_isolated_records["leaders"] / "leader.json").write_text("{}")
    (_isolated_records["followers"] / "follower.json").write_text("{}")

    resolved = resolve_teleoperate_request({"robot_name": "old"})

    assert resolved.teleoperator_type == "leader_arm"
    assert path.read_text() == original


def test_exact_unique_legacy_identity_resolves_leader_record(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot")

    resolved = resolve_teleoperate_request(_legacy_selector())

    assert resolved.selector_kind == SelectorKind.LEGACY
    assert resolved.robot_name == "robot"
    assert resolved.operation == cfg.RobotOperation.LEADER_TELEOPERATION
    assert resolved.legacy_handler_payload() == _legacy_selector()


def test_legacy_identity_with_no_exact_match_fails_closed(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot")
    request = _legacy_selector()
    request["follower_port"] = "/dev/not-canonical"

    with pytest.raises(ControlRequestError) as error:
        resolve_teleoperate_request(request)

    assert error.value.code == ControlRequestErrorCode.LEGACY_NO_MATCH


def test_duplicate_legacy_identity_is_ambiguous(_isolated_records: dict[str, Path]) -> None:
    _save_ready_record(_isolated_records, "alpha")
    _save_ready_record(_isolated_records, "beta")

    with pytest.raises(ControlRequestError) as error:
        resolve_teleoperate_request(_legacy_selector())

    assert error.value.code == ControlRequestErrorCode.LEGACY_AMBIGUOUS
    assert error.value.details["robot_names"] == ("alpha", "beta")


def test_malformed_sibling_blocks_legacy_uniqueness_proof(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "good")
    (_isolated_records["robots"] / "bad.json").write_text("{")

    with pytest.raises(ControlRequestError) as error:
        resolve_teleoperate_request(_legacy_selector())

    assert error.value.code == ControlRequestErrorCode.ROBOT_INVALID
    assert "could not be read as valid JSON" in error.value.details["error"]


def test_legacy_selector_cannot_start_a_stadia_record(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "stadia", mode="stadia")

    with pytest.raises(ControlRequestError) as error:
        resolve_teleoperate_request(_legacy_selector())

    assert error.value.code == ControlRequestErrorCode.LEGACY_MODE_UNSUPPORTED


def test_mixed_canonical_and_legacy_selectors_are_rejected() -> None:
    with pytest.raises(ControlRequestError) as error:
        resolve_teleoperate_request({"robot_name": "robot", **_legacy_selector()})

    assert error.value.code == ControlRequestErrorCode.MIXED_SELECTORS


def test_partial_legacy_and_unknown_fields_are_actionable() -> None:
    partial = _legacy_selector()
    del partial["leader_config"]
    with pytest.raises(ControlRequestError) as partial_error:
        resolve_teleoperate_request(partial)
    assert partial_error.value.code == ControlRequestErrorCode.MISSING_LEGACY_FIELDS
    assert partial_error.value.details["fields"] == ("leader_config",)

    with pytest.raises(ControlRequestError) as unknown_error:
        resolve_teleoperate_request({"robot_name": "robot", "leader_poort": "/dev/x"})
    assert unknown_error.value.code == ControlRequestErrorCode.UNKNOWN_FIELDS
    assert unknown_error.value.details["fields"] == ("leader_poort",)


def test_missing_invalid_unknown_and_corrupt_records_have_typed_errors(
    _isolated_records: dict[str, Path],
) -> None:
    cases = [
        ({}, ControlRequestErrorCode.MISSING_SELECTOR),
        ({"robot_name": "../bad"}, ControlRequestErrorCode.INVALID_ROBOT_NAME),
        ({"robot_name": "missing"}, ControlRequestErrorCode.ROBOT_NOT_FOUND),
    ]
    for request, code in cases:
        with pytest.raises(ControlRequestError) as error:
            resolve_teleoperate_request(request)
        assert error.value.code == code

    (_isolated_records["robots"] / "corrupt.json").write_text(
        json.dumps({"schema_version": 2, "unknown": True})
    )
    with pytest.raises(ControlRequestError) as invalid_error:
        resolve_teleoperate_request({"robot_name": "corrupt"})
    assert invalid_error.value.code == ControlRequestErrorCode.ROBOT_INVALID

    (_isolated_records["robots"] / "malformed.json").write_text("{")
    with pytest.raises(ControlRequestError) as malformed_error:
        resolve_teleoperate_request({"robot_name": "malformed"})
    assert malformed_error.value.code == ControlRequestErrorCode.ROBOT_INVALID


def test_request_data_cannot_override_or_mutate_canonical_configuration(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot")
    request = _legacy_selector()

    resolved = resolve_teleoperate_request(request)
    request["leader_port"] = "/dev/changed-after-resolution"
    exported = resolved.record_dict()
    exported["follower"]["port"] = "/dev/mutated-copy"

    assert resolved.canonical_record["leader"]["port"] == "/dev/leader"
    assert resolved.canonical_record["follower"]["port"] == "/dev/follower"
    assert resolved.legacy_handler_payload()["leader_port"] == "/dev/leader"
    assert isinstance(resolved.canonical_record, MappingProxyType)
    with pytest.raises(TypeError):
        resolved.canonical_record["name"] = "changed"


def test_recording_preserves_and_deep_freezes_session_and_camera_payload(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot")
    settings = _recording_settings()
    expected = _recording_settings()
    request = {"robot_name": "robot", **settings}

    resolved = resolve_recording_request(request)
    request["cameras"]["front"]["camera_index"] = 99
    request["tags"].append("mutated")

    assert resolved.metadata_dict() == expected
    assert resolved.metadata["tags"] == ("demo", "stadia")
    assert resolved.metadata["cameras"]["front"]["camera_index"] == 2
    with pytest.raises(TypeError):
        resolved.metadata["cameras"]["front"]["camera_index"] = 4

    legacy_payload = resolved.legacy_handler_payload()
    assert {key: legacy_payload[key] for key in _legacy_selector()} == _legacy_selector()
    assert {key: legacy_payload[key] for key in expected} == expected


def test_legacy_recording_preserves_session_payload_and_uses_canonical_robot_fields(
    _isolated_records: dict[str, Path],
) -> None:
    _save_ready_record(_isolated_records, "robot")
    request = {**_legacy_selector(), **_recording_settings()}

    resolved = resolve_recording_request(request)

    assert resolved.selector_kind == SelectorKind.LEGACY
    assert resolved.metadata_dict() == _recording_settings()
    assert resolved.legacy_handler_payload() == request
