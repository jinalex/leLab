"""Strict RobotRecordV2 normalization and persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_robot_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from lelab.utils import config as cfg

    robots = tmp_path / "robots"
    robots.mkdir()
    monkeypatch.setattr(cfg, "ROBOTS_PATH", str(robots))
    return robots


def _legacy_record() -> dict:
    return {
        "name": "legacy",
        "leader_port": "/dev/leader",
        "follower_port": "/dev/follower",
        "leader_config": "leader.json",
        "follower_config": "follower.json",
        "cameras": [
            {
                "id": "camera_1",
                "name": "front",
                "type": "opencv",
                "camera_index": 0,
                "device_id": "device-1",
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        ],
    }


def test_legacy_load_normalizes_in_memory_without_writing(_isolated_robot_records: Path) -> None:
    from lelab.utils import config as cfg

    path = _isolated_robot_records / "legacy.json"
    original = json.dumps(_legacy_record(), indent=2)
    path.write_text(original)

    record = cfg.get_robot_record_v2("legacy")

    assert record is not None
    assert record.schema_version == 2
    assert record.teleoperator_type == "leader_arm"
    assert record.leader is not None
    assert record.leader.port == "/dev/leader"
    assert record.follower.calibration == "follower.json"
    assert record.stadia.deadzone == 0.15
    assert path.read_text() == original


def test_legacy_camera_without_index_uses_existing_recorder_default_without_writing(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    legacy = _legacy_record()
    del legacy["cameras"][0]["camera_index"]
    path = _isolated_robot_records / "legacy.json"
    original = json.dumps(legacy, indent=2)
    path.write_text(original)

    record = cfg.get_robot_record_v2("legacy")

    assert record is not None
    assert record.cameras[0].camera_index == 0
    assert path.read_text() == original


def test_legacy_get_keeps_the_existing_flat_route_projection(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    path = _isolated_robot_records / "legacy.json"
    path.write_text(json.dumps(_legacy_record()))

    assert cfg.get_robot_record("legacy") == _legacy_record()


def test_explicit_legacy_save_migrates_disk_to_v2_and_preserves_legacy_api(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    path = _isolated_robot_records / "legacy.json"
    path.write_text(json.dumps(_legacy_record()))

    assert cfg.save_robot_record(
        "legacy",
        {"follower_port": "/dev/new-follower"},
        allow_create=False,
    )

    persisted = json.loads(path.read_text())
    assert persisted["schema_version"] == 2
    assert "leader_port" not in persisted
    assert persisted["leader"] == {
        "port": "/dev/leader",
        "calibration": "leader.json",
    }
    assert persisted["follower"]["port"] == "/dev/new-follower"
    assert cfg.get_robot_record("legacy")["follower_port"] == "/dev/new-follower"


def test_switching_to_stadia_preserves_the_unused_leader(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    (_isolated_robot_records / "legacy.json").write_text(json.dumps(_legacy_record()))

    record = cfg.save_robot_record_v2(
        "legacy",
        {"schema_version": 2, "teleoperator_type": "stadia"},
        allow_create=False,
    )

    assert record is not None
    assert record.teleoperator_type == "stadia"
    assert record.leader is not None
    assert record.leader.port == "/dev/leader"


def test_only_explicit_null_clears_the_unused_leader(_isolated_robot_records: Path) -> None:
    from lelab.utils import config as cfg

    (_isolated_robot_records / "legacy.json").write_text(json.dumps(_legacy_record()))

    record = cfg.save_robot_record_v2(
        "legacy",
        {"schema_version": 2, "teleoperator_type": "stadia", "leader": None},
        allow_create=False,
    )

    assert record is not None
    assert record.leader is None


def test_legacy_stadia_travel_fields_load_without_writing_and_strip_on_save(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    path = _isolated_robot_records / "stadia.json"
    legacy_v2 = {
        "schema_version": 2,
        "name": "stadia",
        "teleoperator_type": "stadia",
        "follower": {"port": "/dev/follower", "calibration": "follower.json"},
        "leader": None,
        "stadia": {
            "guid": "stadia-guid",
            "deadzone": 0.15,
            "max_step_per_tick": 0.35,
            "arm_startup_travel_degrees": 30.0,
            "gripper_startup_travel_percentage_points": 40.0,
        },
        "cameras": [],
    }
    original = json.dumps(legacy_v2, indent=2)
    path.write_text(original)

    loaded = cfg.get_robot_record_v2("stadia")

    assert loaded is not None
    assert loaded.teleoperator_type == "stadia"
    assert loaded.stadia.model_dump(mode="json") == {
        "guid": "stadia-guid",
        "deadzone": 0.15,
        "max_step_per_tick": 0.35,
    }
    assert path.read_text() == original

    saved = cfg.save_robot_record_v2(
        "stadia",
        {"schema_version": 2, "stadia": {"deadzone": 0.2}},
        allow_create=False,
    )

    assert saved is not None
    persisted_stadia = json.loads(path.read_text())["stadia"]
    assert persisted_stadia == {
        "guid": "stadia-guid",
        "deadzone": 0.2,
        "max_step_per_tick": 0.35,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_stadia_numeric_settings_reject_nonfinite_values(value: float) -> None:
    from pydantic import ValidationError

    from lelab.utils.config import StadiaConfig

    with pytest.raises(ValidationError):
        StadiaConfig(deadzone=value)


@pytest.mark.parametrize("value", [True, False, "0.15", None])
def test_stadia_numeric_settings_reject_coerced_non_numbers(value: object) -> None:
    from pydantic import ValidationError

    from lelab.utils.config import StadiaConfig

    with pytest.raises(ValidationError):
        StadiaConfig.model_validate({"deadzone": value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deadzone", -0.01),
        ("deadzone", 1.0),
        ("max_step_per_tick", 0.0),
        ("max_step_per_tick", 0.36),
        ("arm_startup_travel_degrees", 45.01),
        ("gripper_startup_travel_percentage_points", 45.01),
    ],
)
def test_stadia_numeric_settings_enforce_ranges(field: str, value: float) -> None:
    from pydantic import ValidationError

    from lelab.utils.config import StadiaConfig

    with pytest.raises(ValidationError):
        StadiaConfig.model_validate({field: value})


def test_unknown_and_mixed_v2_fields_fail_without_changing_disk(
    _isolated_robot_records: Path,
) -> None:
    from lelab.utils import config as cfg

    path = _isolated_robot_records / "robot.json"
    cfg.save_robot_record("robot", {"leader_port": "/dev/leader"})
    before = path.read_bytes()

    with pytest.raises(cfg.RobotRecordValidationError, match="unknown V2"):
        cfg.save_robot_record_v2(
            "robot",
            {"schema_version": 2, "mystery": True},
            allow_create=False,
        )
    assert path.read_bytes() == before

    with pytest.raises(cfg.RobotRecordValidationError, match="cannot mix"):
        cfg.save_robot_record_v2(
            "robot",
            {
                "schema_version": 2,
                "leader_port": "/dev/legacy",
                "follower": {"port": "/dev/v2"},
            },
            allow_create=False,
        )
    assert path.read_bytes() == before


def test_nested_unknown_fields_and_mismatched_body_name_fail() -> None:
    from lelab.utils import config as cfg

    with pytest.raises(cfg.RobotRecordValidationError, match="extra_forbidden"):
        cfg.normalize_robot_record(
            "robot",
            {
                "schema_version": 2,
                "name": "robot",
                "follower": {"port": "/dev/follower", "unknown": 1},
            },
        )

    with pytest.raises(cfg.RobotRecordValidationError, match="body name"):
        cfg.save_robot_record_v2(
            "robot",
            {"schema_version": 2, "name": "someone-else"},
        )


def test_unknown_legacy_record_field_is_not_silently_discarded() -> None:
    from lelab.utils import config as cfg

    with pytest.raises(cfg.RobotRecordValidationError, match="unknown legacy"):
        cfg.normalize_robot_record("robot", {"leader_port": "/dev/a", "mystery": 1})


def test_v2_deep_patch_preserves_unmentioned_fields(_isolated_robot_records: Path) -> None:
    from lelab.utils import config as cfg

    first = cfg.save_robot_record_v2(
        "robot",
        {
            "schema_version": 2,
            "teleoperator_type": "stadia",
            "follower": {
                "port": "/dev/follower",
                "calibration": "follower.json",
            },
            "stadia": {"guid": "guid-1", "deadzone": 0.2},
        },
    )
    assert first is not None

    updated = cfg.save_robot_record_v2(
        "robot",
        {"schema_version": 2, "stadia": {"deadzone": 0.1}},
        allow_create=False,
    )

    assert updated is not None
    assert updated.follower.port == "/dev/follower"
    assert updated.stadia.guid == "guid-1"
    assert updated.stadia.deadzone == 0.1
