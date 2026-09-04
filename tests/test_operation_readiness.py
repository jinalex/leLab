"""Pure operation-specific RobotRecordV2 readiness tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_calibrations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    from lelab.utils import config as cfg

    leader = tmp_path / "leader"
    follower = tmp_path / "follower"
    leader.mkdir()
    follower.mkdir()
    monkeypatch.setattr(cfg, "LEADER_CONFIG_PATH", str(leader))
    monkeypatch.setattr(cfg, "FOLLOWER_CONFIG_PATH", str(follower))
    return leader, follower


def _record(mode: str = "leader_arm"):
    from lelab.utils.config import DeviceRecord, RobotRecordV2

    return RobotRecordV2(
        name="robot",
        teleoperator_type=mode,
        leader=DeviceRecord(port="/dev/leader", calibration="leader.json"),
        follower=DeviceRecord(port="/dev/follower", calibration="follower.json"),
    )


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_calibration_readiness_does_not_require_an_existing_file() -> None:
    from lelab.utils.config import RobotOperation, evaluate_robot_readiness

    record = _record()

    assert evaluate_robot_readiness(record, RobotOperation.FOLLOWER_CALIBRATION).ready
    assert evaluate_robot_readiness(record, RobotOperation.LEADER_CALIBRATION).ready


def test_calibration_readiness_requires_port_and_destination_name() -> None:
    from lelab.utils.config import DeviceRecord, RobotOperation, evaluate_robot_readiness

    record = _record()
    record.follower = DeviceRecord()

    result = evaluate_robot_readiness(record, RobotOperation.FOLLOWER_CALIBRATION)

    assert not result.ready
    assert _codes(result) == {
        "follower_port_missing",
        "follower_calibration_missing",
    }


def test_leader_live_and_recording_require_both_existing_calibrations(
    _isolated_calibrations: tuple[Path, Path],
) -> None:
    from lelab.utils.config import RobotOperation, evaluate_robot_readiness

    leader_dir, follower_dir = _isolated_calibrations
    record = _record()
    operations = (
        RobotOperation.LEADER_TELEOPERATION,
        RobotOperation.LEADER_RECORDING,
    )

    for operation in operations:
        assert _codes(evaluate_robot_readiness(record, operation)) == {
            "leader_calibration_not_found",
            "follower_calibration_not_found",
        }

    (leader_dir / "leader.json").write_text("{}")
    (follower_dir / "follower.json").write_text("{}")
    for operation in operations:
        assert evaluate_robot_readiness(record, operation).ready


def test_stadia_live_and_recording_are_follower_only(
    _isolated_calibrations: tuple[Path, Path],
) -> None:
    from lelab.utils.config import RobotOperation, evaluate_robot_readiness

    _, follower_dir = _isolated_calibrations
    record = _record("stadia")
    record.leader = None
    (follower_dir / "follower.json").write_text("{}")

    assert evaluate_robot_readiness(record, RobotOperation.STADIA_TELEOPERATION).ready
    assert evaluate_robot_readiness(record, RobotOperation.STADIA_RECORDING).ready


@pytest.mark.parametrize("mode", ["leader_arm", "stadia"])
@pytest.mark.parametrize("operation", ["inference", "replay"])
def test_inference_and_replay_are_follower_only_in_every_mode(
    mode: str,
    operation: str,
    _isolated_calibrations: tuple[Path, Path],
) -> None:
    from lelab.utils.config import evaluate_robot_readiness

    _, follower_dir = _isolated_calibrations
    record = _record(mode)
    record.leader = None
    (follower_dir / "follower.json").write_text("{}")

    assert evaluate_robot_readiness(record, operation).ready


def test_controller_check_is_stadia_only_and_independent_of_follower() -> None:
    from lelab.utils.config import DeviceRecord, RobotOperation, evaluate_robot_readiness

    stadia = _record("stadia")
    stadia.follower = DeviceRecord()
    stadia.leader = None

    assert evaluate_robot_readiness(stadia, RobotOperation.CONTROLLER_CHECK).ready

    leader = _record("leader_arm")
    result = evaluate_robot_readiness(leader, RobotOperation.CONTROLLER_CHECK)
    assert not result.ready
    assert _codes(result) == {"wrong_teleoperator_type"}


@pytest.mark.parametrize(
    ("mode", "operation"),
    [
        ("stadia", "leader_calibration"),
        ("stadia", "leader_teleoperation"),
        ("stadia", "leader_recording"),
        ("leader_arm", "stadia_teleoperation"),
        ("leader_arm", "stadia_recording"),
    ],
)
def test_mode_specific_operations_reject_the_wrong_teleoperator_type(
    mode: str,
    operation: str,
) -> None:
    from lelab.utils.config import evaluate_robot_readiness

    result = evaluate_robot_readiness(_record(mode), operation)

    assert not result.ready
    assert _codes(result) == {"wrong_teleoperator_type"}


def test_missing_and_missing_file_issues_are_distinct() -> None:
    from lelab.utils.config import DeviceRecord, RobotOperation, evaluate_robot_readiness

    missing_name = _record("stadia")
    missing_name.follower = DeviceRecord(port="/dev/follower")
    result = evaluate_robot_readiness(missing_name, RobotOperation.STADIA_TELEOPERATION)
    assert _codes(result) == {"follower_calibration_missing"}

    missing_file = _record("stadia")
    result = evaluate_robot_readiness(missing_file, RobotOperation.STADIA_TELEOPERATION)
    assert _codes(result) == {"follower_calibration_not_found"}


def test_all_readiness_has_the_normalized_nine_operation_shape() -> None:
    from lelab.utils.config import RobotOperation, evaluate_all_robot_readiness

    readiness = evaluate_all_robot_readiness(_record("stadia"))

    assert set(readiness) == {operation.value for operation in RobotOperation}
    controller = readiness["controller_check"].model_dump(mode="json")
    assert controller == {
        "operation": "controller_check",
        "ready": True,
        "issues": [],
    }


def test_readiness_accepts_a_normalized_mapping(
    _isolated_calibrations: tuple[Path, Path],
) -> None:
    from lelab.utils.config import evaluate_robot_readiness

    _, follower_dir = _isolated_calibrations
    (follower_dir / "follower.json").write_text("{}")
    raw = _record("stadia").model_dump(mode="json")

    assert evaluate_robot_readiness(raw, "stadia_teleoperation").ready
