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
"""Tests for lelab.teleoperate — request schema and status handlers."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_teleoperation_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "teleoperation_thread", None)
    monkeypatch.setattr(teleop, "current_robot", None)
    monkeypatch.setattr(teleop, "current_teleop", None)
    monkeypatch.setattr(teleop, "_teleoperation_terminal_status", None)


def test_teleoperate_request_rejects_missing_fields() -> None:
    from pydantic import ValidationError

    from lelab.teleoperate import TeleoperateRequest

    with pytest.raises(ValidationError):
        TeleoperateRequest()


def test_handle_teleoperation_status_returns_dict() -> None:
    from lelab.teleoperate import handle_teleoperation_status

    result = handle_teleoperation_status()
    assert isinstance(result, dict)


def test_handle_get_joint_positions_returns_dict_when_idle() -> None:
    from lelab.teleoperate import handle_get_joint_positions

    result = handle_get_joint_positions()
    assert isinstance(result, dict)


def test_get_joint_positions_from_robot_uses_provided_object() -> None:
    from lelab.teleoperate import get_joint_positions_from_robot
    from tests.mocks import FakeRobot

    robot = FakeRobot()
    robot.connect()
    positions = get_joint_positions_from_robot(robot)
    assert isinstance(positions, dict)


def test_get_joint_positions_maps_degrees_to_radians_without_correction() -> None:
    """Every joint maps its Present_Position (degrees around the calibration
    center) straight to radians, deg * pi/180, with no per-joint offset.
    shoulder_lift/elbow_flex used to be run through a correction table; a
    calibration that would have triggered it is supplied here to prove the
    offset is gone and they now map like every other joint."""
    import math

    from lelab.teleoperate import get_joint_positions_from_robot

    class _Cal:
        # The removed correction derived its offset from range_min/range_max.
        range_min = 0
        range_max = 4095

    class _Robot:
        calibration = {"shoulder_lift": _Cal(), "elbow_flex": _Cal()}

        def get_observation(self):
            return {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 90.0,
                "elbow_flex.pos": -45.0,
                "wrist_flex.pos": 30.0,
                "wrist_roll.pos": 12.0,
                "gripper.pos": 50.0,
            }

    positions = get_joint_positions_from_robot(_Robot())

    assert positions["Rotation"] == pytest.approx(0.0)
    assert positions["Pitch"] == pytest.approx(math.radians(90.0))
    assert positions["Elbow"] == pytest.approx(math.radians(-45.0))
    assert positions["Wrist_Pitch"] == pytest.approx(math.radians(30.0))
    assert positions["Wrist_Roll"] == pytest.approx(math.radians(12.0))
    assert positions["Jaw"] == pytest.approx(math.radians(50.0))


def test_start_teleoperation_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that fails to connect must make the start handler return
    success=False (so the UI surfaces the error and doesn't navigate to an
    empty teleop screen) and reset state so a retry isn't blocked. Previously
    the connect ran in a worker thread and the handler always claimed success.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _Bus:
        def connect(self) -> None:
            raise RuntimeError("serial port unavailable")

    class _Device:
        def __init__(self, config) -> None:
            self.bus = _Bus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    monkeypatch.setattr(teleop, "SO101Follower", _Device)
    monkeypatch.setattr(teleop, "SO101Leader", _Device)

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    # The message must name the arm that failed (the follower connects first).
    assert "follower" in result["message"].lower()
    assert "COM_FOLLOWER" in result["message"]
    # State must be reset so the next attempt isn't blocked by the mutex.
    assert teleop.teleoperation_active is False


def test_start_teleoperation_disconnects_follower_when_leader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-connect path: if the follower connects but the leader then
    fails, the follower must be disconnected so its serial port is released.
    """
    import lelab.teleoperate as teleop

    monkeypatch.setattr(teleop, "teleoperation_active", False)
    monkeypatch.setattr(teleop, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    class _OkBus:
        def connect(self) -> None:
            pass

    class _FailingBus:
        def connect(self) -> None:
            raise RuntimeError("leader offline")

    class _Follower:
        def __init__(self, config) -> None:
            self.bus = _OkBus()
            self.cameras: dict = {}
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class _Leader:
        def __init__(self, config) -> None:
            self.bus = _FailingBus()
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    created: dict = {}
    monkeypatch.setattr(
        teleop, "SO101Follower", lambda config: created.setdefault("follower", _Follower(config))
    )
    monkeypatch.setattr(teleop, "SO101Leader", lambda config: created.setdefault("leader", _Leader(config)))

    request = teleop.TeleoperateRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
    )
    result = teleop.handle_start_teleoperation(request)

    assert result["success"] is False
    assert "leader" in result["message"].lower()
    # The already-connected follower must have been cleaned up.
    assert created["follower"].disconnected is True
    assert teleop.teleoperation_active is False


def test_worker_natural_failure_is_retained_across_status_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import lelab.teleoperate as teleop

    class Bus:
        def __init__(self) -> None:
            self.is_connected = False

        def connect(self) -> None:
            self.is_connected = True

        def write_calibration(self, calibration) -> None:  # type: ignore[no-untyped-def]
            pass

    class Follower:
        def __init__(self, _config) -> None:  # type: ignore[no-untyped-def]
            self.bus = Bus()
            self.cameras: dict = {}
            self.calibration = {}

        @property
        def is_connected(self) -> bool:
            return self.bus.is_connected

        def configure(self) -> None:
            pass

        def disconnect(self) -> None:
            self.bus.is_connected = False

        def send_action(self, _action) -> None:  # type: ignore[no-untyped-def]
            pass

    class Leader(Follower):
        def get_action(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("leader loop failed")

    class InlineThread:
        def __init__(self, *, target, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.target = target
            self.alive = False

        def start(self) -> None:
            self.alive = True
            try:
                self.target()
            finally:
                self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    monkeypatch.setattr(teleop, "setup_calibration_files", lambda *_args: ("leader", "follower"))
    monkeypatch.setattr(teleop, "SO101FollowerConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(teleop, "SO101LeaderConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(teleop, "SO101Follower", Follower)
    monkeypatch.setattr(teleop, "SO101Leader", Leader)
    monkeypatch.setattr(teleop.threading, "Thread", InlineThread)

    result = teleop.handle_start_teleoperation(
        teleop.TeleoperateRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
        )
    )

    assert result["success"] is True
    first = teleop.handle_teleoperation_status()
    second = teleop.handle_teleoperation_status()
    assert first == second
    assert first["teleoperation_active"] is False
    assert first["exited"] is True
    assert first["outcome"] == "failed"
    assert first["error"] == "leader loop failed"
    assert first["cleanup_pending"] is False


def test_worker_unproven_cleanup_is_retained_and_blocks_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.teleoperate as teleop

    class Device:
        is_connected = True

        def disconnect(self) -> None:
            pass

    robot = Device()
    teleoperator = Device()
    robot_error, robot_proven = teleop._safe_disconnect(robot)
    teleop_error, teleop_proven = teleop._safe_disconnect(teleoperator)
    teleop._teleoperation_terminal_status = {
        "exited": True,
        "outcome": "failed",
        "error": "; ".join(error for error in (robot_error, teleop_error) if error),
        "cleanup_pending": not (robot_proven and teleop_proven),
    }
    teleop.current_robot = robot
    teleop.current_teleop = teleoperator

    status = teleop.handle_teleoperation_status()
    assert status["cleanup_pending"] is True
    assert teleop.current_robot is robot
    assert teleop.current_teleop is teleoperator

    monkeypatch.setattr(teleop, "setup_calibration_files", lambda *_args: ("leader", "follower"))
    result = teleop.handle_start_teleoperation(
        teleop.TeleoperateRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
        )
    )
    assert result["success"] is False
    assert "cleanup is unproven" in result["message"]


def test_stop_timeout_keeps_worker_reference_and_returns_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.teleoperate as teleop

    class LingeringThread:
        def __init__(self) -> None:
            self.joins: list[float | None] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            self.joins.append(timeout)

    worker = LingeringThread()
    monkeypatch.setattr(teleop, "teleoperation_active", True)
    monkeypatch.setattr(teleop, "teleoperation_thread", worker)

    result = teleop.handle_stop_teleoperation()

    assert result == {
        "success": False,
        "message": "Teleoperation stop requested, but the worker is still shutting down",
        "stop_pending": True,
        "cleanup_proven": False,
    }
    assert worker.joins == [5.0]
    assert teleop.teleoperation_thread is worker
    status = teleop.handle_teleoperation_status()
    assert status["outcome"] == "stopping"
    assert status["cleanup_pending"] is True
