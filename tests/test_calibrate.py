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
"""Tests for lelab.calibrate — manager initial state and request schema."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "pos,expected",
    [
        (-5, False),  # negative — bad frame
        (0, False),  # lower bound is exclusive (old check: pos > 0)
        (1, True),
        (100, True),
        (2000, True),
        (4095, True),  # encoder max
        (4999, True),
        (5000, False),  # upper bound is exclusive (old check: pos < 5000)
        (6000, False),  # extreme — bad frame
    ],
)
def test_is_valid_position_boundaries(pos, expected) -> None:
    """Pins the plausible-encoder-range filter that replaced three duplicated
    inline `pos > 0 and pos < 5000` checks. Boundaries are exclusive on both ends."""
    from lelab.calibrate import _is_valid_position

    assert _is_valid_position(pos) is expected


def test_calibration_status_defaults_to_idle() -> None:
    from lelab.calibrate import CalibrationStatus

    status = CalibrationStatus()
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.step == 0
    assert status.cleanup_pending is False


def test_calibration_request_dataclass_round_trip() -> None:
    from lelab.calibrate import CalibrationRequest

    req = CalibrationRequest(
        device_type="teleop",
        port="/dev/ttyUSB0",
        config_file="my_calib",
    )
    assert req.device_type == "teleop"
    assert req.port == "/dev/ttyUSB0"
    assert req.config_file == "my_calib"
    assert req.robot_name is None


def test_calibration_manager_starts_idle() -> None:
    from lelab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    assert mgr.status.calibration_active is False
    assert mgr.status.status == "idle"
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_get_status_when_idle_returns_status_object() -> None:
    from lelab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    s = mgr.get_status()
    assert isinstance(s, CalibrationStatus)
    assert s.status == "idle"


def test_get_status_is_a_passive_deep_snapshot() -> None:
    """HTTP polling must neither read the serial bus nor expose mutable state."""
    from lelab.calibrate import CalibrationManager

    class Bus:
        def sync_read(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("status polling must not read the bus")

    class Device:
        bus = Bus()
        is_connected = True

    mgr = CalibrationManager()
    mgr.device = Device()
    mgr._update_status(
        calibration_active=True,
        status="recording",
        current_positions={"joint": 100},
        recorded_ranges={"joint": {"min": 90, "max": 110, "current": 100}},
    )

    snapshot = mgr.get_status()
    snapshot.current_positions["joint"] = 999
    snapshot.recorded_ranges["joint"]["max"] = 999

    unchanged = mgr.get_status()
    assert unchanged.current_positions == {"joint": 100}
    assert unchanged.recorded_ranges == {"joint": {"min": 90, "max": 110, "current": 100}}


def test_range_worker_publishes_positions_and_ranges_from_its_20hz_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.calibrate import CalibrationManager

    class Bus:
        def __init__(self) -> None:
            self.reads = iter(({"joint": 100}, {"joint": 250}))

        def sync_read(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return next(self.reads)

    class Device:
        bus = Bus()

    class OneIterationEvent:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks > 1

        def clear(self) -> None:
            pass

    mgr = CalibrationManager()
    mgr.device = Device()
    mgr._step_complete = OneIterationEvent()
    monkeypatch.setattr("lelab.calibrate.time.sleep", lambda _seconds: None)

    mgr._step_range_recording()

    snapshot = mgr.get_status()
    assert snapshot.current_positions == {"joint": 250}
    assert snapshot.recorded_ranges == {"joint": {"min": 100, "max": 250, "current": 250}}


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate already running

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_cleanup_device_force_releases_and_clears_when_disconnect_fails() -> None:
    """A proven force-close clears the handle while retaining teardown error."""
    from lelab.calibrate import CalibrationManager

    class PortHandler:
        def __init__(self) -> None:
            self.closed = False

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.closed = True

    class Device:
        def __init__(self) -> None:
            self.bus = type("Bus", (), {"port_handler": PortHandler()})()

        def disconnect(self) -> None:
            raise RuntimeError("Failed to write 'Torque_Enable' on id_=6")

        @property
        def is_connected(self) -> bool:
            return not self.bus.port_handler.closed

    mgr = CalibrationManager()
    device = Device()
    mgr.device = device

    assert mgr._cleanup_device() is True

    assert device.bus.port_handler.closed is True  # force-released despite failure
    assert mgr.device is None  # handle cleared so a new calibration can start
    assert "Torque_Enable" in (mgr._cleanup_error or "")


def test_stop_timeout_never_disconnects_worker_owned_device() -> None:
    from lelab.calibrate import CalibrationManager

    class Device:
        def __init__(self) -> None:
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1

    class LingeringWorker:
        def __init__(self) -> None:
            self.join_calls: list[float | None] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            self.join_calls.append(timeout)

    mgr = CalibrationManager()
    device = Device()
    worker = LingeringWorker()
    mgr.device = device
    mgr.calibration_thread = worker
    mgr._update_status(calibration_active=True, status="recording")

    result = mgr.stop_calibration_process()

    assert result == {
        "success": False,
        "message": "Calibration stop requested, but the worker is still shutting down",
        "stop_pending": True,
        "cleanup_proven": False,
    }
    assert worker.join_calls == [5.0]
    assert device.disconnect_calls == 0
    assert mgr.device is device
    assert mgr.get_status().calibration_active is True


def test_stop_join_error_never_disconnects_worker_owned_device() -> None:
    from lelab.calibrate import CalibrationManager

    class Device:
        def __init__(self) -> None:
            self.disconnect_calls = 0

        def disconnect(self) -> None:
            self.disconnect_calls += 1

    class BrokenWorker:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("join failed")

    mgr = CalibrationManager()
    device = Device()
    mgr.device = device
    mgr.calibration_thread = BrokenWorker()
    mgr._update_status(calibration_active=True, status="recording")

    result = mgr.stop_calibration_process()

    assert result == {
        "success": False,
        "message": "join failed",
        "stop_pending": True,
        "cleanup_proven": False,
    }
    assert device.disconnect_calls == 0
    assert mgr.device is device


def test_worker_natural_error_is_retained_across_status_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    class Device:
        def __init__(self) -> None:
            self._connected = False

        @property
        def is_connected(self) -> bool:
            return self._connected

        def connect(self, *, calibrate: bool) -> None:
            self._connected = True
            raise RuntimeError("calibration bus failed")

        def disconnect(self) -> None:
            self._connected = False

    mgr = CalibrationManager()
    mgr._update_status(calibration_active=True, status="connecting")
    monkeypatch.setattr("lelab.calibrate.make_teleoperator_from_config", lambda _config: Device())

    mgr._calibration_worker(CalibrationRequest(device_type="teleop", port="COM_FAKE", config_file="fixture"))

    first = mgr.get_status()
    second = mgr.get_status()
    assert first.status == second.status == "error"
    assert first.error == second.error == "calibration bus failed"
    assert first.calibration_active is second.calibration_active is False
    assert first.cleanup_pending is second.cleanup_pending is False


@pytest.mark.parametrize("postcondition", [True, None])
def test_unproven_worker_cleanup_retains_device_and_active_ownership(postcondition: object) -> None:
    from lelab.calibrate import CalibrationManager

    class Device:
        is_connected = postcondition

        def disconnect(self) -> None:
            pass

    mgr = CalibrationManager()
    device = Device()
    mgr.device = device
    mgr._update_status(calibration_active=True, status="recording")

    mgr._cleanup_and_finish("done")

    status = mgr.get_status()
    assert mgr.device is device
    assert status.calibration_active is True
    assert status.status == "error"
    assert status.cleanup_pending is True
    assert "cleanup" in (status.error or "").lower()
