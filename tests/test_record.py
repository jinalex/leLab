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
"""Tests for lelab.record — request schemas and handler entry points."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_recording_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.record as record

    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(record, "recording_thread", None)
    monkeypatch.setattr(record, "recording_events", None)
    monkeypatch.setattr(record, "recording_config", None)
    monkeypatch.setattr(record, "recording_start_time", None)
    monkeypatch.setattr(record, "session_end_elapsed_seconds", None)
    monkeypatch.setattr(record, "current_episode", 1)
    monkeypatch.setattr(record, "saved_episodes", 0)
    monkeypatch.setattr(record, "current_phase", "preparing")
    monkeypatch.setattr(record, "phase_start_time", None)
    monkeypatch.setattr(record, "last_recording_info", None)


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_recording_status_handler_exposes_state_fields() -> None:
    from lelab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result
    assert result["outcome"] == "idle"
    assert result["cleanup_pending"] is False


def test_handle_stop_recording_when_idle_returns_dict(tmp_lerobot_home) -> None:
    from lelab.record import handle_stop_recording

    result = handle_stop_recording()
    assert isinstance(result, dict)


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import lelab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )

    config = record.create_record_config(request)
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    assert configs["cam"].fourcc is None
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


def test_recording_worker_failure_is_retained_across_status_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record

    class InlineThread:
        def __init__(self, *, target, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(record, "create_record_config", lambda _request: object())
    monkeypatch.setattr(
        record,
        "record_with_web_events",
        lambda _config, _events: (_ for _ in ()).throw(RuntimeError("dataset loop failed")),
    )
    monkeypatch.setattr(record.threading, "Thread", InlineThread)

    result = record.handle_start_recording(
        record.RecordingRequest(
            leader_port="COM_LEADER",
            follower_port="COM_FOLLOWER",
            leader_config="leader",
            follower_config="follower",
            dataset_repo_id="user/fixture",
            single_task="fixture task",
        )
    )

    assert result["success"] is True
    first = record.handle_recording_status()
    second = record.handle_recording_status()
    assert first == second
    assert first["recording_active"] is False
    assert first["session_ended"] is True
    assert first["outcome"] == "failed"
    assert first["error"] == "dataset loop failed"
    assert first["cleanup_pending"] is False


def test_recording_worker_retains_unproven_cleanup_and_blocks_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record
    from lelab.utils.devices import DeviceCleanupError

    class InlineThread:
        def __init__(self, *, target, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.target = target

        def start(self) -> None:
            self.target()

    cleanup_error = DeviceCleanupError(
        "recording cleanup",
        ["serial port still open"],
        cleanup_proven=False,
    )
    monkeypatch.setattr(record, "create_record_config", lambda _request: object())
    monkeypatch.setattr(
        record,
        "record_with_web_events",
        lambda _config, _events: (_ for _ in ()).throw(cleanup_error),
    )
    monkeypatch.setattr(record.threading, "Thread", InlineThread)

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/fixture",
        single_task="fixture task",
    )
    assert record.handle_start_recording(request)["success"] is True

    status = record.handle_recording_status()
    assert status["outcome"] == "failed"
    assert status["cleanup_pending"] is True
    assert "serial port still open" in status["error"]

    retry = record.handle_start_recording(request)
    assert retry["success"] is False
    assert "cleanup is unproven" in retry["message"]


def test_recording_cleanup_attempts_both_devices_and_aggregates_proof() -> None:
    import lelab.record as record
    from lelab.utils.devices import DeviceCleanupError

    events: list[str] = []

    class Device:
        def __init__(self, name: str, *, releases: bool) -> None:
            self.name = name
            self.releases = releases
            self.is_connected = True

        def disconnect(self) -> None:
            events.append(self.name)
            if self.releases:
                self.is_connected = False

    follower = Device("follower", releases=False)
    leader = Device("leader", releases=True)

    with pytest.raises(DeviceCleanupError) as caught:
        record._cleanup_recording_devices(follower, leader)

    assert events == ["follower", "leader"]
    assert caught.value.cleanup_proven is False
    assert "follower" in str(caught.value)


def test_stop_recording_reports_async_cleanup_as_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.record as record

    monkeypatch.setattr(record, "recording_active", True)
    monkeypatch.setattr(record, "recording_events", {"stop_recording": False, "exit_early": False})

    result = record.handle_stop_recording()

    assert result["success"] is True
    assert result["stop_pending"] is True
    assert result["cleanup_proven"] is False
