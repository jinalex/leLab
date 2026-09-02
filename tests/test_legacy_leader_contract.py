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
"""Compatibility oracles for the pre-Stadia leader-arm control paths.

These tests intentionally exercise only fakes.  They pin observable request,
construction, startup-order, control-loop, and recording-loop behavior without
opening a serial port, camera, controller, or calibration file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def test_current_browser_request_shapes_and_recording_defaults_are_accepted() -> None:
    """The v1 frontend payloads remain mandatory compatibility inputs."""
    from lelab.record import RecordingRequest
    from lelab.teleoperate import TeleoperateRequest

    teleoperate_payload = {
        "leader_port": "/dev/leader",
        "follower_port": "/dev/follower",
        "leader_config": "leader-alpha",
        "follower_config": "follower-alpha",
    }
    assert TeleoperateRequest.model_validate(teleoperate_payload).model_dump() == teleoperate_payload

    recording_payload = {
        "leader_port": "/dev/leader",
        "follower_port": "/dev/follower",
        "leader_config": "leader-alpha",
        "follower_config": "follower-alpha",
        "dataset_repo_id": "alex/legacy-demo",
        "single_task": "pick up the cube",
        "num_episodes": 3,
        "episode_time_s": 12,
        "reset_time_s": 4,
        "fps": 30,
        "video": True,
        "push_to_hub": False,
        "resume": False,
        "streaming_encoding": True,
        "cameras": {
            "wrist": {
                "type": "opencv",
                "camera_index": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        },
    }
    request = RecordingRequest.model_validate(recording_payload)

    for field, value in recording_payload.items():
        assert getattr(request, field) == value
    assert request.tags == []
    assert request.private is False
    assert request.test_mode is False

    minimal_request = RecordingRequest(
        leader_port="leader",
        follower_port="follower",
        leader_config="leader-config",
        follower_config="follower-config",
        dataset_repo_id="alex/minimal",
        single_task="move the object",
    )
    assert minimal_request.num_episodes == 5
    assert minimal_request.episode_time_s == 30
    assert minimal_request.reset_time_s == 10
    assert minimal_request.fps == 30
    assert minimal_request.video is True
    assert minimal_request.push_to_hub is False
    assert minimal_request.resume is False
    assert minimal_request.streaming_encoding is True
    assert minimal_request.cameras == {}


def test_record_config_preserves_legacy_calibration_constructors_and_dataset_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record

    calls: list[tuple[str, Any]] = []

    def setup_calibration(leader: str, follower: str) -> tuple[str, str]:
        calls.append(("setup_calibration_files", (leader, follower)))
        return "leader-id", "follower-id"

    def constructor(name: str):
        def construct(**kwargs: Any) -> SimpleNamespace:
            calls.append((name, kwargs))
            return SimpleNamespace(**kwargs)

        return construct

    camera_configs = {"wrist": object()}
    monkeypatch.setattr(record, "setup_calibration_files", setup_calibration)
    monkeypatch.setattr(record, "_platform_backend", lambda: "backend")
    monkeypatch.setattr(record, "_build_camera_configs", lambda cameras, backend: camera_configs)
    monkeypatch.setattr(record, "SO101FollowerConfig", constructor("follower_config"))
    monkeypatch.setattr(record, "SO101LeaderConfig", constructor("leader_config"))
    monkeypatch.setattr(record, "DatasetRecordConfig", constructor("dataset_config"))
    monkeypatch.setattr(record, "RecordConfig", constructor("record_config"))

    request = record.RecordingRequest(
        leader_port="LEADER_PORT",
        follower_port="FOLLOWER_PORT",
        leader_config="leader-file.json",
        follower_config="follower-file.json",
        dataset_repo_id="alex/legacy",
        single_task="stack blocks",
        num_episodes=7,
        episode_time_s=11,
        reset_time_s=5,
        fps=24,
        video=False,
        push_to_hub=False,
        private=True,
        resume=True,
        streaming_encoding=False,
        cameras={"wrist": {"type": "opencv", "camera_index": 0}},
    )

    cfg = record.create_record_config(request)

    assert [name for name, _ in calls] == [
        "setup_calibration_files",
        "follower_config",
        "leader_config",
        "dataset_config",
        "record_config",
    ]
    assert calls[0][1] == ("leader-file.json", "follower-file.json")
    follower_kwargs = calls[1][1]
    leader_kwargs = calls[2][1]
    assert follower_kwargs == {
        "port": "FOLLOWER_PORT",
        "id": "follower-id",
        "cameras": camera_configs,
    }
    assert leader_kwargs == {"port": "LEADER_PORT", "id": "leader-id"}
    assert "max_relative_target" not in follower_kwargs
    assert "max_relative_target" not in leader_kwargs

    dataset_kwargs = calls[3][1]
    assert dataset_kwargs == {
        "repo_id": "alex/legacy",
        "single_task": "stack blocks",
        "num_episodes": 7,
        "episode_time_s": 11,
        "reset_time_s": 5,
        "fps": 24,
        "video": False,
        "push_to_hub": False,
        "tags": None,
        "private": True,
        "streaming_encoding": False,
    }
    assert calls[4][1] == {
        "robot": cfg.robot,
        "teleop": cfg.teleop,
        "dataset": cfg.dataset,
        "resume": True,
        "display_data": False,
        "play_sounds": False,
    }


def test_live_leader_startup_and_loop_keep_legacy_order_and_ignore_sent_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record
    import lelab.rollout as rollout
    import lelab.teleoperate as teleoperate

    events: list[tuple[Any, ...]] = []
    action = {"shoulder_pan.pos": 12.5}

    class _IgnoredSendResult:
        """A marker that is safe only if the live loop does not inspect it."""

        def __bool__(self) -> bool:
            raise AssertionError("SO101Follower.send_action return value must remain ignored")

    class _Bus:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        def connect(self) -> None:
            events.append((f"{self.owner}.bus.connect",))

        def write_calibration(self, calibration: object) -> None:
            events.append((f"{self.owner}.bus.write_calibration", calibration))

    class _Camera:
        def connect(self) -> None:
            events.append(("follower.camera.connect",))

    class _Follower:
        def __init__(self, config: object) -> None:
            events.append(("SO101Follower", config))
            self.bus = _Bus("follower")
            self.calibration = "follower-calibration"
            self.cameras = {"wrist": _Camera()}

        def configure(self) -> None:
            events.append(("follower.configure",))

        def send_action(self, sent_action: dict[str, float]) -> _IgnoredSendResult:
            events.append(("follower.send_action", sent_action))
            return _IgnoredSendResult()

        def disconnect(self) -> None:
            events.append(("follower.disconnect",))

    class _Leader:
        def __init__(self, config: object) -> None:
            events.append(("SO101Leader", config))
            self.bus = _Bus("leader")
            self.calibration = "leader-calibration"

        def configure(self) -> None:
            events.append(("leader.configure",))

        def get_action(self) -> dict[str, float]:
            events.append(("leader.get_action",))
            return action

        def disconnect(self) -> None:
            events.append(("leader.disconnect",))

    class _InlineThread:
        def __init__(self, *, target, name: str, daemon: bool) -> None:
            events.append(("thread.create", name, daemon))
            self.target = target

        def start(self) -> None:
            events.append(("thread.start",))
            self.target()

        def is_alive(self) -> bool:
            return False

    def follower_config(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        events.append(("SO101FollowerConfig", kwargs))
        return "follower-config", kwargs

    def leader_config(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        events.append(("SO101LeaderConfig", kwargs))
        return "leader-config", kwargs

    def setup_calibration(leader: str, follower: str) -> tuple[str, str]:
        events.append(("setup_calibration_files", leader, follower))
        return "leader-id", "follower-id"

    def sleep(seconds: float) -> None:
        events.append(("sleep", seconds))
        teleoperate.teleoperation_active = False

    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_thread", None)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)
    monkeypatch.setattr(teleoperate, "setup_calibration_files", setup_calibration)
    monkeypatch.setattr(teleoperate, "SO101FollowerConfig", follower_config)
    monkeypatch.setattr(teleoperate, "SO101LeaderConfig", leader_config)
    monkeypatch.setattr(teleoperate, "SO101Follower", _Follower)
    monkeypatch.setattr(teleoperate, "SO101Leader", _Leader)
    monkeypatch.setattr(teleoperate.threading, "Thread", _InlineThread)
    monkeypatch.setattr(teleoperate.time, "time", lambda: 0.0)
    monkeypatch.setattr(teleoperate.time, "sleep", sleep)
    monkeypatch.setattr(
        teleoperate,
        "_safe_disconnect",
        lambda device: device.disconnect() if device is not None else None,
    )

    request = teleoperate.TeleoperateRequest(
        leader_port="LEADER_PORT",
        follower_port="FOLLOWER_PORT",
        leader_config="leader-file.json",
        follower_config="follower-file.json",
    )
    result = teleoperate.handle_start_teleoperation(request)

    assert result == {
        "success": True,
        "message": "Teleoperation started successfully",
        "leader_port": "LEADER_PORT",
        "follower_port": "FOLLOWER_PORT",
    }
    assert events == [
        ("setup_calibration_files", "leader-file.json", "follower-file.json"),
        ("SO101FollowerConfig", {"port": "FOLLOWER_PORT", "id": "follower-id"}),
        ("SO101LeaderConfig", {"port": "LEADER_PORT", "id": "leader-id"}),
        ("SO101Follower", ("follower-config", {"port": "FOLLOWER_PORT", "id": "follower-id"})),
        ("SO101Leader", ("leader-config", {"port": "LEADER_PORT", "id": "leader-id"})),
        ("follower.bus.connect",),
        ("leader.bus.connect",),
        ("follower.bus.write_calibration", "follower-calibration"),
        ("leader.bus.write_calibration", "leader-calibration"),
        ("follower.camera.connect",),
        ("follower.configure",),
        ("leader.configure",),
        ("thread.create", "teleoperation-worker", True),
        ("thread.start",),
        ("leader.get_action",),
        ("follower.send_action", action),
        ("sleep", 0.001),
        ("follower.disconnect",),
        ("leader.disconnect",),
    ]
    follower_config_kwargs = events[1][1]
    leader_config_kwargs = events[2][1]
    assert "max_relative_target" not in follower_config_kwargs
    assert "max_relative_target" not in leader_config_kwargs


def test_leader_recording_uses_stock_record_loop_for_episode_and_reset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lelab.record as record
    import lerobot.common.control_utils as control_utils
    import lerobot.datasets as datasets_module
    import lerobot.processor as processor_module
    import lerobot.robots as robots_module
    import lerobot.scripts.lerobot_record as lerobot_record
    import lerobot.teleoperators as teleoperators_module
    import lerobot.utils.feature_utils as feature_utils
    import lerobot.utils.utils as utils_module

    events: list[tuple[Any, ...]] = []
    record_loop_calls: list[dict[str, Any]] = []

    class _Bus:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        def connect(self) -> None:
            events.append((f"{self.owner}.bus.connect",))

        def write_calibration(self, calibration: object) -> None:
            events.append((f"{self.owner}.bus.write_calibration", calibration))

    class _Camera:
        def connect(self) -> None:
            events.append(("follower.camera.connect",))

    class _Robot:
        name = "so101_follower"
        action_features = {"shoulder_pan.pos": float}
        observation_features = {"shoulder_pan.pos": float}
        calibration = "follower-calibration"

        def __init__(self) -> None:
            self.bus = _Bus("follower")
            self.cameras = {"wrist": _Camera()}

        def configure(self) -> None:
            events.append(("follower.configure",))

    class _Teleop:
        calibration = "leader-calibration"

        def __init__(self) -> None:
            self.bus = _Bus("leader")

        def configure(self) -> None:
            events.append(("leader.configure",))

    class _Dataset:
        def __init__(self) -> None:
            self.num_episodes = 0
            self.num_frames = 0
            self.fps = 24
            self.features = {}
            self.meta = SimpleNamespace(robot_type="so101_follower")

        def clear_episode_buffer(self) -> None:
            events.append(("dataset.clear_episode_buffer",))

        def save_episode(self) -> None:
            events.append(("dataset.save_episode",))
            self.num_episodes += 1

        def finalize(self) -> None:
            events.append(("dataset.finalize",))

    robot = _Robot()
    teleop = _Teleop()
    dataset = _Dataset()

    class _DatasetFactory:
        @staticmethod
        def create(*args: Any, **kwargs: Any) -> _Dataset:
            events.append(("dataset.create", args, kwargs))
            return dataset

    episode_attempt = 0

    def stock_record_loop_spy(**kwargs: Any) -> None:
        nonlocal episode_attempt
        record_loop_calls.append(kwargs.copy())
        if "dataset" in kwargs:
            episode_attempt += 1
            events.append(("stock_record_loop.episode", episode_attempt))
            # The first timeout takes the current re-record path.  Subsequent
            # episode passes mimic the web "next/save" event.
            if episode_attempt > 1:
                kwargs["events"]["_exit_early_triggered"] = True
        else:
            events.append(("stock_record_loop.reset",))

    monkeypatch.setattr(robots_module, "make_robot_from_config", lambda config: robot)
    monkeypatch.setattr(teleoperators_module, "make_teleoperator_from_config", lambda config: teleop)
    monkeypatch.setattr(
        processor_module,
        "make_default_processors",
        lambda: ("teleop-processor", "robot-action-processor", "observation-processor"),
    )
    monkeypatch.setattr(
        feature_utils,
        "hw_to_dataset_features",
        lambda features, kind, video: {f"{kind}.fixture": {"video": video}},
    )
    monkeypatch.setattr(datasets_module, "LeRobotDataset", _DatasetFactory)
    monkeypatch.setattr(lerobot_record, "record_loop", stock_record_loop_spy)
    monkeypatch.setattr(control_utils, "sanity_check_dataset_name", lambda repo_id, policy: None)
    monkeypatch.setattr(utils_module, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        record,
        "safe_disconnect_device",
        lambda device, logger, context: events.append(
            ("disconnect", "follower" if device is robot else "leader", context)
        ),
    )

    dataset_cfg = SimpleNamespace(
        repo_id="alex/legacy",
        fps=24,
        root=None,
        video_encoding_batch_size=1,
        rgb_encoder=None,
        depth_encoder=None,
        streaming_encoding=True,
        encoder_queue_maxsize=4,
        encoder_threads=2,
        num_image_writer_processes=0,
        num_image_writer_threads_per_camera=1,
        video=True,
        num_episodes=2,
        episode_time_s=7,
        reset_time_s=3,
        single_task="stack blocks",
        push_to_hub=False,
        tags=None,
        private=False,
    )
    cfg = SimpleNamespace(
        robot="robot-config",
        teleop="leader-config",
        dataset=dataset_cfg,
        resume=False,
        display_data=False,
        play_sounds=False,
    )
    web_events = {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
    }

    returned_dataset = record.record_with_web_events(cfg, web_events)

    assert returned_dataset is dataset
    assert [event[0] for event in events] == [
        "dataset.create",
        "follower.bus.connect",
        "leader.bus.connect",
        "follower.bus.write_calibration",
        "leader.bus.write_calibration",
        "follower.camera.connect",
        "follower.configure",
        "leader.configure",
        "stock_record_loop.episode",
        "dataset.clear_episode_buffer",
        "stock_record_loop.reset",
        "stock_record_loop.episode",
        "dataset.save_episode",
        "stock_record_loop.reset",
        "stock_record_loop.episode",
        "dataset.save_episode",
        "dataset.finalize",
        "disconnect",
        "disconnect",
    ]
    assert dataset.num_episodes == 2

    create_args, create_kwargs = events[0][1], events[0][2]
    assert create_args == ("alex/legacy", 24)
    assert create_kwargs["robot_type"] == "so101_follower"
    assert create_kwargs["use_videos"] is True

    assert len(record_loop_calls) == 5
    episode_calls = [call for call in record_loop_calls if "dataset" in call]
    reset_calls = [call for call in record_loop_calls if "dataset" not in call]
    assert len(episode_calls) == 3
    assert len(reset_calls) == 2
    for call in record_loop_calls:
        assert call["robot"] is robot
        assert call["teleop"] is teleop
        assert call["events"] is web_events
        assert call["fps"] == 24
        assert call["single_task"] == "stack blocks"
        assert call["display_data"] is False
        assert call["teleop_action_processor"] == "teleop-processor"
        assert call["robot_action_processor"] == "robot-action-processor"
        assert call["robot_observation_processor"] == "observation-processor"
    for call in episode_calls:
        assert call["dataset"] is dataset
        assert call["control_time_s"] == 7
    for call in reset_calls:
        assert call["control_time_s"] == 3

    assert web_events == {
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
        "_exit_early_triggered": False,
    }
    assert events[-2:] == [
        ("disconnect", "follower", "recording cleanup"),
        ("disconnect", "leader", "recording cleanup"),
    ]
