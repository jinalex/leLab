"""Device-neutral contracts for generic installed camera plugins and wrappers."""

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException

from lelab.utils.cameras import (
    _fps_matches_requested,
    build_camera_configs,
    camera_cli_config,
    camera_projection,
)
from lelab.utils.config import CameraRecord, RobotRecordV2
from lelab.utils.python_process import python_module_command
from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import Cv2Backends


@CameraConfig.register_subclass("external_test")
@dataclass
class ExternalTestConfig(CameraConfig):
    host: str = "localhost"
    port: int = 7447


def saved_camera():
    return CameraRecord(
        id="remote",
        name="front",
        type="external_test",
        fps=10,
        parameters={"host": "127.0.0.1", "port": 17447},
    )


@pytest.mark.parametrize(
    ("requested", "actual", "matches"),
    [(30, 30.00003, True), (30, 29.97, False), (30, float("nan"), False)],
)
def test_camera_fps_clock_rounding_match(requested, actual, matches):
    assert _fps_matches_requested(requested, actual) is matches


def test_plugin_projection_constructor_and_cli_round_trip(monkeypatch):
    monkeypatch.setattr("lelab.utils.cameras.register_camera_plugins", lambda: None)
    camera = saved_camera()
    projection = camera_projection(camera)
    assert "camera_index" not in projection
    built = build_camera_configs({"front": projection}, Cv2Backends.ANY)["front"]
    assert isinstance(built, ExternalTestConfig)
    assert (built.host, built.port, built.width, built.height, built.fps) == (
        "127.0.0.1",
        17447,
        640,
        480,
        10,
    )
    from lelab.rollout import _format_cameras_arg

    assert json.loads(_format_cameras_arg({"front": projection}))["front"] == camera_cli_config(projection)
    assert CameraRecord.model_validate_json(camera.model_dump_json()) == camera


@pytest.mark.parametrize("parameters", [{"type": "opencv"}, {"width": 99}, {"bad": float("nan")}])
def test_plugin_parameters_cannot_shadow_common_fields_or_store_nonfinite(parameters):
    with pytest.raises(ValueError):
        CameraRecord(id="a", name="a", type="external_test", parameters=parameters)


def test_plugin_inference_binding_uses_saved_id_not_local_index():
    from lelab.server import InferenceCameraBinding, _resolve_inference_cameras

    camera = saved_camera()
    record = RobotRecordV2(name="example", cameras=[camera])
    binding = InferenceCameraBinding(type=camera.type, camera_id=camera.id, width=640, height=480)
    assert _resolve_inference_cameras(record, {"policy_camera": binding}) == {
        "policy_camera": camera_projection(camera)
    }
    with pytest.raises(ValueError):
        _resolve_inference_cameras(record, {"first": binding, "second": binding})
    with pytest.raises(ValueError):
        _resolve_inference_cameras(record, {"front": binding.model_copy(update={"camera_id": "missing"})})


def test_python_wrapper_is_opt_in_and_shell_free(monkeypatch):
    monkeypatch.delenv("LELAB_PYTHON_MODULE_WRAPPER", raising=False)
    assert python_module_command("example.module", ["--name=x y"]) == [
        sys.executable,
        "-m",
        "example.module",
        "--name=x y",
    ]
    prefix = ["-m", "example_wrapper", "--module"]
    monkeypatch.setenv("LELAB_PYTHON_MODULE_WRAPPER", json.dumps(prefix))
    assert python_module_command("example.module", ["--name=x y"]) == [
        sys.executable,
        *prefix,
        "example.module",
        "--",
        "--name=x y",
    ]


@pytest.mark.parametrize("raw", ["{}", "[]", '["ok", 3]', '["ok", ""]', "not-json"])
def test_invalid_wrapper_fails_before_spawn(monkeypatch, raw):
    monkeypatch.setenv("LELAB_PYTHON_MODULE_WRAPPER", raw)
    with pytest.raises(ValueError):
        python_module_command("example", [])


def test_preview_reads_only_matching_active_owner(monkeypatch):
    from lelab import server
    from lelab.control_session import ControlOperation, ControlState

    camera = Mock()
    camera.config = SimpleNamespace(color_mode="rgb")
    camera.read_latest.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    worker = SimpleNamespace(_robot=SimpleNamespace(cameras={"front": camera}))
    status = SimpleNamespace(
        session_id="owned", state=ControlState.RUNNING, operation=ControlOperation.STADIA_TELEOPERATION
    )
    coordinator = Mock()
    coordinator.status.return_value = status
    coordinator.active_managed_worker.return_value = worker
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    with pytest.raises(HTTPException) as error:
        server.teleop_camera_frame("front", "stale")
    assert error.value.status_code == 409
    camera.read_latest.assert_not_called()
    result = server.teleop_camera_frame("front", "owned")
    assert result.media_type == "image/jpeg"
    assert result.body[:2] == b"\xff\xd8"
    camera.connect.assert_not_called()
    status.state = ControlState.STOPPED
    with pytest.raises(HTTPException):
        server.teleop_camera_frame("front", "owned")
    assert camera.read_latest.call_count == 1


def test_dev_reload_rejects_external_wrapper(monkeypatch):
    from lelab.scripts.lelab import main

    monkeypatch.setenv("LELAB_PYTHON_MODULE_WRAPPER", '["-m", "wrapper"]')
    with pytest.raises(SystemExit) as error:
        main(["--dev"])
    assert error.value.code == 2
