"""Software-only API integration tests for shared control ownership and V2 records."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.responses import JSONResponse

from lelab.control_coordinator import ControlCoordinator
from lelab.control_session import (
    NOT_ATTEMPTED_TORQUE,
    ControlManagerClosingError,
    ControlOperation,
    ControlSessionManager,
    ControlState,
)


def _json_body(response: JSONResponse) -> dict[str, Any]:
    return json.loads(response.body)


class _CoordinatorStub:
    def __init__(self) -> None:
        self.manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
        self.calls: list[tuple[str, object]] = []
        self.workers: dict[str, object] = {}

    def start_runtime(self) -> None:
        self.calls.append(("runtime.start", None))

    def shutdown(self, *, timeout_s: float, reason: str):  # type: ignore[no-untyped-def]
        self.calls.append(("runtime.shutdown", (timeout_s, reason)))
        return SimpleNamespace(
            session_id=None,
            teardown_complete=True,
            watchdog_stopped=True,
        )

    def status(self, session_id: str | None = None):  # type: ignore[no-untyped-def]
        if session_id is None:
            return self.manager.current_status(check_expiry=False)
        return self.manager.status_for(session_id, check_expiry=False)

    def renew_lease(self, session_id: str):  # type: ignore[no-untyped-def]
        return self.manager.renew_lease(session_id)

    def request_stop(self, session_id: str, *, reason: str):  # type: ignore[no-untyped-def]
        self.calls.append(("stop", (session_id, reason)))
        return self.manager.request_stop(session_id, reason=reason)

    def set_stadia_speed(self, session_id: str, multiplier: float):  # type: ignore[no-untyped-def]
        self.calls.append(("speed", (session_id, multiplier)))
        return self.manager.merge_details(
            session_id,
            {
                "stadia_speed_multiplier": multiplier,
                "stadia_effective_max_step_per_tick": 0.35 * multiplier,
            },
        )

    def start_teleoperation(self, request, *, websocket_manager=None):  # type: ignore[no-untyped-def]
        self.calls.append(("teleoperation", (dict(request), websocket_manager)))
        return {"success": True, "session_id": "teleop-session"}

    def start_recording(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(("recording", dict(request)))
        return {"success": True, "session_id": "record-session"}

    def start_controller_check(self, robot_name: str) -> dict[str, object]:
        self.calls.append(("controller_check", robot_name))
        return {"success": True, "session_id": "check-session"}

    def start_external_operation(self, operation, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("external", (operation, kwargs)))
        return {"success": True, "session_id": "external-session"}

    def active_managed_worker(self, session_id: str, *, operation=None):  # type: ignore[no-untyped-def]
        status = self.manager.active_status(check_expiry=False)
        if status is None or status.session_id != session_id:
            return None
        if operation is not None and status.operation is not ControlOperation(operation):
            return None
        return self.workers.get(session_id)


@pytest.fixture
def robot_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from lelab import server
    from lelab.utils import config

    robots = tmp_path / "robots"
    followers = tmp_path / "followers"
    leaders = tmp_path / "leaders"
    for path in (robots, followers, leaders):
        path.mkdir()
    monkeypatch.setattr(config, "ROBOTS_PATH", str(robots))
    monkeypatch.setattr(config, "FOLLOWER_CONFIG_PATH", str(followers))
    monkeypatch.setattr(config, "LEADER_CONFIG_PATH", str(leaders))
    monkeypatch.setattr(
        server.job_registry,
        "require_registered_checkpoint_ref",
        lambda policy_ref: policy_ref,
    )
    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    return SimpleNamespace(
        robots=robots,
        followers=followers,
        leaders=leaders,
        coordinator=coordinator,
    )


def _write_calibrations(store) -> None:  # type: ignore[no-untyped-def]
    (store.followers / "follower.json").write_text("{}")
    (store.leaders / "leader.json").write_text("{}")


def _v2_record(*, mode: str = "stadia") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": "desk-arm",
        "teleoperator_type": mode,
        "follower": {"port": "/saved/follower", "calibration": "follower.json"},
        "leader": {"port": "/saved/leader", "calibration": "leader.json"},
        "stadia": {
            "guid": "030000005e040000ea02000000000000",
            "deadzone": 0.15,
            "max_step_per_tick": 0.35,
            "arm_startup_travel_degrees": 45.0,
            "gripper_startup_travel_percentage_points": 45.0,
        },
        "cameras": [],
    }


def test_robot_list_normalizes_legacy_in_memory_without_writing(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    legacy = {
        "name": "ignored-file-name-wins",
        "leader_port": "/saved/leader",
        "follower_port": "/saved/follower",
        "leader_config": "leader.json",
        "follower_config": "follower.json",
        "cameras": [],
    }
    path = robot_store.robots / "desk-arm.json"
    original = json.dumps(legacy, indent=2)
    path.write_text(original)

    result = server.get_robots()

    assert result["status"] == "success"
    record = result["robots"][0]
    assert record["schema_version"] == 2
    assert record["name"] == "desk-arm"
    assert record["teleoperator_type"] == "leader_arm"
    assert set(record["readiness"]) == {operation.value for operation in server.RobotOperation}
    assert record["readiness"]["leader_teleoperation"]["ready"] is True
    assert path.read_text() == original


def test_robot_list_rejects_a_malformed_sibling_instead_of_dropping_it(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    (robot_store.robots / "good.json").write_text(json.dumps({**_v2_record(), "name": "good"}))
    (robot_store.robots / "bad.json").write_text("{not json")

    response = server.get_robots()

    assert isinstance(response, JSONResponse)
    assert response.status_code == 422
    body = _json_body(response)
    assert body["status"] == "error"
    assert body["robots"] == []


def test_nested_v2_patch_preserves_unused_leader_and_rejects_unknown_fields(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    path = robot_store.robots / "desk-arm.json"
    path.write_text(json.dumps(_v2_record(mode="leader_arm")))

    saved = server.upsert_robot(
        "desk-arm",
        {
            "schema_version": 2,
            "teleoperator_type": "stadia",
            "stadia": {"deadzone": 0.2},
        },
    )
    assert saved["status"] == "success"
    assert saved["robot"]["leader"] == {
        "port": "/saved/leader",
        "calibration": "leader.json",
    }
    persisted = json.loads(path.read_text())
    assert persisted["teleoperator_type"] == "stadia"
    assert persisted["leader"]["port"] == "/saved/leader"

    before = path.read_text()
    response = server.upsert_robot(
        "desk-arm",
        {"schema_version": 2, "stadia": {"mystery_limit": 1}},
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert path.read_text() == before


def test_controller_readiness_is_blocked_by_any_active_owner(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server
    from lelab.utils.config import RobotRecordV2

    record = RobotRecordV2.model_validate(_v2_record())
    robot_store.coordinator.manager.claim(ControlOperation.INFERENCE)

    payload = server._record_for_api(record)

    readiness = payload["readiness"]["controller_check"]
    assert readiness["ready"] is False
    assert any(issue["code"] == "control_session_busy" for issue in readiness["issues"])


def test_generic_control_routes_require_exact_session_and_return_stopping(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    manager = ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0)
    coordinator = ControlCoordinator(manager, monitor_interval_s=0.002)
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    claim = manager.claim(ControlOperation.CONTROLLER_CHECK, teleoperator_type="stadia")
    manager.mark_running(claim.session_id)

    current = server.control_status(claim.session_id)
    assert current["status"]["state"] == "running"
    renewed = server.renew_control_lease(server.ControlSessionBody(session_id=claim.session_id))
    assert renewed["session_id"] == claim.session_id

    stale = server.stop_control(server.ControlSessionBody(session_id="wrong-session"))
    assert isinstance(stale, JSONResponse)
    assert stale.status_code == 409
    assert manager.active_status(check_expiry=False).state is ControlState.RUNNING

    stopping = server.stop_control(server.ControlSessionBody(session_id=claim.session_id))
    assert stopping["status"]["state"] == "stopping"
    assert stopping["status"]["teardown_completed_at_utc"] is None


def test_stadia_speed_route_uses_exact_session_and_returns_revision(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    claim = robot_store.coordinator.manager.claim(
        ControlOperation.STADIA_TELEOPERATION,
        teleoperator_type="stadia",
    )
    robot_store.coordinator.manager.mark_running(claim.session_id)

    response = server.set_control_speed(server.StadiaSpeedBody(session_id=claim.session_id, multiplier=1.75))

    assert response["success"] is True
    assert response["session_id"] == claim.session_id
    assert response["status"]["session_id"] == claim.session_id
    assert response["status"]["details"]["stadia_speed_multiplier"] == 1.75
    assert robot_store.coordinator.calls[-1] == ("speed", (claim.session_id, 1.75))


def test_failed_runtime_is_not_replaced_by_an_ordinary_request(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    failed = ControlCoordinator()
    failed.manager.begin_shutdown(reason="watchdog failed", terminal_error=True)
    monkeypatch.setattr(server, "_control_coordinator", failed)

    assert server._control() is failed
    with pytest.raises(ControlManagerClosingError):
        server._control().manager.claim(ControlOperation.CONTROLLER_CHECK)

    replacement = server._control_for_lifespan_start()
    assert replacement is failed
    assert replacement.manager.closing


def test_clean_lifespan_shutdown_can_recreate_coordinator(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    closed = ControlCoordinator()
    closed.manager.begin_shutdown(reason="clean server shutdown")
    monkeypatch.setattr(server, "_control_coordinator", closed)

    replacement = server._control_for_lifespan_start()

    assert replacement is not closed
    assert not replacement.manager.closing


def test_stadia_recording_routes_dispatch_exact_worker_and_never_read_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from lelab import server

    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    monkeypatch.setattr(
        server,
        "handle_recording_status",
        lambda: pytest.fail("Stadia status must not read legacy globals"),
    )
    claim = coordinator.manager.claim(
        ControlOperation.STADIA_RECORDING,
        teleoperator_type="stadia",
        details={"robot_name": "desk-arm"},
    )
    coordinator.manager.mark_running(claim.session_id)
    calls: list[str] = []

    def worker_status() -> dict[str, Any]:
        return {
            "session_id": claim.session_id,
            "recording_active": True,
            "current_phase": "recording",
            "session_ended": False,
            "dataset_repo_id": "alex/data_20260902_040000",
            "dataset_safe": True,
            "dataset_finalized": False,
            "dataset_uploaded": False,
            "upload_available": False,
            "camera_feed_available": False,
            "cameras": ["front"],
            "available_controls": {
                "stop_recording": True,
                "exit_early": True,
                "rerecord_episode": True,
            },
            "error": None,
        }

    worker = SimpleNamespace(
        recording_status=lambda: pytest.fail("route must use one immutable manager snapshot"),
        finish_episode=lambda: calls.append("finish") or {"success": True},
        rerecord_episode=lambda: calls.append("rerecord") or {"success": True},
    )
    coordinator.workers[claim.session_id] = worker
    coordinator.manager.merge_details(claim.session_id, {"recording": worker_status()})

    status = server.recording_status()
    assert status["session_id"] == claim.session_id
    assert status["control_status"]["operation"] == "stadia_recording"
    assert status["dataset_safe"] is True
    assert status["camera_feed_available"] is False

    wrong = server.recording_exit_early(server.ControlSessionBody(session_id="wrong"))
    assert isinstance(wrong, JSONResponse)
    assert wrong.status_code == 409
    assert calls == []

    finished = server.recording_exit_early(server.ControlSessionBody(session_id=claim.session_id))
    rerecorded = server.recording_rerecord_episode(server.ControlSessionBody(session_id=claim.session_id))
    assert finished["success"] is True
    assert rerecorded["success"] is True
    assert calls == ["finish", "rerecord"]

    with pytest.raises(HTTPException) as camera_error:
        server.camera_feed("front")
    assert camera_error.value.status_code == 404


def test_stadia_stop_recording_requires_exact_session_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab import server

    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    claim = coordinator.manager.claim(ControlOperation.STADIA_RECORDING, teleoperator_type="stadia")
    coordinator.manager.mark_running(claim.session_id)

    missing = server.stop_recording()
    wrong = server.stop_recording(server.ControlSessionBody(session_id="stale"))

    assert isinstance(missing, JSONResponse) and missing.status_code == 409
    assert isinstance(wrong, JSONResponse) and wrong.status_code == 409
    assert coordinator.manager.active_status(check_expiry=False).state is ControlState.RUNNING

    exact = server.stop_recording(server.ControlSessionBody(session_id=claim.session_id))
    assert exact["session_id"] == claim.session_id
    assert exact["status"]["state"] == "stopping"


def test_stadia_terminal_status_persists_and_unsafe_dataset_cannot_use_legacy_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server

    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    upload_calls: list[str] = []
    monkeypatch.setattr(
        server,
        "handle_upload_dataset",
        lambda request: upload_calls.append(request.dataset_repo_id) or {"success": True},
    )
    claim = coordinator.manager.claim(
        ControlOperation.STADIA_RECORDING,
        teleoperator_type="stadia",
        details={"robot_name": "desk-arm"},
    )
    coordinator.manager.mark_running(claim.session_id)
    terminal_recording = {
        "session_id": claim.session_id,
        "recording_active": False,
        "current_phase": "error",
        "session_ended": True,
        "dataset_repo_id": "alex/unsafe_20260902_040000",
        "dataset_safe": False,
        "dataset_finalized": False,
        "dataset_uploaded": False,
        "upload_available": False,
        "camera_feed_available": False,
        "cameras": [],
        "available_controls": {
            "stop_recording": False,
            "exit_early": False,
            "rerecord_episode": False,
        },
        "error": "rollback could not prove video cleanup",
    }
    coordinator.manager.merge_details(claim.session_id, {"recording": terminal_recording})
    coordinator.manager.request_stop(claim.session_id, reason="dataset failure")
    coordinator.manager.finish_teardown(
        claim.session_id,
        terminal_state=ControlState.ERROR,
        torque=NOT_ATTEMPTED_TORQUE,
        reason="dataset failure",
    )

    status = server.recording_status()
    assert status["session_id"] == claim.session_id
    assert status["dataset_safe"] is False
    assert status["dataset_finalized"] is False
    assert "video cleanup" in status["error"]

    blocked = server.upload_dataset(server.UploadRequest(dataset_repo_id="alex/unsafe_20260902_040000"))
    assert isinstance(blocked, JSONResponse)
    assert blocked.status_code == 409
    assert upload_calls == []

    unrelated = server.upload_dataset(server.UploadRequest(dataset_repo_id="alex/legacy-safe"))
    assert unrelated["success"] is True
    assert upload_calls == ["alex/legacy-safe"]


def test_durable_stadia_safety_blocks_upload_after_history_eviction_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server
    from lelab.stadia.dataset_safety import DatasetSafetyManifest, write_dataset_safety_manifest

    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    repo_id = "alex/unsafe_20260902_050000"
    write_dataset_safety_manifest(
        DatasetSafetyManifest(
            dataset_repo_id=repo_id,
            session_id="old-recording",
            dataset_safe=False,
            dataset_finalized=False,
            dataset_uploaded=False,
            saved_episodes=1,
            error="video cleanup was not proven",
        )
    )
    # A fresh coordinator models a process/lifespan restart with no retained
    # in-memory terminal history.
    monkeypatch.setattr(server, "_control", lambda: _CoordinatorStub())
    uploads: list[str] = []
    monkeypatch.setattr(
        server,
        "handle_upload_dataset",
        lambda request: uploads.append(request.dataset_repo_id) or {"success": True},
    )

    blocked = server.upload_dataset(server.UploadRequest(dataset_repo_id=repo_id))

    assert isinstance(blocked, JSONResponse)
    assert blocked.status_code == 409
    assert "video cleanup" in _json_body(blocked)["message"]
    assert uploads == []


def test_successful_manual_stadia_upload_is_durably_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server
    from lelab.stadia.dataset_safety import DatasetSafetyManifest, write_dataset_safety_manifest

    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    repo_id = "alex/safe_20260902_050000"
    write_dataset_safety_manifest(
        DatasetSafetyManifest(
            dataset_repo_id=repo_id,
            session_id="old-recording",
            dataset_safe=True,
            dataset_finalized=True,
            dataset_uploaded=False,
            saved_episodes=1,
        )
    )
    uploads: list[str] = []
    monkeypatch.setattr(
        server,
        "handle_upload_dataset",
        lambda request: uploads.append(request.dataset_repo_id) or {"success": True},
    )

    first = server.upload_dataset(server.UploadRequest(dataset_repo_id=repo_id))
    second = server.upload_dataset(server.UploadRequest(dataset_repo_id=repo_id))

    assert first["success"] is True
    assert isinstance(second, JSONResponse) and second.status_code == 409
    assert uploads == [repo_id]


def test_recording_status_uses_one_manager_revision_not_a_live_worker_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server

    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    claim = coordinator.manager.claim(ControlOperation.STADIA_RECORDING, teleoperator_type="stadia")
    coordinator.manager.mark_running(claim.session_id)
    published = {
        "session_id": claim.session_id,
        "recording_active": True,
        "current_phase": "recording",
        "current_episode": 1,
        "total_episodes": 1,
        "saved_episodes": 0,
        "session_ended": False,
        "dataset_repo_id": "alex/atomic_20260902_050000",
        "dataset_safe": True,
        "dataset_finalized": False,
        "dataset_uploaded": False,
        "upload_available": False,
        "camera_feed_available": False,
        "cameras": [],
        "available_controls": {
            "stop_recording": True,
            "exit_early": True,
            "rerecord_episode": True,
        },
        "error": None,
    }
    immutable = coordinator.manager.merge_details(claim.session_id, {"recording": published})
    coordinator.workers[claim.session_id] = SimpleNamespace(
        recording_status=lambda: pytest.fail("mutable worker read must not occur")
    )

    response = server.recording_status()

    assert response["current_phase"] == "recording"
    assert response["dataset_finalized"] is False
    assert response["control_status"]["revision"] == immutable.revision
    assert response["control_status"]["details"]["recording"] == published


def test_start_routes_forward_canonical_robot_name_only(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    teleop = server.teleoperate_arm({"robot_name": "desk-arm"})
    recording = server.start_recording(
        {
            "robot_name": "desk-arm",
            "dataset_repo_id": "alex/data",
            "single_task": "pick up cube",
        }
    )
    controller = server.controller_check(server.ControllerCheckBody(robot_name="desk-arm"))

    assert teleop["success"] is True
    assert recording["success"] is True
    assert controller["success"] is True
    assert robot_store.coordinator.calls[:3] == [
        ("teleoperation", ({"robot_name": "desk-arm"}, server.manager)),
        (
            "recording",
            {
                "robot_name": "desk-arm",
                "dataset_repo_id": "alex/data",
                "single_task": "pick up cube",
            },
        ),
        ("controller_check", "desk-arm"),
    ]


def test_canonical_inference_uses_saved_follower_identity(
    robot_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))
    received: list[object] = []
    monkeypatch.setattr(
        server,
        "handle_start_inference",
        lambda request: received.append(request) or {"success": True},
    )
    monkeypatch.setattr(
        server,
        "handle_inference_status",
        lambda: {"inference_active": False},
    )

    result = server.start_inference(
        {
            "robot_name": "desk-arm",
            "policy_ref": "/models/policy",
            "task": "move cube",
            "duration_s": 12,
        }
    )

    assert result["success"] is True
    _, (operation, kwargs) = robot_store.coordinator.calls[-1]
    assert operation is ControlOperation.INFERENCE
    assert kwargs["teleoperator_type"] == "stadia"
    kwargs["start"]()
    request = received[0]
    assert request.follower_port == "/saved/follower"
    assert request.follower_config == "follower.json"


def test_inference_rejects_a_policy_ref_not_authorized_by_the_job_registry(
    robot_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))
    monkeypatch.setattr(
        server.job_registry,
        "require_registered_checkpoint_ref",
        lambda _policy_ref: (_ for _ in ()).throw(
            ValueError("policy_ref must identify a registered model checkpoint")
        ),
    )

    response = server.start_inference(
        {
            "robot_name": "desk-arm",
            "policy_ref": "/private/arbitrary-model",
            "task": "move cube",
            "duration_s": 12,
        }
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert "registered model checkpoint" in _json_body(response)["message"]
    assert robot_store.coordinator.calls == []


def test_inference_uses_only_exact_saved_camera_configuration(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    record = _v2_record()
    record["cameras"] = [
        {
            "id": "front-id",
            "name": "Front",
            "type": "opencv",
            "camera_index": 2,
            "device_id": "browser-device-id",
            "width": 640,
            "height": 480,
            "fps": 25,
            "fourcc": "MJPG",
            "backend": "AVFOUNDATION",
        }
    ]
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(record))

    _, resolved = server._resolve_inference_request(
        {
            "robot_name": "desk-arm",
            "policy_ref": "/models/policy",
            "cameras": {
                "observation.images.front": {
                    "type": "opencv",
                    "camera_index": 2,
                    "width": 640,
                    "height": 480,
                }
            },
        }
    )

    assert resolved.cameras == {
        "observation.images.front": {
            "type": "opencv",
            "camera_index": 2,
            "width": 640,
            "height": 480,
            "fps": 25,
            "fourcc": "MJPG",
            "backend": "AVFOUNDATION",
        }
    }


@pytest.mark.parametrize(
    "cameras",
    [
        {
            "front": {
                "type": "opencv",
                "camera_index": 99,
                "width": 640,
                "height": 480,
                "fps": 25,
            }
        },
        {
            "front": {
                "type": "opencv",
                "camera_index": 2,
                "width": 1280,
                "height": 720,
                "fps": 25,
            }
        },
        {
            "front": {
                "type": "opencv",
                "camera_index": 2,
                "width": 640,
                "height": 480,
                "fps": 25,
            },
            "wrist": {
                "type": "opencv",
                "camera_index": 2,
                "width": 640,
                "height": 480,
                "fps": 25,
            },
        },
        {
            "front}, injected: {camera": {
                "type": "opencv",
                "camera_index": 2,
                "width": 640,
                "height": 480,
                "fps": 25,
            }
        },
    ],
)
def test_inference_rejects_unsaved_changed_or_reused_camera_bindings(
    robot_store,
    cameras: dict[str, dict[str, object]],
) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    record = _v2_record()
    record["cameras"] = [
        {
            "id": "front-id",
            "name": "Front",
            "type": "opencv",
            "camera_index": 2,
            "device_id": "browser-device-id",
            "width": 640,
            "height": 480,
            "fps": 25,
            "fourcc": None,
            "backend": None,
        }
    ]
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(record))

    with pytest.raises(ValueError):
        server._resolve_inference_request(
            {
                "robot_name": "desk-arm",
                "policy_ref": "/models/policy",
                "cameras": cameras,
            }
        )


def test_inference_rejects_unsafe_saved_fourcc_before_cli_serialization(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    record = _v2_record()
    record["cameras"] = [
        {
            "id": "front-id",
            "name": "Front",
            "type": "opencv",
            "camera_index": 2,
            "device_id": "browser-device-id",
            "width": 640,
            "height": 480,
            "fps": 25,
            "fourcc": "}:x,",
            "backend": None,
        }
    ]
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(record))

    with pytest.raises(ValueError, match="unsafe FOURCC"):
        server._resolve_inference_request(
            {
                "robot_name": "desk-arm",
                "policy_ref": "/models/policy",
                "cameras": {
                    "front": {
                        "type": "opencv",
                        "camera_index": 2,
                        "width": 640,
                        "height": 480,
                        "fps": 25,
                    }
                },
            }
        )


def test_legacy_inference_camera_binding_still_uses_saved_record_authority(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    record = _v2_record()
    record["cameras"] = [
        {
            "id": "front-id",
            "name": "Front",
            "type": "opencv",
            "camera_index": 2,
            "device_id": "browser-device-id",
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": None,
            "backend": None,
        }
    ]
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(record))

    with pytest.raises(ValueError, match="does not match a camera"):
        server._resolve_inference_request(
            {
                "follower_port": "/saved/follower",
                "follower_config": "follower.json",
                "policy_ref": "/models/policy",
                "cameras": {
                    "front": {
                        "type": "opencv",
                        "camera_index": 99,
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    }
                },
            }
        )


def test_inference_owner_retains_process_liveness_after_legacy_stop_clears_global(
    robot_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))

    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode

    def start_legacy(_request):  # type: ignore[no-untyped-def]
        server._rollout._inference_proc = process
        return {"success": True}

    def stop_legacy():  # type: ignore[no-untyped-def]
        server._rollout._inference_proc = None
        return {"success": True}

    monkeypatch.setattr(server, "handle_start_inference", start_legacy)
    monkeypatch.setattr(server, "handle_stop_inference", stop_legacy)
    monkeypatch.setattr(
        server,
        "handle_inference_status",
        lambda: {"inference_active": False},
    )

    result = server.start_inference(
        {
            "robot_name": "desk-arm",
            "policy_ref": "/models/policy",
        }
    )
    assert result["success"] is True
    _, (_, callbacks) = robot_store.coordinator.calls[-1]

    assert callbacks["start"]()["success"] is True
    assert callbacks["stop"]()["success"] is True
    assert callbacks["is_active"]() is True

    process.returncode = 0
    assert callbacks["is_active"]() is False


def test_natural_inference_exit_is_retained_after_monitor_consumes_legacy_status(
    robot_store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    _write_calibrations(robot_store)
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))
    coordinator = ControlCoordinator(
        ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0),
        monitor_interval_s=0.002,
    )
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    consumed = False
    lock = threading.Lock()

    def start_legacy(_request):  # type: ignore[no-untyped-def]
        server._rollout._inference_proc = process
        return {"success": True, "message": "started", "log_path": "/tmp/inference.log"}

    def status_legacy() -> dict[str, Any]:
        nonlocal consumed
        with lock:
            if process.returncode is None:
                return {
                    "inference_active": True,
                    "started_at": 1.0,
                    "rollout_started_at": 2.0,
                    "elapsed_s": 3.0,
                    "rollout_elapsed_s": 2.0,
                    "duration_s": 5,
                    "policy_ref": "/models/policy",
                    "log_path": "/tmp/inference.log",
                }
            if not consumed:
                consumed = True
                server._rollout._inference_proc = None
                return {
                    "inference_active": False,
                    "exited": True,
                    "exit_code": 7,
                    "outcome": "failed",
                    "error": "policy process failed",
                    "hint": None,
                    "started_at": 1.0,
                    "rollout_started_at": 2.0,
                    "elapsed_s": 3.0,
                    "rollout_elapsed_s": 2.0,
                    "duration_s": 5,
                    "policy_ref": "/models/policy",
                    "log_path": "/tmp/inference.log",
                }
            return {
                "inference_active": False,
                "started_at": None,
                "rollout_started_at": None,
                "elapsed_s": 0.0,
                "rollout_elapsed_s": 0.0,
                "duration_s": None,
                "policy_ref": None,
                "log_path": None,
            }

    monkeypatch.setattr(server, "handle_start_inference", start_legacy)
    monkeypatch.setattr(server, "handle_inference_status", status_legacy)

    started = server.start_inference(
        {
            "robot_name": "desk-arm",
            "policy_ref": "/models/policy",
        }
    )
    session_id = started["session_id"]
    process.returncode = 7
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        terminal = coordinator.status(session_id)
        if terminal is not None and terminal.terminal:
            break
        time.sleep(0.002)
    else:
        pytest.fail("inference owner did not become terminal")

    retained = server.inference_status()
    assert retained["session_id"] == session_id
    assert retained["inference_active"] is False
    assert retained["exited"] is True
    assert retained["exit_code"] == 7
    assert retained["error"] == "policy process failed"
    terminal = coordinator.status(session_id)
    assert terminal is not None
    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "policy process failed"


def test_calibration_start_requires_saved_identity_and_enters_shared_owner(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    record = _v2_record()
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(record))
    request = server.CalibrationRequest(
        device_type="robot",
        port="/saved/follower",
        config_file="follower",
        robot_name="desk-arm",
    )

    result = server.start_calibration(request)

    assert result["success"] is True
    _, (operation, kwargs) = robot_store.coordinator.calls[-1]
    assert operation is ControlOperation.FOLLOWER_CALIBRATION
    assert kwargs["teleoperator_type"] == "stadia"
    assert kwargs["details"]["robot_name"] == "desk-arm"

    mismatch = server.start_calibration(
        server.CalibrationRequest(
            device_type="robot",
            port="/browser/substitution",
            config_file="follower",
            robot_name="desk-arm",
        )
    )
    assert isinstance(mismatch, JSONResponse)
    assert mismatch.status_code == 400


def test_legacy_calibration_identity_resolves_exact_saved_record(
    robot_store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server

    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))
    started: list[server.CalibrationRequest] = []
    monkeypatch.setattr(
        server.calibration_manager,
        "start_calibration",
        lambda request: started.append(request) or {"success": True},
    )

    result = server.start_calibration(
        server.CalibrationRequest(
            device_type="robot",
            port="/saved/follower",
            config_file="follower",
        )
    )

    assert result["success"] is True
    _, (_operation, kwargs) = robot_store.coordinator.calls[-1]
    assert kwargs["details"]["robot_name"] == "desk-arm"
    kwargs["start"]()
    assert started[0].robot_name == "desk-arm"


def test_legacy_calibration_identity_rejects_no_match_ambiguity_and_malformed_sibling(
    robot_store,  # type: ignore[no-untyped-def]
) -> None:
    from lelab import server

    request = server.CalibrationRequest(
        device_type="robot",
        port="/saved/follower",
        config_file="follower",
    )
    no_match = server.start_calibration(request)
    assert isinstance(no_match, JSONResponse)
    assert no_match.status_code == 400

    first = _v2_record()
    second = {**_v2_record(), "name": "desk-arm-2"}
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(first))
    (robot_store.robots / "desk-arm-2.json").write_text(json.dumps(second))
    ambiguous = server.start_calibration(request)
    assert isinstance(ambiguous, JSONResponse)
    assert ambiguous.status_code == 400
    assert "ambiguous" in _json_body(ambiguous)["message"]

    (robot_store.robots / "desk-arm-2.json").write_text("not-json")
    malformed = server.start_calibration(request)
    assert isinstance(malformed, JSONResponse)
    assert malformed.status_code == 422
    assert "valid JSON" in _json_body(malformed)["message"]


def test_legacy_leader_calibration_rejects_stadia_mode(robot_store) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record(mode="stadia")))

    result = server.start_calibration(
        server.CalibrationRequest(
            device_type="teleop",
            port="/saved/leader",
            config_file="leader",
        )
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 400
    assert "not ready" in _json_body(result)["message"]


def test_calibration_owner_includes_lingering_worker_liveness(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from lelab import server

    worker = SimpleNamespace(is_alive=lambda: True)
    with server.calibration_manager._status_lock:
        monkeypatch.setattr(server.calibration_manager.status, "calibration_active", False)
        monkeypatch.setattr(server.calibration_manager, "calibration_thread", worker)

    assert server._calibration_is_active() is True


def test_natural_calibration_worker_error_becomes_shared_terminal_error(
    robot_store,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server
    from lelab.calibrate import CalibrationStatus

    _write_calibrations(robot_store)
    (robot_store.robots / "desk-arm.json").write_text(json.dumps(_v2_record()))
    coordinator = ControlCoordinator(
        ControlSessionManager(lease_ttl_s=10.0, lease_renew_interval_s=1.0),
        monitor_interval_s=0.002,
    )
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    monkeypatch.setattr(server.calibration_manager, "status", CalibrationStatus())
    monkeypatch.setattr(server.calibration_manager, "calibration_thread", None)

    def start_fake(_request):  # type: ignore[no-untyped-def]
        server.calibration_manager._update_status(
            calibration_active=True,
            status="recording",
            error=None,
            cleanup_pending=False,
        )
        return {"success": True, "message": "started"}

    monkeypatch.setattr(server.calibration_manager, "start_calibration", start_fake)
    monkeypatch.setattr(
        server.calibration_manager,
        "stop_calibration_process",
        lambda: {"success": True},
    )

    started = server.start_calibration(
        server.CalibrationRequest(
            device_type="robot",
            port="/saved/follower",
            config_file="follower",
            robot_name="desk-arm",
        )
    )
    session_id = started["session_id"]
    server.calibration_manager._update_status(
        calibration_active=False,
        status="error",
        error="calibration motor read failed",
        message="failed",
        cleanup_pending=False,
    )

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        terminal = coordinator.status(session_id)
        if terminal is not None and terminal.terminal:
            break
        time.sleep(0.002)
    else:
        pytest.fail("calibration owner did not become terminal")

    assert terminal.state is ControlState.ERROR
    assert terminal.stop_reason == "calibration motor read failed"
    assert terminal.details["calibration"]["status"] == "error"
    assert coordinator.manager.quarantine_reason is None


def test_complete_calibration_step_requires_exact_session_and_returns_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lelab import server

    coordinator = _CoordinatorStub()
    monkeypatch.setattr(server, "_control", lambda: coordinator)
    claim = coordinator.manager.claim(
        ControlOperation.FOLLOWER_CALIBRATION,
        teleoperator_type="stadia",
    )
    running = coordinator.manager.mark_running(claim.session_id)
    calls: list[str] = []
    monkeypatch.setattr(
        server.calibration_manager,
        "complete_step",
        lambda: calls.append("complete") or {"success": True, "message": "advanced"},
    )

    stale = server.complete_calibration_step(server.ControlSessionBody(session_id="stale"))
    assert isinstance(stale, JSONResponse) and stale.status_code == 409
    assert calls == []

    exact = server.complete_calibration_step(server.ControlSessionBody(session_id=claim.session_id))
    assert exact["success"] is True
    assert exact["session_id"] == claim.session_id
    assert exact["status"]["revision"] == running.revision
    assert calls == ["complete"]
