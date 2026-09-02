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

import asyncio
import contextlib
import glob
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

# Import our custom recording functionality
from . import datasets as dataset_browser, record as _record, rollout as _rollout

# Import our custom calibration functionality
from .calibrate import CalibrationRequest, calibration_manager
from .control_coordinator import ControlCoordinator
from .control_session import ControlOperation, ControlSessionError, ControlState
from .jobs import (
    JobAlreadyRunningError,
    JobNotFoundError,
    JobNotRunningError,
    JobTarget,
    job_registry,
)
from .record import (
    DatasetInfoRequest,
    UploadRequest,
    handle_delete_dataset,
    handle_exit_early,
    handle_get_dataset_info,
    handle_recording_status,
    handle_rerecord_episode,
    handle_stop_recording,
    handle_upload_dataset,
)
from .rollout import (
    InferenceRequest,
    handle_inference_status,
    handle_start_inference,
    handle_stop_inference,
)
from .stadia.dataset_safety import (
    delete_dataset_safety_manifest,
    read_dataset_safety_manifest,
    write_dataset_safety_manifest,
)

# Import our custom teleoperation functionality
from .teleoperate import (
    handle_get_joint_positions,
    handle_stop_teleoperation,
    handle_teleoperation_status,
)

# Training is now job-based; see app/jobs.py.
from .train import TrainingRequest
from .update import handle_run_update, handle_update_check
from .utils import config
from .utils.config import (
    FOLLOWER_CONFIG_PATH,
    LEADER_CONFIG_PATH,
    RobotOperation,
    RobotRecordV2,
    RobotRecordValidationError,
    delete_robot_record,
    detect_port_after_disconnect,
    evaluate_all_robot_readiness,
    evaluate_robot_readiness,
    find_available_ports,
    find_robot_port,
    get_default_robot_port,
    get_robot_record_v2,
    get_saved_robot_port,
    is_robot_record_clean,
    is_valid_robot_name,
    save_robot_port,
    save_robot_record_v2,
)
from .utils.hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login, shared_hf_api
from .utils.system import (
    handle_get_cuda_status,
    handle_get_policy_extra,
    handle_get_training_extra,
    handle_get_wandb_extra,
    handle_install_policy_extra,
    handle_install_policy_extra_status,
    handle_install_training_extra,
    handle_install_training_extra_status,
    handle_install_wandb_extra,
    handle_install_wandb_extra_status,
    warn_if_cuda_mismatch,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StartTrainingBody(BaseModel):
    """Wrapping body for POST /jobs/training. Adds optional target spec."""

    config: TrainingRequest
    target: JobTarget | None = None

    @classmethod
    def from_legacy(cls, raw: dict) -> "StartTrainingBody":
        """Accept the old request shape (TrainingRequest fields at top level)
        as well as the new shape ({config: ..., target: ...}).
        """
        if "config" in raw and isinstance(raw["config"], dict):
            return cls.model_validate(raw)
        # Legacy: top-level training fields, no target.
        return cls(config=TrainingRequest.model_validate(raw))


# Cache for HF Jobs hardware flavors (5-minute TTL)
_flavors_cache: dict = {"data": None, "fetched_at": 0.0}
_FLAVOR_CACHE_TTL_SECONDS = 300.0


@contextlib.asynccontextmanager
async def _app_lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
    """Start the lease watchdog and wait boundedly for real owner teardown."""

    coordinator = _control_for_lifespan_start()
    coordinator.start_runtime()
    try:
        warn_if_cuda_mismatch()
        yield
    finally:
        logger.info("🔄 FastAPI shutting down, cleaning up...")
        result = await asyncio.to_thread(
            coordinator.shutdown,
            timeout_s=8.0,
            reason="FastAPI server shutdown",
        )
        if not result.teardown_complete:
            logger.error(
                "Control teardown did not finish before shutdown timeout; "
                "session %s remains unsafe and must not be bypassed%s",
                result.session_id,
                (f": {result.quarantine_reason}" if result.quarantine_reason is not None else ""),
            )
        if not result.watchdog_stopped:
            logger.error("Control lease watchdog did not stop before shutdown timeout")
        manager.stop_broadcast_thread()
        logger.info(
            "Shutdown cleanup attempt finished (control_teardown=%s, watchdog_stopped=%s)",
            result.teardown_complete,
            result.watchdog_stopped,
        )


app = FastAPI(lifespan=_app_lifespan)

# In dev mode the React app runs on :8080 while the API runs on :8000; in
# prod they share an origin and CORS is unnecessary. allow_credentials with
# a wildcard origin is rejected by browsers, so we drop it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Get the path to the lerobot root directory (3 levels up from this script)
LEROBOT_PATH = str(Path(__file__).parent.parent.parent.parent)
logger.info(f"LeRobot path: {LEROBOT_PATH}")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.broadcast_queue = queue.Queue()
        self.broadcast_thread = None
        self.is_running = False
        # Guards `active_connections` since the broadcast worker thread also
        # mutates it on send failure.
        self._connections_lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._connections_lock:
            self.active_connections.append(websocket)
            count = len(self.active_connections)
        logger.info(f"WebSocket connected. Total connections: {count}")

        if not self.is_running:
            self.start_broadcast_thread()

    def disconnect(self, websocket: WebSocket):
        with self._connections_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                count = len(self.active_connections)
                logger.info(f"WebSocket disconnected. Total connections: {count}")
            else:
                count = len(self.active_connections)

        if count == 0 and self.is_running:
            self.stop_broadcast_thread()

    def start_broadcast_thread(self):
        """Start the background thread for broadcasting data"""
        if self.is_running:
            return

        self.is_running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_worker, daemon=True)
        self.broadcast_thread.start()
        logger.info("📡 Broadcast thread started")

    def stop_broadcast_thread(self):
        """Stop the background thread"""
        self.is_running = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=1.0)
            logger.info("📡 Broadcast thread stopped")

    def _broadcast_worker(self):
        """Background worker thread for broadcasting WebSocket data"""
        import asyncio

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    # Get data from queue with timeout
                    data = self.broadcast_queue.get(timeout=0.1)
                    if data is None:  # Poison pill to stop
                        break

                    # Broadcast to all connections
                    if self.active_connections:
                        loop.run_until_complete(self._send_to_all_connections(data))

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in broadcast worker: {e}")

        finally:
            loop.close()

    async def _send_to_all_connections(self, data: dict[str, Any]):
        """Send data to all active WebSocket connections"""
        with self._connections_lock:
            connections = list(self.active_connections)
        if not connections:
            return

        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending data to WebSocket: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

    def broadcast_joint_data_sync(self, data: dict[str, Any]):
        """Thread-safe method to queue data for broadcasting"""
        if self.is_running and self.active_connections:
            try:
                self.broadcast_queue.put_nowait(data)
            except queue.Full:
                logger.warning("Broadcast queue is full, dropping data")

    def notify_jobs_changed(self) -> None:
        """Push a 'jobs_changed' event to all WS clients so they refetch.

        Called from JobRegistry on submit / watchdog finalisation / delete.
        Skipped silently if no clients are connected — the frontend does an
        initial fetch on mount, so a missed broadcast is self-healing.
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait({"type": "jobs_changed", "timestamp": time.time()})

    def notify_job_progress(self, snapshots: list[dict]) -> None:
        """Push a 'job_progress' event with per-running-job snapshots.

        Fired from the JobRegistry watchdog (~1Hz) while jobs are running so
        the dashboard's progress bar updates live without refetching /jobs
        (let alone /jobs/hub, which hits the HF API on every call).
        """
        if self.is_running and self.active_connections:
            with contextlib.suppress(queue.Full):
                self.broadcast_queue.put_nowait(
                    {"type": "job_progress", "jobs": snapshots, "timestamp": time.time()}
                )


manager = ConnectionManager()
job_registry.set_on_change(manager.notify_jobs_changed)
job_registry.set_on_progress(manager.notify_job_progress)

_control_lock = threading.Lock()
_control_coordinator = ControlCoordinator()


def _control() -> ControlCoordinator:
    """Return the process lifecycle's shared control coordinator.

    Ordinary requests never replace a failed/closing coordinator. Replacing it
    mid-lifespan would create a fresh manager whose lease watchdog was never
    started and could bypass the fail-closed state.
    """

    with _control_lock:
        return _control_coordinator


def _control_for_lifespan_start() -> ControlCoordinator:
    """Return a coordinator whose runtime will be started by this lifespan."""

    global _control_coordinator
    with _control_lock:
        if (
            _control_coordinator.manager.closing
            and _control_coordinator.manager.recyclable_after_shutdown
            and _control_coordinator.manager.active_status(check_expiry=False) is None
        ):
            _control_coordinator = ControlCoordinator()
        return _control_coordinator


class ControlSessionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


class StadiaSpeedBody(ControlSessionBody):
    model_config = ConfigDict(extra="forbid", strict=True)

    multiplier: float = Field(ge=0.25, le=2.0)


class ControllerCheckBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_name: str


class InferenceCameraBinding(BaseModel):
    """Policy alias bound to one camera from the saved robot record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["opencv"]
    camera_index: int = Field(ge=0)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    fps: int | None = Field(default=None, ge=1, le=240)


class CanonicalInferenceBody(BaseModel):
    """Inference settings whose robot identity comes from a saved V2 record."""

    model_config = ConfigDict(extra="forbid")

    robot_name: str
    policy_ref: str
    task: str = ""
    cameras: dict[str, InferenceCameraBinding] = Field(default_factory=dict)
    duration_s: int = Field(default=60, gt=0)


class LegacyInferenceBody(BaseModel):
    """Temporary compatibility shape for the pre-V2 inference frontend."""

    model_config = ConfigDict(extra="forbid")

    follower_port: str
    follower_config: str
    policy_ref: str
    task: str = ""
    cameras: dict[str, InferenceCameraBinding] = Field(default_factory=dict)
    duration_s: int = Field(default=60, gt=0)


def _control_failure(
    error: Exception,
    *,
    requested_session_id: str | None = None,
) -> JSONResponse:
    if isinstance(error, ControlSessionError):
        status_code = 409
    elif isinstance(error, FileNotFoundError):
        status_code = 404
    elif isinstance(error, RobotRecordValidationError):
        status_code = 422
    elif isinstance(error, (ValidationError, ValueError, TypeError)):
        status_code = 400
    else:
        status_code = 500
    content: dict[str, Any] = {"success": False, "message": str(error)}
    if requested_session_id is not None:
        content["requested_session_id"] = requested_session_id
    if isinstance(error, ControlSessionError):
        try:
            coordinator = _control()
        except Exception:
            coordinator = globals().get("_control_coordinator")
        if isinstance(coordinator, ControlCoordinator):
            status = coordinator.manager.active_status(check_expiry=False)
            if status is None and requested_session_id is not None:
                status = coordinator.manager.status_for(
                    requested_session_id,
                    check_expiry=False,
                )
            if status is not None:
                content["session_id"] = status.session_id
                content["status"] = status.as_dict()
    return JSONResponse(status_code=status_code, content=content)


_REQUEST_ERROR_STATUS = {
    "robot_not_found": 404,
    "robot_invalid": 422,
    "robot_not_ready": 409,
    "legacy_no_match": 409,
    "legacy_ambiguous": 409,
    "legacy_mode_unsupported": 409,
}


def _control_result(result: Mapping[str, Any], *, default_error_status: int = 500):
    """Return a successful mapping or a truthful non-2xx JSON failure."""

    payload = dict(result)
    if payload.get("success") is True:
        return payload
    status_code = payload.get("status_code")
    if not isinstance(status_code, int) or not 400 <= status_code <= 599:
        error = payload.get("error")
        code = error.get("code") if isinstance(error, Mapping) else None
        status_code = _REQUEST_ERROR_STATUS.get(str(code), default_error_status)
    return JSONResponse(status_code=status_code, content=payload)


def _require_saved_robot(name: str) -> RobotRecordV2:
    """Load one exact saved record and distinguish missing from malformed."""

    if not is_valid_robot_name(name):
        raise ValueError("robot_name is not a valid saved robot name")
    path = Path(config.ROBOTS_PATH) / f"{name}.json"
    existed = path.is_file()
    record = get_robot_record_v2(name)
    if record is not None:
        return record
    if existed:
        raise RobotRecordValidationError(f"saved robot record {name!r} exists but is not valid JSON")
    raise FileNotFoundError(f"saved robot record {name!r} was not found")


def _strict_robot_records() -> list[RobotRecordV2]:
    """Load every saved record without silently dropping malformed siblings."""

    root = Path(config.ROBOTS_PATH)
    if not root.exists():
        return []
    try:
        paths = sorted(path for path in root.iterdir() if path.suffix == ".json")
    except OSError as error:
        raise RobotRecordValidationError(f"could not list saved robot records: {error}") from error
    records: list[RobotRecordV2] = []
    for path in paths:
        if not is_valid_robot_name(path.stem):
            raise RobotRecordValidationError(f"saved robot filename {path.name!r} is invalid")
        records.append(_require_saved_robot(path.stem))
    return records


def _status_response(status) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "success": True,
        "session_id": status.session_id,
        "status": status.as_dict(),
    }


def _matching_control_status(operations: set[ControlOperation]):
    status = _control().status()
    if status is None or status.operation not in operations:
        return None
    return status


_RECORDING_OPERATIONS = {
    ControlOperation.LEADER_RECORDING,
    ControlOperation.STADIA_RECORDING,
}


def _stadia_recording_payload(status) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Return worker-published recording truth or a conservative fallback."""

    published = status.details.get("recording")
    if isinstance(published, Mapping):
        payload = dict(published)
        if payload.get("session_id") != status.session_id:
            return {
                "session_id": status.session_id,
                "recording_active": not status.terminal,
                "current_phase": "error",
                "session_ended": status.terminal,
                "dataset_safe": False,
                "dataset_finalized": False,
                "dataset_uploaded": False,
                "upload_available": False,
                "camera_feed_available": False,
                "cameras": [],
                "message": "Recording status is invalid; inspect the session error.",
                "available_controls": {
                    "stop_recording": not status.terminal,
                    "exit_early": False,
                    "rerecord_episode": False,
                },
                "error": "recording worker published a contradictory session identity",
            }
        return payload
    return {
        "session_id": status.session_id,
        "recording_active": not status.terminal,
        "current_phase": "preparing" if not status.terminal else "error",
        "session_ended": status.terminal,
        "dataset_safe": False,
        "dataset_finalized": False,
        "dataset_uploaded": False,
        "upload_available": False,
        "camera_feed_available": False,
        "cameras": [],
        "message": (
            "Recording session ended without a complete worker status."
            if status.terminal
            else "Recording is preparing; waiting for the worker status."
        ),
        "available_controls": {
            "stop_recording": not status.terminal,
            "exit_early": False,
            "rerecord_episode": False,
        },
        "error": (
            status.stop_reason
            if status.terminal
            else "recording worker has not published its initial status yet"
        ),
    }


def _recording_command(
    body: ControlSessionBody | None,
    *,
    worker_method: str,
    fallback,
):  # type: ignore[no-untyped-def]
    """Dispatch one recording event without crossing the active owner."""

    coordinator = _control()
    status = coordinator.manager.active_status()
    if status is None:
        if body is None:
            return fallback()
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": "No active recording session owns this event.",
                "session_id": body.session_id,
            },
        )
    if status.operation not in _RECORDING_OPERATIONS:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": f"Control is owned by {status.operation.value}, not recording.",
                "session_id": status.session_id,
                "status": status.as_dict(),
            },
        )
    if body is not None and body.session_id != status.session_id:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": f"Session {body.session_id!r} does not own recording.",
                "session_id": status.session_id,
                "status": status.as_dict(),
            },
        )
    if status.operation is ControlOperation.LEADER_RECORDING:
        return fallback()
    if body is None:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "session_id is required for a Stadia recording event.",
                "session_id": status.session_id,
                "status": status.as_dict(),
            },
        )
    worker = coordinator.active_managed_worker(
        body.session_id,
        operation=ControlOperation.STADIA_RECORDING,
    )
    if worker is None:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": "The Stadia recording owner is no longer available.",
                "session_id": status.session_id,
                "status": status.as_dict(),
            },
        )
    method = getattr(worker, worker_method, None)
    if not callable(method):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"The recording owner does not support {worker_method}.",
                "session_id": status.session_id,
                "status": status.as_dict(),
            },
        )
    try:
        raw_result = method()
        if not isinstance(raw_result, Mapping):
            raise TypeError("recording event handlers must return a mapping")
        result = dict(raw_result)
    except Exception as error:
        latest = coordinator.status(status.session_id) or status
        failure = _control_failure(error)
        content = json.loads(failure.body)
        content.update(
            {
                "session_id": latest.session_id,
                "status": latest.as_dict(),
            }
        )
        return JSONResponse(status_code=failure.status_code, content=content)
    latest = coordinator.status(status.session_id) or status
    return _control_result(
        {
            **result,
            "session_id": latest.session_id,
            "status": latest.as_dict(),
        },
        default_error_status=409,
    )


def _stadia_dataset_guard(dataset_repo_id: str) -> str | None:
    """Explain why a retained Stadia dataset must not use the legacy uploader."""

    try:
        durable = read_dataset_safety_manifest(dataset_repo_id)
    except Exception as error:
        return f"The Stadia dataset safety record is invalid: {type(error).__name__}: {error}"
    if durable is not None:
        if durable.dataset_uploaded:
            return "The Stadia dataset was already uploaded by its recording owner."
        if not durable.dataset_safe:
            return durable.error or "The Stadia dataset is not proven safe."
        if not durable.dataset_finalized:
            return "The Stadia dataset is not proven finalized."
        if durable.saved_episodes < 1:
            return "The Stadia dataset contains no proven saved episodes."
        return None

    coordinator = _control()
    active = coordinator.manager.active_status(check_expiry=False)
    candidates = ([active] if active is not None else []) + list(
        reversed(coordinator.manager.terminal_history())
    )
    for status in candidates:
        if status.operation is not ControlOperation.STADIA_RECORDING:
            continue
        recording = status.details.get("recording")
        if not isinstance(recording, Mapping):
            continue
        if recording.get("dataset_repo_id") != dataset_repo_id:
            continue
        if not status.terminal:
            return "The Stadia recording is still active or tearing down."
        if recording.get("dataset_safe") is not True:
            return str(recording.get("error") or "The Stadia dataset is not proven safe.")
        if recording.get("dataset_finalized") is not True:
            return "The Stadia dataset is not proven finalized."
        return None
    return None


def _stop_matching_operation(
    operations: set[ControlOperation],
    *,
    reason: str,
    fallback,
):  # type: ignore[no-untyped-def]
    coordinator = _control()
    active = coordinator.manager.active_status()
    if active is None:
        return fallback()
    if active.operation not in operations:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": (f"Control is owned by {active.operation.value}; this endpoint cannot stop it."),
                "session_id": active.session_id,
                "status": active.as_dict(),
            },
        )
    try:
        return _status_response(coordinator.request_stop(active.session_id, reason=reason))
    except Exception as error:
        return _control_failure(error)


def _calibration_is_active() -> bool:
    # Avoid CalibrationManager.get_status(), which may perform a serial read.
    # The ownership monitor needs only the already-published lifecycle flag and
    # the worker's liveness. A stop handler may publish idle before a lingering
    # thread has actually released its resources.
    snapshot = calibration_manager.get_status()
    worker = calibration_manager.calibration_thread
    if worker is not None and worker.is_alive():
        return True
    return bool(snapshot.calibration_active and not snapshot.cleanup_pending)


def _capture_calibration_status() -> dict[str, Any]:
    """Retain passive legacy outcome evidence in the exact shared session."""

    from dataclasses import asdict

    snapshot = asdict(calibration_manager.get_status())
    coordinator = _control()
    active = coordinator.manager.active_status(check_expiry=False)
    if (
        active is not None
        and active.operation
        in {
            ControlOperation.FOLLOWER_CALIBRATION,
            ControlOperation.LEADER_CALIBRATION,
        }
        and not active.terminal
    ):
        coordinator.manager.merge_details(active.session_id, {"calibration": snapshot})
    return snapshot


def _resolve_inference_cameras(
    record: RobotRecordV2,
    requested: Mapping[str, InferenceCameraBinding],
) -> dict[str, dict[str, Any]]:
    """Bind policy aliases only to exact cameras in the saved robot record.

    The alias names belong to the selected policy, but every physical camera
    field is server-owned. Requiring the checkpoint dimensions to match the
    saved record prevents a browser from silently changing capture settings.
    """

    resolved: dict[str, dict[str, Any]] = {}
    used_camera_ids: set[str] = set()
    for alias, binding in requested.items():
        if (
            not alias
            or alias.strip() != alias
            or any(
                not (character.isascii() and (character.isalnum() or character in "_.-"))
                for character in alias
            )
        ):
            raise ValueError(
                "inference camera aliases may contain only ASCII letters, digits, dot, dash, and underscore"
            )
        matches = [camera for camera in record.cameras if camera.camera_index == binding.camera_index]
        if not matches:
            raise ValueError(
                f"inference camera {alias!r} does not match a camera in saved robot {record.name!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"inference camera index {binding.camera_index} is ambiguous in saved robot {record.name!r}"
            )
        camera = matches[0]
        if camera.id in used_camera_ids:
            raise ValueError(f"saved camera {camera.id!r} cannot be bound to more than one policy alias")
        if (
            binding.width != camera.width
            or binding.height != camera.height
            or (binding.fps is not None and binding.fps != camera.fps)
        ):
            raise ValueError(
                f"inference camera {alias!r} settings do not exactly match saved camera {camera.name!r}"
            )
        if camera.fourcc is not None and any(
            not (character.isascii() and (character.isalnum() or character == " "))
            for character in camera.fourcc
        ):
            raise ValueError(f"saved camera {camera.name!r} has an unsafe FOURCC value")

        used_camera_ids.add(camera.id)
        camera_config: dict[str, Any] = {
            "type": camera.type,
            "camera_index": camera.camera_index,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
        }
        if camera.fourcc is not None:
            camera_config["fourcc"] = camera.fourcc
        if camera.backend is not None:
            camera_config["backend"] = camera.backend
        resolved[alias] = camera_config
    return resolved


def _resolve_inference_request(
    raw: Mapping[str, Any],
) -> tuple[RobotRecordV2, InferenceRequest]:
    """Resolve canonical or temporary follower-only compatibility input."""

    if "robot_name" in raw:
        parsed = CanonicalInferenceBody.model_validate(raw)
        record = _require_saved_robot(parsed.robot_name)
    else:
        parsed_legacy = LegacyInferenceBody.model_validate(raw)
        matches = [
            candidate
            for candidate in _strict_robot_records()
            if candidate.follower.port == parsed_legacy.follower_port
            and candidate.follower.calibration == parsed_legacy.follower_config
        ]
        if not matches:
            raise ValueError("legacy inference identity does not exactly match a saved robot record")
        if len(matches) != 1:
            raise ValueError("legacy inference identity is ambiguous; use robot_name")
        record = matches[0]
        parsed = CanonicalInferenceBody(
            robot_name=record.name,
            policy_ref=parsed_legacy.policy_ref,
            task=parsed_legacy.task,
            cameras=parsed_legacy.cameras,
            duration_s=parsed_legacy.duration_s,
        )

    readiness = evaluate_robot_readiness(record, RobotOperation.INFERENCE)
    if not readiness.ready:
        reasons = " ".join(issue.message for issue in readiness.issues)
        raise ValueError(f"saved robot is not ready for inference: {reasons}")
    cameras = _resolve_inference_cameras(record, parsed.cameras)
    return record, InferenceRequest(
        follower_port=record.follower.port,
        follower_config=record.follower.calibration,
        policy_ref=parsed.policy_ref,
        task=parsed.task,
        cameras=cameras,
        duration_s=parsed.duration_s,
    )


@app.get("/get-configs")
def get_configs():
    # Get all available calibration configs
    leader_configs = [os.path.basename(f) for f in glob.glob(os.path.join(LEADER_CONFIG_PATH, "*.json"))]
    follower_configs = [os.path.basename(f) for f in glob.glob(os.path.join(FOLLOWER_CONFIG_PATH, "*.json"))]

    return {"leader_configs": leader_configs, "follower_configs": follower_configs}


@app.post("/move-arm")
def teleoperate_arm(request: dict[str, Any]):
    """Resolve one saved robot and start its leader or Stadia owner."""

    try:
        result = _control().start_teleoperation(request, websocket_manager=manager)
    except Exception as error:
        return _control_failure(error)
    return _control_result(result)


@app.post("/stop-teleoperation")
def stop_teleoperation(body: ControlSessionBody | None = None):
    """Compatibility stop route backed by the shared exact owner."""

    active = _control().manager.active_status()
    if active is not None and active.operation is ControlOperation.STADIA_TELEOPERATION:
        if body is None or body.session_id != active.session_id:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message": "Stadia teleoperation stop requires its exact active session_id.",
                    "session_id": active.session_id,
                    "status": active.as_dict(),
                },
            )
        try:
            return _status_response(
                _control().request_stop(
                    body.session_id,
                    reason="teleoperation stop requested by UI",
                )
            )
        except Exception as error:
            return _control_failure(error)

    return _stop_matching_operation(
        {ControlOperation.LEADER_TELEOPERATION},
        reason="teleoperation stop requested by UI",
        fallback=handle_stop_teleoperation,
    )


@app.get("/teleoperation-status")
def teleoperation_status():
    """Keep legacy fields while attaching the persistent shared status."""

    result = handle_teleoperation_status()
    status = _matching_control_status(
        {
            ControlOperation.LEADER_TELEOPERATION,
            ControlOperation.STADIA_TELEOPERATION,
        }
    )
    if status is not None:
        result.update(
            {
                "session_id": status.session_id,
                "control_status": status.as_dict(),
                "teleoperation_active": not status.terminal,
            }
        )
        result["available_controls"]["stop_teleoperation"] = not status.terminal
    return result


@app.get("/control-status")
def control_status(session_id: str | None = None):
    """Return the exact active or retained terminal session status."""

    try:
        status = _control().status(session_id)
    except Exception as error:
        return _control_failure(error)
    if status is None:
        content: dict[str, Any] = {
            "success": False,
            "message": "Control session status was not found.",
        }
        if session_id is not None:
            content["session_id"] = session_id
        return JSONResponse(
            status_code=404,
            content=content,
        )
    return _status_response(status)


@app.post("/control-lease/renew")
def renew_control_lease(body: ControlSessionBody):
    """Renew only the exact starting/running owner lease."""

    try:
        return _status_response(_control().renew_lease(body.session_id))
    except Exception as error:
        return _control_failure(error, requested_session_id=body.session_id)


@app.post("/control-stop")
def stop_control(body: ControlSessionBody):
    """Signal one exact owner; cleanup remains worker-owned and asynchronous."""

    try:
        return _status_response(
            _control().request_stop(
                body.session_id,
                reason="stop requested by control UI",
            )
        )
    except Exception as error:
        return _control_failure(error, requested_session_id=body.session_id)


@app.post("/control-speed")
def set_control_speed(body: StadiaSpeedBody):
    """Set a bounded live Stadia speed for one exact running owner."""

    try:
        return _status_response(_control().set_stadia_speed(body.session_id, body.multiplier))
    except Exception as error:
        return _control_failure(error, requested_session_id=body.session_id)


@app.post("/controller-check")
def controller_check(body: ControllerCheckBody):
    """Start the pygame-only controller check without importing a robot path."""

    try:
        result = _control().start_controller_check(body.robot_name)
    except Exception as error:
        return _control_failure(error)
    return _control_result(result, default_error_status=400)


@app.get("/joint-positions")
def get_joint_positions():
    """Get current robot joint positions"""
    return handle_get_joint_positions()


@app.post("/start-inference")
def start_inference(request: dict[str, Any]):
    try:
        record, legacy_request = _resolve_inference_request(request)
        coordinator = _control()
        owned_process: object | None = None
        retained_terminal_snapshot: dict[str, Any] | None = None

        def capture_inference_status() -> dict[str, Any]:
            nonlocal retained_terminal_snapshot
            raw_status = handle_inference_status()
            if not isinstance(raw_status, Mapping):
                raise TypeError("inference status handler must return a mapping")
            snapshot = dict(raw_status)
            if retained_terminal_snapshot is not None and not snapshot.get("inference_active"):
                snapshot = dict(retained_terminal_snapshot)
            elif not snapshot.get("inference_active") and (
                snapshot.get("exited") is True
                or snapshot.get("outcome") in {"failed", "ran_with_warning", "stopped", "completed"}
            ):
                retained_terminal_snapshot = dict(snapshot)
            active_status = coordinator.manager.active_status(check_expiry=False)
            if (
                active_status is not None
                and active_status.operation is ControlOperation.INFERENCE
                and not active_status.terminal
            ):
                coordinator.manager.merge_details(
                    active_status.session_id,
                    {"inference": snapshot},
                )
            return snapshot

        def start_owned_inference() -> Mapping[str, Any]:
            nonlocal owned_process
            result = handle_start_inference(legacy_request)
            owned_process = _rollout._inference_proc
            capture_inference_status()
            return result

        def stop_owned_inference() -> Mapping[str, Any]:
            nonlocal owned_process
            owned_process = _rollout._inference_proc or owned_process
            return handle_stop_inference()

        def owned_inference_is_active() -> bool:
            snapshot = capture_inference_status()
            active = bool(snapshot.get("inference_active"))
            if owned_process is None:
                return active
            poll = getattr(owned_process, "poll", None)
            if not callable(poll):
                raise TypeError("inference process must expose poll()")
            return active or poll() is None

        result = coordinator.start_external_operation(
            ControlOperation.INFERENCE,
            teleoperator_type=record.teleoperator_type,
            start=start_owned_inference,
            stop=stop_owned_inference,
            is_active=owned_inference_is_active,
            terminal_status=capture_inference_status,
            details={
                "robot_name": record.name,
                "policy_ref": legacy_request.policy_ref,
            },
        )
    except Exception as error:
        return _control_failure(error)
    return _control_result(result)


@app.post("/stop-inference")
def stop_inference():
    return _stop_matching_operation(
        {ControlOperation.INFERENCE},
        reason="inference stop requested by UI",
        fallback=handle_stop_inference,
    )


@app.get("/inference-status")
def inference_status():
    status = _matching_control_status({ControlOperation.INFERENCE})
    if status is not None:
        retained = status.details.get("inference")
        if isinstance(retained, Mapping):
            result = dict(retained)
        else:
            result = {
                "inference_active": not status.terminal,
                "started_at": None,
                "rollout_started_at": None,
                "elapsed_s": 0.0,
                "rollout_elapsed_s": 0.0,
                "duration_s": None,
                "policy_ref": status.details.get("policy_ref"),
                "log_path": None,
            }
        result.update(
            {
                "session_id": status.session_id,
                "control_status": status.as_dict(),
                "inference_active": not status.terminal,
            }
        )
        return result
    raw_result = handle_inference_status()
    return dict(raw_result) if isinstance(raw_result, Mapping) else {}


@app.get("/health")
def health_check():
    """Simple health check endpoint to verify server is running"""
    return {"status": "ok", "message": "FastAPI server is running"}


@app.get("/hf-auth-status")
def hf_auth_status():
    """Check whether the local HF CLI is authenticated and return user info."""
    return handle_hf_auth_status()


class HfLoginBody(BaseModel):
    token: str


@app.post("/hf-auth/login")
def hf_auth_login(body: HfLoginBody):
    """Persist a pasted HF token (validated against whoami) for this user."""
    try:
        return handle_hf_login(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/datasets")
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()


@app.get("/ws-test")
def websocket_test():
    """Test endpoint to verify WebSocket support"""
    return {"websocket_endpoint": "/ws/joint-data", "status": "available"}


@app.websocket("/ws/joint-data")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔗 New WebSocket connection attempt")
    try:
        await manager.connect(websocket)
        logger.info("✅ WebSocket connection established")

        while True:
            # Keep the connection alive and wait for messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Handle any incoming messages if needed
                logger.debug(f"Received WebSocket message: {data}")
            except TimeoutError:
                # No message received, continue
                pass
            except WebSocketDisconnect:
                logger.info("🔌 WebSocket client disconnected")
                break

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info("🧹 WebSocket connection cleaned up")


@app.post("/start-recording")
def start_recording(request: dict[str, Any]):
    """Resolve one saved robot and start its leader or Stadia recorder."""

    try:
        result = _control().start_recording(request)
    except Exception as error:
        return _control_failure(error)
    return _control_result(result)


@app.post("/stop-recording")
def stop_recording(body: ControlSessionBody | None = None):
    """Compatibility stop route backed by the shared exact owner."""

    active = _control().manager.active_status()
    if active is not None and active.operation is ControlOperation.STADIA_RECORDING:
        if body is None or body.session_id != active.session_id:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message": "Stadia recording stop requires its exact active session_id.",
                    "session_id": active.session_id,
                    "status": active.as_dict(),
                },
            )
        try:
            return _status_response(
                _control().request_stop(
                    body.session_id,
                    reason="recording stop requested by UI",
                )
            )
        except Exception as error:
            return _control_failure(error)

    return _stop_matching_operation(
        {ControlOperation.LEADER_RECORDING},
        reason="recording stop requested by UI",
        fallback=handle_stop_recording,
    )


@app.get("/recording-status")
def recording_status():
    """Keep legacy fields while attaching the persistent shared status."""

    status = _matching_control_status(_RECORDING_OPERATIONS)
    if status is not None and status.operation is ControlOperation.STADIA_RECORDING:
        # One immutable manager snapshot supplies both the compatibility fields
        # and nested control status. A second mutable worker read could combine
        # two different revisions in one response.
        result = _stadia_recording_payload(status)
        result.update(
            {
                "session_id": status.session_id,
                "control_status": status.as_dict(),
            }
        )
        return result

    result = handle_recording_status()
    if status is not None:
        active = not status.terminal
        result.update(
            {
                "session_id": status.session_id,
                "control_status": status.as_dict(),
                "recording_active": active,
                "session_ended": status.terminal,
            }
        )
        result["available_controls"]["stop_recording"] = active
    return result


@app.get("/camera-feed/{cam_key}")
def camera_feed(cam_key: str):
    """Live MJPEG preview of one configured camera during an active recording.

    Browsers render `multipart/x-mixed-replace` directly in an <img>, so the
    frontend just points an <img> at this URL. Only valid while a session is
    active; the generator ends itself when recording stops.
    """
    status = _matching_control_status(_RECORDING_OPERATIONS)
    if status is not None and status.operation is ControlOperation.STADIA_RECORDING:
        raise HTTPException(
            status_code=404,
            detail="Live camera preview is not exposed by the Stadia recording owner.",
        )
    if not _record.recording_active:
        raise HTTPException(status_code=409, detail="No active recording session")
    return StreamingResponse(
        _record.camera_feed_frames(cam_key),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/recording-exit-early")
def recording_exit_early(body: ControlSessionBody | None = None):
    """Skip to next episode (replaces right arrow key)"""
    return _recording_command(
        body,
        worker_method="finish_episode",
        fallback=handle_exit_early,
    )


@app.post("/recording-rerecord-episode")
def recording_rerecord_episode(body: ControlSessionBody | None = None):
    """Re-record current episode (replaces left arrow key)"""
    return _recording_command(
        body,
        worker_method="rerecord_episode",
        fallback=handle_rerecord_episode,
    )


@app.post("/upload-dataset")
def upload_dataset(request: UploadRequest):
    """Upload dataset to HuggingFace Hub"""
    blocked = _stadia_dataset_guard(request.dataset_repo_id)
    if blocked is not None:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": blocked,
                "dataset_repo_id": request.dataset_repo_id,
            },
        )
    durable = read_dataset_safety_manifest(request.dataset_repo_id)
    result = handle_upload_dataset(request)
    if durable is not None and isinstance(result, Mapping) and result.get("success") is True:
        try:
            write_dataset_safety_manifest(replace(durable, dataset_uploaded=True))
        except Exception as error:
            return {
                **dict(result),
                "success": False,
                "message": (
                    "Dataset upload completed, but durable upload state could not be recorded: "
                    f"{type(error).__name__}: {error}"
                ),
            }
    return result


@app.post("/dataset-info")
def get_dataset_info(request: DatasetInfoRequest):
    """Get information about a saved dataset"""
    return handle_get_dataset_info(request)


@app.post("/delete-dataset")
def delete_dataset(request: DatasetInfoRequest):
    """Remove a recorded dataset directory from local disk."""

    result = handle_delete_dataset(request)
    if isinstance(result, Mapping) and result.get("success") is True:
        try:
            delete_dataset_safety_manifest(request.dataset_repo_id)
        except Exception as error:
            return {
                "success": False,
                "message": (
                    "Dataset was removed, but its Stadia safety record could not be removed: "
                    f"{type(error).__name__}: {error}"
                ),
            }
    return result


# ============================================================================
# JOB ENDPOINTS
# ============================================================================


@app.post("/jobs/training", status_code=201)
async def create_training_job(req: Request):
    raw = await req.json()
    body = StartTrainingBody.from_legacy(raw)
    try:
        record = job_registry.start(body.config, body.target)
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job already running: {exc}") from exc
    except ValueError as exc:
        # e.g. "flavor is required when runner is hf_cloud"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


class ImportModelRequest(BaseModel):
    source: str
    name: str | None = None


@app.post("/jobs/import", status_code=201)
def import_model(body: ImportModelRequest):
    """Register an external model (local dir or HF repo) as a pseudo-job."""
    try:
        return job_registry.register_imported(body.source, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs")
def list_jobs(limit: int = 10):
    return {"jobs": job_registry.list(limit=limit)}


@app.get("/jobs/hub")
def list_hub_jobs():
    """List the user's HF Cloud compute Jobs and their uploaded LeRobot model
    repos on huggingface.co.

    Returns 200 with empty lists when no token is configured so the frontend
    can render an unauthenticated empty state without surfacing an error.

    Declared before `/jobs/{job_id}` so FastAPI's first-match routing doesn't
    treat "hub" as a job id.
    """
    info = cached_whoami()
    if info is None:
        return {"authenticated": False, "jobs": [], "models": []}
    api = shared_hf_api()

    authors: list[str] = []
    if info.get("name"):
        authors.append(info["name"])
    for o in info.get("orgs", []) or []:
        if isinstance(o, dict) and o.get("name"):
            authors.append(o["name"])

    try:
        jobs = api.list_jobs()
    except Exception as exc:
        logger.warning("list_jobs failed: %s", exc)
        jobs = []

    seen_models: set[str] = set()
    models: list[dict] = []
    for author in authors:
        try:
            for m in api.list_models(author=author, filter="LeRobot", limit=200):
                if m.id in seen_models:
                    continue
                seen_models.add(m.id)
                models.append(
                    {
                        "repo_id": m.id,
                        "last_modified": m.last_modified.isoformat() if m.last_modified else None,
                        "private": bool(getattr(m, "private", False)),
                    }
                )
        except Exception as exc:
            logger.warning("list_models(%s) failed: %s", author, exc)
    models.sort(key=lambda m: m["last_modified"] or "", reverse=True)

    return {
        "authenticated": True,
        "jobs": [
            {
                "id": ji.id,
                "created_at": ji.created_at.isoformat() if ji.created_at else None,
                "docker_image": ji.docker_image,
                "space_id": ji.space_id,
                "flavor": ji.flavor,
                "status": ({"stage": ji.status.stage, "message": ji.status.message} if ji.status else None),
                "owner": ji.owner.name if ji.owner else None,
                "url": ji.url,
            }
            for ji in jobs
        ],
        "models": models,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_registry.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        logs = job_registry.drain_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"logs": logs}


@app.get("/jobs/{job_id}/log-file")
def get_job_log_file(job_id: str):
    """Return the entire on-disk log file for a job. Drains the live queue too
    so the next /logs poll returns only lines that arrived after this call."""
    try:
        logs = job_registry.read_persisted_logs(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    # Best-effort drain so the frontend doesn't double-display.
    with contextlib.suppress(JobNotFoundError):
        job_registry.drain_logs(job_id)
    return {"logs": logs}


@app.get("/jobs/{job_id}/metrics-history")
def get_job_metrics_history(job_id: str):
    """Return the per-step loss/lr/grad-norm series reconstructed from the
    job's log.jsonl. Used to seed the monitoring charts so curves persist
    across page reloads, navigation, and lelab restarts."""
    try:
        points = job_registry.read_metrics_history(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    return {"points": points}


@app.get("/jobs/{job_id}/checkpoints")
def get_job_checkpoints(job_id: str):
    """List the checkpoints saved for this job, ascending by step."""
    try:
        return {"checkpoints": job_registry.list_checkpoints(job_id)}
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc


@app.get("/jobs/{job_id}/checkpoints/{step}/policy-config")
def get_checkpoint_policy_config(job_id: str, step: int):
    """Return the UX-relevant slice of a checkpoint's pretrained_model config:
    policy_type, image_features (per-camera height/width), and requires_task."""
    try:
        return job_registry.get_policy_config_summary(job_id, step)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    try:
        return job_registry.stop(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is not running") from exc


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        job_registry.delete(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found") from exc
    except JobNotRunningError as exc:
        raise HTTPException(status_code=409, detail=f"Job {job_id!r} is running; stop it first") from exc


@app.get("/jobs/runners/hardware")
def get_runners_hardware():
    """Return HF Jobs flavor catalog + auth state for the TargetCard.

    Both the flavors list and the whoami result are cached in-process to
    keep this endpoint cheap (it can be re-fetched whenever auth state
    changes). The whoami cache is invalidated on login.
    """
    info = cached_whoami()
    if info is None or not info.get("name"):
        return {"authenticated": False, "username": None, "flavors": []}
    username: str = info["name"]
    api = shared_hf_api()

    now = time.time()
    if _flavors_cache["data"] is None or now - _flavors_cache["fetched_at"] > _FLAVOR_CACHE_TTL_SECONDS:
        try:
            hw_list = api.list_jobs_hardware()
        except Exception as exc:
            logger.warning("list_jobs_hardware failed: %s", exc)
            return {"authenticated": True, "username": username, "flavors": []}
        _flavors_cache["data"] = [
            {
                "name": h.name,
                "pretty_name": h.pretty_name,
                "cpu": h.cpu,
                "ram": h.ram,
                "accelerator": h.accelerator,
                "unit_cost_usd": h.unit_cost_usd,
                "unit_label": h.unit_label,
            }
            for h in hw_list
        ]
        _flavors_cache["fetched_at"] = now

    return {
        "authenticated": True,
        "username": username,
        "flavors": _flavors_cache["data"],
    }


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================


@app.get("/system/cuda-status")
def get_cuda_status():
    """Report whether an NVIDIA GPU is present but PyTorch is CPU-only (issue #30)."""
    return handle_get_cuda_status()


@app.get("/system/training-extra")
def get_training_extra():
    """Return whether the LeRobot training extra (accelerate) is importable."""
    return handle_get_training_extra()


@app.post("/system/training-extra/install")
def install_training_extra():
    """Spawn `pip install accelerate` as a background subprocess. No-op if already running."""
    return handle_install_training_extra()


@app.get("/system/training-extra/install-status")
def install_training_extra_status():
    """Return current install state plus any pending log lines (drained on read)."""
    return handle_install_training_extra_status()


@app.get("/system/wandb-extra")
def get_wandb_extra():
    """Return whether the `wandb` package is importable in this lelab process."""
    return handle_get_wandb_extra()


@app.post("/system/wandb-extra/install")
def install_wandb_extra():
    """Spawn `pip install wandb` as a background subprocess. No-op if already running."""
    return handle_install_wandb_extra()


@app.get("/system/wandb-extra/install-status")
def install_wandb_extra_status():
    """Return current wandb install state plus any pending log lines (drained on read)."""
    return handle_install_wandb_extra_status()


@app.get("/system/policy-extra/{policy_type}")
def get_policy_extra(policy_type: str):
    """Whether the optional LeRobot extra a policy needs (e.g. transformers for
    smolvla/pi0, diffusers for diffusion) is importable. Core policies report available."""
    return handle_get_policy_extra(policy_type)


@app.post("/system/policy-extra/{policy_type}/install")
def install_policy_extra(policy_type: str):
    """Spawn `pip install lerobot[<extra>]` for the policy's extra in the background."""
    return handle_install_policy_extra(policy_type)


@app.get("/system/policy-extra/{policy_type}/install-status")
def install_policy_extra_status(policy_type: str):
    """Return the policy extra's install state plus any pending log lines (drained on read)."""
    return handle_install_policy_extra_status(policy_type)


@app.get("/system/update-check")
def update_check():
    """Report whether a newer LeLab commit exists on GitHub (cached, silent on failure)."""
    return handle_update_check()


@app.post("/system/update")
def run_update():
    """Run the pip upgrade in-process; the user must restart lelab afterwards."""
    return handle_run_update()


# Replay is rendered by the embedded lerobot/visualize_dataset Space; no backend routes needed.


# ============================================================================
# Calibration endpoints
@app.post("/start-calibration")
def start_calibration(request: CalibrationRequest):
    """Start one calibration owner while preserving the legacy worker."""

    try:
        if not is_valid_robot_name(request.config_file):
            raise ValueError("config_file must be a safe non-empty basename")
        operation = (
            ControlOperation.FOLLOWER_CALIBRATION
            if request.device_type == "robot"
            else ControlOperation.LEADER_CALIBRATION
        )
        resolved_request = request
        details: dict[str, Any] = {
            "device_type": request.device_type,
            "config_file": request.config_file,
        }
        target_calibration = f"{request.config_file}.json"
        if request.robot_name is None:
            matches: list[RobotRecordV2] = []
            for candidate in _strict_robot_records():
                device = candidate.follower if request.device_type == "robot" else candidate.leader
                if (
                    device is not None
                    and device.port == request.port
                    and device.calibration == target_calibration
                ):
                    matches.append(candidate)
            if not matches:
                raise ValueError("legacy calibration identity does not match a saved robot; send robot_name")
            if len(matches) != 1:
                raise ValueError("legacy calibration identity is ambiguous; send robot_name")
            record = matches[0]
            resolved_request = replace(request, robot_name=record.name)
        else:
            record = _require_saved_robot(request.robot_name)

        robot_operation = (
            RobotOperation.FOLLOWER_CALIBRATION
            if request.device_type == "robot"
            else RobotOperation.LEADER_CALIBRATION
        )
        readiness = evaluate_robot_readiness(record, robot_operation)
        if not readiness.ready:
            reasons = " ".join(issue.message for issue in readiness.issues)
            raise ValueError(f"saved robot is not ready for calibration: {reasons}")
        device = record.follower if request.device_type == "robot" else record.leader
        if device is None:
            raise ValueError("saved robot has no leader configuration")
        if request.port != device.port:
            raise ValueError("calibration port must match the saved robot record")
        if target_calibration != device.calibration:
            raise ValueError("calibration destination must match the saved robot record")
        teleoperator_type = record.teleoperator_type
        details["robot_name"] = record.name

        result = _control().start_external_operation(
            operation,
            teleoperator_type=teleoperator_type,
            start=lambda: calibration_manager.start_calibration(resolved_request),
            stop=calibration_manager.stop_calibration_process,
            is_active=_calibration_is_active,
            terminal_status=_capture_calibration_status,
            details=details,
        )
    except Exception as error:
        return _control_failure(error)
    return _control_result(result)


@app.post("/stop-calibration")
def stop_calibration(body: ControlSessionBody | None = None):
    """Signal the exact calibration owner and let it clean up."""

    active = _control().manager.active_status()
    calibration_operations = {
        ControlOperation.FOLLOWER_CALIBRATION,
        ControlOperation.LEADER_CALIBRATION,
    }
    if active is not None and active.operation in calibration_operations:
        if body is None or body.session_id != active.session_id:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message": "Calibration stop requires its exact active session_id.",
                    "session_id": active.session_id,
                    "status": active.as_dict(),
                },
            )
        try:
            return _status_response(
                _control().request_stop(
                    body.session_id,
                    reason="calibration stop requested by UI",
                )
            )
        except Exception as error:
            return _control_failure(error)

    return _stop_matching_operation(
        calibration_operations,
        reason="calibration stop requested by UI",
        fallback=calibration_manager.stop_calibration_process,
    )


@app.get("/calibration-status")
def calibration_status():
    """Get legacy calibration data plus the shared lifecycle status."""
    from dataclasses import asdict

    status = calibration_manager.get_status()
    result = asdict(status)
    control = _matching_control_status(
        {
            ControlOperation.FOLLOWER_CALIBRATION,
            ControlOperation.LEADER_CALIBRATION,
        }
    )
    if control is not None:
        result.update(
            {
                "session_id": control.session_id,
                "control_status": control.as_dict(),
                "calibration_active": not control.terminal,
            }
        )
    return result


@app.post("/complete-calibration-step")
def complete_calibration_step(body: ControlSessionBody):
    """Advance only the exact active calibration owner."""

    active = _control().manager.active_status()
    if (
        active is None
        or active.operation
        not in {
            ControlOperation.FOLLOWER_CALIBRATION,
            ControlOperation.LEADER_CALIBRATION,
        }
        or active.session_id != body.session_id
        or active.state not in {ControlState.STARTING, ControlState.RUNNING}
    ):
        content: dict[str, Any] = {
            "success": False,
            "message": "Calibration step requires its exact active session_id.",
            "session_id": body.session_id,
        }
        if active is not None:
            content["active_session_id"] = active.session_id
            content["status"] = active.as_dict()
        return JSONResponse(status_code=409, content=content)
    try:
        raw = calibration_manager.complete_step()
        result = dict(raw) if isinstance(raw, Mapping) else {"success": False, "message": "Invalid result"}
        latest = _control().status(body.session_id)
        if latest is None:
            raise RuntimeError("calibration session status disappeared")
        return _control_result(
            {
                **result,
                "session_id": body.session_id,
                "status": latest.as_dict(),
            }
        )
    except Exception as error:
        return _control_failure(error)


@app.get("/calibration-configs/{device_type}")
def get_calibration_configs(device_type: str):
    """Get all calibration config files for a specific device type"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # Get all JSON files in the config directory
        configs = []
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(".json"):
                    config_name = os.path.splitext(file)[0]
                    file_path = os.path.join(config_path, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)

                    configs.append(
                        {
                            "name": config_name,
                            "filename": file,
                            "size": file_size,
                            "modified": modified_time,
                        }
                    )

        return {"success": True, "configs": configs, "device_type": device_type}

    except Exception as e:
        logger.error(f"Error getting calibration configs: {e}")
        return {"success": False, "message": str(e)}


@app.delete("/calibration-configs/{device_type}/{config_name}")
def delete_calibration_config(device_type: str, config_name: str):
    """Delete a calibration config file"""
    try:
        if device_type == "robot":
            config_path = FOLLOWER_CONFIG_PATH
        elif device_type == "teleop":
            config_path = LEADER_CONFIG_PATH
        else:
            return {"success": False, "message": "Invalid device type"}

        # config_name is interpolated into a filename, so reject path-traversal
        # characters (/, \, ..) before touching the filesystem. Defense-in-depth:
        # FastAPI path params already block a literal "/", but not "\" or "..".
        # Reuses the same guard already applied to robot-record deletes.
        if not is_valid_robot_name(config_name):
            return {"success": False, "message": "Invalid configuration name"}

        # Construct the file path
        filename = f"{config_name}.json"
        file_path = os.path.join(config_path, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": "Configuration file not found"}

        # Delete the file
        os.remove(file_path)
        logger.info(f"Deleted calibration config: {file_path}")

        return {
            "success": True,
            "message": f"Configuration '{config_name}' deleted successfully",
        }

    except Exception as e:
        logger.error(f"Error deleting calibration config: {e}")
        return {"success": False, "message": str(e)}


# ============================================================================
# PORT DETECTION ENDPOINTS
# ============================================================================


@app.get("/available-ports")
def get_available_ports():
    """Get all available serial ports"""
    try:
        ports = find_available_ports()
        return {"status": "success", "ports": ports}
    except Exception as e:
        logger.error(f"Error getting available ports: {e}")
        return {"status": "error", "message": str(e)}


# Runs in a fresh Python — see _avfoundation_cameras_in_cv2_order for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def _avfoundation_cameras_in_cv2_order() -> list[dict[str, Any]]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def _generic_cv2_cameras(backend) -> list[dict[str, Any]]:
    """Last-resort enumeration: probe cv2 indices with placeholder names."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        cap.release()
        if opened:
            cameras.append({"index": i, "name": f"Camera {i}", "available": True})
    return cameras


def _windows_cameras() -> list[dict[str, Any]]:
    """Enumerate Windows cameras with their real DirectShow names.

    pygrabber lists DirectShow video devices in the same order cv2's DSHOW
    backend indexes them (which recording is pinned to), so the returned index
    matches what ``cv2.VideoCapture(i, CAP_DSHOW)`` opens. The real names let the
    frontend match each index to the browser's ``MediaDeviceInfo.label`` for the
    live preview. Falls back to generic names if pygrabber is unavailable.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
    except Exception as e:  # ImportError, or a COM/DirectShow failure
        logger.warning("pygrabber unavailable; using generic camera names: %s", e)
        import cv2

        return _generic_cv2_cameras(cv2.CAP_DSHOW)
    return [{"index": i, "name": name, "available": True} for i, name in enumerate(names)]


def _v4l2_camera_name(index: int) -> str | None:
    """Real camera name for /dev/video{index} from sysfs (Linux, no deps)."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _linux_cameras() -> list[dict[str, Any]]:
    """Enumerate Linux cameras, naming each from sysfs (no extra deps)."""
    import cv2

    cameras: list[dict[str, Any]] = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        cameras.append({"index": i, "name": _v4l2_camera_name(i) or f"Camera {i}", "available": True})
    return cameras


@app.get("/available-cameras")
def get_available_cameras():
    """List cameras with the same index ordering cv2 will use to record.

    Each platform enumerates in the order its cv2 backend indexes devices, and
    pairs each index with the device's real name so the frontend can match it to
    the browser's ``MediaDeviceInfo.label`` for the live preview:
      - macOS: AVFoundation ``localizedName`` (via a PyObjC subprocess);
      - Windows: DirectShow FriendlyName (via pygrabber; recording pinned DSHOW);
      - Linux: the v4l2 device name from sysfs.
    Without real names the frontend can't match a camera and shows "No browser
    match" with an empty device_id (issues #12, #16).
    """
    try:
        import platform

        system = platform.system()

        if system == "Darwin":
            cameras = _avfoundation_cameras_in_cv2_order()
            for cam in cameras:
                cam["available"] = True
            return {"status": "success", "cameras": cameras}
        if system == "Windows":
            return {"status": "success", "cameras": _windows_cameras()}
        if system == "Linux":
            return {"status": "success", "cameras": _linux_cameras()}

        import cv2

        return {"status": "success", "cameras": _generic_cv2_cameras(cv2.CAP_ANY)}
    except ImportError:
        logger.warning("OpenCV not available for camera detection")
        return {"status": "success", "cameras": []}
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {"status": "error", "message": str(e), "cameras": []}


RobotSideLiteral = Literal["leader", "follower"]


class PortDetectionBody(BaseModel):
    robot_type: RobotSideLiteral = "follower"


class PortDisconnectBody(BaseModel):
    ports_before: list[str]


class SaveRobotPortBody(BaseModel):
    robot_type: RobotSideLiteral
    port: str


class SaveRobotConfigBody(BaseModel):
    robot_type: RobotSideLiteral
    config_name: str


@app.post("/start-port-detection")
def start_port_detection(body: PortDetectionBody):
    """Snapshot available ports so the follow-up /detect-port-after-disconnect
    call can diff them."""
    result = find_robot_port(body.robot_type)
    return {"status": "success", "data": result}


@app.post("/detect-port-after-disconnect")
def detect_port_after_disconnect_endpoint(body: PortDisconnectBody):
    """Block up to 15s waiting for one port from `ports_before` to disappear."""
    try:
        detected_port = detect_port_after_disconnect(body.ports_before)
    except OSError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    return {"status": "success", "port": detected_port}


@app.post("/save-robot-port")
def save_robot_port_endpoint(body: SaveRobotPortBody):
    """Save a robot port for future use"""
    save_robot_port(body.robot_type, body.port)
    return {"status": "success", "message": f"Port {body.port} saved for {body.robot_type}"}


@app.get("/robot-port/{robot_type}")
def get_robot_port(robot_type: RobotSideLiteral):
    """Get the saved port for a robot type"""
    saved_port = get_saved_robot_port(robot_type)
    default_port = get_default_robot_port(robot_type)
    return {"status": "success", "saved_port": saved_port, "default_port": default_port}


@app.post("/save-robot-config")
def save_robot_config_endpoint(body: SaveRobotConfigBody):
    """Save a robot configuration for future use"""
    if not config.save_robot_config(body.robot_type, body.config_name):
        raise HTTPException(status_code=500, detail="Failed to save configuration")
    return {"status": "success", "message": f"Configuration saved for {body.robot_type}"}


@app.get("/robot-config/{robot_type}")
def get_robot_config(robot_type: RobotSideLiteral, available_configs: str = ""):
    """Get the saved configuration for a robot type"""
    available_configs_list = [c.strip() for c in available_configs.split(",") if c.strip()]
    saved_config = config.get_saved_robot_config(robot_type)
    default_config = config.get_default_robot_config(robot_type, available_configs_list)
    return {"status": "success", "saved_config": saved_config, "default_config": default_config}


# ============================================================================
# Robot config records (named robots)


def _record_for_api(record: RobotRecordV2) -> dict[str, Any]:
    """Attach static readiness and controller-check exclusivity to V2 JSON."""

    readiness = {
        operation: result.model_dump(mode="json")
        for operation, result in evaluate_all_robot_readiness(record).items()
    }
    active = _control().manager.active_status()
    controller = readiness[RobotOperation.CONTROLLER_CHECK.value]
    if active is not None:
        controller["ready"] = False
        controller["issues"] = [
            *controller["issues"],
            {
                "code": "control_session_busy",
                "field": None,
                "message": (
                    f"Control is currently owned by {active.operation.value}; wait for teardown to finish."
                ),
            },
        ]
    return {
        **record.model_dump(mode="json"),
        "readiness": readiness,
        # Temporary compatibility for screens not yet migrated to an exact
        # operation readiness key.
        "is_clean": is_robot_record_clean(record),
    }


@app.get("/robots")
def get_robots():
    """List every saved record as canonical V2 or fail closed."""
    try:
        records = [_record_for_api(record) for record in _strict_robot_records()]
        return {"status": "success", "robots": records}
    except Exception as error:
        logger.error("Error listing robots: %s", error)
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(error), "robots": []},
        )


@app.get("/robots/{name}")
def get_robot(name: str):
    """Get one strict canonical V2 record by name."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    try:
        record = _require_saved_robot(name)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})
    except RobotRecordValidationError as error:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(error)},
        )
    return {"status": "success", "robot": _record_for_api(record)}


@app.post("/robots/{name}")
def upsert_robot(name: str, data: dict, create: bool = False):
    """
    Upsert a robot record.

    - `?create=true` is the "Add Robot" path: returns 409 if a record with that
      name already exists; otherwise creates with empty fields then merges body.
    - Without `?create=true` is the "patch" path (e.g., calibration write-back):
      merges body into existing record. If no record exists, no-ops and returns
      success — see deletion-during-calibration edge case in the spec.
    """
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    try:
        path = Path(config.ROBOTS_PATH) / f"{name}.json"
        if create:
            if path.exists():
                return JSONResponse(
                    status_code=409,
                    content={"status": "error", "message": "A robot with this name already exists"},
                )
            record = save_robot_record_v2(name, data or {}, allow_create=True)
        else:
            record = save_robot_record_v2(name, data or {}, allow_create=False)
        if record is None:
            return {"status": "success", "robot": None}
        return {"status": "success", "robot": _record_for_api(record)}
    except RobotRecordValidationError as error:
        logger.error("Invalid robot update for %s: %s", name, error)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(error)},
        )
    except Exception as error:
        logger.error("Error upserting robot %s: %s", name, error)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(error)})


@app.delete("/robots/{name}")
def delete_robot(name: str):
    """Delete a robot record."""
    if not is_valid_robot_name(name):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid robot name"})
    if delete_robot_record(name):
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Robot not found"})


def _accepts_html(accept: str) -> bool:
    """Whether an Accept header explicitly wants text/html (quality > 0).

    Browser navigations list `text/html` with a positive quality value, so
    they get the SPA shell. A `text/html;q=0` entry is an explicit refusal and
    must not count — a plain substring check would wrongly treat it as a yes.
    `*/*` (curl, XHR, API clients) is deliberately not treated as wanting HTML.
    """
    for part in accept.split(","):
        media_type, _, params = part.strip().partition(";")
        if media_type.strip().lower() != "text/html":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown client-side routes.

    The frontend is a single-page app: routes like /recording and /calibration
    exist only in the browser's router, not as files on disk. A hard reload or
    deep link to one of those URLs asks the server for a file that isn't there;
    plain StaticFiles answers 404 ({"detail":"Not Found"}), so the page breaks.

    Here we fall back to index.html on 404 so the SPA boots and its router
    renders the route. Only requests that accept HTML (i.e. browser navigations)
    get the fallback — API typos, XHR, and curl still receive a JSON 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and _accepts_html(Headers(scope=scope).get("accept", "")):
                return await super().get_response("index.html", scope)
            raise


# Serve the built frontend at /. Must be mounted last so API routes win.
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    logger.warning(
        f"frontend/dist not found at {FRONTEND_DIST}; run `npm run build` in frontend/ or use `lelab --dev`."
    )
