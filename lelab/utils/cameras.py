"""Camera configuration shared by recording, teleoperation and subprocesses.

Only installed LeRobot plugins are discovered. A saved record never names a
Python module to import, and local OpenCV defaults remain unchanged.
"""

import logging
import math
import platform
from collections.abc import Mapping
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)


def _fps_matches_requested(requested: object, actual: object) -> bool:
    """Treat tiny backend clock rounding as the requested frame rate."""

    try:
        requested_fps = float(requested)
        actual_fps = float(actual)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(requested_fps)
        and math.isfinite(actual_fps)
        and math.isclose(
            requested_fps,
            actual_fps,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )
    )


@lru_cache(maxsize=1)
def install_macos_opencv_fps_compatibility() -> None:
    """Accept AVFoundation's false setter result when the FPS readback matches.

    Some UVC cameras, including the Opal C1, return ``False`` from
    ``CAP_PROP_FPS`` while immediately reporting a clock such as 30.00003 FPS.
    LeRobot otherwise rejects this usable stream before robot construction.
    """

    if platform.system() != "Darwin":
        return

    import cv2

    from lerobot.cameras.opencv import OpenCVCamera

    original = OpenCVCamera._validate_fps
    if getattr(original, "_lelab_macos_fps_compatibility", False):
        return

    @wraps(original)
    def validate_fps(self) -> None:
        try:
            original(self)
        except RuntimeError:
            capture = getattr(self, "videocapture", None)
            requested = getattr(self, "fps", None)
            actual = capture.get(cv2.CAP_PROP_FPS) if capture is not None else None
            if not _fps_matches_requested(requested, actual):
                raise
            logger.warning(
                "AVFoundation rejected the FPS setter but confirmed a matching rate "
                "(requested=%s, actual=%s); continuing",
                requested,
                actual,
            )

    validate_fps._lelab_macos_fps_compatibility = True  # type: ignore[attr-defined]
    OpenCVCamera._validate_fps = validate_fps


@lru_cache(maxsize=1)
def register_camera_plugins():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()


def camera_projection(camera):
    values = {
        "type": camera.type,
        "width": camera.width,
        "height": camera.height,
        "fps": camera.fps,
    }
    if camera.type == "opencv":
        values["camera_index"] = camera.camera_index
        for key in ("fourcc", "backend"):
            if getattr(camera, key, None) is not None:
                values[key] = getattr(camera, key)
    else:
        values["parameters"] = dict(camera.parameters)
    return values


def camera_cli_config(raw):
    """Flatten plugin parameters into LeRobot's native config, without shadowing."""
    if not isinstance(raw, Mapping):
        raise ValueError("camera configuration must be an object")
    if raw.get("type", "opencv") == "opencv":
        return {
            ("index_or_path" if k == "camera_index" else k): v
            for k, v in raw.items()
            if v is not None and k != "parameters"
        }
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping) or set(parameters) & {"type", "width", "height", "fps"}:
        raise ValueError("plugin parameters cannot override type, width, height or fps")
    unknown = set(raw) - {"type", "width", "height", "fps", "parameters"}
    if unknown:
        raise ValueError(f"unexpected plugin camera fields: {sorted(unknown)}")
    return {**parameters, **{k: raw[k] for k in ("type", "width", "height", "fps") if k in raw}}


def build_camera_configs(cameras, default_backend):
    install_macos_opencv_fps_compatibility()

    import draccus

    from lerobot.cameras import CameraConfig
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv import OpenCVCameraConfig

    result = {}
    for name, raw in cameras.items():
        if raw.get("type", "opencv") == "opencv":
            result[name] = OpenCVCameraConfig(
                index_or_path=raw.get("camera_index", 0),
                backend=Cv2Backends[raw["backend"]] if raw.get("backend") else default_backend,
                fourcc=raw.get("fourcc") or None,
                width=raw.get("width"),
                height=raw.get("height"),
                fps=raw.get("fps"),
            )
        else:
            register_camera_plugins()
            result[name] = draccus.decode(CameraConfig, camera_cli_config(raw))
    return result
