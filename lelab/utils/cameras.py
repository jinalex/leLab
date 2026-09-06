"""Camera configuration shared by recording, teleoperation and subprocesses.

Only installed LeRobot plugins are discovered. A saved record never names a
Python module to import, and local OpenCV defaults remain unchanged.
"""

from collections.abc import Mapping
from functools import lru_cache


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
