"""Device cleanup helpers for LeRobot hardware wrappers.

Serial ports are a trust boundary on Windows: if a normal disconnect fails
while disabling torque, the COM handle can stay open until the Python process
exits. These helpers preserve LeRobot's normal disconnect behavior, then force
close the underlying port/cameras as a last resort.
"""

from __future__ import annotations

import logging
from typing import Any


class DeviceCleanupError(RuntimeError):
    """Raised after best-effort cleanup reports any teardown error."""

    def __init__(self, context: str, errors: list[str], *, cleanup_proven: bool) -> None:
        self.context = context
        self.errors = tuple(errors)
        self.cleanup_proven = cleanup_proven
        proof = "cleanup proven" if cleanup_proven else "cleanup unproven"
        super().__init__(f"{context} reported errors ({proof}): {'; '.join(errors)}")


def _disconnect_postcondition(
    device: Any,
    camera_threads: dict[str, object],
) -> tuple[bool, list[str]]:
    """Require every available pinned LeRobot connection flag to be False."""

    errors: list[str] = []
    sentinel = object()
    try:
        device_connected = getattr(device, "is_connected", sentinel)
    except Exception as exc:
        errors.append(f"device disconnect postcondition raised {type(exc).__name__}: {exc}")
        device_connected = sentinel

    bus = getattr(device, "bus", None)
    try:
        bus_connected = getattr(bus, "is_connected", sentinel)
    except Exception as exc:
        errors.append(f"bus disconnect postcondition raised {type(exc).__name__}: {exc}")
        bus_connected = sentinel

    if device_connected is sentinel and bus_connected is sentinel:
        errors.append("disconnect postcondition is unavailable")
    elif device_connected is not sentinel and device_connected is not False:
        errors.append(f"device disconnect postcondition is not false: {device_connected!r}")
    elif bus_connected is not sentinel and bus_connected is not False:
        errors.append(f"bus disconnect postcondition is not false: {bus_connected!r}")

    cameras = getattr(device, "cameras", None)
    if isinstance(cameras, dict):
        for name, camera in cameras.items():
            try:
                connected = getattr(camera, "is_connected", sentinel)
            except Exception as exc:
                errors.append(f"camera {name!r} disconnect postcondition raised {type(exc).__name__}: {exc}")
                continue
            if connected is sentinel:
                errors.append(f"camera {name!r} disconnect postcondition is unavailable")
            elif connected is not False:
                errors.append(f"camera {name!r} disconnect postcondition is not false: {connected!r}")

    for name, thread in camera_threads.items():
        is_alive = getattr(thread, "is_alive", None)
        if not callable(is_alive):
            errors.append(f"camera {name!r} read-thread liveness is unavailable")
            continue
        try:
            if is_alive():
                errors.append(f"camera {name!r} read thread is still alive")
        except Exception as exc:
            errors.append(f"camera {name!r} read-thread liveness raised {type(exc).__name__}: {exc}")

    return not errors, errors


def _capture_camera_threads(device: Any) -> tuple[dict[str, object], list[str]]:
    threads: dict[str, object] = {}
    errors: list[str] = []
    cameras = getattr(device, "cameras", None)
    if not isinstance(cameras, dict):
        return threads, errors
    for name, camera in cameras.items():
        try:
            thread = getattr(camera, "thread", None)
        except Exception as exc:
            errors.append(f"camera {name!r} read-thread capture raised {type(exc).__name__}: {exc}")
            continue
        if thread is not None:
            threads[name] = thread
    return threads, errors


def _join_camera_threads(camera_threads: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for name, thread in camera_threads.items():
        is_alive = getattr(thread, "is_alive", None)
        join = getattr(thread, "join", None)
        if not callable(is_alive) or not callable(join):
            errors.append(f"camera {name!r} read-thread join contract is unavailable")
            continue
        try:
            if is_alive():
                join(timeout=2)
        except Exception as exc:
            errors.append(f"camera {name!r} read-thread join failed with {type(exc).__name__}: {exc}")
    return errors


def safe_disconnect_device(device: Any, logger: logging.Logger, context: str = "cleanup") -> bool:
    """Best-effort all release paths, then return only strict cleanup proof.

    Any normal-disconnect, force-close, camera, or postcondition failure is
    aggregated and raised after all cleanup attempts. Callers can therefore
    retain/quarantine ownership instead of silently reusing uncertain hardware.
    """
    if device is None:
        return True

    errors: list[str] = []
    camera_threads, capture_errors = _capture_camera_threads(device)
    errors.extend(capture_errors)
    try:
        device.disconnect()
    except Exception as exc:
        logger.warning("Error disconnecting device during %s: %s", context, exc)
        errors.append(f"disconnect failed with {type(exc).__name__}: {exc}")

    errors.extend(_join_camera_threads(camera_threads))
    cleanup_proven, proof_errors = _disconnect_postcondition(device, camera_threads)
    cleanup_proven = cleanup_proven and not capture_errors
    if not cleanup_proven:
        errors.extend(proof_errors)
        # A disconnect may return without releasing every resource, and a
        # failed disconnect may still have partially closed the device. In
        # either case, run every bounded force-close attempt, then recheck.
        errors.extend(_force_close_device_resources(device, logger))
        errors.extend(_join_camera_threads(camera_threads))
        cleanup_proven, final_proof_errors = _disconnect_postcondition(device, camera_threads)
        cleanup_proven = cleanup_proven and not capture_errors
        if not cleanup_proven:
            errors.extend(f"after force-close: {error}" for error in final_proof_errors)

    if errors:
        raise DeviceCleanupError(context, errors, cleanup_proven=cleanup_proven)
    return True


def _force_close_device_resources(device: Any, logger: logging.Logger) -> list[str]:
    """Best-effort release for serial/camera resources after disconnect fails."""
    errors: list[str] = []
    bus = getattr(device, "bus", None)
    port_handler = getattr(bus, "port_handler", None)
    if port_handler is not None:
        try:
            port_handler.clearPort()
        except Exception as exc:
            errors.append(f"serial clear failed with {type(exc).__name__}: {exc}")
        try:
            port_handler.is_using = False
        except Exception as exc:
            errors.append(f"serial ownership reset failed with {type(exc).__name__}: {exc}")
        try:
            port_handler.closePort()
            logger.info("Force-closed serial port after disconnect failure")
        except Exception as exc:
            logger.warning("Failed to force-close serial port after disconnect failure: %s", exc)
            errors.append(f"serial close failed with {type(exc).__name__}: {exc}")

    cameras = getattr(device, "cameras", None)
    if isinstance(cameras, dict):
        for name, cam in cameras.items():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect camera after device cleanup failure: %s", exc)
                errors.append(f"camera {name!r} disconnect failed with {type(exc).__name__}: {exc}")
    return errors
