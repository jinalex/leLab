"""Thread-owned, lazy pygame reader for the exact Stadia controller profile."""

from __future__ import annotations

import importlib
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import STADIA_LAYOUT, STADIA_PRODUCT_NAME, ControllerLayout, StadiaSnapshot


class StadiaReaderError(RuntimeError):
    """The controller reader cannot safely publish a usable sample."""


@dataclass(frozen=True)
class _DeviceIdentity:
    product_name: str
    guid: str
    instance_id: int
    layout: ControllerLayout


class StadiaDeviceReader:
    """Own pygame and publish immutable controller snapshots from one thread."""

    def __init__(
        self,
        *,
        expected_guid: str | None = None,
        poll_interval_s: float = 1 / 120,
        reconnect_interval_s: float = 0.25,
        pygame_module: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if expected_guid is not None and not expected_guid:
            raise ValueError("expected_guid must be non-empty when provided")
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and positive")
        if not math.isfinite(reconnect_interval_s) or reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s must be finite and non-negative")

        self.expected_guid = expected_guid
        self.poll_interval_s = float(poll_interval_s)
        self.reconnect_interval_s = float(reconnect_interval_s)
        self._injected_pygame = pygame_module
        self._pygame: Any | None = None
        self._clock = clock
        self._sleeper = sleeper

        self._snapshot_condition = threading.Condition()
        self._latest = StadiaSnapshot(
            sequence=0,
            sampled_at=0.0,
            connected=False,
            product_name=None,
            guid=None,
            instance_id=None,
            connection_generation=0,
            read_error="Stadia reader not started",
        )
        self._sequence = 0
        self._generation = 0

        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._initialized = False
        self._cleanup_error: str | None = None

        # These fields are exclusively read or written by the reader thread.
        self._selected: Any | None = None
        self._selected_identity: _DeviceIdentity | None = None
        self._last_identity: _DeviceIdentity | None = None
        self._pinned_guid: str | None = expected_guid
        self._last_error = "No Google Stadia Controller detected"
        self._next_connect_at = -float("inf")

    @property
    def initialized(self) -> bool:
        with self._lifecycle_lock:
            return self._initialized

    @property
    def is_alive(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the sole pygame-owning reader thread."""

        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("Stadia reader instances may only be started once")
            if self._stop_event.is_set():
                raise RuntimeError("Stadia reader is already closed")
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="lelab-stadia-reader",
                daemon=False,
            )
            self._thread.start()

    def snapshot(self) -> StadiaSnapshot:
        """Return the latest immutable publication without touching pygame."""

        with self._snapshot_condition:
            return self._latest

    def wait_for_snapshot(self, *, after_sequence: int, timeout: float) -> StadiaSnapshot:
        """Wait for a newer publication; intended for session startup and tests."""

        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and non-negative")
        with self._snapshot_condition:
            advanced = self._snapshot_condition.wait_for(
                lambda: self._latest.sequence > after_sequence,
                timeout=timeout,
            )
            if not advanced:
                raise TimeoutError("timed out waiting for a Stadia controller sample")
            return self._latest

    def stop(self, *, timeout: float = 2.0) -> None:
        """Request shutdown and join after the reader thread closes pygame."""

        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and non-negative")
        self._stop_event.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is None:
            return
        if thread is threading.current_thread():
            raise RuntimeError("the Stadia reader cannot join itself")
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("Stadia reader did not stop before the join timeout")
        with self._lifecycle_lock:
            cleanup_error = self._cleanup_error
        if cleanup_error is not None:
            raise StadiaReaderError(cleanup_error)

    close = stop

    def _run(self) -> None:
        stopped_normally = False
        try:
            self._initialize_pygame()
            while not self._stop_event.is_set():
                try:
                    self._cycle()
                except Exception as error:
                    identity = self._selected_identity or self._last_identity
                    cleanup_errors = self._disconnect_selected(schedule_reconnect=True)
                    self._last_error = self._join_errors(
                        self._format_error("Controller read failed", error),
                        cleanup_errors,
                    )
                    self._publish_disconnected(self._last_error, identity=identity)
                if not self._stop_event.is_set():
                    self._sleeper(self.poll_interval_s)
            stopped_normally = True
        except Exception as error:
            self._last_error = self._format_error("Stadia reader failed", error)
            self._publish_disconnected(self._last_error)
        finally:
            identity = self._selected_identity or self._last_identity
            cleanup_errors = [
                *self._disconnect_selected(schedule_reconnect=False),
                *self._shutdown_pygame(),
            ]
            if cleanup_errors:
                cleanup_error = self._join_errors("Stadia reader cleanup failed", cleanup_errors)
                with self._lifecycle_lock:
                    self._cleanup_error = cleanup_error
                self._publish_disconnected(cleanup_error, identity=identity)
            elif stopped_normally:
                self._publish_disconnected("Stadia reader stopped", identity=identity)

    def _initialize_pygame(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        pygame = self._injected_pygame
        if pygame is None:
            try:
                pygame = importlib.import_module("pygame")
            except ImportError as error:
                raise StadiaReaderError("pygame is not installed") from error
        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        with self._lifecycle_lock:
            self._initialized = True

    def _shutdown_pygame(self) -> tuple[str, ...]:
        errors: list[str] = []
        pygame = self._pygame
        if pygame is not None:
            try:
                pygame.joystick.quit()
            except Exception as error:
                errors.append(self._format_error("pygame joystick quit failed", error))
            try:
                pygame.quit()
            except Exception as error:
                errors.append(self._format_error("pygame quit failed", error))
        with self._lifecycle_lock:
            self._initialized = False
        self._pygame = None
        return tuple(errors)

    def _cycle(self) -> None:
        pygame = self._pygame
        if pygame is None:
            raise StadiaReaderError("pygame is not initialized")

        pygame.event.pump()
        events = pygame.event.get(pump=False)
        for event in events:
            if event.type == pygame.JOYDEVICEREMOVED:
                if (
                    self._selected_identity is not None
                    and getattr(event, "instance_id", None) == self._selected_identity.instance_id
                ):
                    identity = self._selected_identity
                    cleanup_errors = self._disconnect_selected(schedule_reconnect=True)
                    self._last_error = self._join_errors(
                        "Selected Stadia controller was removed",
                        cleanup_errors,
                    )
                    self._publish_disconnected(self._last_error, identity=identity)
                    return
            elif event.type == pygame.JOYDEVICEADDED:
                if self._selected is None:
                    self._next_connect_at = -float("inf")
                elif self._added_device_conflicts(getattr(event, "device_index", None)):
                    identity = self._selected_identity
                    cleanup_errors = self._disconnect_selected(schedule_reconnect=True)
                    self._last_error = self._join_errors(
                        "Another matching Stadia controller was added",
                        cleanup_errors,
                    )
                    self._publish_disconnected(self._last_error, identity=identity)
                    return

        if self._selected is None:
            now = float(self._clock())
            if now >= self._next_connect_at:
                self._next_connect_at = now + self.reconnect_interval_s
                self._select_controller()

        if self._selected is None:
            self._publish_disconnected(self._last_error)
            return
        self._read_selected()

    def _select_controller(self) -> None:
        pygame = self._pygame
        if pygame is None:
            raise StadiaReaderError("pygame is not initialized")

        handles: list[Any] = []
        candidates: list[tuple[Any, _DeviceIdentity]] = []
        try:
            for index in range(pygame.joystick.get_count()):
                handle = pygame.joystick.Joystick(index)
                handles.append(handle)
                if handle.get_name() != STADIA_PRODUCT_NAME:
                    continue
                candidates.append((handle, self._describe(handle)))

            selected_guid = self._pinned_guid
            if selected_guid is None:
                eligible = candidates
                if len(eligible) > 1:
                    raise StadiaReaderError(
                        "Multiple Google Stadia Controller devices detected; configure a GUID"
                    )
            else:
                eligible = [candidate for candidate in candidates if candidate[1].guid == selected_guid]
                if not eligible:
                    qualifier = "Configured" if self.expected_guid is not None else "Pinned"
                    raise StadiaReaderError(f"{qualifier} Stadia GUID {selected_guid!r} was not found")
                if len(eligible) > 1:
                    raise StadiaReaderError(
                        f"Multiple Stadia controllers match selected GUID {selected_guid!r}"
                    )

            if not eligible:
                self._last_error = "No Google Stadia Controller detected"
                return

            chosen, identity = eligible[0]
            self._validate_layout(identity.layout)
            other_handles = [handle for handle in handles if handle is not chosen]
            cleanup_errors = self._close_handles(other_handles)
            handles = [chosen]
            if cleanup_errors:
                raise StadiaReaderError(self._join_errors("Controller handle cleanup failed", cleanup_errors))
            self._selected = chosen
            self._selected_identity = identity
            self._last_identity = identity
            self._pinned_guid = identity.guid
            self._generation += 1
            self._last_error = ""
            handles = []
        except Exception as error:
            self._last_error = self._format_error("Controller selection failed", error)
        finally:
            cleanup_errors = self._close_handles(handles)
            if cleanup_errors:
                self._last_error = self._join_errors(self._last_error, cleanup_errors)

    def _added_device_conflicts(self, device_index: object) -> bool:
        if not isinstance(device_index, int):
            return False
        pygame = self._pygame
        if pygame is None:
            return False
        handle: Any | None = None
        try:
            handle = pygame.joystick.Joystick(device_index)
            if handle.get_name() != STADIA_PRODUCT_NAME:
                return False
            identity = self._describe(handle)
            selected_guid = self._pinned_guid
            return selected_guid is None or identity.guid == selected_guid
        except Exception:
            return True
        finally:
            if handle is not None:
                self._close_handle(handle)

    def _describe(self, handle: Any) -> _DeviceIdentity:
        guid = str(handle.get_guid())
        if not guid:
            raise StadiaReaderError("Stadia controller GUID is missing")
        layout = ControllerLayout(
            axes=int(handle.get_numaxes()),
            buttons=int(handle.get_numbuttons()),
            hats=int(handle.get_numhats()),
        )
        return _DeviceIdentity(
            product_name=str(handle.get_name()),
            guid=guid,
            instance_id=int(handle.get_instance_id()),
            layout=layout,
        )

    def _read_selected(self) -> None:
        handle = self._selected
        identity = self._selected_identity
        if handle is None or identity is None:
            raise StadiaReaderError("no selected Stadia controller")
        if hasattr(handle, "get_attached") and not handle.get_attached():
            raise StadiaReaderError("selected Stadia controller is detached")

        layout = ControllerLayout(
            axes=int(handle.get_numaxes()),
            buttons=int(handle.get_numbuttons()),
            hats=int(handle.get_numhats()),
        )
        self._validate_layout(layout)
        axes = tuple(float(handle.get_axis(index)) for index in range(layout.axes))
        if not all(math.isfinite(value) for value in axes):
            raise StadiaReaderError("Stadia controller returned a non-finite axis value")
        buttons = tuple(bool(handle.get_button(index)) for index in range(layout.buttons))
        hats = tuple(self._read_hat(handle.get_hat(index)) for index in range(layout.hats))
        identity = _DeviceIdentity(
            product_name=identity.product_name,
            guid=identity.guid,
            instance_id=identity.instance_id,
            layout=layout,
        )
        self._selected_identity = identity
        self._last_identity = identity
        self._publish(
            connected=True,
            identity=identity,
            axes=axes,
            buttons=buttons,
            hats=hats,
            read_error=None,
        )

    @staticmethod
    def _validate_layout(layout: ControllerLayout) -> None:
        if layout.axes < STADIA_LAYOUT.axes:
            raise StadiaReaderError("Stadia controller requires at least 6 axes")
        if layout.buttons < STADIA_LAYOUT.buttons:
            raise StadiaReaderError("Stadia controller requires at least 15 buttons")
        if layout.hats < 0:
            raise StadiaReaderError("Stadia controller reported an invalid hat count")

    @staticmethod
    def _read_hat(raw_hat: object) -> tuple[int, int]:
        try:
            x, y = raw_hat  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise StadiaReaderError("Stadia controller returned a malformed D-pad hat") from error
        return int(x), int(y)

    def _disconnect_selected(self, *, schedule_reconnect: bool) -> tuple[str, ...]:
        handle = self._selected
        if self._selected_identity is not None:
            self._last_identity = self._selected_identity
        self._selected = None
        self._selected_identity = None
        if schedule_reconnect:
            self._next_connect_at = float(self._clock()) + self.reconnect_interval_s
        return self._close_handles([handle] if handle is not None else [])

    @staticmethod
    def _close_handle(handle: Any) -> None:
        handle.quit()

    @classmethod
    def _close_handles(cls, handles: list[Any]) -> tuple[str, ...]:
        errors: list[str] = []
        for handle in handles:
            try:
                cls._close_handle(handle)
            except Exception as error:
                errors.append(cls._format_error("joystick handle quit failed", error))
        return tuple(errors)

    def _publish_disconnected(
        self,
        error: str,
        *,
        identity: _DeviceIdentity | None = None,
    ) -> None:
        identity = identity or self._last_identity
        self._publish(
            connected=False,
            identity=identity,
            axes=(),
            buttons=(),
            hats=(),
            read_error=error,
        )

    def _publish(
        self,
        *,
        connected: bool,
        identity: _DeviceIdentity | None,
        axes: tuple[float, ...],
        buttons: tuple[bool, ...],
        hats: tuple[tuple[int, int], ...],
        read_error: str | None,
    ) -> None:
        sampled_at = float(self._clock())
        if not math.isfinite(sampled_at):
            raise StadiaReaderError("monotonic clock returned a non-finite value")
        layout = identity.layout if identity is not None else ControllerLayout(0, 0, 0)
        with self._snapshot_condition:
            self._sequence += 1
            self._latest = StadiaSnapshot(
                sequence=self._sequence,
                sampled_at=sampled_at,
                connected=connected,
                product_name=identity.product_name if identity is not None else None,
                guid=identity.guid if identity is not None else None,
                instance_id=identity.instance_id if identity is not None else None,
                connection_generation=self._generation,
                axes=axes,
                buttons=buttons,
                hats=hats,
                layout=layout,
                read_error=read_error,
            )
            self._snapshot_condition.notify_all()

    @staticmethod
    def _format_error(prefix: str, error: Exception) -> str:
        return f"{prefix}: {type(error).__name__}: {error}"

    @staticmethod
    def _join_errors(prefix: str, errors: tuple[str, ...] | list[str]) -> str:
        details = tuple(error for error in errors if error)
        return prefix if not details else f"{prefix}; " + "; ".join(details)
