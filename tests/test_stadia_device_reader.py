from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lelab.stadia.device_reader import StadiaDeviceReader, StadiaReaderError
from lelab.stadia.types import STADIA_PRODUCT_NAME, ControllerLayout, StadiaSnapshot


@dataclass(frozen=True)
class DeviceSpec:
    name: str = STADIA_PRODUCT_NAME
    guid: str = "stadia-guid"
    instance_id: int = 7
    axes: tuple[float, ...] = (0.1, -0.2, 0.3, -0.4, -1.0, -1.0)
    buttons: tuple[bool, ...] = (False,) * 17
    hats: tuple[tuple[int, int], ...] = ()
    quit_error: str | None = None


class CallLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[tuple[str, int]] = []

    def add(self, name: str) -> None:
        with self._lock:
            self.entries.append((name, threading.get_ident()))

    def thread_ids(self) -> set[int]:
        with self._lock:
            return {thread_id for _, thread_id in self.entries}


class FakeJoystickHandle:
    def __init__(self, spec: DeviceSpec, log: CallLog) -> None:
        self.spec = spec
        self.log = log
        self.attached = True
        self.axis_error: Exception | None = None
        self.quit_count = 0

    def _record(self, name: str) -> None:
        self.log.add(f"handle.{name}")

    def get_name(self) -> str:
        self._record("get_name")
        return self.spec.name

    def get_guid(self) -> str:
        self._record("get_guid")
        return self.spec.guid

    def get_instance_id(self) -> int:
        self._record("get_instance_id")
        return self.spec.instance_id

    def get_numaxes(self) -> int:
        self._record("get_numaxes")
        return len(self.spec.axes)

    def get_numbuttons(self) -> int:
        self._record("get_numbuttons")
        return len(self.spec.buttons)

    def get_numhats(self) -> int:
        self._record("get_numhats")
        return len(self.spec.hats)

    def get_axis(self, index: int) -> float:
        self._record("get_axis")
        if self.axis_error is not None:
            raise self.axis_error
        return self.spec.axes[index]

    def get_button(self, index: int) -> bool:
        self._record("get_button")
        return self.spec.buttons[index]

    def get_hat(self, index: int) -> tuple[int, int]:
        self._record("get_hat")
        return self.spec.hats[index]

    def get_attached(self) -> bool:
        self._record("get_attached")
        return self.attached

    def quit(self) -> None:
        self._record("quit")
        self.quit_count += 1
        self.attached = False
        if self.spec.quit_error is not None:
            raise OSError(self.spec.quit_error)


class FakeJoystickModule:
    def __init__(self, devices: list[DeviceSpec], log: CallLog) -> None:
        self._lock = threading.Lock()
        self._devices = list(devices)
        self.log = log
        self.opened: list[tuple[int, FakeJoystickHandle]] = []
        self.quit_count = 0
        self.quit_error: str | None = None

    def set_devices(self, devices: list[DeviceSpec]) -> None:
        with self._lock:
            self._devices = list(devices)

    def init(self) -> None:
        self.log.add("joystick.init")

    def quit(self) -> None:
        self.log.add("joystick.quit")
        self.quit_count += 1
        if self.quit_error is not None:
            raise OSError(self.quit_error)

    def get_count(self) -> int:
        self.log.add("joystick.get_count")
        with self._lock:
            return len(self._devices)

    def Joystick(self, index: int) -> FakeJoystickHandle:  # noqa: N802 - pygame API
        self.log.add(f"joystick.Joystick({index})")
        with self._lock:
            handle = FakeJoystickHandle(self._devices[index], self.log)
            self.opened.append((index, handle))
        return handle

    def selected_handle(self, instance_id: int) -> FakeJoystickHandle:
        with self._lock:
            return next(
                handle
                for _, handle in reversed(self.opened)
                if handle.spec.instance_id == instance_id and handle.quit_count == 0
            )


class FakeEventModule:
    def __init__(self, log: CallLog) -> None:
        self._lock = threading.Lock()
        self._events: list[SimpleNamespace] = []
        self.log = log

    def push(self, event: SimpleNamespace) -> None:
        with self._lock:
            self._events.append(event)

    def pump(self) -> None:
        self.log.add("event.pump")

    def get(self, *, pump: bool = True) -> list[SimpleNamespace]:
        self.log.add(f"event.get(pump={pump})")
        with self._lock:
            events, self._events = self._events, []
        return events


class FakePygame:
    JOYDEVICEADDED = 1
    JOYDEVICEREMOVED = 2

    def __init__(self, devices: list[DeviceSpec]) -> None:
        self.log = CallLog()
        self.joystick = FakeJoystickModule(devices, self.log)
        self.event = FakeEventModule(self.log)
        self.quit_count = 0
        self.quit_error: str | None = None

    def init(self) -> None:
        self.log.add("pygame.init")

    def quit(self) -> None:
        self.log.add("pygame.quit")
        self.quit_count += 1
        if self.quit_error is not None:
            raise OSError(self.quit_error)


class FakeClock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 100.0

    def __call__(self) -> float:
        with self._lock:
            self._value += 0.001
            return self._value


class YieldingSleep:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        with self._lock:
            self.calls.append(seconds)
        threading.Event().wait(0.0005)


def make_reader(
    pygame: FakePygame,
    *,
    expected_guid: str | None = None,
    reconnect_interval_s: float = 100.0,
) -> StadiaDeviceReader:
    return StadiaDeviceReader(
        expected_guid=expected_guid,
        poll_interval_s=0.001,
        reconnect_interval_s=reconnect_interval_s,
        pygame_module=pygame,
        clock=FakeClock(),
        sleeper=YieldingSleep(),
    )


def wait_until(
    reader: StadiaDeviceReader,
    predicate: Callable[[StadiaSnapshot], bool],
    *,
    timeout: float = 1.0,
) -> StadiaSnapshot:
    deadline = time.monotonic() + timeout
    snapshot = reader.snapshot()
    while not predicate(snapshot):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"condition not reached; latest snapshot: {snapshot!r}")
        snapshot = reader.wait_for_snapshot(
            after_sequence=snapshot.sequence,
            timeout=remaining,
        )
    return snapshot


def test_import_and_construction_are_lazy_when_pygame_is_forbidden() -> None:
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "pygame" or name.startswith("pygame."):
        raise AssertionError(f"forbidden pygame import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from lelab.stadia import StadiaDeviceReader
reader = StadiaDeviceReader()
assert reader.snapshot().sequence == 0
reader.stop()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_enumerates_every_index_and_selects_only_exact_stadia_product() -> None:
    pygame = FakePygame(
        [
            DeviceSpec(name="Generic Gamepad", guid="generic", instance_id=1),
            DeviceSpec(instance_id=8),
        ]
    )
    reader = make_reader(pygame)
    reader.start()
    try:
        snapshot = wait_until(reader, lambda value: value.connected)
        assert snapshot.product_name == STADIA_PRODUCT_NAME
        assert snapshot.guid == "stadia-guid"
        assert snapshot.instance_id == 8
        assert snapshot.layout == ControllerLayout(axes=6, buttons=17, hats=0)
        assert snapshot.axes == pytest.approx((0.1, -0.2, 0.3, -0.4, -1.0, -1.0))
        assert [index for index, _ in pygame.joystick.opened[:2]] == [0, 1]
        assert pygame.joystick.opened[0][1].quit_count == 1
        assert pygame.log.thread_ids() == {pygame.log.entries[0][1]}
        assert threading.get_ident() not in pygame.log.thread_ids()
    finally:
        reader.stop()


def test_multiple_matching_devices_fail_closed_without_configured_guid() -> None:
    pygame = FakePygame([DeviceSpec(guid="guid-a", instance_id=1), DeviceSpec(guid="guid-b", instance_id=2)])
    reader = make_reader(pygame)
    reader.start()
    try:
        snapshot = wait_until(reader, lambda value: "Multiple" in (value.read_error or ""))
        assert not snapshot.connected
        assert snapshot.connection_generation == 0
        assert all(handle.quit_count == 1 for _, handle in pygame.joystick.opened[:2])
    finally:
        reader.stop()


def test_configured_guid_uniquely_resolves_multiple_stadia_devices() -> None:
    pygame = FakePygame([DeviceSpec(guid="guid-a", instance_id=1), DeviceSpec(guid="guid-b", instance_id=2)])
    reader = make_reader(pygame, expected_guid="guid-b")
    reader.start()
    try:
        snapshot = wait_until(reader, lambda value: value.connected)
        assert snapshot.guid == "guid-b"
        assert snapshot.instance_id == 2
        assert snapshot.connection_generation == 1
        assert pygame.joystick.opened[0][1].quit_count == 1
        assert pygame.joystick.opened[1][1].quit_count == 0
    finally:
        reader.stop()


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (DeviceSpec(axes=(0.0,) * 5), "at least 6 axes"),
        (DeviceSpec(buttons=(False,) * 14), "at least 15 buttons"),
    ],
)
def test_invalid_stadia_layout_fails_closed(spec: DeviceSpec, message: str) -> None:
    pygame = FakePygame([spec])
    reader = make_reader(pygame)
    reader.start()
    try:
        snapshot = wait_until(reader, lambda value: message in (value.read_error or ""))
        assert not snapshot.connected
        assert snapshot.connection_generation == 0
        assert pygame.joystick.opened[0][1].quit_count == 1
    finally:
        reader.stop()


def test_removal_event_is_filtered_to_the_selected_instance_id() -> None:
    pygame = FakePygame([DeviceSpec(instance_id=7)])
    reader = make_reader(pygame)
    reader.start()
    try:
        connected = wait_until(reader, lambda value: value.connected)
        handle = pygame.joystick.selected_handle(7)
        pygame.event.push(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=999))
        after_unrelated = reader.wait_for_snapshot(
            after_sequence=connected.sequence,
            timeout=1.0,
        )
        assert after_unrelated.connected
        assert after_unrelated.instance_id == 7
        assert handle.quit_count == 0

        pygame.joystick.set_devices([])
        pygame.event.push(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=7))
        removed = wait_until(
            reader,
            lambda value: not value.connected and "removed" in (value.read_error or ""),
        )
        assert removed.guid == "stadia-guid"
        assert removed.instance_id == 7
        assert removed.connection_generation == 1
        assert handle.quit_count == 1
    finally:
        reader.stop()


def test_reconnect_gets_new_instance_id_and_increments_generation() -> None:
    pygame = FakePygame([DeviceSpec(instance_id=7)])
    reader = make_reader(pygame, reconnect_interval_s=0.0)
    reader.start()
    try:
        first = wait_until(reader, lambda value: value.connected)
        assert first.connection_generation == 1
        pygame.joystick.set_devices([DeviceSpec(instance_id=12)])
        pygame.event.push(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=7))
        reconnected = wait_until(
            reader,
            lambda value: value.connected and value.connection_generation == 2,
        )
        assert reconnected.guid == first.guid
        assert reconnected.instance_id == 12
        assert reconnected.sequence > first.sequence
    finally:
        reader.stop()


def test_unconfigured_reader_pins_first_guid_and_rejects_different_reconnect() -> None:
    pygame = FakePygame([DeviceSpec(guid="guid-a", instance_id=7)])
    reader = make_reader(pygame, reconnect_interval_s=0.0)
    reader.start()
    try:
        first = wait_until(reader, lambda value: value.connected)
        pygame.joystick.set_devices([DeviceSpec(guid="guid-b", instance_id=12)])
        pygame.event.push(SimpleNamespace(type=pygame.JOYDEVICEREMOVED, instance_id=7))

        rejected = wait_until(
            reader,
            lambda value: not value.connected and "Pinned Stadia GUID 'guid-a'" in (value.read_error or ""),
        )

        assert first.guid == "guid-a"
        assert rejected.connection_generation == 1
        assert rejected.guid == "guid-a"
        assert not any(
            handle.quit_count == 0 and handle.spec.guid == "guid-b" for _, handle in pygame.joystick.opened
        )
    finally:
        reader.stop()


def test_selected_handle_read_error_publishes_fail_closed_snapshot() -> None:
    pygame = FakePygame([DeviceSpec(instance_id=7)])
    reader = make_reader(pygame)
    reader.start()
    try:
        connected = wait_until(reader, lambda value: value.connected)
        handle = pygame.joystick.selected_handle(7)
        pygame.joystick.set_devices([])
        handle.axis_error = OSError("lost input report")
        failed = wait_until(
            reader,
            lambda value: value.sequence > connected.sequence and not value.connected,
        )
        assert "lost input report" in (failed.read_error or "")
        assert failed.guid == "stadia-guid"
        assert failed.instance_id == 7
        assert failed.layout == ControllerLayout(axes=6, buttons=17, hats=0)
        assert failed.axes == ()
        assert failed.buttons == ()
        assert handle.quit_count == 1
    finally:
        reader.stop()


def test_publication_is_immutable_and_heartbeat_advances_independently() -> None:
    pygame = FakePygame([DeviceSpec()])
    reader = make_reader(pygame)
    reader.start()
    try:
        first = wait_until(reader, lambda value: value.connected)
        second = reader.wait_for_snapshot(after_sequence=first.sequence, timeout=1.0)
        assert second.sequence > first.sequence
        assert second.sampled_at > first.sampled_at
        assert first.sequence < second.sequence
        assert isinstance(second.axes, tuple)
        with pytest.raises(FrozenInstanceError):
            second.sequence = 999  # type: ignore[misc]
    finally:
        reader.stop()


def test_stop_closes_handle_pygame_and_joins_reader_thread() -> None:
    pygame = FakePygame([DeviceSpec(instance_id=44)])
    reader = make_reader(pygame)
    reader.start()
    connected = wait_until(reader, lambda value: value.connected)
    handle = pygame.joystick.selected_handle(44)

    reader.stop()

    assert connected.connected
    assert not reader.is_alive
    assert not reader.initialized
    assert handle.quit_count == 1
    assert pygame.joystick.quit_count == 1
    assert pygame.quit_count == 1
    assert reader.snapshot().read_error == "Stadia reader stopped"


def test_stop_attempts_all_cleanup_and_reports_aggregated_failures() -> None:
    pygame = FakePygame([DeviceSpec(instance_id=44, quit_error="handle stuck")])
    reader = make_reader(pygame)
    reader.start()
    wait_until(reader, lambda value: value.connected)
    handle = pygame.joystick.selected_handle(44)
    pygame.joystick.quit_error = "joystick shutdown failed"
    pygame.quit_error = "pygame shutdown failed"

    with pytest.raises(StadiaReaderError) as raised:
        reader.stop()

    message = str(raised.value)
    assert "handle stuck" in message
    assert "joystick shutdown failed" in message
    assert "pygame shutdown failed" in message
    assert handle.quit_count == 1
    assert pygame.joystick.quit_count == 1
    assert pygame.quit_count == 1
    assert not reader.is_alive
    assert not reader.initialized
    assert reader.snapshot().read_error == message
