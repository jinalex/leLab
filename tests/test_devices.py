from __future__ import annotations

import logging

import pytest


def test_safe_disconnect_force_closes_serial_port_after_disconnect_failure() -> None:
    from lelab.utils.devices import safe_disconnect_device

    class PortHandler:
        def __init__(self) -> None:
            self.cleared = False
            self.closed = False
            self.is_using = True

        def clearPort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.cleared = True

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.closed = True

    class Camera:
        def __init__(self) -> None:
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    class Device:
        def __init__(self) -> None:
            self.bus = type("Bus", (), {"port_handler": PortHandler()})()
            self.cameras = {"cam": Camera()}

        def disconnect(self) -> None:
            raise RuntimeError("Failed to write 'Torque_Enable' on id_=6")

    device = Device()
    with pytest.raises(Exception, match="disconnect failed"):
        safe_disconnect_device(device, logging.getLogger(__name__))

    assert device.bus.port_handler.cleared is True
    assert device.bus.port_handler.is_using is False
    assert device.bus.port_handler.closed is True
    assert device.cameras["cam"].disconnected is True


def test_safe_disconnect_uses_normal_disconnect_when_it_succeeds() -> None:
    from lelab.utils.devices import safe_disconnect_device

    class PortHandler:
        def __init__(self) -> None:
            self.closed = False

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.closed = True

    class Device:
        def __init__(self) -> None:
            self.bus = type("Bus", (), {"port_handler": PortHandler()})()
            self.disconnected = False

        @property
        def is_connected(self) -> bool:
            return not self.disconnected

        def disconnect(self) -> None:
            self.disconnected = True

    device = Device()
    assert safe_disconnect_device(device, logging.getLogger(__name__)) is True

    assert device.disconnected is True
    assert device.bus.port_handler.closed is False


def test_safe_disconnect_aggregates_all_force_close_failures() -> None:
    from lelab.utils.devices import DeviceCleanupError, safe_disconnect_device

    events: list[str] = []

    class PortHandler:
        is_open = True

        def clearPort(self) -> None:  # noqa: N802 - mirrors LeRobot API
            events.append("clear")
            raise RuntimeError("clear boom")

        @property
        def is_using(self) -> bool:
            return True

        @is_using.setter
        def is_using(self, value: bool) -> None:
            events.append("ownership")
            raise RuntimeError("ownership boom")

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot API
            events.append("close")
            raise RuntimeError("close boom")

    class Camera:
        def disconnect(self) -> None:
            events.append("camera")
            raise RuntimeError("camera boom")

    class Bus:
        port_handler = PortHandler()

        @property
        def is_connected(self) -> bool:
            return self.port_handler.is_open

    class Device:
        bus = Bus()
        cameras = {"front": Camera()}

        @property
        def is_connected(self) -> bool:
            return self.bus.is_connected

        def disconnect(self) -> None:
            raise RuntimeError("disconnect boom")

    with pytest.raises(DeviceCleanupError) as caught:
        safe_disconnect_device(Device(), logging.getLogger(__name__))

    assert events == ["clear", "ownership", "close", "camera"]
    assert set(caught.value.errors) >= {
        "disconnect failed with RuntimeError: disconnect boom",
        "serial clear failed with RuntimeError: clear boom",
        "serial ownership reset failed with RuntimeError: ownership boom",
        "serial close failed with RuntimeError: close boom",
        "camera 'front' disconnect failed with RuntimeError: camera boom",
        "device disconnect postcondition is not false: True",
    }


@pytest.mark.parametrize("postcondition", [None, 0, "", object()])
def test_safe_disconnect_requires_exact_false_postcondition(postcondition: object) -> None:
    from lelab.utils.devices import DeviceCleanupError, safe_disconnect_device

    class Device:
        is_connected = postcondition

        def disconnect(self) -> None:
            pass

    with pytest.raises(DeviceCleanupError, match="device disconnect postcondition is not false"):
        safe_disconnect_device(Device(), logging.getLogger(__name__))


def test_safe_disconnect_rejects_missing_postcondition() -> None:
    from lelab.utils.devices import DeviceCleanupError, safe_disconnect_device

    class Device:
        def disconnect(self) -> None:
            pass

    with pytest.raises(DeviceCleanupError, match="postcondition is unavailable"):
        safe_disconnect_device(Device(), logging.getLogger(__name__))


def test_safe_disconnect_tracks_camera_thread_even_when_camera_drops_reference() -> None:
    from lelab.utils.devices import DeviceCleanupError, safe_disconnect_device

    class LingeringThread:
        def __init__(self) -> None:
            self.joins: list[int] = []

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: int) -> None:
            self.joins.append(timeout)

    class Camera:
        def __init__(self, thread: LingeringThread) -> None:
            self.thread = thread
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False
            self.thread = None

    class Bus:
        is_connected = True

    thread = LingeringThread()
    camera = Camera(thread)

    class Device:
        bus = Bus()
        cameras = {"front": camera}

        @property
        def is_connected(self) -> bool:
            return self.bus.is_connected

        def disconnect(self) -> None:
            self.bus.is_connected = False
            camera.disconnect()

    with pytest.raises(DeviceCleanupError) as caught:
        safe_disconnect_device(Device(), logging.getLogger(__name__))

    assert caught.value.cleanup_proven is False
    assert thread.joins == [2, 2]
    assert "read thread is still alive" in str(caught.value)


def test_safe_disconnect_proves_captured_camera_thread_joined() -> None:
    from lelab.utils.devices import safe_disconnect_device

    class JoinableThread:
        def __init__(self) -> None:
            self.alive = True
            self.joins: list[int] = []

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: int) -> None:
            self.joins.append(timeout)
            self.alive = False

    class Camera:
        def __init__(self, thread: JoinableThread) -> None:
            self.thread = thread
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False
            self.thread = None

    class Bus:
        is_connected = True

    thread = JoinableThread()
    camera = Camera(thread)

    class Device:
        bus = Bus()
        cameras = {"front": camera}

        @property
        def is_connected(self) -> bool:
            return self.bus.is_connected

        def disconnect(self) -> None:
            self.bus.is_connected = False
            camera.disconnect()

    assert safe_disconnect_device(Device(), logging.getLogger(__name__)) is True
    assert thread.joins == [2]
