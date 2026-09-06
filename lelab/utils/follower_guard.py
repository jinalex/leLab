"""Opt-in bridge boundary for the pinned SO-101 Stadia command path.

No vendor files or direct-USB behavior are changed. Other bridge workflows are
rejected before connecting until their command ownership/freshness is covered.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib.metadata
import json
import logging
import math
import os
import time
from collections import deque
from numbers import Real
from pathlib import Path

PORTS_ENV = "LELAB_GUARDED_FOLLOWER_PORTS"
MAX_COMMAND_AGE_S = 0.250
logger = logging.getLogger(__name__)
_supported_construction = contextvars.ContextVar("guarded_stadia_construction", default=False)


class FollowerGuardError(RuntimeError):
    """Rejected before a goal write; the session must not continue."""


class CommandCancelledError(RuntimeError):
    """Current controller authorization changed during a blocking read."""


def guarded_ports() -> tuple[str, ...]:
    raw = os.environ.get(PORTS_ENV)
    if raw is None:
        return ()
    ports = json.loads(raw)
    if (
        not isinstance(ports, list)
        or not ports
        or any(
            not isinstance(port, str) or not port or "\0" in port or not Path(port).is_absolute()
            for port in ports
        )
    ):
        raise ValueError(f"{PORTS_ENV} must be a non-empty JSON array of absolute device paths")
    return tuple(str(Path(port).resolve()) for port in ports)


def selected_port(port: str) -> bool:
    return str(Path(port).resolve()) in guarded_ports()


@contextlib.contextmanager
def stadia_follower_construction():
    token = _supported_construction.set(True)
    try:
        yield
    finally:
        _supported_construction.reset(token)


def install_guarded_followers() -> None:
    """Install once per process, before constructors, only when opted in."""
    if not guarded_ports():
        return
    if importlib.metadata.version("lerobot") != "0.6.0":
        raise FollowerGuardError("guarded followers require the reviewed LeRobot 0.6.0 API")
    from lerobot.robots.so_follower.so_follower import SOFollower

    original = SOFollower.__init__
    if getattr(original, "_lelab_guarded", False):
        return

    @functools.wraps(original)
    def initialize(self, config):
        enabled = selected_port(config.port)
        if enabled and not _supported_construction.get():
            raise FollowerGuardError(
                "This bridge release supports guarded Stadia teleoperation/recording only. "
                "Leader teleop, calibration, and policy inference on this port are not yet validated."
            )
        original(self, config)
        if enabled:
            # A numeric plausibility check cannot replace transaction validation.
            from tailbridge.feetech_consumer import feetech_reply_integrity_status, record_feetech_context

            status = feetech_reply_integrity_status(self.bus.port_handler)
            if (
                not status.get("enabled")
                or status.get("quarantined")
                or not selected_port(status.get("selected_port", ""))
                or str(Path(status["selected_port"]).resolve()) != str(Path(config.port).resolve())
            ):
                raise FollowerGuardError("selected bridge port lacks active strict Feetech reply validation")
            self._lelab_follower_guard = FollowerWriteGuard(
                self,
                record_context=functools.partial(record_feetech_context, self.bus.port_handler),
                protocol_status=functools.partial(feetech_reply_integrity_status, self.bus.port_handler),
            )

    initialize._lelab_guarded = True
    SOFollower.__init__ = initialize


class FollowerWriteGuard:
    """Validate raw feedback and final goals at the instance bus boundary."""

    def __init__(self, follower, *, clock=time.monotonic, record_context=None, protocol_status=None):
        from lelab.stadia.session import _endpoint_bounds_from_scales, _position_scales

        self.follower = follower
        self.bus = follower.bus
        self.scales = _position_scales(follower)
        self.bounds = _endpoint_bounds_from_scales(self.scales)
        caps = follower.config.max_relative_target
        if not isinstance(caps, dict) or set(caps) != set(self.scales):
            raise FollowerGuardError("guarded follower requires explicit per-joint relative limits")
        self.caps = {name: self._finite(value) for name, value in caps.items()}
        if any(value <= 0 for value in self.caps.values()):
            raise FollowerGuardError("relative limits must be positive")
        self.clock = clock
        self.record_context = record_context or (lambda **kwargs: None)
        self.protocol_status = protocol_status or (lambda: {"enabled": False})
        self.bus._lelab_follower_guard = self
        self.before_write = None
        self.fault = None
        self.events = deque(maxlen=64)
        self.feedback = {}
        self.action_started = None
        self.action_feedback = {}
        self.action_request = {}
        self.last_written = {}
        self._read = self.bus.read
        self._sync_read = self.bus.sync_read
        self._write = self.bus.write
        self._sync_write = self.bus.sync_write
        self._send_action = follower.send_action
        self.bus.read = self.read
        self.bus.sync_read = self.sync_read
        self.bus.write = self.write
        self.bus.sync_write = self.sync_write
        follower.send_action = self.send_action

    @staticmethod
    def _finite(value):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise FollowerGuardError(f"non-finite/non-numeric motor value: {value!r}")
        return float(value)

    def _reject(self, message):
        if self.fault is None:
            self.fault = message
            logger.error("Follower guard fault: %s; recent_events=%r", message, list(self.events))
        raise FollowerGuardError(self.fault)

    def _raw(self, name, value):
        numeric = self._finite(value)
        scale = self.scales[name]
        if not numeric.is_integer() or not (
            0 <= numeric <= scale.native_max and scale.calibration_min <= numeric <= scale.calibration_max
        ):
            raise FollowerGuardError(f"{name} raw position {value!r} outside calibrated/native range")
        return int(numeric)

    def _positions(self, values, names, normalize):
        self.events.append({"event": "feedback", "at": self.clock(), "raw": dict(values)})
        self.record_context(raw_feedback=values)
        try:
            if set(values) != set(names):
                raise FollowerGuardError("incomplete or unexpected position batch")
            raw = {name: self._raw(name, value) for name, value in values.items()}
        except (ValueError, KeyError, FollowerGuardError) as error:
            self._reject(str(error))
        normalized = {name: self.scales[name].normalize(value) for name, value in raw.items()}
        self.record_context(normalized_feedback=normalized)
        self.feedback.update(raw)
        if self.action_started is not None:
            self.action_feedback.update(normalized)
        return normalized if normalize else raw

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        if data_name != "Present_Position":
            return self._read(data_name, motor, normalize=normalize, num_retry=0)
        value = self._read(data_name, motor, normalize=False, num_retry=0)
        return self._positions({motor: value}, [motor], normalize)[motor]

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        if data_name != "Present_Position":
            return self._sync_read(data_name, motors, normalize=normalize, num_retry=0)
        names = self.bus._get_motors_list(motors)
        values = self._sync_read(data_name, motors, normalize=False, num_retry=0)
        return self._positions(values, names, normalize)

    def _check_write(self, data_name, values, normalize):
        if data_name != "Goal_Position":
            if data_name == "Torque_Enable" and any(values.values()):
                if self.fault:
                    self._reject(self.fault)
                if set(self.last_written) != set(self.scales):
                    self._reject("all six goals must be seeded before torque enable")
                if self.before_write is None:
                    self._reject("missing controller authorization before torque enable")
                self.before_write()
            return
        self.events.append(
            {"event": "goal", "at": self.clock(), "values": dict(values), "normalize": normalize}
        )
        self.record_context(final_goals={"values": values, "normalize": normalize})
        try:
            if self.fault:
                raise FollowerGuardError(self.fault)
            for name, value in values.items():
                number = self._finite(value)
                lower, upper = self.bounds[f"{name}.pos"]
                if normalize:
                    if not lower <= number <= upper:
                        raise FollowerGuardError(f"{name} final goal {number} outside calibrated bounds")
                    raw = self.bus._unnormalize({self.bus.motors[name].id: number})[self.bus.motors[name].id]
                    self._raw(name, raw)
                else:
                    raw = self._raw(name, value)
                    number = self.scales[name].normalize(raw)
                if self.action_started is None:
                    if normalize or self.feedback.get(name) != raw:
                        raise FollowerGuardError("startup goal must exactly seed validated raw feedback")
                elif (
                    name not in self.action_feedback
                    or abs(number - self.action_feedback[name]) > self.caps[name] + 1e-8
                ):
                    raise FollowerGuardError(
                        f"{name} final goal lacks fresh feedback or exceeds relative limit"
                    )
                if self.action_started is not None:
                    # A corrupt but in-range feedback value must not let the
                    # limiter replace a small user step with a distant goal.
                    if abs(number - self.action_request[f"{name}.pos"]) > self.caps[name] + 1e-8:
                        raise FollowerGuardError(
                            f"{name} limiter displaced requested goal beyond relative limit"
                        )
                    if (
                        name in self.last_written
                        and abs(number - self.last_written[name]) > self.caps[name] + 1e-8
                    ):
                        raise FollowerGuardError(f"{name} final goal discontinuity exceeds relative limit")
            if self.action_started is not None and self.clock() - self.action_started > MAX_COMMAND_AGE_S:
                raise FollowerGuardError("command expired during follower I/O; no goal written")
            if self.before_write is None:
                raise FollowerGuardError("missing current controller authorization before goal write")
        except (ValueError, KeyError, FollowerGuardError) as error:
            self._reject(str(error))
        # Deliberately outside the latching fault handler: a normal RB release
        # cancels this step and can hold without reconnecting the robot.
        self.before_write()

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        self._check_write(data_name, {motor: value}, normalize)
        result = self._write(data_name, motor, value, normalize=normalize, num_retry=0)
        self._record_written(data_name, {motor: value}, normalize)
        return result

    def sync_write(self, data_name, values, *, normalize=True, num_retry=0):
        mapping = values if isinstance(values, dict) else dict.fromkeys(self.scales, values)
        self._check_write(data_name, mapping, normalize)
        result = self._sync_write(data_name, values, normalize=normalize, num_retry=0)
        self._record_written(data_name, mapping, normalize)
        return result

    def _record_written(self, data_name, values, normalize):
        if data_name == "Goal_Position":
            self.last_written.update(
                {
                    name: float(value) if normalize else self.scales[name].normalize(value)
                    for name, value in values.items()
                }
            )

    def send_action(self, action):
        if self.fault:
            self._reject(self.fault)
        self.action_started = self.clock()
        self.action_feedback = {}
        self.action_request = dict(action)
        self.events.append({"event": "request", "at": self.action_started, "action": dict(action)})
        self.record_context(requested_goals=action)
        try:
            if set(action) != {f"{name}.pos" for name in self.scales}:
                self._reject("command must contain exactly the six follower positions")
            for key, value in action.items():
                lower, upper = self.bounds[key]
                if not lower <= self._finite(value) <= upper:
                    self._reject(f"requested goal {key} outside calibrated bounds")
            return self._send_action(action)
        except CommandCancelledError:
            raise
        except Exception as error:
            self._reject(f"follower transaction failed: {error}")
        finally:
            self.action_started = None
            self.action_feedback = {}
            self.action_request = {}
