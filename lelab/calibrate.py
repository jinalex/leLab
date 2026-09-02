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

"""
Calibration module for the web interface.

This module provides calibration functionality similar to the CLI calibrate.py,
but adapted for the web interface with step-by-step guidance.
"""

import copy
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import (
    Robot,
    make_robot_from_config,
)
from lerobot.teleoperators import (
    Teleoperator,
    make_teleoperator_from_config,
)
from lerobot.utils.utils import init_logging

from .utils.devices import DeviceCleanupError, safe_disconnect_device

logger = logging.getLogger(__name__)

# Feetech sts3215 Present_Position readings are 12-bit (0-4095). A reading at or
# below 0, or at/above this loose ceiling, is a bad frame (disconnected motor or
# an encoder wrap-around) rather than a real joint position, so it's filtered out
# everywhere a raw position is consumed.
_MIN_VALID_POSITION = 0
_MAX_VALID_POSITION = 5000

# A single-frame jump larger than this is the encoder wrapping past 0/4095
# (a ~4096-step delta), not real motion — see CalibrationDiscontinuityError.
_MAX_POSITION_JUMP = 2000

# A recorded min..max sweep smaller than this many encoder steps means the joint
# barely moved; the user is warned their range of motion looks insufficient.
_MIN_CALIBRATION_RANGE = 100


def _is_valid_position(pos: float) -> bool:
    """True when a raw Present_Position reading sits within the plausible encoder
    range, filtering out 0/negative/extreme bad frames before they pollute the
    recorded min/max."""
    return _MIN_VALID_POSITION < pos < _MAX_VALID_POSITION


class CalibrationDiscontinuityError(Exception):
    """Raised when a motor position reading jumps across the encoder wrap-around.

    The Feetech encoder is 12-bit (0-4095); if calibration starts with a joint
    near a boundary, moving it past 0 or 4095 produces a single-frame delta of
    ~4096. The user-side fix is to start with all joints in the middle of their
    range, as documented in the SO-101 docs.
    """


@dataclass
class CalibrationStatus:
    """Status information for calibration process"""

    calibration_active: bool = False
    status: str = "idle"  # "idle", "connecting", "recording", "completed", "error", "stopping"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0  # Current calibration step
    total_steps: int = 1  # Total number of calibration steps
    current_positions: dict[str, float] = None
    recorded_ranges: dict[str, dict[str, float]] = None  # {motor: {min: val, max: val, current: val}}
    cleanup_pending: bool = False


@dataclass
class CalibrationRequest:
    """Request parameters for starting calibration"""

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None  # When set, write port + config back into the robot record on success


class CalibrationManager:
    """Manages calibration process for the web interface"""

    def __init__(self):
        self.status = CalibrationStatus()
        self.device: Robot | Teleoperator | None = None
        self.calibration_thread: threading.Thread | None = None
        self.stop_calibration = False
        self._status_lock = threading.Lock()
        self._step_complete = threading.Event()
        self._recording_active = False
        self._start_positions = {}
        self._mins = {}
        self._maxes = {}
        self._homing_offsets = {}
        self._current_request: CalibrationRequest | None = None
        self._cleanup_error: str | None = None

        # Initialize logging
        init_logging()

    def get_status(self) -> CalibrationStatus:
        """Return a copied status snapshot without touching the device bus.

        The calibration worker is the sole bus reader while it owns the device.
        Keeping this accessor passive prevents HTTP polling from racing the
        worker's 20 Hz range-recording reads on the serial port.
        """
        with self._status_lock:
            return copy.deepcopy(self.status)

    def _update_status(self, **kwargs):
        """Update calibration status thread-safely"""
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    def start_calibration(self, request: CalibrationRequest) -> dict[str, Any]:
        """Start calibration process"""
        try:
            if self.status.calibration_active:
                return {"success": False, "message": "Calibration already active"}

            # Reset status and clear any previous calibration data
            self._start_positions = {}
            self._mins = {}
            self._maxes = {}
            self._homing_offsets = {}
            self._cleanup_error = None

            self._update_status(
                calibration_active=True,
                status="connecting",
                device_type=request.device_type,
                error=None,
                message=f"Starting calibration for {request.device_type}",
                step=0,
                current_positions=None,
                recorded_ranges=None,
                cleanup_pending=False,
            )
            self._current_request = request

            # Start calibration in a separate thread
            self.calibration_thread = threading.Thread(
                target=self._calibration_worker, args=(request,), daemon=True
            )
            self.stop_calibration = False
            self._step_complete.clear()
            self.calibration_thread.start()

            return {"success": True, "message": "Calibration started"}

        except Exception as e:
            logger.error(f"Error starting calibration: {e}")
            self._update_status(
                calibration_active=False, status="error", error=str(e), message="Failed to start calibration"
            )
            return {"success": False, "message": str(e)}

    def complete_step(self) -> dict[str, Any]:
        """Complete the current calibration step"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            if self.status.status == "recording":
                # Complete recording step
                self._recording_active = False
                self._step_complete.set()
                return {"success": True, "message": "Range recording completed"}

            else:
                return {"success": False, "message": f"Cannot complete step in status: {self.status.status}"}

        except Exception as e:
            logger.error(f"Error completing step: {e}")
            return {"success": False, "message": str(e)}

    def stop_calibration_process(self) -> dict[str, Any]:
        """Signal the calibration worker and wait boundedly for its cleanup.

        This request-thread method never disconnects the device. The worker's
        ``finally`` block is the sole cleanup owner, including when this join
        times out or raises.
        """
        try:
            if not self.get_status().calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping calibration process...")
            self.stop_calibration = True
            self._recording_active = False
            self._step_complete.set()  # Unblock any waiting step

            self._update_status(status="stopping", message="Stopping calibration...")

            # Wait for thread to finish
            if self.calibration_thread and self.calibration_thread.is_alive():
                self.calibration_thread.join(timeout=5.0)

            if self.calibration_thread and self.calibration_thread.is_alive():
                logger.warning(
                    "Calibration thread did not finish within timeout; cleanup remains worker-owned"
                )
                return {
                    "success": False,
                    "message": "Calibration stop requested, but the worker is still shutting down",
                    "stop_pending": True,
                    "cleanup_proven": False,
                }

            terminal = self.get_status()
            if terminal.calibration_active:
                logger.error("Calibration worker exited without publishing a terminal status")
                return {
                    "success": False,
                    "message": "Calibration worker exited without confirming cleanup",
                    "stop_pending": True,
                    "cleanup_proven": False,
                }

            if terminal.status == "error":
                return {
                    "success": False,
                    "message": terminal.error or terminal.message,
                    "stop_pending": terminal.cleanup_pending,
                    "cleanup_proven": not terminal.cleanup_pending,
                }

            logger.info("Calibration stop completed")
            return {
                "success": True,
                "message": "Calibration stopped",
                "cleanup_proven": True,
            }

        except Exception as e:
            logger.error(f"Error stopping calibration: {e}")
            return {
                "success": False,
                "message": str(e),
                "stop_pending": True,
                "cleanup_proven": False,
            }

    def _calibration_worker(self, request: CalibrationRequest):
        """Worker thread for calibration process"""
        terminal_message = "Calibration completed successfully"
        terminal_status = "completed"
        terminal_error: str | None = None
        try:
            logger.info(f"Starting calibration worker for {request.device_type}")

            # Create device configuration
            if request.device_type == "robot":
                from lerobot.robots.so_follower import SO101FollowerConfig

                config = SO101FollowerConfig(port=request.port, id=request.config_file)
            elif request.device_type == "teleop":
                from lerobot.teleoperators.so_leader import SO101LeaderConfig

                config = SO101LeaderConfig(port=request.port, id=request.config_file)
            else:
                raise ValueError(f"Unknown device type: {request.device_type}")

            self._update_status(status="connecting", message="Connecting to device...")

            # Create and connect device
            if request.device_type == "robot":
                self.device = make_robot_from_config(config)
            else:
                self.device = make_teleoperator_from_config(config)

            logger.info("Connecting to device...")
            self.device.connect(calibrate=False)

            if self.stop_calibration:
                logger.info("Calibration stopped after device connection")
                terminal_message = "Calibration cancelled"
                terminal_status = "idle"
                return

            # Start Step 1: Homing
            self._step_homing()

            if self.stop_calibration:
                logger.info("Calibration stopped after homing step")
                terminal_message = "Calibration cancelled"
                terminal_status = "idle"
                return

            # Start Step 2: Range Recording
            self._step_range_recording()

            if self.stop_calibration:
                logger.info("Calibration stopped after recording step")
                terminal_message = "Calibration cancelled"
                terminal_status = "idle"
                return

            # Complete calibration
            self._complete_calibration()

            logger.info("Calibration completed successfully")

        except CalibrationDiscontinuityError as e:
            logger.error(f"Calibration discontinuity: {e}")
            terminal_message = str(e)
            terminal_status = "error"
            terminal_error = str(e)
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            logger.error(traceback.format_exc())
            terminal_message = f"Calibration failed: {e}"
            terminal_status = "error"
            terminal_error = str(e)
        finally:
            logger.info("Calibration worker thread finishing")
            if self.stop_calibration and terminal_status == "completed":
                terminal_message = "Calibration cancelled"
                terminal_status = "idle"
            self._cleanup_and_finish(
                terminal_message,
                status=terminal_status,
                error=terminal_error,
            )

    def _step_homing(self):
        """Auto-capture homing offsets from the device's current position."""
        logger.info("Setting homing offsets from current position")

        # Disable torque to allow manual movement during recording
        self.device.bus.disable_torque()
        for motor in self.device.bus.motors:
            self.device.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self.device.bus.reset_calibration()
        actual_positions = self.device.bus.sync_read("Present_Position", normalize=False)
        logger.info(f"Current positions for homing: {actual_positions}")

        self._homing_offsets = self.device.bus._get_half_turn_homings(actual_positions)
        logger.info(f"Calculated homing offsets: {self._homing_offsets}")

        for motor, offset in self._homing_offsets.items():
            self.device.bus.write("Homing_Offset", motor, offset)

    def _step_range_recording(self):
        """Record range of motion as the user moves all joints."""
        logger.info("Starting range recording step")

        # Initialize range tracking with retry and validation
        self._start_positions = {}
        for attempt in range(5):  # Try multiple times to get valid initial positions
            try:
                positions = self.device.bus.sync_read("Present_Position", normalize=False)
                # Validate initial positions
                valid_positions = {}
                for motor, pos in positions.items():
                    if _is_valid_position(pos):
                        valid_positions[motor] = pos

                if len(valid_positions) == len(positions):  # All positions are valid
                    self._start_positions = valid_positions
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: Got invalid initial positions, retrying...")
                    time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to read initial positions: {e}")
                time.sleep(0.1)

        if not self._start_positions:
            raise RuntimeError("Could not get valid initial positions after multiple attempts")

        logger.info(f"Starting positions for range recording: {self._start_positions}")

        self._mins = self._start_positions.copy()
        self._maxes = self._start_positions.copy()
        logger.info(f"Initialized mins: {self._mins}")
        logger.info(f"Initialized maxes: {self._maxes}")

        self._update_status(
            status="recording",
            step=1,
            message="Move ALL joints through their FULL ranges of motion - from minimum to maximum positions. Ensure each joint moves significantly from its starting position.",
            current_positions=dict(self._start_positions),
            recorded_ranges={
                motor: {"min": pos, "max": pos, "current": pos}
                for motor, pos in self._start_positions.items()
            },
        )

        self._recording_active = True
        prev_positions: dict[str, int] = dict(self._start_positions)

        # Record positions until user completes step
        while not self._step_complete.is_set() and not self.stop_calibration:
            try:
                # Try reading positions with retry on port contention
                positions = None
                for attempt in range(3):  # Try up to 3 times
                    try:
                        positions = self.device.bus.sync_read("Present_Position", normalize=False)
                        break  # Success, exit retry loop
                    except Exception as read_error:
                        if "Port is in use" in str(read_error) and attempt < 2:
                            time.sleep(0.01)  # Short delay before retry
                            continue
                        else:
                            raise read_error  # Re-raise if not port contention or final attempt

                if positions:
                    # Validate the readings - filter out invalid/zero values
                    valid_positions = {}
                    for motor, pos in positions.items():
                        # Filter out clearly invalid readings (0, negative, or extreme values)
                        if _is_valid_position(pos):
                            valid_positions[motor] = pos
                        else:
                            logger.debug(f"Filtered invalid position for {motor}: {pos}")

                    # Only update if we have valid readings
                    if valid_positions:
                        for motor, pos in valid_positions.items():
                            if (
                                motor in prev_positions
                                and abs(pos - prev_positions[motor]) > _MAX_POSITION_JUMP
                            ):
                                raise CalibrationDiscontinuityError(
                                    "Motor discontinuity detected. Make sure to start "
                                    "the calibration with the robot in a middle position "
                                    "- all joints in the middle of their ranges."
                                )
                            prev_positions[motor] = pos
                            if motor in self._mins:
                                self._mins[motor] = min(self._mins[motor], pos)
                                self._maxes[motor] = max(self._maxes[motor], pos)

                        # Publish one coherent snapshot from the worker-owned
                        # serial read. Status polling only copies these values.
                        self._update_status(
                            current_positions=dict(prev_positions),
                            recorded_ranges={
                                motor: {
                                    "min": self._mins[motor],
                                    "max": self._maxes[motor],
                                    "current": prev_positions[motor],
                                }
                                for motor in self._mins
                            },
                        )

                time.sleep(0.05)  # 20Hz update rate
            except CalibrationDiscontinuityError:
                raise
            except Exception as e:
                if "Port is in use" in str(e):
                    logger.debug(f"Port busy during position read: {e}")
                else:
                    logger.warning(f"Error reading positions during recording: {e}")
                # Increase sleep time on error to reduce port contention
                time.sleep(0.2)

        if self.stop_calibration:
            logger.info("Range recording step cancelled due to stop request")
            return

        # Log the final recorded ranges for debugging
        logger.info("Final recorded ranges:")
        for motor in self._mins:
            logger.info(
                f"  {motor}: min={self._mins[motor]}, max={self._maxes[motor]}, range={self._maxes[motor] - self._mins[motor]}"
            )

        # Validate ranges
        same_min_max = [motor for motor in self._mins if self._mins[motor] == self._maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values: {same_min_max}")

        # Check for insufficient range movement (less than 100 motor steps)
        insufficient_range = []
        for motor in self._mins:
            range_diff = self._maxes[motor] - self._mins[motor]
            if range_diff < _MIN_CALIBRATION_RANGE:
                insufficient_range.append(f"{motor}: {range_diff}")

        if insufficient_range:
            logger.warning(
                f"Some motors may not have been moved through sufficient range: {insufficient_range}"
            )
            logger.warning("Consider moving all joints through their full range of motion during calibration")

        self._step_complete.clear()
        logger.info("Range recording step completed")

    def _complete_calibration(self):
        """Complete the calibration and save results"""
        logger.info("Completing calibration...")

        # Log motor information for debugging
        logger.info("Motor configuration:")
        for motor, m in self.device.bus.motors.items():
            logger.info(f"  {motor}: ID={m.id}, Model={m.model}")

        # Create calibration dict
        calibration = {}
        for motor, m in self.device.bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=self._homing_offsets[motor],
                range_min=self._mins[motor],
                range_max=self._maxes[motor],
            )
            logger.info(
                f"Calibration for {motor}: "
                f"ID={m.id}, "
                f"homing_offset={self._homing_offsets[motor]}, "
                f"range_min={self._mins[motor]}, "
                f"range_max={self._maxes[motor]}"
            )

        # Write and save calibration
        self.device.calibration = calibration
        self.device.bus.write_calibration(calibration)
        self.device._save_calibration()

        logger.info(f"Calibration saved to {self.device.calibration_fpath}")

        # Robot-record write-back: if this calibration was launched from a tile,
        # update the robot's port + config field for the side that was just calibrated.
        request = self._current_request
        if request is not None and request.robot_name:
            from .utils.config import save_robot_record

            if request.device_type == "teleop":
                patch = {"leader_port": request.port, "leader_config": f"{request.config_file}.json"}
            else:
                patch = {"follower_port": request.port, "follower_config": f"{request.config_file}.json"}
            try:
                save_robot_record(request.robot_name, patch, allow_create=False)
            except Exception as e:
                logger.warning(f"Robot-record write-back failed for {request.robot_name}: {e}")

    def _cleanup_and_finish(
        self,
        message: str,
        status: str = "completed",
        error: str | None = None,
    ):
        """Clean up and finish calibration"""
        self._recording_active = False
        cleanup_proven = self._cleanup_device()
        if not cleanup_proven:
            cleanup_error = self._cleanup_error or "Calibration device cleanup could not be verified"
            if error:
                cleanup_error = f"{error}; {cleanup_error}"
            # Fail closed: retain the device reference and active ownership so
            # the shared coordinator cannot hand the serial resource to a new
            # operation after an unproven disconnect.
            self._update_status(
                calibration_active=True,
                status="error",
                error=cleanup_error,
                message=cleanup_error,
                cleanup_pending=True,
            )
            return
        if self._cleanup_error:
            error = f"{error}; {self._cleanup_error}" if error else self._cleanup_error
            message = error
            status = "error"
        self._update_status(
            calibration_active=False,
            status=status,
            error=error,
            message=message,
            cleanup_pending=False,
        )

    def _cleanup_device(self) -> bool:
        """Clean up device connection.

        Uses safe_disconnect_device so a failed disconnect (flaky USB/serial)
        force-releases the port/cameras instead of leaving the device busy and
        blocking the next calibration/teleop/record run.
        """
        device = self.device
        if device is None:
            return True

        logger.info("Disconnecting device...")
        try:
            safe_disconnect_device(device, logger, context="calibration cleanup")
        except DeviceCleanupError as exc:
            logger.error("Calibration device cleanup failed: %s", exc)
            self._cleanup_error = str(exc)
            if exc.cleanup_proven:
                self.device = None
            return exc.cleanup_proven
        except Exception as exc:
            logger.error("Calibration device cleanup failed unexpectedly: %s", exc)
            self._cleanup_error = str(exc)
            return False

        # LeRobot's SO-101 wrappers expose `is_connected`; its bus fallback is
        # useful for narrow fakes and compatible wrappers. If a postcondition
        # exists, require it to say disconnected. An unreadable or still-true
        # postcondition is unproven cleanup and must retain ownership.
        sentinel = object()
        try:
            connected: object = getattr(device, "is_connected", sentinel)
            if connected is sentinel:
                connected = getattr(getattr(device, "bus", None), "is_connected", sentinel)
            if connected is not False:
                logger.error("Calibration device disconnect postcondition is not exact false")
                return False
        except Exception as exc:
            logger.error("Could not verify calibration device cleanup: %s", exc)
            return False

        self.device = None
        self._cleanup_error = None
        return True


# Global calibration manager instance
calibration_manager = CalibrationManager()
