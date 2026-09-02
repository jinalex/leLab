"""Exact Stadia mapping and fail-closed neutral/release gate."""

from __future__ import annotations

import math

from .types import (
    STADIA_LAYOUT,
    STADIA_PRODUCT_NAME,
    MappedStadiaInput,
    NeutralGateDecision,
    NeutralGateState,
    StadiaSnapshot,
    TriggerNeutralState,
)

LEFT_X_AXIS = 0
LEFT_Y_AXIS = 1
RIGHT_X_AXIS = 2
RIGHT_Y_AXIS = 3
LEFT_TRIGGER_AXIS = 4
RIGHT_TRIGGER_AXIS = 5
RB_BUTTON = 10
DPAD_UP_BUTTON = 11
DPAD_DOWN_BUTTON = 12
DPAD_LEFT_BUTTON = 13
DPAD_RIGHT_BUTTON = 14


class StadiaProfileError(ValueError):
    """The snapshot does not match the exact supported Stadia profile."""


def validate_stadia_profile(snapshot: StadiaSnapshot, *, expected_guid: str | None = None) -> None:
    if not snapshot.connected:
        raise StadiaProfileError("Stadia controller is disconnected")
    if snapshot.read_error:
        raise StadiaProfileError(f"Stadia read failed: {snapshot.read_error}")
    if snapshot.product_name != STADIA_PRODUCT_NAME:
        raise StadiaProfileError(
            f"unsupported controller product {snapshot.product_name!r}; expected {STADIA_PRODUCT_NAME!r}"
        )
    if not snapshot.guid:
        raise StadiaProfileError("Stadia controller GUID is missing")
    if expected_guid is not None and snapshot.guid != expected_guid:
        raise StadiaProfileError(f"Stadia controller GUID {snapshot.guid!r} does not match configured GUID")
    actual_layout = (len(snapshot.axes), len(snapshot.buttons), len(snapshot.hats))
    published_layout = (
        snapshot.layout.axes,
        snapshot.layout.buttons,
        snapshot.layout.hats,
    )
    if actual_layout != published_layout:
        raise StadiaProfileError("published layout does not match snapshot values")
    if snapshot.layout.axes < STADIA_LAYOUT.axes:
        raise StadiaProfileError("Stadia profile requires at least 6 axes")
    if snapshot.layout.buttons < STADIA_LAYOUT.buttons:
        raise StadiaProfileError("Stadia profile requires at least 15 buttons")


def rescaled_deadzone(value: float, deadzone: float = 0.15) -> float:
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("axis value must be finite")
    value = max(-1.0, min(1.0, value))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


def normalize_signed_trigger(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("trigger value must be finite")
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _dpad_x(snapshot: StadiaSnapshot) -> int:
    return int(snapshot.button(DPAD_RIGHT_BUTTON)) - int(snapshot.button(DPAD_LEFT_BUTTON))


def _dpad_y(snapshot: StadiaSnapshot) -> int:
    return int(snapshot.button(DPAD_UP_BUTTON)) - int(snapshot.button(DPAD_DOWN_BUTTON))


def _observe_trigger(value: float, previous: TriggerNeutralState) -> TriggerNeutralState:
    """Treat a signed-axis midpoint as ambiguous until a released sample is seen."""

    value = float(value)
    if not math.isfinite(value):
        return TriggerNeutralState(exercised=previous.exercised, released=False)
    if value <= -0.85:
        return TriggerNeutralState(exercised=previous.exercised, released=True)
    return TriggerNeutralState(exercised=True, released=False)


class NeutralReleaseGate:
    """Require neutral controls and RB-up after each unsafe connection state."""

    def __init__(
        self,
        *,
        expected_guid: str | None = None,
        deadzone: float = 0.15,
        stable_samples_required: int = 3,
    ) -> None:
        if stable_samples_required < 2:
            raise ValueError("stable_samples_required must be at least 2")
        self.expected_guid = expected_guid
        self.deadzone = deadzone
        self.stable_samples_required = stable_samples_required

    def evaluate(self, snapshot: StadiaSnapshot, state: NeutralGateState) -> NeutralGateDecision:
        generation_changed = state.connection_generation != snapshot.connection_generation
        if generation_changed:
            state = NeutralGateState(connection_generation=snapshot.connection_generation)

        try:
            validate_stadia_profile(snapshot, expected_guid=self.expected_guid)
        except (StadiaProfileError, ValueError) as error:
            failed_state = NeutralGateState(connection_generation=state.connection_generation)
            return NeutralGateDecision(failed_state, False, False, False, str(error))

        left_trigger = _observe_trigger(snapshot.axis(LEFT_TRIGGER_AXIS), state.left_trigger)
        right_trigger = _observe_trigger(snapshot.axis(RIGHT_TRIGGER_AXIS), state.right_trigger)
        try:
            sticks_neutral = all(
                rescaled_deadzone(snapshot.axis(index), self.deadzone) == 0.0
                for index in (LEFT_X_AXIS, LEFT_Y_AXIS, RIGHT_X_AXIS, RIGHT_Y_AXIS)
            )
        except ValueError as error:
            failed_state = NeutralGateState(connection_generation=state.connection_generation)
            return NeutralGateDecision(failed_state, False, False, False, str(error))
        dpad_neutral = _dpad_x(snapshot) == 0 and _dpad_y(snapshot) == 0
        triggers_released = left_trigger.released and right_trigger.released
        controls_neutral = sticks_neutral and dpad_neutral and triggers_released
        rb_held = snapshot.button(RB_BUTTON)

        sequence_advanced = state.last_sequence is None or snapshot.sequence > state.last_sequence
        last_sequence = snapshot.sequence if sequence_advanced else state.last_sequence
        release_seen = state.release_seen or (sequence_advanced and not rb_held)
        neutral_armed = state.neutral_armed
        stable_neutral_samples = state.stable_neutral_samples
        if not rb_held:
            if controls_neutral:
                if sequence_advanced:
                    stable_neutral_samples = min(
                        stable_neutral_samples + 1,
                        self.stable_samples_required,
                    )
                neutral_armed = stable_neutral_samples >= self.stable_samples_required
            else:
                stable_neutral_samples = 0
                neutral_armed = False
        elif not neutral_armed:
            stable_neutral_samples = 0
        motion_enabled = release_seen and neutral_armed and rb_held and not generation_changed

        next_state = NeutralGateState(
            connection_generation=snapshot.connection_generation,
            release_seen=release_seen,
            neutral_armed=neutral_armed,
            stable_neutral_samples=stable_neutral_samples,
            last_sequence=last_sequence,
            left_trigger=left_trigger,
            right_trigger=right_trigger,
        )
        if generation_changed:
            reason = "connection changed; release RB and neutralize controls"
        elif not release_seen:
            reason = "release RB before enabling motion"
        elif not triggers_released:
            reason = "exercise and release both triggers"
        elif controls_neutral and not rb_held and not neutral_armed:
            reason = (
                "hold controls neutral with RB released "
                f"({stable_neutral_samples}/{self.stable_samples_required} stable samples)"
            )
        elif not neutral_armed:
            reason = "neutralize sticks, D-pad, and triggers while RB is released"
        elif not rb_held:
            reason = "RB released; holding accepted target"
        else:
            reason = None
        return NeutralGateDecision(next_state, True, controls_neutral, motion_enabled, reason)


def map_stadia_input(
    snapshot: StadiaSnapshot,
    *,
    motion_enabled: bool,
    deadzone: float = 0.15,
    max_step_per_tick: float = 0.35,
    expected_guid: str | None = None,
) -> MappedStadiaInput:
    validate_stadia_profile(snapshot, expected_guid=expected_guid)
    if not math.isfinite(max_step_per_tick) or max_step_per_tick <= 0:
        raise ValueError("max_step_per_tick must be finite and positive")

    inputs = (
        ("left_x", rescaled_deadzone(snapshot.axis(LEFT_X_AXIS), deadzone)),
        ("left_y", -rescaled_deadzone(snapshot.axis(LEFT_Y_AXIS), deadzone)),
        ("right_x", rescaled_deadzone(snapshot.axis(RIGHT_X_AXIS), deadzone)),
        ("right_y", -rescaled_deadzone(snapshot.axis(RIGHT_Y_AXIS), deadzone)),
        ("dpad_x", float(_dpad_x(snapshot))),
        ("dpad_y", float(_dpad_y(snapshot))),
        ("left_trigger", normalize_signed_trigger(snapshot.axis(LEFT_TRIGGER_AXIS))),
        ("right_trigger", normalize_signed_trigger(snapshot.axis(RIGHT_TRIGGER_AXIS))),
    )
    values = dict(inputs)
    scale = max_step_per_tick if motion_enabled and snapshot.button(RB_BUTTON) else 0.0
    deltas = (
        ("shoulder_pan.pos", values["left_x"] * scale),
        ("shoulder_lift.pos", values["left_y"] * scale),
        ("elbow_flex.pos", values["right_y"] * scale),
        ("wrist_flex.pos", values["right_x"] * scale),
        ("wrist_roll.pos", values["dpad_x"] * scale),
        ("gripper.pos", (values["left_trigger"] - values["right_trigger"]) * scale),
    )
    return MappedStadiaInput(inputs, deltas)
