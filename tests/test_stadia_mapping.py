from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lelab.stadia.mapping import (
    NeutralReleaseGate,
    StadiaProfileError,
    map_stadia_input,
    rescaled_deadzone,
    validate_stadia_profile,
)
from lelab.stadia.types import (
    STADIA_PRODUCT_NAME,
    ControllerLayout,
    NeutralGateState,
    StadiaSnapshot,
)


def test_stadia_core_imports_without_pygame_or_lerobot() -> None:
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "pygame" or name.startswith("pygame."):
        raise AssertionError(f"forbidden pygame import: {name}")
    if name == "lerobot" or name.startswith("lerobot."):
        raise AssertionError(f"forbidden lerobot import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import lelab.stadia
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def snapshot(
    *,
    sequence: int = 4,
    generation: int = 1,
    rb: bool = False,
    axes: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, -1.0, -1.0),
    dpad_left: bool = False,
    dpad_right: bool = False,
    product_name: str = STADIA_PRODUCT_NAME,
    guid: str = "stadia-guid",
    connected: bool = True,
    error: str | None = None,
) -> StadiaSnapshot:
    buttons = [False] * 17
    buttons[10] = rb
    buttons[13] = dpad_left
    buttons[14] = dpad_right
    return StadiaSnapshot(
        sequence=sequence,
        sampled_at=10.0,
        connected=connected,
        product_name=product_name,
        guid=guid,
        instance_id=7,
        connection_generation=generation,
        axes=axes,
        buttons=tuple(buttons),
        hats=(),
        layout=ControllerLayout(len(axes), len(buttons), 0),
        read_error=error,
    )


def test_snapshot_is_immutable_and_carries_exact_stadia_identity_and_layout() -> None:
    value = snapshot()
    assert value.product_name == "Google Stadia Controller"
    assert value.layout == ControllerLayout(axes=6, buttons=17, hats=0)
    assert value.guid == "stadia-guid"
    with pytest.raises(FrozenInstanceError):
        value.sequence = 5  # type: ignore[misc]

    coerced = StadiaSnapshot(
        sequence=1,
        sampled_at=1.0,
        connected=True,
        product_name=STADIA_PRODUCT_NAME,
        guid="guid",
        instance_id=1,
        connection_generation=1,
        axes=[0.0] * 6,  # type: ignore[arg-type]
        buttons=[False] * 15,  # type: ignore[arg-type]
        hats=[],  # type: ignore[arg-type]
        layout=ControllerLayout(6, 15, 0),
    )
    assert isinstance(coerced.axes, tuple)
    assert isinstance(coerced.buttons, tuple)


def test_exact_profile_rejects_wrong_product_guid_and_layout() -> None:
    validate_stadia_profile(snapshot(), expected_guid="stadia-guid")
    with pytest.raises(StadiaProfileError, match="unsupported controller"):
        validate_stadia_profile(snapshot(product_name="Generic Gamepad"))
    with pytest.raises(StadiaProfileError, match="configured GUID"):
        validate_stadia_profile(snapshot(), expected_guid="other-guid")
    with pytest.raises(StadiaProfileError, match="at least 6 axes"):
        validate_stadia_profile(snapshot(axes=(0.0,) * 5))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"connected": False}, "disconnected"),
        ({"error": "reader blocked"}, "read failed"),
    ],
)
def test_disconnected_or_read_error_fails_closed(change: dict[str, object], message: str) -> None:
    decision = NeutralReleaseGate().evaluate(snapshot(**change), NeutralGateState())
    assert not decision.profile_valid
    assert not decision.motion_enabled
    assert not decision.state.release_seen
    assert message in (decision.reason or "")


def test_rb_up_maps_to_exactly_six_zero_deltas() -> None:
    mapped = map_stadia_input(snapshot(rb=False), motion_enabled=True)
    assert len(mapped.joint_deltas) == 6
    assert set(mapped.deltas_dict().values()) == {0.0}


def test_release_suppresses_motion_and_requires_neutral_before_repress() -> None:
    gate = NeutralReleaseGate()
    state = NeutralGateState()
    neutral = gate.evaluate(snapshot(sequence=1, rb=False), state)
    assert not neutral.motion_enabled
    neutral = gate.evaluate(snapshot(sequence=2, rb=False), neutral.state)
    assert not neutral.state.neutral_armed
    neutral = gate.evaluate(snapshot(sequence=3, rb=False), neutral.state)
    assert neutral.state.neutral_armed
    moving_axes = (1.0, 0.0, 0.0, 0.0, -1.0, -1.0)
    held = gate.evaluate(snapshot(sequence=4, rb=True, axes=moving_axes), neutral.state)
    assert held.motion_enabled

    released = gate.evaluate(snapshot(sequence=5, rb=False, axes=moving_axes), held.state)
    assert not released.motion_enabled
    assert not released.state.neutral_armed
    repressed = gate.evaluate(snapshot(sequence=6, rb=True, axes=moving_axes), released.state)
    assert not repressed.motion_enabled

    rearmed = gate.evaluate(snapshot(sequence=7, rb=False), repressed.state)
    rearmed = gate.evaluate(snapshot(sequence=8, rb=False), rearmed.state)
    rearmed = gate.evaluate(snapshot(sequence=9, rb=False), rearmed.state)
    assert rearmed.state.neutral_armed
    assert gate.evaluate(snapshot(sequence=10, rb=True, axes=moving_axes), rearmed.state).motion_enabled


def test_reconnect_generation_resets_release_and_rejects_held_rb() -> None:
    gate = NeutralReleaseGate()
    state = NeutralGateState()
    for sequence in range(1, 4):
        state = gate.evaluate(snapshot(sequence=sequence, rb=False), state).state
    assert state.neutral_armed
    assert gate.evaluate(snapshot(sequence=4, rb=True), state).motion_enabled

    reconnect = gate.evaluate(snapshot(sequence=1, generation=2, rb=True), state)
    assert reconnect.state.connection_generation == 2
    assert not reconnect.state.release_seen
    assert reconnect.state.stable_neutral_samples == 0
    assert not reconnect.motion_enabled
    still_held = gate.evaluate(snapshot(sequence=2, generation=2, rb=True), reconnect.state)
    assert not still_held.motion_enabled
    released = gate.evaluate(snapshot(sequence=3, generation=2, rb=False), still_held.state)
    assert released.state.release_seen
    assert not released.state.neutral_armed
    released = gate.evaluate(snapshot(sequence=4, generation=2, rb=False), released.state)
    released = gate.evaluate(snapshot(sequence=5, generation=2, rb=False), released.state)
    assert released.state.neutral_armed
    assert gate.evaluate(snapshot(sequence=6, generation=2, rb=True), released.state).motion_enabled


def test_ambiguous_trigger_startup_requires_exercise_and_release() -> None:
    gate = NeutralReleaseGate()
    midpoint = snapshot(axes=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    decision = gate.evaluate(midpoint, NeutralGateState())
    assert not decision.controls_neutral
    assert not decision.state.neutral_armed
    assert decision.reason == "connection changed; release RB and neutralize controls"

    decision = gate.evaluate(midpoint, decision.state)
    assert decision.reason == "exercise and release both triggers"
    exercised = gate.evaluate(snapshot(axes=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)), decision.state)
    released = gate.evaluate(snapshot(sequence=5), exercised.state)
    assert released.state.left_trigger.exercised
    assert released.state.right_trigger.exercised
    assert released.controls_neutral
    assert released.state.stable_neutral_samples == 1
    assert not released.state.neutral_armed


def test_sticks_and_dpad_must_be_neutral_while_rb_is_up_to_arm() -> None:
    gate = NeutralReleaseGate()
    stick = gate.evaluate(snapshot(axes=(0.5, 0.0, 0.0, 0.0, -1.0, -1.0)), NeutralGateState())
    assert not stick.controls_neutral
    assert not stick.state.neutral_armed
    dpad = gate.evaluate(snapshot(sequence=5, dpad_left=True), stick.state)
    assert not dpad.controls_neutral
    neutral = gate.evaluate(snapshot(sequence=6), dpad.state)
    assert neutral.controls_neutral
    assert neutral.state.stable_neutral_samples == 1
    assert not neutral.state.neutral_armed


def test_neutral_arming_requires_three_distinct_advancing_samples() -> None:
    gate = NeutralReleaseGate()
    first = gate.evaluate(snapshot(sequence=10), NeutralGateState())
    assert first.state.stable_neutral_samples == 1
    assert not first.state.neutral_armed

    duplicate = gate.evaluate(snapshot(sequence=10), first.state)
    older = gate.evaluate(snapshot(sequence=9), duplicate.state)
    assert duplicate.state.stable_neutral_samples == 1
    assert older.state.stable_neutral_samples == 1

    second = gate.evaluate(snapshot(sequence=11), older.state)
    third = gate.evaluate(snapshot(sequence=12), second.state)
    assert second.state.stable_neutral_samples == 2
    assert not second.state.neutral_armed
    assert third.state.stable_neutral_samples == 3
    assert third.state.neutral_armed


def test_unstable_or_held_sample_resets_partial_neutral_progress() -> None:
    gate = NeutralReleaseGate()
    first = gate.evaluate(snapshot(sequence=1), NeutralGateState())
    moving = gate.evaluate(
        snapshot(
            sequence=2,
            axes=(0.5, 0.0, 0.0, 0.0, -1.0, -1.0),
        ),
        first.state,
    )
    assert moving.state.stable_neutral_samples == 0

    second_first = gate.evaluate(snapshot(sequence=3), moving.state)
    held = gate.evaluate(snapshot(sequence=4, rb=True), second_first.state)
    assert held.state.stable_neutral_samples == 0
    assert not held.motion_enabled


def test_deadzone_axis_signs_and_stadia_button_layout_match_reference() -> None:
    assert rescaled_deadzone(0.15) == 0.0
    assert rescaled_deadzone(0.575) == pytest.approx(0.5)
    axes = (0.575, -0.575, 0.575, -0.575, 1.0, -1.0)
    mapped = map_stadia_input(
        snapshot(rb=True, axes=axes, dpad_right=True),
        motion_enabled=True,
        max_step_per_tick=0.4,
    )
    assert mapped.deltas_dict() == pytest.approx(
        {
            "shoulder_pan.pos": 0.2,
            "shoulder_lift.pos": 0.2,
            "elbow_flex.pos": 0.2,
            "wrist_flex.pos": 0.2,
            "wrist_roll.pos": 0.4,
            "gripper.pos": 0.4,
        }
    )


def test_nonfinite_axis_fails_closed() -> None:
    bad = snapshot(axes=(float("nan"), 0.0, 0.0, 0.0, -1.0, -1.0))
    decision = NeutralReleaseGate().evaluate(bad, NeutralGateState())
    assert not decision.profile_valid
    assert not decision.motion_enabled
    with pytest.raises(ValueError, match="finite"):
        map_stadia_input(bad, motion_enabled=True)
