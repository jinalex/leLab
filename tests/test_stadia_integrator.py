from __future__ import annotations

import math

import pytest

from lelab.stadia.integrator import BoundedStadiaIntegrator, validate_returned_action
from lelab.stadia.types import (
    ACTION_KEYS,
    DEFAULT_JOINT_SPECS,
    JointControlSpec,
    PositionUnit,
)


def action(value: float) -> dict[str, float]:
    return dict.fromkeys(ACTION_KEYS, value)


def bounds(lower: float = -180.0, upper: float = 180.0) -> dict[str, tuple[float, float]]:
    result = dict.fromkeys(ACTION_KEYS, (lower, upper))
    result["gripper.pos"] = (0.0, 100.0)
    return result


def integrator(initial: float = 0.0) -> BoundedStadiaIntegrator:
    initial_action = action(initial)
    initial_action["gripper.pos"] = max(0.0, initial)
    return BoundedStadiaIntegrator(
        initial_action=initial_action,
        startup_anchor=initial_action,
        endpoint_bounds=bounds(),
    )


def test_default_specs_use_explicit_degrees_and_gripper_percentage_points() -> None:
    assert [spec.unit for spec in DEFAULT_JOINT_SPECS[:5]] == [PositionUnit.DEGREES] * 5
    assert DEFAULT_JOINT_SPECS[5].unit is PositionUnit.GRIPPER_PERCENTAGE_POINTS
    assert {spec.max_step_per_tick for spec in DEFAULT_JOINT_SPECS} == {0.35}
    assert {spec.startup_travel for spec in DEFAULT_JOINT_SPECS} == {45.0}


def test_integrates_no_more_than_one_clamped_step_and_counts_saturation() -> None:
    control = integrator()
    result = control.integrate_one_step(action(5.0), enabled=True)
    values = result.action_dict()
    assert {values[key] for key in ACTION_KEYS[:5]} == {0.35}
    assert values["gripper.pos"] == 0.35
    assert result.counters.step_saturations == 6


def test_disabled_integrator_holds_without_advancing() -> None:
    control = integrator(10.0)
    result = control.integrate_one_step(action(0.35), enabled=False)
    assert result.action_dict() == action(10.0)
    assert control.target == action(10.0)


def test_startup_travel_and_calibrated_endpoint_clamps_are_counted() -> None:
    small_specs = tuple(
        JointControlSpec(spec.action_key, spec.unit, 0.35, 0.2) for spec in DEFAULT_JOINT_SPECS
    )
    start = action(0.0)
    control = BoundedStadiaIntegrator(
        initial_action=start,
        startup_anchor=start,
        endpoint_bounds=bounds(),
        specs=small_specs,
    )
    travel = control.integrate_one_step(action(0.35), enabled=True)
    assert set(travel.action_dict().values()) == {0.2}
    assert travel.counters.travel_saturations == 6

    endpoint_bounds = bounds(-0.1, 0.1)
    endpoint_bounds["gripper.pos"] = (0.0, 0.1)
    endpoint = BoundedStadiaIntegrator(
        initial_action=start,
        startup_anchor=start,
        endpoint_bounds=endpoint_bounds,
    ).integrate_one_step(action(0.35), enabled=True)
    assert set(endpoint.action_dict().values()) == {0.1}
    assert endpoint.counters.endpoint_saturations == 6


def test_gripper_is_always_clamped_to_zero_through_one_hundred() -> None:
    start = action(99.9)
    start["gripper.pos"] = 99.9
    control = BoundedStadiaIntegrator(
        initial_action=start,
        startup_anchor=start,
        endpoint_bounds=bounds(),
    )
    result = control.integrate_one_step(action(0.35), enabled=True)
    assert result.action_dict()["gripper.pos"] == 100.0


@pytest.mark.parametrize(
    "bad_action",
    [
        dict.fromkeys(ACTION_KEYS[:-1], 0.0),
        {**action(0.0), "extra.pos": 1.0},
        {**action(0.0), "wrist_roll.pos": math.nan},
    ],
)
def test_returned_action_requires_exactly_six_finite_position_keys(
    bad_action: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        validate_returned_action(bad_action)


def test_returned_command_is_counted_and_adopted_as_next_integrator_target() -> None:
    control = integrator()
    requested = control.integrate_one_step(action(0.35), enabled=True).action_dict()
    returned = dict(requested)
    returned["shoulder_pan.pos"] = 0.1
    control.accept_returned_action(returned, requested_action=requested)
    assert control.counters.returned_clippings == 1
    assert control.target == returned

    next_result = control.integrate_one_step(action(0.1), enabled=True)
    assert next_result.action_dict()["shoulder_pan.pos"] == pytest.approx(0.2)
