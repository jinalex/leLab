"""Exact returned-action validation and comparison tests."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from lelab.stadia.action import (
    ActionValidationError,
    NormalizedAction,
    compare_requested_returned,
    validate_returned_action,
)
from lelab.stadia.types import ACTION_KEYS


def _action(value: object = 0.0) -> dict[str, object]:
    return dict.fromkeys(ACTION_KEYS, value)


def test_key_order_is_irrelevant_but_result_order_is_canonical() -> None:
    unordered = {
        "gripper.pos": 6,
        "wrist_roll.pos": 5,
        "wrist_flex.pos": 4,
        "elbow_flex.pos": 3,
        "shoulder_lift.pos": 2,
        "shoulder_pan.pos": 1,
    }

    normalized = validate_returned_action(unordered)

    assert tuple(normalized) == ACTION_KEYS
    assert normalized.as_items() == tuple(zip(ACTION_KEYS, range(1, 7), strict=True))
    assert normalized.positions == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_integer_fraction_and_float_values_normalize_to_floats() -> None:
    action = _action(1)
    action["gripper.pos"] = Fraction(1, 2)

    normalized = validate_returned_action(action)

    assert normalized["shoulder_pan.pos"] == 1.0
    assert normalized["gripper.pos"] == 0.5
    assert all(isinstance(value, float) for value in normalized.positions)


@pytest.mark.parametrize("returned", [None, [], (), "action", object()])
def test_returned_action_must_be_a_mapping(returned: object) -> None:
    with pytest.raises(ActionValidationError, match="must be a mapping"):
        validate_returned_action(returned)


def test_missing_key_is_rejected() -> None:
    returned = _action()
    del returned["gripper.pos"]

    with pytest.raises(ActionValidationError, match="missing=.*gripper.pos"):
        validate_returned_action(returned)


def test_extra_key_is_rejected() -> None:
    returned = _action()
    returned["extra.pos"] = 1.0

    with pytest.raises(ActionValidationError, match="extra=.*extra.pos"):
        validate_returned_action(returned)


def test_missing_and_extra_keys_are_rejected_together() -> None:
    returned = _action()
    del returned["gripper.pos"]
    returned["other.pos"] = 1.0

    with pytest.raises(ActionValidationError, match="missing=.*gripper.pos.*extra=.*other.pos"):
        validate_returned_action(returned)


@pytest.mark.parametrize("value", [True, False])
def test_bool_is_rejected_even_though_it_is_an_int_subclass(value: bool) -> None:
    returned = _action()
    returned["shoulder_pan.pos"] = value

    with pytest.raises(ActionValidationError, match="not bool"):
        validate_returned_action(returned)


@pytest.mark.parametrize("value", [None, "1.0", 1 + 0j, object()])
def test_non_real_values_are_rejected(value: object) -> None:
    returned = _action()
    returned["shoulder_pan.pos"] = value

    with pytest.raises(ActionValidationError, match="real number"):
        validate_returned_action(returned)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value: float) -> None:
    returned = _action()
    returned["shoulder_pan.pos"] = value

    with pytest.raises(ActionValidationError, match="must be finite"):
        validate_returned_action(returned)


def test_normalized_action_is_immutable_and_copies_do_not_alias() -> None:
    source = _action(1.0)
    normalized = validate_returned_action(source)
    source["shoulder_pan.pos"] = 99.0
    copy = normalized.as_dict()
    copy["shoulder_pan.pos"] = 88.0

    assert normalized["shoulder_pan.pos"] == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        normalized._positions = (2.0,) * 6
    with pytest.raises(TypeError):
        normalized["shoulder_pan.pos"] = 2.0


def test_normalized_action_constructor_enforces_its_invariants() -> None:
    with pytest.raises(ActionValidationError, match="exactly 6"):
        NormalizedAction((1.0,))
    with pytest.raises(ActionValidationError, match="finite"):
        NormalizedAction((0.0, 0.0, 0.0, 0.0, 0.0, float("nan")))
    with pytest.raises(ActionValidationError, match="real numbers"):
        NormalizedAction((0.0, 0.0, 0.0, 0.0, 0.0, True))


def test_comparison_counts_each_clipped_joint_beyond_absolute_tolerance() -> None:
    requested = _action(10.0)
    returned = _action(10.0)
    returned["shoulder_pan.pos"] = 10.000000002
    returned["gripper.pos"] = 9.5
    returned["wrist_roll.pos"] = 10.0000000005

    result = compare_requested_returned(requested, returned)

    assert result.clipped_keys == ("shoulder_pan.pos", "gripper.pos")
    assert result.clipping_count == 2
    assert result.was_clipped


def test_comparison_uses_strictly_greater_than_tolerance() -> None:
    requested = _action(0.0)
    returned = _action(0.0)
    returned["shoulder_pan.pos"] = 0.25

    result = compare_requested_returned(requested, returned, tolerance=0.25)

    assert not result.was_clipped
    assert result.clipping_count == 0


def test_returned_command_is_the_explicit_adoption_affordance() -> None:
    requested = _action(1.0)
    returned = _action(1.0)
    returned["elbow_flex.pos"] = 0.7

    result = compare_requested_returned(requested, returned)

    assert result.adopted_action is result.returned
    assert result.adopted_action["elbow_flex.pos"] == 0.7
    assert result.adopted_action["elbow_flex.pos"] != result.requested["elbow_flex.pos"]


def test_comparison_does_not_mutate_or_alias_inputs() -> None:
    requested = _action(1.0)
    returned = _action(0.5)
    requested_before = requested.copy()
    returned_before = returned.copy()

    result = compare_requested_returned(requested, returned)
    requested["shoulder_pan.pos"] = 50.0
    returned["shoulder_pan.pos"] = 60.0

    assert requested_before == _action(1.0)
    assert returned_before == _action(0.5)
    assert result.requested["shoulder_pan.pos"] == 1.0
    assert result.returned["shoulder_pan.pos"] == 0.5


@pytest.mark.parametrize(
    "tolerance",
    [True, "0.1", -0.1, float("nan"), float("inf"), float("-inf")],
)
def test_comparison_rejects_invalid_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        compare_requested_returned(_action(), _action(), tolerance=tolerance)


def test_comparison_validates_requested_and_returned_contracts() -> None:
    bad_requested = _action()
    del bad_requested["wrist_roll.pos"]
    with pytest.raises(ActionValidationError, match="requested action"):
        compare_requested_returned(bad_requested, _action())

    bad_returned = _action()
    bad_returned["gripper.pos"] = True
    with pytest.raises(ActionValidationError, match="returned action"):
        compare_requested_returned(_action(), bad_returned)
