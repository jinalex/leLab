"""Bounded one-step Stadia target integration and follower-return validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real

from .action import compare_requested_returned, validate_returned_action
from .types import (
    ACTION_KEYS,
    DEFAULT_JOINT_SPECS,
    IntegrationResult,
    IntegratorCounters,
    JointControlSpec,
)


def _exact_finite_values(values: Mapping[str, float], keys: tuple[str, ...], label: str) -> dict[str, float]:
    if set(values) != set(keys) or len(values) != len(keys):
        raise ValueError(f"{label} must contain exactly {list(keys)}")
    result = {key: float(values[key]) for key in keys}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} contains a non-finite value")
    return result


class BoundedStadiaIntegrator:
    def __init__(
        self,
        *,
        initial_action: Mapping[str, float],
        startup_anchor: Mapping[str, float],
        endpoint_bounds: Mapping[str, tuple[float, float]],
        specs: Sequence[JointControlSpec] = DEFAULT_JOINT_SPECS,
    ) -> None:
        self.specs = tuple(specs)
        self.keys = tuple(spec.action_key for spec in self.specs)
        if self.keys != ACTION_KEYS:
            raise ValueError(f"specs must define exactly {list(ACTION_KEYS)} in order")
        if any(
            not math.isfinite(spec.max_step_per_tick)
            or spec.max_step_per_tick <= 0
            or not math.isfinite(spec.startup_travel)
            or spec.startup_travel <= 0
            for spec in self.specs
        ):
            raise ValueError("step and startup-travel limits must be finite and positive")
        self.target = _exact_finite_values(initial_action, self.keys, "initial action")
        self.anchor = _exact_finite_values(startup_anchor, self.keys, "startup anchor")
        if set(endpoint_bounds) != set(self.keys) or len(endpoint_bounds) != len(self.keys):
            raise ValueError(f"endpoint bounds must contain exactly {list(self.keys)}")
        self.endpoint_bounds: dict[str, tuple[float, float]] = {}
        for key in self.keys:
            lower, upper = map(float, endpoint_bounds[key])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("endpoint bounds must be finite and increasing")
            if key == "gripper.pos":
                lower, upper = max(0.0, lower), min(100.0, upper)
                if lower >= upper:
                    raise ValueError("gripper endpoint bounds must overlap 0-100")
            self.endpoint_bounds[key] = (lower, upper)
        self.counters = IntegratorCounters()

    def set_max_step_per_tick(self, value: float) -> None:
        """Update the live per-tick cap without changing the accepted target."""

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("max_step_per_tick must be a finite positive number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("max_step_per_tick must be a finite positive number")
        self.specs = tuple(replace(spec, max_step_per_tick=numeric) for spec in self.specs)

    def integrate_one_step(self, deltas: Mapping[str, float], *, enabled: bool) -> IntegrationResult:
        values = _exact_finite_values(deltas, self.keys, "joint deltas")
        step_saturations = travel_saturations = endpoint_saturations = 0
        requested: dict[str, float] = {}
        for spec in self.specs:
            key = spec.action_key
            delta = values[key] if enabled else 0.0
            bounded_delta = max(-spec.max_step_per_tick, min(spec.max_step_per_tick, delta))
            step_saturations += int(bounded_delta != delta)
            candidate = self.target[key] + bounded_delta
            travel_lower = self.anchor[key] - spec.startup_travel
            travel_upper = self.anchor[key] + spec.startup_travel
            travel_target = max(travel_lower, min(travel_upper, candidate))
            travel_saturations += int(travel_target != candidate)
            lower, upper = self.endpoint_bounds[key]
            if key == "gripper.pos":
                lower, upper = max(0.0, lower), min(100.0, upper)
            final_target = max(lower, min(upper, travel_target))
            endpoint_saturations += int(final_target != travel_target)
            requested[key] = final_target

        self.counters = IntegratorCounters(
            step_saturations=self.counters.step_saturations + step_saturations,
            travel_saturations=self.counters.travel_saturations + travel_saturations,
            endpoint_saturations=self.counters.endpoint_saturations + endpoint_saturations,
            returned_clippings=self.counters.returned_clippings,
        )
        if enabled:
            self.target = requested
        else:
            requested = dict(self.target)
        return IntegrationResult(tuple(requested.items()), self.counters)

    def accept_returned_action(
        self,
        returned_action: Mapping[str, float],
        *,
        requested_action: Mapping[str, float] | None = None,
        tolerance: float = 1e-9,
    ) -> dict[str, float]:
        requested = (
            _exact_finite_values(requested_action, self.keys, "requested action")
            if requested_action is not None
            else dict(self.target)
        )
        comparison = compare_requested_returned(
            requested,
            validate_returned_action(returned_action),
            tolerance=tolerance,
        )
        self.counters = IntegratorCounters(
            step_saturations=self.counters.step_saturations,
            travel_saturations=self.counters.travel_saturations,
            endpoint_saturations=self.counters.endpoint_saturations,
            returned_clippings=(self.counters.returned_clippings + comparison.clipping_count),
        )
        self.target = comparison.adopted_action.as_dict()
        return comparison.adopted_action.as_dict()
