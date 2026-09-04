"""Exact, immutable follower-returned action validation."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .types import ACTION_KEYS

DEFAULT_CLIPPING_TOLERANCE = 1e-9


class ActionValidationError(ValueError):
    """An action does not satisfy the exact six-joint finite contract."""


@dataclass(frozen=True, slots=True)
class NormalizedAction(Mapping[str, float]):
    """Canonical-order, deeply immutable six-joint action values."""

    _positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self._positions) != len(ACTION_KEYS):
            raise ActionValidationError(f"normalized action must contain exactly {len(ACTION_KEYS)} values")
        positions = []
        for value in self._positions:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ActionValidationError("normalized action values must be real numbers")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ActionValidationError("normalized action values must be finite")
            positions.append(normalized)
        object.__setattr__(self, "_positions", tuple(positions))

    def __getitem__(self, key: str) -> float:
        try:
            index = ACTION_KEYS.index(key)
        except ValueError as error:
            raise KeyError(key) from error
        return self._positions[index]

    def __iter__(self) -> Iterator[str]:
        return iter(ACTION_KEYS)

    def __len__(self) -> int:
        return len(ACTION_KEYS)

    @property
    def positions(self) -> tuple[float, ...]:
        """Return values in canonical ``ACTION_KEYS`` order."""
        return self._positions

    def as_dict(self) -> dict[str, float]:
        """Return a fresh mutable copy for follower or dataset APIs."""
        return dict(zip(ACTION_KEYS, self._positions, strict=True))

    def as_items(self) -> tuple[tuple[str, float], ...]:
        """Return immutable canonical key/value pairs."""
        return tuple(zip(ACTION_KEYS, self._positions, strict=True))


def _format_keys(keys: tuple[Any, ...]) -> str:
    return "[" + ", ".join(sorted(repr(key) for key in keys)) + "]"


def _normalize_exact_action(action: object, *, label: str) -> NormalizedAction:
    if isinstance(action, NormalizedAction):
        return action
    if not isinstance(action, Mapping):
        raise ActionValidationError(f"{label} must be a mapping")

    keys = tuple(action.keys())
    try:
        actual_keys = set(keys)
    except TypeError as error:
        raise ActionValidationError(f"{label} contains an invalid key") from error
    expected_keys = set(ACTION_KEYS)
    if len(action) != len(ACTION_KEYS) or actual_keys != expected_keys:
        missing = tuple(key for key in ACTION_KEYS if key not in actual_keys)
        extra = tuple(key for key in keys if key not in expected_keys)
        detail = []
        if missing:
            detail.append(f"missing={_format_keys(missing)}")
        if extra:
            detail.append(f"extra={_format_keys(extra)}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        raise ActionValidationError(f"{label} must contain exactly the six canonical action keys{suffix}")

    positions = []
    for key in ACTION_KEYS:
        value = action[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ActionValidationError(f"{label}[{key!r}] must be a real number, not bool")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ActionValidationError(f"{label}[{key!r}] cannot be represented as a float") from error
        if not math.isfinite(normalized):
            raise ActionValidationError(f"{label}[{key!r}] must be finite")
        positions.append(normalized)
    return NormalizedAction(tuple(positions))


def validate_returned_action(action: object) -> NormalizedAction:
    """Validate and normalize the authoritative ``send_action`` result."""
    return _normalize_exact_action(action, label="returned action")


def _normalize_tolerance(tolerance: object) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a real number, not bool")
    normalized = float(tolerance)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("tolerance must be finite and non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class ActionComparison:
    """Immutable requested-versus-returned clipping evidence."""

    requested: NormalizedAction
    returned: NormalizedAction
    tolerance: float = DEFAULT_CLIPPING_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.requested, NormalizedAction):
            raise TypeError("requested must be a NormalizedAction")
        if not isinstance(self.returned, NormalizedAction):
            raise TypeError("returned must be a NormalizedAction")
        object.__setattr__(self, "tolerance", _normalize_tolerance(self.tolerance))

    @property
    def clipped_keys(self) -> tuple[str, ...]:
        """Keys whose absolute returned delta exceeds the fixed tolerance."""
        return tuple(
            key for key in ACTION_KEYS if abs(self.returned[key] - self.requested[key]) > self.tolerance
        )

    @property
    def clipping_count(self) -> int:
        """Count clipped joints, not merely clipped calls."""
        return len(self.clipped_keys)

    @property
    def was_clipped(self) -> bool:
        return bool(self.clipped_keys)

    @property
    def adopted_action(self) -> NormalizedAction:
        """Expose the authoritative returned command for worker adoption."""
        return self.returned


def compare_requested_returned(
    requested: object,
    returned: object,
    *,
    tolerance: object = DEFAULT_CLIPPING_TOLERANCE,
) -> ActionComparison:
    """Validate both actions and report deterministic per-joint clipping."""
    return ActionComparison(
        requested=_normalize_exact_action(requested, label="requested action"),
        returned=validate_returned_action(returned),
        tolerance=_normalize_tolerance(tolerance),
    )
