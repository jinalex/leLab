"""Immutable types and explicit units for Stadia control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

STADIA_PRODUCT_NAME = "Google Stadia Controller"


class PositionUnit(StrEnum):
    DEGREES = "degrees"
    GRIPPER_PERCENTAGE_POINTS = "gripper_percentage_points"


@dataclass(frozen=True)
class ControllerLayout:
    axes: int
    buttons: int
    hats: int


STADIA_LAYOUT = ControllerLayout(axes=6, buttons=15, hats=0)


@dataclass(frozen=True)
class StadiaSnapshot:
    """One immutable reader publication from a single connection generation."""

    sequence: int
    sampled_at: float
    connected: bool
    product_name: str | None
    guid: str | None
    instance_id: int | None
    connection_generation: int
    axes: tuple[float, ...] = ()
    buttons: tuple[bool, ...] = ()
    hats: tuple[tuple[int, int], ...] = ()
    layout: ControllerLayout = ControllerLayout(0, 0, 0)
    read_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(float(value) for value in self.axes))
        object.__setattr__(self, "buttons", tuple(bool(value) for value in self.buttons))
        object.__setattr__(
            self,
            "hats",
            tuple((int(x), int(y)) for x, y in self.hats),
        )

    def axis(self, index: int) -> float:
        return float(self.axes[index]) if 0 <= index < len(self.axes) else 0.0

    def button(self, index: int) -> bool:
        return bool(self.buttons[index]) if 0 <= index < len(self.buttons) else False


@dataclass(frozen=True)
class JointControlSpec:
    action_key: str
    unit: PositionUnit
    max_step_per_tick: float = 0.35
    startup_travel: float = 45.0


DEFAULT_JOINT_SPECS = (
    JointControlSpec("shoulder_pan.pos", PositionUnit.DEGREES),
    JointControlSpec("shoulder_lift.pos", PositionUnit.DEGREES),
    JointControlSpec("elbow_flex.pos", PositionUnit.DEGREES),
    JointControlSpec("wrist_flex.pos", PositionUnit.DEGREES),
    JointControlSpec("wrist_roll.pos", PositionUnit.DEGREES),
    JointControlSpec("gripper.pos", PositionUnit.GRIPPER_PERCENTAGE_POINTS),
)
ACTION_KEYS = tuple(spec.action_key for spec in DEFAULT_JOINT_SPECS)


@dataclass(frozen=True)
class TriggerNeutralState:
    exercised: bool = False
    released: bool = False
    requires_exercise: bool = False


@dataclass(frozen=True)
class NeutralGateState:
    connection_generation: int | None = None
    release_seen: bool = False
    neutral_armed: bool = False
    stable_neutral_samples: int = 0
    last_sequence: int | None = None
    left_trigger: TriggerNeutralState = TriggerNeutralState()
    right_trigger: TriggerNeutralState = TriggerNeutralState()


@dataclass(frozen=True)
class NeutralGateDecision:
    state: NeutralGateState
    profile_valid: bool
    controls_neutral: bool
    motion_enabled: bool
    reason: str | None


@dataclass(frozen=True)
class MappedStadiaInput:
    normalized_inputs: tuple[tuple[str, float], ...]
    joint_deltas: tuple[tuple[str, float], ...]

    def inputs_dict(self) -> dict[str, float]:
        return dict(self.normalized_inputs)

    def deltas_dict(self) -> dict[str, float]:
        return dict(self.joint_deltas)


@dataclass(frozen=True)
class IntegratorCounters:
    step_saturations: int = 0
    travel_saturations: int = 0
    endpoint_saturations: int = 0
    returned_clippings: int = 0


@dataclass(frozen=True)
class IntegrationResult:
    requested_action: tuple[tuple[str, float], ...]
    counters: IntegratorCounters

    def action_dict(self) -> dict[str, float]:
        return dict(self.requested_action)
