"""Software-only decision-table tests for the Stadia recording loop."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from lelab.stadia import recording
from lelab.stadia.action import NormalizedAction, validate_returned_action
from lelab.stadia.recording import (
    AttemptCleanupError,
    AttemptCleanupReport,
    ControllerHealth,
    ControlTick,
    HardInvalidationReason,
    MotionBehavior,
    RecordingEvent,
    RecordingEventKind,
    RecordingOutcome,
    RecordingPhase,
    SaturationDelta,
    record_stadia_loop,
)
from lelab.stadia.types import ACTION_KEYS


def _action(value: float = 0.0, **overrides: float) -> NormalizedAction:
    values = dict.fromkeys(ACTION_KEYS, value)
    values.update(overrides)
    return validate_returned_action(values)


def _tick(
    action: NormalizedAction,
    *,
    motion: MotionBehavior = MotionBehavior.HOLD,
    health: ControllerHealth = ControllerHealth.HEALTHY,
    guard_ready: bool = True,
    saturations: SaturationDelta = SaturationDelta(),
) -> ControlTick:
    return ControlTick(
        requested_action=action,
        motion=motion,
        controller_health=health,
        guard_ready=guard_ready,
        saturations=saturations,
    )


class LoggedReturned(dict):
    pass


@dataclass
class FakePacer:
    trace: list[str]
    missed: deque[int]

    def wait_for_next_tick(self) -> int:
        self.trace.append("pace")
        if not self.missed:
            raise AssertionError("unexpected recording-loop iteration")
        return self.missed.popleft()


@dataclass
class FakeEvents:
    trace: list[str]
    values: deque[RecordingEvent | None]
    phases: list[RecordingPhase] = field(default_factory=list)

    def poll(self, phase: RecordingPhase) -> RecordingEvent | None:
        self.trace.append("event")
        self.phases.append(phase)
        if not self.values:
            raise AssertionError("event script exhausted")
        return self.values.popleft()


@dataclass
class FakeThermal:
    trace: list[str]
    fail_on_call: int | None = None
    calls: int = 0

    def check(self) -> None:
        self.trace.append("thermal")
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("confirmed heat")


@dataclass
class FakeFollower:
    trace: list[str]
    returned: deque[object]
    observations: int = 0
    sent: list[dict[str, float]] = field(default_factory=list)

    def get_observation(self) -> Mapping[str, Any]:
        self.trace.append("observation")
        self.observations += 1
        return {"state": self.observations, "image": object()}

    def send_action(self, action: Mapping[str, float]) -> object:
        self.trace.append("send")
        self.sent.append(dict(action))
        if not self.returned:
            raise AssertionError("returned-action script exhausted")
        result = self.returned.popleft()
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class FakeControl:
    trace: list[str]
    ticks: deque[ControlTick]
    accepted: list[NormalizedAction] = field(default_factory=list)
    calls: list[tuple[RecordingPhase, RecordingEvent | None]] = field(default_factory=list)

    def target_or_hold(
        self,
        observation: Mapping[str, Any],
        *,
        phase: RecordingPhase,
        event: RecordingEvent | None,
    ) -> ControlTick:
        del observation
        self.trace.append("target")
        self.calls.append((phase, event))
        if not self.ticks:
            raise AssertionError("control script exhausted")
        return self.ticks.popleft()

    def accept_sent_target(self, action: NormalizedAction) -> None:
        self.trace.append("accept")
        self.accepted.append(action)


@dataclass
class FakeAudit:
    trace: list[str]
    frames: list[recording.FrameAudit] = field(default_factory=list)

    def record_frame(self, frame: recording.FrameAudit) -> None:
        self.trace.append("audit")
        self.frames.append(frame)


@dataclass
class FakeFrameBuilder:
    trace: list[str]
    calls: list[tuple[Mapping[str, Any], NormalizedAction, str]] = field(default_factory=list)

    def __call__(
        self,
        observation: Mapping[str, Any],
        returned_action: NormalizedAction,
        task: str,
    ) -> Mapping[str, Any]:
        self.trace.append("build")
        self.calls.append((observation, returned_action, task))
        return {
            "observation": dict(observation),
            "action": returned_action.as_dict(),
            "task": task,
        }


@dataclass
class FakeDataset:
    trace: list[str]
    forced_unproven: set[str] = field(default_factory=set)
    discard_error: Exception | None = None
    save_error: Exception | None = None
    current_frames: list[dict[str, Any]] = field(default_factory=list)
    saved: list[list[dict[str, Any]]] = field(default_factory=list)
    all_added: list[dict[str, Any]] = field(default_factory=list)
    streaming_fragments: int = 0
    episode_rows: int = 0
    video_references: int = 0
    metadata_references: int = 0
    discard_calls: int = 0
    begin_calls: int = 0

    def begin_attempt(self) -> tuple[int, int, int]:
        self.trace.append("begin")
        if self.current_frames:
            raise AssertionError("new attempt began with uncleared frames")
        self.begin_calls += 1
        return (self.episode_rows, self.video_references, self.metadata_references)

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        self.trace.append("add")
        copied = dict(frame)
        copied["action"] = dict(copied["action"])
        self.current_frames.append(copied)
        self.all_added.append(copied)
        self.streaming_fragments += 1

    def save_episode(self) -> None:
        self.trace.append("save")
        if self.save_error is not None:
            # Model a writer that failed after creating references. Cleanup must
            # roll these back or report that it cannot prove the attempt clean.
            self.episode_rows += 1
            self.video_references += 1
            self.metadata_references += 1
            raise self.save_error
        self.saved.append(list(self.current_frames))
        self.episode_rows += 1
        self.video_references += int(bool(self.current_frames))
        self.metadata_references += 1
        self.current_frames.clear()
        self.streaming_fragments = 0

    def discard_attempt(self, checkpoint: object) -> AttemptCleanupReport:
        self.trace.append("discard")
        self.discard_calls += 1
        if self.discard_error is not None:
            raise self.discard_error
        episode_rows, video_references, metadata_references = checkpoint
        self.current_frames.clear()
        self.streaming_fragments = 0
        self.episode_rows = episode_rows
        self.video_references = video_references
        self.metadata_references = metadata_references
        values = {
            "memory_frames_cleared": not self.current_frames,
            "streaming_fragments_cleared": self.streaming_fragments == 0,
            "episode_rows_unchanged": self.episode_rows == episode_rows,
            "video_references_unchanged": self.video_references == video_references,
            "metadata_references_unchanged": self.metadata_references == metadata_references,
        }
        for field_name in self.forced_unproven:
            values[field_name] = False
        return AttemptCleanupReport(**values)


@dataclass
class Harness:
    trace: list[str]
    follower: FakeFollower
    control: FakeControl
    dataset: FakeDataset
    events: FakeEvents
    builder: FakeFrameBuilder
    audit: FakeAudit
    thermal: FakeThermal
    pacer: FakePacer


def _harness(
    *,
    ticks: list[ControlTick],
    returned: list[object],
    events: list[RecordingEvent | None],
    missed: list[int] | None = None,
    dataset: FakeDataset | None = None,
    thermal_fail_on_call: int | None = None,
) -> Harness:
    trace = dataset.trace if dataset is not None else []
    dataset = dataset or FakeDataset(trace)
    return Harness(
        trace=trace,
        follower=FakeFollower(trace, deque(returned)),
        control=FakeControl(trace, deque(ticks)),
        dataset=dataset,
        events=FakeEvents(trace, deque(events)),
        builder=FakeFrameBuilder(trace),
        audit=FakeAudit(trace),
        thermal=FakeThermal(trace, fail_on_call=thermal_fail_on_call),
        pacer=FakePacer(trace, deque(missed or [0] * len(events))),
    )


def _run(
    harness: Harness,
    *,
    num_episodes: int = 1,
    thermal_check_interval_ticks: int = 30,
):
    return record_stadia_loop(
        follower=harness.follower,
        control=harness.control,
        dataset=harness.dataset,
        events=harness.events,
        frame_builder=harness.builder,
        audit=harness.audit,
        thermal_guard=harness.thermal,
        initial_action=_action(),
        task="Pick up the block",
        num_episodes=num_episodes,
        pacer=harness.pacer,
        thermal_check_interval_ticks=thermal_check_interval_ticks,
    )


def test_event_types_are_distinct_and_hard_invalidation_requires_a_reason() -> None:
    events = (
        RecordingEvent.finish_episode(),
        RecordingEvent.rerecord_episode(),
        RecordingEvent.stop_session(),
        RecordingEvent.hard_invalidation(HardInvalidationReason.CONTROLLER_STALE, "old sample"),
    )

    assert tuple(event.kind for event in events) == tuple(RecordingEventKind)
    assert events[-1].reason is HardInvalidationReason.CONTROLLER_STALE
    assert events[-1].detail == "old sample"
    with pytest.raises(ValueError, match="requires a typed reason"):
        RecordingEvent(RecordingEventKind.HARD_INVALIDATION)
    with pytest.raises(ValueError, match="only hard invalidation"):
        RecordingEvent(RecordingEventKind.STOP_SESSION, HardInvalidationReason.SEND_FAILURE)


def test_mandatory_order_and_dataset_action_use_validated_returned_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _action(1.0)
    returned_values = requested.as_dict()
    returned_values["shoulder_pan.pos"] = 0.6
    returned = LoggedReturned(returned_values)
    accepted_hold = validate_returned_action(returned_values)
    harness = _harness(
        ticks=[
            _tick(requested, motion=MotionBehavior.STEP),
            _tick(accepted_hold),
        ],
        returned=[returned, accepted_hold.as_dict()],
        events=[None, RecordingEvent.finish_episode()],
    )
    real_validate = recording.validate_returned_action

    def logged_validate(value: object) -> NormalizedAction:
        if isinstance(value, LoggedReturned):
            harness.trace.append("validate")
        return real_validate(value)

    monkeypatch.setattr(recording, "validate_returned_action", logged_validate)

    result = _run(harness)

    start = harness.trace.index("observation")
    assert harness.trace[start : start + 8] == [
        "observation",
        "target",
        "send",
        "validate",
        "accept",
        "audit",
        "build",
        "add",
    ]
    assert result.outcome is RecordingOutcome.COMPLETED
    assert harness.dataset.saved[0][0]["action"] == returned_values
    assert harness.dataset.saved[0][0]["action"] != requested.as_dict()
    assert harness.control.accepted[0].as_dict() == returned_values
    assert harness.builder.calls[0][1] is harness.control.accepted[0]


def test_rb_up_intentional_hold_is_a_valid_recorded_frame() -> None:
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), hold.as_dict()],
        events=[None, RecordingEvent.finish_episode()],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.COMPLETED
    assert len(harness.dataset.saved[0]) == 1
    assert harness.audit.frames[0].motion is MotionBehavior.HOLD
    assert harness.audit.frames[0].dataset_frame_planned


def test_saturation_and_relative_clipping_are_counted_surfaced_adopted_and_saved() -> None:
    requested = _action(0.35)
    returned_values = requested.as_dict()
    returned_values["wrist_flex.pos"] = 0.2
    returned_values["gripper.pos"] = 0.1
    returned = validate_returned_action(returned_values)
    harness = _harness(
        ticks=[
            _tick(
                requested,
                motion=MotionBehavior.STEP,
                saturations=SaturationDelta(step=3, travel=2, endpoint=1),
            ),
            _tick(returned),
        ],
        returned=[returned_values, returned_values],
        events=[None, RecordingEvent.finish_episode()],
    )

    result = _run(harness)

    assert result.counters.step_saturations == 3
    assert result.counters.travel_saturations == 2
    assert result.counters.endpoint_saturations == 1
    assert result.counters.returned_clippings == 2
    assert harness.audit.frames[0].comparison.clipped_keys == (
        "wrist_flex.pos",
        "gripper.pos",
    )
    assert harness.dataset.saved[0][0]["action"] == returned_values
    assert harness.control.accepted[0] == returned


@pytest.mark.parametrize(
    ("health", "reason"),
    [
        (ControllerHealth.DISCONNECTED, HardInvalidationReason.CONTROLLER_DISCONNECTED),
        (ControllerHealth.STALE, HardInvalidationReason.CONTROLLER_STALE),
        (ControllerHealth.READ_ERROR, HardInvalidationReason.CONTROLLER_READ_ERROR),
    ],
)
def test_controller_fault_hard_invalidates_clears_and_requires_guarded_recovery(
    health: ControllerHealth,
    reason: HardInvalidationReason,
) -> None:
    hold = _action()
    harness = _harness(
        ticks=[
            _tick(hold),
            _tick(hold, health=health, guard_ready=False),
            _tick(hold, health=health, guard_ready=False),
            _tick(hold, guard_ready=False),
            _tick(hold, guard_ready=True),
            _tick(hold),
            _tick(hold),
        ],
        returned=[hold.as_dict()] * 7,
        events=[None, None, None, None, None, None, RecordingEvent.finish_episode()],
        missed=[0, 0, 5, 0, 0, 0, 0],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.COMPLETED
    assert result.attempts_started == 2
    assert result.counters.hard_invalidations == 1
    assert result.counters.missed_ticks == 5
    assert result.event_history[0].kind is RecordingEventKind.HARD_INVALIDATION
    assert result.event_history[0].reason is reason
    assert harness.dataset.discard_calls == 1
    assert harness.dataset.begin_calls == 2
    assert len(harness.dataset.saved) == 1
    assert len(harness.dataset.saved[0]) == 1
    assert harness.events.phases[2:5] == [RecordingPhase.RECOVERY] * 3
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


def test_reset_has_no_dataset_writes_and_loss_blocks_next_episode_without_catch_up() -> None:
    hold = _action()
    harness = _harness(
        ticks=[
            _tick(hold),
            _tick(hold),
            _tick(hold, motion=MotionBehavior.STEP),
            _tick(hold, health=ControllerHealth.DISCONNECTED, guard_ready=False),
            _tick(hold, guard_ready=False),
            _tick(hold, guard_ready=True),
            _tick(hold),
            _tick(hold),
        ],
        returned=[hold.as_dict()] * 8,
        events=[
            None,
            RecordingEvent.finish_episode(),
            None,
            None,
            None,
            None,
            None,
            RecordingEvent.finish_episode(),
        ],
        missed=[0, 0, 7, 0, 0, 0, 0, 0],
    )

    result = _run(harness, num_episodes=2)

    assert result.outcome is RecordingOutcome.COMPLETED
    assert result.saved_episodes == 2
    assert result.attempts_started == 2
    assert result.counters.missed_ticks == 7
    assert len(harness.control.calls) == 8
    assert [frame.phase for frame in harness.audit.frames] == [
        RecordingPhase.RECORDING,
        RecordingPhase.RECORDING,
        RecordingPhase.RESET,
        RecordingPhase.RESET,
        RecordingPhase.RECOVERY,
        RecordingPhase.RECOVERY,
        RecordingPhase.RECORDING,
        RecordingPhase.RECORDING,
    ]
    assert len(harness.dataset.all_added) == 2
    assert [len(episode) for episode in harness.dataset.saved] == [1, 1]


def test_explicit_rerecord_clears_attempt_runs_reset_without_writes_then_records_again() -> None:
    first = _action()
    reset_target = _action(0.1)
    harness = _harness(
        ticks=[
            _tick(first),
            _tick(first),
            _tick(reset_target, motion=MotionBehavior.STEP),
            _tick(reset_target),
            _tick(reset_target),
            _tick(reset_target),
        ],
        returned=[
            first.as_dict(),
            first.as_dict(),
            reset_target.as_dict(),
            reset_target.as_dict(),
            reset_target.as_dict(),
            reset_target.as_dict(),
        ],
        events=[
            None,
            RecordingEvent.rerecord_episode(),
            None,
            RecordingEvent.finish_episode(),
            None,
            RecordingEvent.finish_episode(),
        ],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.COMPLETED
    assert result.attempts_started == 2
    assert harness.dataset.discard_calls == 1
    assert len(harness.dataset.saved) == 1
    assert len(harness.dataset.saved[0]) == 1
    assert harness.dataset.saved[0][0]["action"] == reset_target.as_dict()
    assert [event.kind for event in result.event_history] == [
        RecordingEventKind.RERECORD_EPISODE,
        RecordingEventKind.FINISH_EPISODE,
        RecordingEventKind.FINISH_EPISODE,
    ]
    assert sum(frame.dataset_frame_planned for frame in harness.audit.frames) == 2


def test_explicit_stop_clears_unsaved_attempt_and_sends_nothing_after_stop() -> None:
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold)],
        returned=[hold.as_dict()],
        events=[None, RecordingEvent.stop_session()],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.STOPPED
    assert result.terminal_event.kind is RecordingEventKind.STOP_SESSION
    assert harness.dataset.saved == []
    assert harness.dataset.discard_calls == 1
    assert len(harness.follower.sent) == 1
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


@pytest.mark.parametrize(
    "returned",
    [
        dict.fromkeys(ACTION_KEYS[:-1], 0.0),
        {**dict.fromkeys(ACTION_KEYS, 0.0), "gripper.pos": float("nan")},
        {**dict.fromkeys(ACTION_KEYS, 0.0), "wrist_roll.pos": True},
    ],
)
def test_malformed_or_nonfinite_returned_action_clears_attempt_and_ends(returned: object) -> None:
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), returned],
        events=[None, None],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.ERROR
    assert result.terminal_event.reason is HardInvalidationReason.RETURNED_ACTION_INVALID
    assert harness.dataset.discard_calls == 1
    assert len(harness.control.accepted) == 1
    assert len(harness.audit.frames) == 1
    assert len(harness.dataset.all_added) == 1
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


def test_send_failure_clears_attempt_and_ends_without_further_processing() -> None:
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), OSError("serial write failed")],
        events=[None, None],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.ERROR
    assert result.terminal_event.reason is HardInvalidationReason.SEND_FAILURE
    assert harness.dataset.discard_calls == 1
    assert len(harness.control.accepted) == 1
    assert len(harness.audit.frames) == 1
    assert len(harness.dataset.all_added) == 1
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


def test_thermal_stop_clears_buffered_attempt_and_sends_nothing_after_stop() -> None:
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), hold.as_dict()],
        events=[None, None],
        thermal_fail_on_call=2,
    )

    result = _run(harness, thermal_check_interval_ticks=1)

    assert result.outcome is RecordingOutcome.ERROR
    assert result.terminal_event.reason is HardInvalidationReason.THERMAL_STOP
    assert harness.dataset.discard_calls == 1
    assert harness.follower.observations == 1
    assert len(harness.follower.sent) == 1
    assert len(harness.dataset.all_added) == 1
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


@pytest.mark.parametrize(
    "unproven_field",
    [
        "memory_frames_cleared",
        "streaming_fragments_cleared",
        "episode_rows_unchanged",
        "video_references_unchanged",
        "metadata_references_unchanged",
    ],
)
def test_invalid_attempt_fails_closed_when_any_cleanup_proof_is_missing(
    unproven_field: str,
) -> None:
    trace: list[str] = []
    dataset = FakeDataset(trace, forced_unproven={unproven_field})
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold)],
        returned=[hold.as_dict()],
        events=[None, RecordingEvent.stop_session()],
        dataset=dataset,
    )

    with pytest.raises(AttemptCleanupError) as error:
        _run(harness)

    assert error.value.report is not None
    assert not getattr(error.value.report, unproven_field)
    assert harness.dataset.discard_calls == 1
    assert len(harness.follower.sent) == 1


def test_cleanup_exception_fails_closed_and_no_more_control_ticks_run() -> None:
    trace: list[str] = []
    dataset = FakeDataset(trace, discard_error=OSError("cannot remove fragment"))
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold)],
        returned=[hold.as_dict()],
        events=[None, RecordingEvent.stop_session()],
        dataset=dataset,
    )

    with pytest.raises(AttemptCleanupError, match="cleanup raised OSError"):
        _run(harness)

    assert harness.dataset.discard_calls == 1
    assert len(harness.follower.sent) == 1


def test_partial_save_failure_is_rolled_back_before_terminal_error() -> None:
    trace: list[str] = []
    dataset = FakeDataset(trace, save_error=RuntimeError("metadata write failed"))
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), hold.as_dict()],
        events=[None, RecordingEvent.finish_episode()],
        dataset=dataset,
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.ERROR
    assert result.terminal_event.reason is HardInvalidationReason.DATASET_FAILURE
    assert harness.dataset.episode_rows == 0
    assert harness.dataset.video_references == 0
    assert harness.dataset.metadata_references == 0
    assert harness.dataset.current_frames == []
    assert harness.dataset.streaming_fragments == 0


def test_partial_save_reference_residue_fails_closed() -> None:
    trace: list[str] = []
    dataset = FakeDataset(
        trace,
        forced_unproven={"metadata_references_unchanged"},
        save_error=RuntimeError("metadata write failed"),
    )
    hold = _action()
    harness = _harness(
        ticks=[_tick(hold), _tick(hold)],
        returned=[hold.as_dict(), hold.as_dict()],
        events=[None, RecordingEvent.finish_episode()],
        dataset=dataset,
    )

    with pytest.raises(AttemptCleanupError) as error:
        _run(harness)

    assert error.value.trigger.kind is RecordingEventKind.HARD_INVALIDATION
    assert error.value.trigger.reason is HardInvalidationReason.DATASET_FAILURE
    assert error.value.report is not None
    assert not error.value.report.metadata_references_unchanged


def test_changed_target_labeled_as_hold_fails_before_send_and_cleans_attempt() -> None:
    changed = _action(0.1)
    harness = _harness(
        ticks=[_tick(changed)],
        returned=[changed.as_dict()],
        events=[None],
    )

    result = _run(harness)

    assert result.outcome is RecordingOutcome.ERROR
    assert result.terminal_event.reason is HardInvalidationReason.CONTROL_FAILURE
    assert harness.follower.sent == []
    assert harness.dataset.discard_calls == 1
