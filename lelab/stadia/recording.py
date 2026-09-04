"""Dependency-neutral Stadia recording loop and attempt-cleanup contract.

The legacy leader recording path deliberately does not import this module.  A
route/session adapter supplies robot, dataset, control, audit, and event ports;
this module owns only the safety-critical ordering and recording state machine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .action import (
    ActionComparison,
    NormalizedAction,
    compare_requested_returned,
    validate_returned_action,
)
from .timing import NoCatchUpScheduler


class RecordingEventKind(StrEnum):
    FINISH_EPISODE = "finish_episode"
    RERECORD_EPISODE = "rerecord_episode"
    STOP_SESSION = "stop_session"
    HARD_INVALIDATION = "hard_invalidation"


class HardInvalidationReason(StrEnum):
    CONTROLLER_DISCONNECTED = "controller_disconnected"
    CONTROLLER_STALE = "controller_stale"
    CONTROLLER_READ_ERROR = "controller_read_error"
    CONTROLLER_GUARD_NOT_READY = "controller_guard_not_ready"
    OBSERVATION_FAILURE = "observation_failure"
    CONTROL_FAILURE = "control_failure"
    SEND_FAILURE = "send_failure"
    RETURNED_ACTION_INVALID = "returned_action_invalid"
    ADOPTION_FAILURE = "adoption_failure"
    AUDIT_FAILURE = "audit_failure"
    DATASET_FAILURE = "dataset_failure"
    THERMAL_STOP = "thermal_stop"

    @property
    def recoverable(self) -> bool:
        return self in {
            HardInvalidationReason.CONTROLLER_DISCONNECTED,
            HardInvalidationReason.CONTROLLER_STALE,
            HardInvalidationReason.CONTROLLER_READ_ERROR,
            HardInvalidationReason.CONTROLLER_GUARD_NOT_READY,
        }


class RecordingPhase(StrEnum):
    RECORDING = "recording"
    RESET = "resetting"
    RECOVERY = "recovery"


class MotionBehavior(StrEnum):
    STEP = "step"
    HOLD = "hold"


class ControllerHealth(StrEnum):
    HEALTHY = "healthy"
    DISCONNECTED = "disconnected"
    STALE = "stale"
    READ_ERROR = "read_error"


class RecordingOutcome(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RecordingEvent:
    kind: RecordingEventKind
    reason: HardInvalidationReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RecordingEventKind):
            raise TypeError("kind must be a RecordingEventKind")
        if self.kind is RecordingEventKind.HARD_INVALIDATION:
            if not isinstance(self.reason, HardInvalidationReason):
                raise ValueError("hard invalidation requires a typed reason")
        elif self.reason is not None:
            raise ValueError("only hard invalidation events may have a reason")
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("event detail must be a string or None")

    @classmethod
    def finish_episode(cls) -> RecordingEvent:
        return cls(RecordingEventKind.FINISH_EPISODE)

    @classmethod
    def rerecord_episode(cls) -> RecordingEvent:
        return cls(RecordingEventKind.RERECORD_EPISODE)

    @classmethod
    def stop_session(cls) -> RecordingEvent:
        return cls(RecordingEventKind.STOP_SESSION)

    @classmethod
    def hard_invalidation(
        cls,
        reason: HardInvalidationReason,
        detail: str | None = None,
    ) -> RecordingEvent:
        return cls(RecordingEventKind.HARD_INVALIDATION, reason, detail)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SaturationDelta:
    step: int = 0
    travel: int = 0
    endpoint: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _nonnegative_int(self.step, field="step"))
        object.__setattr__(self, "travel", _nonnegative_int(self.travel, field="travel"))
        object.__setattr__(self, "endpoint", _nonnegative_int(self.endpoint, field="endpoint"))


@dataclass(frozen=True, slots=True)
class ControlTick:
    """One already-bounded target or intentional hold from the control core."""

    requested_action: NormalizedAction
    motion: MotionBehavior
    controller_health: ControllerHealth = ControllerHealth.HEALTHY
    guard_ready: bool = True
    saturations: SaturationDelta = SaturationDelta()

    def __post_init__(self) -> None:
        if not isinstance(self.requested_action, NormalizedAction):
            raise TypeError("requested_action must be a NormalizedAction")
        if not isinstance(self.motion, MotionBehavior):
            raise TypeError("motion must be a MotionBehavior")
        if not isinstance(self.controller_health, ControllerHealth):
            raise TypeError("controller_health must be a ControllerHealth")
        if not isinstance(self.guard_ready, bool):
            raise TypeError("guard_ready must be a bool")
        if not isinstance(self.saturations, SaturationDelta):
            raise TypeError("saturations must be a SaturationDelta")
        if self.controller_health is not ControllerHealth.HEALTHY and self.motion is not MotionBehavior.HOLD:
            raise ValueError("an unhealthy controller may only request an intentional hold")


@dataclass(frozen=True, slots=True)
class AttemptCleanupReport:
    """Positive proof that an unsaved attempt left no dataset residue."""

    memory_frames_cleared: bool
    streaming_fragments_cleared: bool
    episode_rows_unchanged: bool
    video_references_unchanged: bool
    metadata_references_unchanged: bool
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "memory_frames_cleared",
            "streaming_fragments_cleared",
            "episode_rows_unchanged",
            "video_references_unchanged",
            "metadata_references_unchanged",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be a bool")
        object.__setattr__(self, "details", tuple(str(detail) for detail in self.details))

    @property
    def proven_clean(self) -> bool:
        return all(
            (
                self.memory_frames_cleared,
                self.streaming_fragments_cleared,
                self.episode_rows_unchanged,
                self.video_references_unchanged,
                self.metadata_references_unchanged,
            )
        )


class AttemptCleanupError(RuntimeError):
    """Discarding an invalid attempt failed or could not be proven complete.

    The owning worker must treat this as terminal and must not finalize or push
    the dataset after it is raised.
    """

    def __init__(
        self,
        trigger: RecordingEvent,
        *,
        report: AttemptCleanupReport | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.trigger = trigger
        self.report = report
        self.cause = cause
        if cause is not None:
            detail = f"cleanup raised {type(cause).__name__}: {cause}"
        elif report is None:
            detail = "cleanup returned no proof"
        else:
            failed = [
                field
                for field in (
                    "memory_frames_cleared",
                    "streaming_fragments_cleared",
                    "episode_rows_unchanged",
                    "video_references_unchanged",
                    "metadata_references_unchanged",
                )
                if not getattr(report, field)
            ]
            detail = f"unproven cleanup fields: {', '.join(failed)}"
        super().__init__(f"recording attempt cleanup failed closed after {trigger.kind.value}: {detail}")


@dataclass(frozen=True, slots=True)
class RecordingCounters:
    successful_sends: int = 0
    dataset_frames_added: int = 0
    step_saturations: int = 0
    travel_saturations: int = 0
    endpoint_saturations: int = 0
    returned_clippings: int = 0
    hard_invalidations: int = 0
    missed_ticks: int = 0


@dataclass(frozen=True, slots=True)
class FrameAudit:
    phase: RecordingPhase
    motion: MotionBehavior
    controller_health: ControllerHealth
    guard_ready: bool
    requested_action: NormalizedAction
    returned_action: NormalizedAction
    comparison: ActionComparison
    saturations: SaturationDelta
    counters: RecordingCounters
    transition_event: RecordingEvent | None
    dataset_frame_planned: bool
    missed_ticks: int


@dataclass(frozen=True, slots=True)
class RecordingLoopResult:
    outcome: RecordingOutcome
    terminal_event: RecordingEvent
    saved_episodes: int
    attempts_started: int
    counters: RecordingCounters
    event_history: tuple[RecordingEvent, ...]


class FollowerPort(Protocol):
    def get_observation(self) -> Mapping[str, Any]: ...

    def send_action(self, action: Mapping[str, float]) -> object: ...


class StadiaControlPort(Protocol):
    def target_or_hold(
        self,
        observation: Mapping[str, Any],
        *,
        phase: RecordingPhase,
        event: RecordingEvent | None,
    ) -> ControlTick: ...

    def accept_sent_target(self, action: NormalizedAction) -> None: ...


class RecordingEventSource(Protocol):
    def poll(self, phase: RecordingPhase) -> RecordingEvent | None: ...


class RecordingDatasetPort(Protocol):
    """Adapter boundary around LeRobot's writer and attempt residue checks.

    ``begin_attempt`` must capture a read-only checkpoint before the attempt
    writes anything; taking the checkpoint must not itself create dataset
    references.  ``discard_attempt`` must clear the in-memory buffer,
    cancel/remove pending streaming fragments, remove any references created
    after ``checkpoint``, and independently inspect every report field.
    """

    def begin_attempt(self) -> object: ...

    def add_frame(self, frame: Mapping[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def discard_attempt(self, checkpoint: object) -> AttemptCleanupReport: ...


class DatasetFrameBuilder(Protocol):
    """Build a frame whose action comes only from the exact returned command."""

    def __call__(
        self,
        observation: Mapping[str, Any],
        returned_action: NormalizedAction,
        task: str,
    ) -> Mapping[str, Any]: ...


class RecordingAuditPort(Protocol):
    def record_frame(self, frame: FrameAudit) -> None: ...


class ThermalGuardPort(Protocol):
    def check(self) -> object: ...


class TickPacer(Protocol):
    def wait_for_next_tick(self) -> int:
        """Wait for one future tick and return the number of dropped ticks."""
        ...


class FixedRateNoCatchUpPacer:
    """Production 30 Hz pacer that emits at most one step after an overrun."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleeper = sleeper
        self._scheduler = NoCatchUpScheduler(rate_hz=30.0, start_time=clock())

    def wait_for_next_tick(self) -> int:
        while True:
            now = self._clock()
            decision = self._scheduler.poll(now)
            if decision.should_step:
                return decision.missed_ticks
            self._sleeper(max(0.0, decision.next_deadline - now))


_NO_ATTEMPT = object()


def _health_invalidation(health: ControllerHealth) -> RecordingEvent:
    reasons = {
        ControllerHealth.DISCONNECTED: HardInvalidationReason.CONTROLLER_DISCONNECTED,
        ControllerHealth.STALE: HardInvalidationReason.CONTROLLER_STALE,
        ControllerHealth.READ_ERROR: HardInvalidationReason.CONTROLLER_READ_ERROR,
    }
    return RecordingEvent.hard_invalidation(reasons[health])


def _error_event(reason: HardInvalidationReason, error: object) -> RecordingEvent:
    return RecordingEvent.hard_invalidation(reason, f"{type(error).__name__}: {error}")


def _increment_hard(counters: RecordingCounters) -> RecordingCounters:
    return RecordingCounters(
        successful_sends=counters.successful_sends,
        dataset_frames_added=counters.dataset_frames_added,
        step_saturations=counters.step_saturations,
        travel_saturations=counters.travel_saturations,
        endpoint_saturations=counters.endpoint_saturations,
        returned_clippings=counters.returned_clippings,
        hard_invalidations=counters.hard_invalidations + 1,
        missed_ticks=counters.missed_ticks,
    )


def _increment_missed(counters: RecordingCounters, missed: int) -> RecordingCounters:
    return RecordingCounters(
        successful_sends=counters.successful_sends,
        dataset_frames_added=counters.dataset_frames_added,
        step_saturations=counters.step_saturations,
        travel_saturations=counters.travel_saturations,
        endpoint_saturations=counters.endpoint_saturations,
        returned_clippings=counters.returned_clippings,
        hard_invalidations=counters.hard_invalidations,
        missed_ticks=counters.missed_ticks + missed,
    )


def _increment_success(
    counters: RecordingCounters,
    tick: ControlTick,
    comparison: ActionComparison,
) -> RecordingCounters:
    return RecordingCounters(
        successful_sends=counters.successful_sends + 1,
        dataset_frames_added=counters.dataset_frames_added,
        step_saturations=counters.step_saturations + tick.saturations.step,
        travel_saturations=counters.travel_saturations + tick.saturations.travel,
        endpoint_saturations=counters.endpoint_saturations + tick.saturations.endpoint,
        returned_clippings=counters.returned_clippings + comparison.clipping_count,
        hard_invalidations=counters.hard_invalidations,
        missed_ticks=counters.missed_ticks,
    )


def _increment_frame(counters: RecordingCounters) -> RecordingCounters:
    return RecordingCounters(
        successful_sends=counters.successful_sends,
        dataset_frames_added=counters.dataset_frames_added + 1,
        step_saturations=counters.step_saturations,
        travel_saturations=counters.travel_saturations,
        endpoint_saturations=counters.endpoint_saturations,
        returned_clippings=counters.returned_clippings,
        hard_invalidations=counters.hard_invalidations,
        missed_ticks=counters.missed_ticks,
    )


def _discard_attempt(
    dataset: RecordingDatasetPort,
    checkpoint: object,
    trigger: RecordingEvent,
) -> None:
    try:
        report = dataset.discard_attempt(checkpoint)
    except Exception as error:
        raise AttemptCleanupError(trigger, cause=error) from error
    if not isinstance(report, AttemptCleanupReport):
        raise AttemptCleanupError(trigger)
    if not report.proven_clean:
        raise AttemptCleanupError(trigger, report=report)


def _result(
    *,
    outcome: RecordingOutcome,
    terminal_event: RecordingEvent,
    saved_episodes: int,
    attempts_started: int,
    counters: RecordingCounters,
    event_history: list[RecordingEvent],
) -> RecordingLoopResult:
    return RecordingLoopResult(
        outcome=outcome,
        terminal_event=terminal_event,
        saved_episodes=saved_episodes,
        attempts_started=attempts_started,
        counters=counters,
        event_history=tuple(event_history),
    )


def record_stadia_loop(
    *,
    follower: FollowerPort,
    control: StadiaControlPort,
    dataset: RecordingDatasetPort,
    events: RecordingEventSource,
    frame_builder: DatasetFrameBuilder,
    audit: RecordingAuditPort,
    thermal_guard: ThermalGuardPort,
    initial_action: Mapping[str, float] | NormalizedAction,
    task: str,
    num_episodes: int,
    pacer: TickPacer | None = None,
    thermal_check_interval_ticks: int = 30,
) -> RecordingLoopResult:
    """Run the Stadia-only multi-episode recording state machine.

    For every successfully sent tick the order is fixed: observation, bounded
    target/hold, ``send_action``, strict returned-action validation, returned
    target adoption, audit, and only then (during a valid recording attempt)
    dataset-frame construction and buffering.
    """

    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int) or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    thermal_check_interval_ticks = _nonnegative_int(
        thermal_check_interval_ticks,
        field="thermal_check_interval_ticks",
    )
    if thermal_check_interval_ticks < 1:
        raise ValueError("thermal_check_interval_ticks must be at least 1")

    last_accepted = validate_returned_action(initial_action)
    pacer = pacer or FixedRateNoCatchUpPacer()
    phase = RecordingPhase.RECORDING
    saved_episodes = 0
    attempts_started = 0
    frames_in_attempt = 0
    counters = RecordingCounters()
    event_history: list[RecordingEvent] = []
    schedule_ticks_since_thermal = thermal_check_interval_ticks

    try:
        checkpoint: object = dataset.begin_attempt()
        attempts_started = 1
    except Exception as error:
        event = _error_event(HardInvalidationReason.DATASET_FAILURE, error)
        return _result(
            outcome=RecordingOutcome.ERROR,
            terminal_event=event,
            saved_episodes=saved_episodes,
            attempts_started=attempts_started,
            counters=_increment_hard(counters),
            event_history=[event],
        )

    def end_hard(event: RecordingEvent) -> RecordingLoopResult:
        nonlocal checkpoint, counters
        event_history.append(event)
        counters = _increment_hard(counters)
        if checkpoint is not _NO_ATTEMPT:
            _discard_attempt(dataset, checkpoint, event)
            checkpoint = _NO_ATTEMPT
        return _result(
            outcome=RecordingOutcome.ERROR,
            terminal_event=event,
            saved_episodes=saved_episodes,
            attempts_started=attempts_started,
            counters=counters,
            event_history=event_history,
        )

    def begin_next_attempt() -> RecordingLoopResult | None:
        nonlocal checkpoint, attempts_started, frames_in_attempt, counters
        try:
            checkpoint = dataset.begin_attempt()
        except Exception as error:
            checkpoint = _NO_ATTEMPT
            return end_hard(_error_event(HardInvalidationReason.DATASET_FAILURE, error))
        attempts_started += 1
        frames_in_attempt = 0
        return None

    while True:
        try:
            missed_ticks = pacer.wait_for_next_tick()
            missed_ticks = _nonnegative_int(missed_ticks, field="missed_ticks")
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))
        counters = _increment_missed(counters, missed_ticks)
        schedule_ticks_since_thermal += missed_ticks + 1

        try:
            event = events.poll(phase)
            if event is not None and not isinstance(event, RecordingEvent):
                raise TypeError("event source must return RecordingEvent or None")
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))

        if event is not None and event.kind is RecordingEventKind.STOP_SESSION:
            event_history.append(event)
            if checkpoint is not _NO_ATTEMPT:
                _discard_attempt(dataset, checkpoint, event)
                checkpoint = _NO_ATTEMPT
            return _result(
                outcome=RecordingOutcome.STOPPED,
                terminal_event=event,
                saved_episodes=saved_episodes,
                attempts_started=attempts_started,
                counters=counters,
                event_history=event_history,
            )

        if (
            event is not None
            and event.kind is RecordingEventKind.HARD_INVALIDATION
            and event.reason is not None
            and not event.reason.recoverable
        ):
            return end_hard(event)

        if schedule_ticks_since_thermal >= thermal_check_interval_ticks:
            try:
                thermal_guard.check()
            except Exception as error:
                return end_hard(_error_event(HardInvalidationReason.THERMAL_STOP, error))
            schedule_ticks_since_thermal = 0

        try:
            observation = follower.get_observation()
            if not isinstance(observation, Mapping):
                raise TypeError("follower observation must be a mapping")
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.OBSERVATION_FAILURE, error))

        try:
            tick = control.target_or_hold(observation, phase=phase, event=event)
            if not isinstance(tick, ControlTick):
                raise TypeError("control port must return ControlTick")
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))

        hard_event: RecordingEvent | None = None
        if tick.controller_health is not ControllerHealth.HEALTHY and phase is not RecordingPhase.RECOVERY:
            hard_event = _health_invalidation(tick.controller_health)
        elif (
            tick.controller_health is ControllerHealth.HEALTHY
            and not tick.guard_ready
            and phase is RecordingPhase.RECORDING
        ):
            hard_event = RecordingEvent.hard_invalidation(HardInvalidationReason.CONTROLLER_GUARD_NOT_READY)
        elif event is not None and event.kind is RecordingEventKind.HARD_INVALIDATION:
            hard_event = event

        transition_event = hard_event or event
        transition_requires_hold = transition_event is not None and transition_event.kind in {
            RecordingEventKind.FINISH_EPISODE,
            RecordingEventKind.RERECORD_EPISODE,
            RecordingEventKind.HARD_INVALIDATION,
        }
        unsafe_step = not tick.guard_ready and tick.motion is MotionBehavior.STEP
        recovery_step = phase is RecordingPhase.RECOVERY and tick.motion is MotionBehavior.STEP
        hold_changed_target = (
            tick.motion is MotionBehavior.HOLD and tick.requested_action.positions != last_accepted.positions
        )
        if transition_requires_hold and tick.motion is not MotionBehavior.HOLD:
            error = ValueError("finish, re-record, and invalidation transitions must request a hold")
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))
        if unsafe_step or recovery_step:
            error = ValueError("motion is forbidden until the controller guard is ready for a new episode")
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))
        if hold_changed_target:
            error = ValueError("an intentional hold must equal the last accepted follower target")
            return end_hard(_error_event(HardInvalidationReason.CONTROL_FAILURE, error))

        try:
            returned_raw = follower.send_action(tick.requested_action.as_dict())
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.SEND_FAILURE, error))

        try:
            returned = validate_returned_action(returned_raw)
            comparison = compare_requested_returned(tick.requested_action, returned)
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.RETURNED_ACTION_INVALID, error))

        try:
            control.accept_sent_target(returned)
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.ADOPTION_FAILURE, error))
        last_accepted = returned
        counters = _increment_success(counters, tick, comparison)
        audit_counters = _increment_hard(counters) if hard_event is not None else counters

        dataset_frame_planned = phase is RecordingPhase.RECORDING and transition_event is None
        frame_audit = FrameAudit(
            phase=phase,
            motion=tick.motion,
            controller_health=tick.controller_health,
            guard_ready=tick.guard_ready,
            requested_action=tick.requested_action,
            returned_action=returned,
            comparison=comparison,
            saturations=tick.saturations,
            counters=audit_counters,
            transition_event=transition_event,
            dataset_frame_planned=dataset_frame_planned,
            missed_ticks=missed_ticks,
        )
        try:
            audit.record_frame(frame_audit)
        except Exception as error:
            return end_hard(_error_event(HardInvalidationReason.AUDIT_FAILURE, error))
        counters = audit_counters

        if dataset_frame_planned:
            try:
                frame = frame_builder(observation, returned, task)
                if not isinstance(frame, Mapping):
                    raise TypeError("frame builder must return a mapping")
                dataset.add_frame(dict(frame))
            except Exception as error:
                return end_hard(_error_event(HardInvalidationReason.DATASET_FAILURE, error))
            counters = _increment_frame(counters)
            frames_in_attempt += 1

        if hard_event is not None:
            event_history.append(hard_event)
            if checkpoint is not _NO_ATTEMPT:
                _discard_attempt(dataset, checkpoint, hard_event)
                checkpoint = _NO_ATTEMPT
            if hard_event.reason is not None and hard_event.reason.recoverable:
                frames_in_attempt = 0
                phase = RecordingPhase.RECOVERY
                continue
            return _result(
                outcome=RecordingOutcome.ERROR,
                terminal_event=hard_event,
                saved_episodes=saved_episodes,
                attempts_started=attempts_started,
                counters=counters,
                event_history=event_history,
            )

        if event is not None:
            event_history.append(event)
            if event.kind is RecordingEventKind.RERECORD_EPISODE:
                if checkpoint is not _NO_ATTEMPT:
                    _discard_attempt(dataset, checkpoint, event)
                    checkpoint = _NO_ATTEMPT
                frames_in_attempt = 0
                if phase is RecordingPhase.RECORDING:
                    phase = RecordingPhase.RESET
                continue

            if event.kind is RecordingEventKind.FINISH_EPISODE:
                if phase is RecordingPhase.RECORDING:
                    if frames_in_attempt < 1:
                        error = ValueError("cannot save an episode without a validated dataset frame")
                        return end_hard(_error_event(HardInvalidationReason.DATASET_FAILURE, error))
                    try:
                        dataset.save_episode()
                    except Exception as error:
                        return end_hard(_error_event(HardInvalidationReason.DATASET_FAILURE, error))
                    checkpoint = _NO_ATTEMPT
                    saved_episodes += 1
                    frames_in_attempt = 0
                    if saved_episodes >= num_episodes:
                        return _result(
                            outcome=RecordingOutcome.COMPLETED,
                            terminal_event=event,
                            saved_episodes=saved_episodes,
                            attempts_started=attempts_started,
                            counters=counters,
                            event_history=event_history,
                        )
                    phase = RecordingPhase.RESET
                    continue

                if tick.controller_health is ControllerHealth.HEALTHY and tick.guard_ready:
                    started = begin_next_attempt()
                    if started is not None:
                        return started
                    phase = RecordingPhase.RECORDING
                else:
                    phase = RecordingPhase.RECOVERY
                continue

        if (
            phase is RecordingPhase.RECOVERY
            and tick.controller_health is ControllerHealth.HEALTHY
            and tick.guard_ready
        ):
            started = begin_next_attempt()
            if started is not None:
                return started
            phase = RecordingPhase.RECORDING
