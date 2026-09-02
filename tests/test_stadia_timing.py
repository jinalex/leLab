from __future__ import annotations

import pytest

from lelab.stadia.timing import NoCatchUpScheduler


def test_fixed_30_hz_scheduler_emits_at_most_one_step() -> None:
    scheduler = NoCatchUpScheduler(start_time=0.0)
    first = scheduler.poll(0.0)
    assert first.should_step
    assert first.missed_ticks == 0
    assert first.next_deadline == pytest.approx(1 / 30)
    assert not scheduler.poll(0.01).should_step


def test_overrun_drops_missed_ticks_and_moves_deadline_into_future() -> None:
    scheduler = NoCatchUpScheduler(start_time=0.0)
    late = scheduler.poll(0.2)
    assert late.should_step
    assert late.missed_ticks == 6
    assert late.next_deadline > 0.2
    assert not scheduler.poll(0.2).should_step


def test_stale_hold_and_recovery_never_catch_up() -> None:
    scheduler = NoCatchUpScheduler(start_time=0.0)
    assert scheduler.poll(0.0).should_step
    stale = scheduler.poll(5.0, ready=False)
    assert not stale.should_step
    assert stale.missed_ticks == 0
    recovered = scheduler.poll(10.0, ready=True)
    assert not recovered.should_step
    assert recovered.missed_ticks == 0
    assert recovered.next_deadline == pytest.approx(10.0 + 1 / 30)
    assert scheduler.poll(10.0 + 1 / 30, ready=True).should_step
