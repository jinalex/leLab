"""Fixed-rate scheduling that drops missed work instead of catching up."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TickDecision:
    should_step: bool
    missed_ticks: int
    next_deadline: float


class NoCatchUpScheduler:
    """Issue at most one tick and always advance to a future deadline."""

    def __init__(self, *, rate_hz: float = 30.0, start_time: float = 0.0) -> None:
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        if not math.isfinite(start_time):
            raise ValueError("start_time must be finite")
        self.period = 1.0 / rate_hz
        self.next_deadline = float(start_time)
        self._ready = True

    def poll(self, now: float, *, ready: bool = True) -> TickDecision:
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if not ready:
            self._ready = False
            self.next_deadline = now + self.period
            return TickDecision(False, 0, self.next_deadline)
        if not self._ready:
            self._ready = True
            self.next_deadline = now + self.period
            return TickDecision(False, 0, self.next_deadline)
        if now < self.next_deadline:
            return TickDecision(False, 0, self.next_deadline)

        elapsed = max(0.0, now - self.next_deadline)
        missed_ticks = int(math.floor(elapsed / self.period + 1e-12))
        self.next_deadline += (missed_ticks + 1) * self.period
        while self.next_deadline <= now:
            self.next_deadline += self.period
            missed_ticks += 1
        return TickDecision(True, missed_ticks, self.next_deadline)
