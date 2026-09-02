"""Confirmed, fail-closed temperature monitoring for Stadia sessions."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol


class TemperatureBus(Protocol):
    def sync_read(
        self,
        data_name: str,
        motors: list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class ThermalSafetyConfig:
    warning_c: float = 50.0
    stop_c: float = 60.0
    confirmation_reads: int = 2
    required_hot_samples: int = 2
    confirmation_interval_s: float = 0.05
    num_retry: int = 5

    def __post_init__(self) -> None:
        if not 40 <= self.warning_c < self.stop_c <= 60:
            raise ValueError("temperature thresholds must satisfy 40 <= warning < stop <= 60")
        if not 1 <= self.confirmation_reads <= 4:
            raise ValueError("confirmation_reads must be between 1 and 4")
        if not 2 <= self.required_hot_samples <= self.confirmation_reads + 1:
            raise ValueError("required_hot_samples must fit the initial plus confirmation samples")
        if not 0.02 <= self.confirmation_interval_s <= 0.2:
            raise ValueError("confirmation_interval_s must be between 0.02 and 0.2")
        if not 0 <= self.num_retry <= 10:
            raise ValueError("num_retry must be between 0 and 10")


@dataclass(frozen=True)
class ThermalSnapshot:
    temperatures: dict[str, float]
    reported_peaks: dict[str, float]
    confirmed_peaks: dict[str, float]
    spike_counts: dict[str, int]
    invalid_sample_counts: dict[str, int]
    last_invalid_values: dict[str, float | None]
    warning_motors: tuple[str, ...]


class ConfirmedTemperatureStopError(RuntimeError):
    def __init__(
        self,
        motor_samples: Mapping[str, tuple[float, ...]],
        *,
        stop_c: float,
        required_hot_samples: int,
    ) -> None:
        self.motor_samples = dict(motor_samples)
        self.stop_c = stop_c
        self.required_hot_samples = required_hot_samples
        detail = "; ".join(
            f"{motor} readings {', '.join(f'{value:.0f}' for value in samples)}C"
            for motor, samples in self.motor_samples.items()
        )
        super().__init__(
            f"Confirmed heat stop: {detail} ({required_hot_samples} readings at or above {stop_c:.0f}C)"
        )


class ConfirmedTemperatureGuard:
    """Confirm hot packets while retaining spikes and rejecting bad sensor data."""

    def __init__(
        self,
        bus: TemperatureBus,
        motors: tuple[str, ...],
        config: ThermalSafetyConfig | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.bus = bus
        self.motors = motors
        self.config = config or ThermalSafetyConfig()
        self.sleeper = sleeper
        self._reported_peaks = dict.fromkeys(motors, float("-inf"))
        self._confirmed_peaks = dict.fromkeys(motors, float("-inf"))
        self._spike_counts = dict.fromkeys(motors, 0)
        self._invalid_sample_counts = dict.fromkeys(motors, 0)
        self._last_invalid_values: dict[str, float | None] = dict.fromkeys(motors)
        self._temperatures = dict.fromkeys(motors, float("nan"))
        self._latched_error: ConfirmedTemperatureStopError | None = None

    def check(self) -> ThermalSnapshot:
        if self._latched_error is not None:
            raise self._latched_error

        first = self._read()
        self._temperatures = first
        self._record_reported(first)
        initially_hot = tuple(motor for motor in self.motors if first[motor] >= self.config.stop_c)
        if not initially_hot:
            self._record_confirmed(first)
            self._temperatures = first
            return self.snapshot()

        samples = {motor: [first[motor]] for motor in self.motors}
        latest = first
        for _ in range(self.config.confirmation_reads):
            self.sleeper(self.config.confirmation_interval_s)
            latest = self._read()
            self._temperatures = latest
            self._record_reported(latest)
            for motor in self.motors:
                samples[motor].append(latest[motor])

        confirmed_hot = {
            motor: tuple(values)
            for motor, values in samples.items()
            if sum(value >= self.config.stop_c for value in values) >= self.config.required_hot_samples
        }
        if confirmed_hot:
            for motor in self.motors:
                below_stop = [value for value in samples[motor] if value < self.config.stop_c]
                if below_stop:
                    self._confirmed_peaks[motor] = max(self._confirmed_peaks[motor], max(below_stop))
            for motor, values in confirmed_hot.items():
                self._confirmed_peaks[motor] = max(self._confirmed_peaks[motor], max(values))
            self._temperatures = latest
            self._latched_error = ConfirmedTemperatureStopError(
                confirmed_hot,
                stop_c=self.config.stop_c,
                required_hot_samples=self.config.required_hot_samples,
            )
            raise self._latched_error

        for motor in initially_hot:
            self._spike_counts[motor] += 1
        for motor in self.motors:
            below_stop = [value for value in samples[motor] if value < self.config.stop_c]
            if below_stop:
                self._confirmed_peaks[motor] = max(self._confirmed_peaks[motor], max(below_stop))
        self._temperatures = latest
        return self.snapshot()

    def snapshot(self) -> ThermalSnapshot:
        warnings = tuple(
            motor
            for motor in self.motors
            if math.isfinite(self._temperatures[motor])
            and self.config.warning_c <= self._temperatures[motor] < self.config.stop_c
        )
        return ThermalSnapshot(
            temperatures=dict(self._temperatures),
            reported_peaks=dict(self._reported_peaks),
            confirmed_peaks=dict(self._confirmed_peaks),
            spike_counts=dict(self._spike_counts),
            invalid_sample_counts=dict(self._invalid_sample_counts),
            last_invalid_values=dict(self._last_invalid_values),
            warning_motors=warnings,
        )

    def _read(self) -> dict[str, float]:
        invalid_attempts: dict[str, list[float]] = {motor: [] for motor in self.motors}
        for attempt in range(self.config.confirmation_reads + 1):
            values = self.bus.sync_read(
                "Present_Temperature",
                list(self.motors),
                normalize=False,
                num_retry=self.config.num_retry,
            )
            missing = sorted(set(self.motors) - set(values))
            if missing:
                raise RuntimeError(f"missing temperature values for: {', '.join(missing)}")
            result = {motor: float(values[motor]) for motor in self.motors}
            invalid = [
                motor
                for motor, value in result.items()
                if not math.isfinite(value) or not -30 <= value <= 125
            ]
            if not invalid:
                return result
            for motor in invalid:
                self._invalid_sample_counts[motor] += 1
                self._last_invalid_values[motor] = result[motor]
                invalid_attempts[motor].append(result[motor])
            if attempt < self.config.confirmation_reads:
                self.sleeper(self.config.confirmation_interval_s)

        details = "; ".join(
            f"{motor} readings "
            + ", ".join("non-finite" if not math.isfinite(value) else f"{value:g}C" for value in samples)
            for motor, samples in invalid_attempts.items()
            if samples
        )
        raise RuntimeError(
            "invalid temperature values after "
            f"{self.config.confirmation_reads + 1} reads: {details} "
            "(valid sensor range -30C to 125C)"
        )

    def _record_reported(self, values: Mapping[str, float]) -> None:
        for motor, value in values.items():
            self._reported_peaks[motor] = max(self._reported_peaks[motor], value)

    def _record_confirmed(self, values: Mapping[str, float]) -> None:
        for motor, value in values.items():
            self._confirmed_peaks[motor] = max(self._confirmed_peaks[motor], value)
