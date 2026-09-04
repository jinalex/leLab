from __future__ import annotations

import pytest

from lelab.stadia.thermal_safety import (
    ConfirmedTemperatureGuard,
    ConfirmedTemperatureStopError,
    ThermalSafetyConfig,
)

MOTORS = ("j1", "j2")


class FakeTemperatureBus:
    def __init__(self, readings: list[dict[str, float]]) -> None:
        self.readings = list(readings)
        self.calls = 0

    def sync_read(
        self,
        data_name: str,
        motors: list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, float]:
        assert data_name == "Present_Temperature"
        assert motors == list(MOTORS)
        assert normalize is False
        assert num_retry == 5
        self.calls += 1
        return dict(self.readings.pop(0))


def test_50c_is_warning_and_requires_only_one_read() -> None:
    bus = FakeTemperatureBus([{"j1": 50, "j2": 49}])
    snapshot = ConfirmedTemperatureGuard(bus, MOTORS).check()
    assert bus.calls == 1
    assert snapshot.warning_motors == ("j1",)
    assert snapshot.confirmed_peaks == {"j1": 50, "j2": 49}


def test_below_warning_does_not_warn() -> None:
    snapshot = ConfirmedTemperatureGuard(FakeTemperatureBus([{"j1": 49.9, "j2": 20}]), MOTORS).check()
    assert snapshot.warning_motors == ()


def test_isolated_60c_spike_is_retained_without_stopping() -> None:
    sleeps: list[float] = []
    guard = ConfirmedTemperatureGuard(
        FakeTemperatureBus([{"j1": 60, "j2": 34}, {"j1": 32, "j2": 34}, {"j1": 32, "j2": 34}]),
        MOTORS,
        sleeper=sleeps.append,
    )
    snapshot = guard.check()
    assert snapshot.reported_peaks["j1"] == 60
    assert snapshot.confirmed_peaks["j1"] == 32
    assert snapshot.spike_counts["j1"] == 1
    assert sleeps == [0.05, 0.05]


def test_two_of_three_60c_samples_confirm_stop_with_actual_evidence() -> None:
    bus = FakeTemperatureBus([{"j1": 60, "j2": 34}, {"j1": 61, "j2": 34}, {"j1": 34, "j2": 34}])
    guard = ConfirmedTemperatureGuard(
        bus,
        MOTORS,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ConfirmedTemperatureStopError, match="j1 readings 60, 61, 34C") as first_stop:
        guard.check()
    assert guard.snapshot().confirmed_peaks["j1"] == 61

    with pytest.raises(ConfirmedTemperatureStopError) as latched_stop:
        guard.check()
    assert latched_stop.value is first_stop.value
    assert bus.calls == 3


def test_new_thermal_guard_starts_with_a_clear_latch() -> None:
    hot_bus = FakeTemperatureBus([{"j1": 60, "j2": 34}, {"j1": 61, "j2": 34}, {"j1": 34, "j2": 34}])
    guard = ConfirmedTemperatureGuard(hot_bus, MOTORS, sleeper=lambda _seconds: None)
    with pytest.raises(ConfirmedTemperatureStopError):
        guard.check()

    cool_bus = FakeTemperatureBus([{"j1": 35, "j2": 34}])
    snapshot = ConfirmedTemperatureGuard(cool_bus, MOTORS).check()
    assert snapshot.temperatures == {"j1": 35.0, "j2": 34.0}
    assert cool_bus.calls == 1


def test_invalid_reading_is_retried_and_counted() -> None:
    guard = ConfirmedTemperatureGuard(
        FakeTemperatureBus([{"j1": 255, "j2": 32}, {"j1": 33, "j2": 32}]),
        MOTORS,
        sleeper=lambda _seconds: None,
    )
    snapshot = guard.check()
    assert snapshot.invalid_sample_counts == {"j1": 1, "j2": 0}
    assert snapshot.last_invalid_values == {"j1": 255, "j2": None}


def test_repeated_invalid_readings_fail_closed() -> None:
    guard = ConfirmedTemperatureGuard(
        FakeTemperatureBus([{"j1": 255, "j2": 32}, {"j1": 255, "j2": 32}, {"j1": 255, "j2": 32}]),
        MOTORS,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="invalid temperature values after 3 reads"):
        guard.check()


def test_missing_temperature_fails_closed() -> None:
    guard = ConfirmedTemperatureGuard(FakeTemperatureBus([{"j1": 31}]), MOTORS)
    with pytest.raises(RuntimeError, match="missing temperature"):
        guard.check()


def test_valid_hot_sample_remains_actual_evidence_if_confirmation_packet_is_missing() -> None:
    guard = ConfirmedTemperatureGuard(
        FakeTemperatureBus([{"j1": 61, "j2": 31}, {"j2": 31}]),
        MOTORS,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="missing temperature values for: j1"):
        guard.check()

    assert guard.snapshot().temperatures == {"j1": 61.0, "j2": 31.0}
    assert guard.snapshot().reported_peaks == {"j1": 61.0, "j2": 31.0}


def test_thermal_configuration_preserves_confirmed_guard_contract() -> None:
    config = ThermalSafetyConfig()
    assert config == ThermalSafetyConfig(
        warning_c=50.0,
        stop_c=60.0,
        confirmation_reads=2,
        required_hot_samples=2,
        confirmation_interval_s=0.05,
        num_retry=5,
    )
    with pytest.raises(ValueError, match="thresholds"):
        ThermalSafetyConfig(stop_c=61)
