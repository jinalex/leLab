from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from lelab.control_session import (
    SO101_ACTION_KEYS,
    SO101_MOTOR_NAMES,
    ControlResourceKey,
    ControlSessionManager,
    ControlState,
    JointStatusSpec,
    JointStatusUnit,
    MotionState,
    ThermalStatus,
    TorqueOutcome,
    classify_torque_outcome,
)


class UtcClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def manager(*, history_limit: int = 2) -> ControlSessionManager:
    return ControlSessionManager(
        history_limit=history_limit,
        monotonic_clock=lambda: 20.0,
        utc_clock=UtcClock(),
    )


def finish(sessions: ControlSessionManager, session_id: str) -> None:
    sessions.request_stop(session_id, reason=f"finish {session_id}")
    sessions.finish_teardown(
        session_id,
        terminal_state=ControlState.STOPPED,
        torque=classify_torque_outcome(disable_attempted=False),
    )


def joint_specs() -> tuple[JointStatusSpec, ...]:
    return tuple(
        JointStatusSpec(
            action_key=key,
            unit=(
                JointStatusUnit.GRIPPER_PERCENTAGE_POINTS if key == "gripper.pos" else JointStatusUnit.DEGREES
            ),
            max_step_per_tick=0.35,
            max_relative_target=5.0,
            calibrated_min=0.0 if key == "gripper.pos" else -90.0,
            calibrated_max=100.0 if key == "gripper.pos" else 90.0,
        )
        for key in SO101_ACTION_KEYS
    )


def thermal_status() -> ThermalStatus:
    temperatures = dict.fromkeys(SO101_MOTOR_NAMES, 42.0)
    return ThermalStatus.from_mappings(
        temperatures=temperatures,
        reported_peaks=dict.fromkeys(SO101_MOTOR_NAMES, 43.0),
        confirmed_peaks=temperatures,
        spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
    )


def test_terminal_history_is_bounded_immutable_and_latest_persists() -> None:
    sessions = manager(history_limit=2)
    terminals = []
    for index in range(3):
        claim = sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id=f"session-{index}")
        finish(sessions, claim.session_id)
        terminals.append(sessions.current_status())

    history = sessions.terminal_history()
    assert isinstance(history, tuple)
    assert [status.session_id for status in history] == ["session-1", "session-2"]
    assert sessions.current_status() is terminals[-1]
    assert sessions.status_for("session-1") is terminals[-2]
    assert sessions.status_for("session-0") is None
    with pytest.raises(FrozenInstanceError):
        history[-1].stop_reason = "changed"  # type: ignore[misc]
    mutated_copy = history[-1].as_dict()
    mutated_copy["details"]["local"] = True
    assert "local" not in history[-1].details
    with pytest.raises(ValueError, match="never be reused"):
        sessions.claim(ControlResourceKey.STADIA_TELEOPERATION, session_id="session-0")


def test_status_uses_injected_utc_clock_and_rejects_non_json_values() -> None:
    sessions = manager()
    claim = sessions.claim(
        ControlResourceKey.CONTROLLER_CHECK,
        session_id="check-1",
        details={"sample_age_s": 0.1, "layout": [6, 17, 0]},
    )
    started = sessions.active_status(check_expiry=False)
    assert started is not None
    assert started.claimed_at_utc == "2026-09-02T08:00:00+00:00"
    json.dumps(started.as_dict(), allow_nan=False)

    with pytest.raises(ValueError, match="finite JSON-safe"):
        sessions.update_details(claim.session_id, {"sample_age_s": float("nan")})
    with pytest.raises(ValueError, match="finite JSON-safe"):
        sessions.update_details(claim.session_id, {"bad": object()})

    updated = sessions.update_details(claim.session_id, {"sample_age_s": 0.2})
    assert updated.updated_at_utc == "2026-09-02T08:00:01+00:00"
    assert updated.details == {"sample_age_s": 0.2}

    merged = sessions.merge_details(claim.session_id, {"controller_ready": True})
    assert merged.details == {
        "sample_age_s": 0.2,
        "controller_ready": True,
    }


def test_runtime_contract_fields_are_typed_atomic_and_json_safe() -> None:
    sessions = manager()
    claim = sessions.claim(
        ControlResourceKey.STADIA_TELEOPERATION,
        session_id="live-1",
        teleoperator_type="stadia",
    )

    updated = sessions.update_runtime_status(
        claim.session_id,
        controller_connected=True,
        controller_error=None,
        controller_product_name="Google Stadia Controller rev. A",
        controller_guid="guid-1",
        controller_instance_id=9,
        controller_generation=2,
        controller_layout=(6, 17, 0),
        sample_sequence=31,
        sample_age_s=0.012,
        rb_held=True,
        release_observed=True,
        controls_neutral=False,
        motion_state=MotionState.ENABLED,
        joint_specs=joint_specs(),
        saturation_count=3,
        relative_clipping_count=2,
        thermal_snapshot=thermal_status(),
        details_patch={"controller_ready": True, "controller_gate_reason": "ready"},
    )

    assert updated.controller_connected is True
    assert updated.controller_error is None
    assert updated.controller_guid == "guid-1"
    assert updated.controller_layout == (6, 17, 0)
    assert updated.motion_state is MotionState.ENABLED
    assert updated.details["controller_monitoring_active"] is True
    assert updated.details["controller_ready"] is True
    assert updated.details["controller_gate_reason"] == "ready"
    assert tuple(spec.action_key for spec in updated.joint_specs) == SO101_ACTION_KEYS
    payload = updated.as_dict()
    assert payload["controller_connected"] is True
    assert payload["controller_error"] is None
    assert payload["controller_layout"] == {"axes": 6, "buttons": 17, "hats": 0}
    assert payload["joint_limits"]["gripper.pos"]["calibrated_max"] == 100.0
    assert "startup_min" not in payload["joint_limits"]["gripper.pos"]
    assert "startup_max" not in payload["joint_specs"][-1]
    assert payload["thermal_snapshot"]["temperatures"]["gripper"] == 42.0
    json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError, match="non-negative finite"):
        sessions.update_runtime_status(
            claim.session_id,
            controller_product_name=None,
            controller_guid=None,
            controller_instance_id=None,
            controller_generation=None,
            controller_layout=None,
            sample_sequence=None,
            sample_age_s=float("nan"),
            rb_held=None,
            release_observed=False,
            controls_neutral=None,
            motion_state=MotionState.HOLD,
            joint_specs=(),
            saturation_count=0,
            relative_clipping_count=0,
            thermal_snapshot=None,
        )


def test_controller_connection_and_error_evidence_preserves_stale_and_disconnected_truth() -> None:
    sessions = manager()
    claim = sessions.claim(
        ControlResourceKey.CONTROLLER_CHECK,
        session_id="controller-truth",
        teleoperator_type="stadia",
    )

    disconnected = sessions.update_runtime_status(
        claim.session_id,
        controller_connected=False,
        controller_error="controller sample is stale",
        controller_product_name="Google Stadia Controller",
        controller_guid="guid-1",
        controller_instance_id=9,
        controller_generation=2,
        controller_layout=(6, 15, 0),
        sample_sequence=31,
        sample_age_s=0.2,
        rb_held=False,
        release_observed=False,
        controls_neutral=False,
        motion_state=MotionState.DISARMED,
        joint_specs=(),
        saturation_count=0,
        relative_clipping_count=0,
        thermal_snapshot=None,
    )
    assert disconnected.controller_connected is False
    assert disconnected.controller_error == "controller sample is stale"
    assert disconnected.controller_guid == "guid-1"

    common = {
        "controller_product_name": None,
        "controller_guid": None,
        "controller_instance_id": None,
        "controller_generation": None,
        "controller_layout": None,
        "sample_sequence": None,
        "sample_age_s": None,
        "rb_held": None,
        "release_observed": False,
        "controls_neutral": None,
        "motion_state": MotionState.DISARMED,
        "joint_specs": (),
        "saturation_count": 0,
        "relative_clipping_count": 0,
        "thermal_snapshot": None,
    }
    stale = sessions.update_runtime_status(
        claim.session_id,
        controller_connected=True,
        controller_error="controller sample is stale",
        **common,
    )
    assert stale.controller_connected is True
    assert stale.controller_error == "controller sample is stale"
    with pytest.raises(ValueError, match="disconnected controller"):
        sessions.update_runtime_status(
            claim.session_id,
            controller_connected=False,
            controller_error=None,
            **common,
        )
    with pytest.raises(ValueError, match="requires an explicit disconnected"):
        sessions.update_runtime_status(
            claim.session_id,
            controller_connected=None,
            controller_error="read failed",
            **common,
        )


def test_joint_and_thermal_status_reject_incomplete_or_nonfinite_evidence() -> None:
    partial_specs = joint_specs()[:-1]
    sessions = manager()
    claim = sessions.claim(
        ControlResourceKey.STADIA_TELEOPERATION,
        session_id="live-invalid",
        teleoperator_type="stadia",
    )

    with pytest.raises(ValueError, match="each SO-101 action key"):
        sessions.update_runtime_status(
            claim.session_id,
            controller_product_name=None,
            controller_guid=None,
            controller_instance_id=None,
            controller_generation=None,
            controller_layout=None,
            sample_sequence=None,
            sample_age_s=None,
            rb_held=None,
            release_observed=False,
            controls_neutral=None,
            motion_state=MotionState.DISARMED,
            joint_specs=partial_specs,
            saturation_count=0,
            relative_clipping_count=0,
            thermal_snapshot=None,
        )

    with pytest.raises(ValueError, match="exactly the six"):
        ThermalStatus.from_mappings(
            temperatures={"gripper": 42.0},
            reported_peaks=dict.fromkeys(SO101_MOTOR_NAMES, 42.0),
            confirmed_peaks=dict.fromkeys(SO101_MOTOR_NAMES, 42.0),
            spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
            invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
            last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
        )

    with pytest.raises(ValueError, match="finite"):
        ThermalStatus.from_mappings(
            temperatures=dict.fromkeys(SO101_MOTOR_NAMES, float("nan")),
            reported_peaks=dict.fromkeys(SO101_MOTOR_NAMES, 42.0),
            confirmed_peaks=dict.fromkeys(SO101_MOTOR_NAMES, 42.0),
            spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
            invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
            last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
        )

    unavailable = ThermalStatus.from_mappings(
        temperatures=dict.fromkeys(SO101_MOTOR_NAMES),
        reported_peaks=dict.fromkeys(SO101_MOTOR_NAMES),
        confirmed_peaks=dict.fromkeys(SO101_MOTOR_NAMES),
        spike_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        invalid_sample_counts=dict.fromkeys(SO101_MOTOR_NAMES, 0),
        last_invalid_values=dict.fromkeys(SO101_MOTOR_NAMES),
        stop_reason="missing temperature values",
    )
    assert unavailable.as_dict()["temperatures"]["shoulder_pan"] is None
    assert unavailable.as_dict()["last_invalid_values"]["gripper"] is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"disable_attempted": True, "readback": dict.fromkeys(SO101_MOTOR_NAMES, 0)},
            TorqueOutcome.VERIFIED_OFF,
        ),
        (
            {
                "disable_attempted": True,
                "readback": {**dict.fromkeys(SO101_MOTOR_NAMES, 0), "gripper": 1},
            },
            TorqueOutcome.FAILED,
        ),
        (
            {
                "disable_attempted": True,
                "readback": dict.fromkeys(SO101_MOTOR_NAMES, 0),
                "disable_errors": {"gripper": "write failed"},
            },
            TorqueOutcome.FAILED,
        ),
        (
            {"disable_attempted": True, "readback": {"shoulder_pan": 0}},
            TorqueOutcome.UNKNOWN,
        ),
        (
            {
                "disable_attempted": True,
                "readback": dict.fromkeys(SO101_MOTOR_NAMES, 0),
                "verification_supported": False,
            },
            TorqueOutcome.UNKNOWN,
        ),
        (
            {"disable_attempted": False, "readback": dict.fromkeys(SO101_MOTOR_NAMES, 1)},
            TorqueOutcome.FAILED,
        ),
        (
            {"disable_attempted": False, "readback": dict.fromkeys(SO101_MOTOR_NAMES, 0)},
            TorqueOutcome.VERIFIED_OFF,
        ),
        (
            {"disable_attempted": False},
            TorqueOutcome.NOT_ATTEMPTED,
        ),
    ],
)
def test_six_motor_torque_classification_is_conservative_and_evidenced(
    kwargs: dict[str, object], expected: TorqueOutcome
) -> None:
    evidence = classify_torque_outcome(**kwargs)  # type: ignore[arg-type]

    assert evidence.outcome is expected
    assert [motor for motor, _ in evidence.readback] == list(SO101_MOTOR_NAMES)
    json.dumps(evidence.as_dict(), allow_nan=False)


def test_incomplete_and_invalid_readback_names_missing_evidence() -> None:
    evidence = classify_torque_outcome(
        disable_attempted=True,
        readback={"shoulder_pan": 0, "shoulder_lift": 7, "extra": 0},
    )

    assert evidence.outcome is TorqueOutcome.UNKNOWN
    assert evidence.invalid_motors == ("shoulder_lift",)
    assert evidence.missing_motors == SO101_MOTOR_NAMES[2:]
    assert evidence.unexpected_motors == ("extra",)
