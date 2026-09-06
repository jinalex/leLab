"""Real LeRobot normalization/send_action, fake transport; never opens devices."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lelab.control_session import SO101_MOTOR_NAMES
from lelab.stadia.integrator import BoundedStadiaIntegrator
from lelab.utils.follower_guard import (
    PORTS_ENV,
    CommandCancelledError,
    FollowerGuardError,
    FollowerWriteGuard,
    guarded_ports,
    install_guarded_followers,
    selected_port,
    stadia_follower_construction,
)
from lelab.utils.python_process import python_module_command
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower.so_follower import SOFollower


@pytest.fixture
def guarded(monkeypatch):
    names = SO101_MOTOR_NAMES
    bus = FeetechMotorsBus(
        port="/fake/never-open",
        motors={
            name: Motor(
                index, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES
            )
            for index, name in enumerate(names, 1)
        },
        calibration={
            name: MotorCalibration(id=index, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000)
            for index, name in enumerate(names, 1)
        },
    )
    monkeypatch.setattr(bus.port_handler, "is_open", True)
    writes = []
    raw = dict.fromkeys(names, 2000)
    now = [100.0]
    hook = [lambda: None]

    def read_batch(_addr, _length, ids, **kwargs):
        assert kwargs["num_retry"] == 0
        hook[0]()
        return {id_: raw[bus._id_to_name(id_)] for id_ in ids}, 0

    def read_one(_addr, _length, id_, **kwargs):
        assert kwargs["num_retry"] == 0
        return raw[bus._id_to_name(id_)], 0, 0

    monkeypatch.setattr(bus, "_sync_read", read_batch)
    monkeypatch.setattr(bus, "_read", read_one)
    monkeypatch.setattr(bus, "_sync_write", lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(bus, "_write", lambda *args, **kwargs: writes.append((args, kwargs)))
    follower = SimpleNamespace(
        bus=bus, is_connected=True, config=SimpleNamespace(max_relative_target=dict.fromkeys(names, 5.0))
    )
    follower.send_action = SOFollower.send_action.__get__(follower)
    guard = FollowerWriteGuard(follower, clock=lambda: now[0])
    guard.before_write = lambda: None
    action = {f"{name}.pos": 50.0 if name == "gripper" else 0.0 for name in names}
    yield SimpleNamespace(
        bus=bus, follower=follower, guard=guard, writes=writes, raw=raw, now=now, hook=hook, action=action
    )
    bus.port_handler.is_open = False


@pytest.mark.parametrize(
    "motor,raw",
    [
        ("shoulder_pan", -23328),
        ("shoulder_lift", -23072),
        ("elbow_flex", -23071),
        ("wrist_flex", -22560),
        ("wrist_roll", -21794),
        ("gripper", -23071),
        ("gripper", 4000),
        ("elbow_flex", 999),
        ("elbow_flex", 4096),
    ],
)
def test_invalid_feedback_never_reaches_real_goal_write(guarded, motor, raw):
    guarded.raw[motor] = raw
    with pytest.raises(FollowerGuardError, match="outside calibrated/native"):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []
    guarded.raw[motor] = 2000
    with pytest.raises(FollowerGuardError):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []  # fault does not auto-recover


def test_valid_action_retains_native_relative_clip(guarded):
    guarded.action["elbow_flex.pos"] = 10
    returned = guarded.follower.send_action(guarded.action)
    assert returned["elbow_flex.pos"] == 5
    assert len(guarded.writes) == 1
    assert guarded.writes[0][1]["num_retry"] == 0


def test_in_range_feedback_jump_cannot_relocate_the_requested_goal(guarded):
    guarded.follower.send_action(guarded.action)
    guarded.raw["elbow_flex"] = 2800  # valid raw range, implausible new feedback
    with pytest.raises(FollowerGuardError, match="displaced requested goal"):
        guarded.follower.send_action(guarded.action)
    assert len(guarded.writes) == 1


def test_torque_enable_requires_complete_seed(guarded):
    with pytest.raises(FollowerGuardError, match="six goals must be seeded"):
        guarded.bus.write("Torque_Enable", "elbow_flex", 1)
    assert guarded.writes == []


@pytest.mark.parametrize("value", [-2191.307692308, 999, float("nan"), float("inf"), True])
def test_invalid_requested_goals_rejected_before_io(guarded, value):
    guarded.action["elbow_flex.pos"] = value
    with pytest.raises(FollowerGuardError):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []


def test_final_goal_revalidated_after_relative_calculation(guarded):
    guarded.guard._send_action = lambda action: guarded.bus.sync_write(
        "Goal_Position", {"elbow_flex": -2191.307692308}
    )
    with pytest.raises(FollowerGuardError, match="final goal"):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []


def test_blocking_read_expires_command_without_write_or_retry(guarded):
    guarded.hook[0] = lambda: guarded.now.__setitem__(0, 100.251)
    with pytest.raises(FollowerGuardError, match="expired"):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []


def test_blocking_read_rechecks_rb_and_does_not_latch_normal_release(guarded):
    authorized = [True]
    guarded.hook[0] = lambda: authorized.__setitem__(0, False)

    def authorize():
        if not authorized[0]:
            raise CommandCancelledError("RB released")

    guarded.guard.before_write = authorize
    with pytest.raises(CommandCancelledError):
        guarded.follower.send_action(guarded.action)
    assert guarded.writes == []
    assert guarded.guard.fault is None
    guarded.hook[0] = lambda: None
    authorized[0] = True
    guarded.follower.send_action(guarded.action)
    assert len(guarded.writes) == 1


def test_startup_seed_requires_validated_feedback_and_authorization(guarded):
    pose = guarded.bus.sync_read("Present_Position", normalize=False, num_retry=5)
    guarded.bus.write("Goal_Position", "elbow_flex", pose["elbow_flex"], normalize=False, num_retry=5)
    assert len(guarded.writes) == 1
    assert guarded.writes[0][1]["num_retry"] == 0
    with pytest.raises(FollowerGuardError, match="exactly seed"):
        guarded.bus.write("Goal_Position", "elbow_flex", 2100, normalize=False)
    assert len(guarded.writes) == 1


def test_startup_seed_cancellation_precedes_write(guarded):
    guarded.bus.sync_read("Present_Position", normalize=False)

    def cancel():
        raise CommandCancelledError("controller changed during pose read")

    guarded.guard.before_write = cancel
    with pytest.raises(CommandCancelledError):
        guarded.bus.write("Goal_Position", "elbow_flex", 2000, normalize=False)
    assert guarded.writes == []


def test_integrator_cannot_adopt_impossible_returned_goal(guarded):
    from lelab.stadia.action import ActionValidationError

    integrator = BoundedStadiaIntegrator(initial_action=guarded.action, endpoint_bounds=guarded.guard.bounds)
    bad = dict(guarded.action, **{"elbow_flex.pos": -2191.307692308})
    with pytest.raises(ActionValidationError, match="outside calibrated"):
        integrator.accept_returned_action(bad)
    assert integrator.target == guarded.action
    assert integrator.counters.returned_clippings == 0


def test_opt_in_exact_resolved_port_and_child_bootstrap(monkeypatch, tmp_path):
    device = tmp_path / "device"
    alias = tmp_path / "alias"
    alias.symlink_to(device)
    monkeypatch.setenv(PORTS_ENV, json.dumps([str(alias)]))
    assert selected_port(str(device))
    assert not selected_port(str(tmp_path / "other"))
    monkeypatch.delenv("LELAB_PYTHON_MODULE_WRAPPER", raising=False)
    assert python_module_command("lerobot.scripts.lerobot_record", ["--test=hello world"]) == [
        sys.executable,
        "-m",
        "lelab.scripts.guarded_module",
        "lerobot.scripts.lerobot_record",
        "--",
        "--test=hello world",
    ]
    monkeypatch.setenv("LELAB_PYTHON_MODULE_WRAPPER", '["-m", "tailbridge", "run-python", "--module"]')
    command = python_module_command("target", ["--value=x y"])
    assert command == [
        sys.executable,
        "-m",
        "tailbridge",
        "run-python",
        "--module",
        "lelab.scripts.guarded_module",
        "--",
        "target",
        "--",
        "--value=x y",
    ]


@pytest.mark.parametrize("value", ["[]", "{}", '["relative"]', "[true]"])
def test_invalid_port_configuration_fails_closed(monkeypatch, value):
    monkeypatch.setenv(PORTS_ENV, value)
    with pytest.raises(ValueError):
        guarded_ports()


def test_uncovered_follower_workflows_block_before_constructor(monkeypatch, tmp_path):
    original = SOFollower.__init__
    monkeypatch.setenv(PORTS_ENV, '["/fake/guarded"]')
    try:
        install_guarded_followers()
        for degrees in [False, True]:
            with pytest.raises(FollowerGuardError, match="Stadia teleoperation/recording only"):
                SOFollower.__init__(
                    SimpleNamespace(), SimpleNamespace(port="/fake/guarded", use_degrees=degrees)
                )
    finally:
        SOFollower.__init__ = original


def test_selected_stadia_requires_protocol_guard_before_device_open(monkeypatch):
    monkeypatch.setenv(PORTS_ENV, '["/fake/guarded"]')
    original = SOFollower.__init__

    # Replace construction itself to avoid any filesystem/device dependencies.
    def construct(self, config):
        self.bus = SimpleNamespace(port_handler=object())

    monkeypatch.setattr(SOFollower, "__init__", construct)
    monkeypatch.setitem(
        sys.modules,
        "tailbridge.feetech_consumer",
        SimpleNamespace(
            feetech_reply_integrity_status=lambda port: {"enabled": False},
            record_feetech_context=lambda **kwargs: None,
        ),
    )
    try:
        install_guarded_followers()
        with stadia_follower_construction(), pytest.raises(FollowerGuardError, match="strict Feetech"):
            SOFollower.__init__(SimpleNamespace(), SimpleNamespace(port="/fake/guarded"))
    finally:
        SOFollower.__init__ = original


@pytest.mark.parametrize("change", ["release", "disconnect", "stale", "identity", "hold"])
def test_worker_checks_current_authorization_and_original_input_age(change):
    from dataclasses import replace

    from tests.test_stadia_session import make_harness, stadia_snapshot

    harness = make_harness()
    worker = harness.worker
    worker._wait_for_neutral_startup()
    origin = stadia_snapshot(10, rb=True, left_x=1.0)
    harness.clock.value = origin.sampled_at
    decision, healthy, _, _ = worker._evaluate_snapshot(origin)
    assert healthy and decision.motion_enabled
    worker._movement_enabled = True
    worker._pending_command_snapshot = origin
    current = replace(origin, sequence=11, sampled_at=origin.sampled_at + 0.001)
    if change == "release":
        current = stadia_snapshot(11, rb=False, sampled_at=current.sampled_at)
    elif change == "disconnect":
        current = stadia_snapshot(11, connected=False, sampled_at=current.sampled_at)
    elif change == "identity":
        current = replace(current, connection_generation=2, instance_id=8)
    elif change == "hold":
        harness.claim.hold_requested.set()
    elif change == "stale":
        # Reader itself is fresh; the command it authorized is now too old.
        current = replace(current, sampled_at=origin.sampled_at + 0.151)
    harness.clock.value = current.sampled_at
    harness.reader.runtime = []
    harness.reader._checkpoint_pending = False
    harness.reader.latest = current
    with pytest.raises(CommandCancelledError):
        worker._require_current_write_authorization()
    assert harness.follower.sent_actions == []


def test_cancelled_teleop_step_rolls_back_unsent_integration():
    from tests.test_stadia_session import make_harness, stadia_snapshot

    harness = make_harness(
        runtime=[
            stadia_snapshot(5, rb=True, left_x=1.0),
            stadia_snapshot(6, rb=True, left_x=1.0),
            stadia_snapshot(7, rb=True, left_x=1.0),
        ]
    )
    harness.follower.send_errors = [CommandCancelledError("released during read")]
    harness.follower.stop_after_sends = 2
    result = harness.worker.run()
    assert result.commands_sent == 1
    assert harness.follower.sent_actions[0] == harness.follower.sent_actions[1]


@pytest.mark.parametrize("corrupt", [False, True])
def test_combined_tailbridge_sdk_and_lelab_write_boundary(monkeypatch, tmp_path, corrupt):
    integrity = pytest.importorskip("tailbridge.feetech_consumer")
    import scservo_sdk as sdk

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    port = "/fake/combined-guard-no-device"
    monkeypatch.setenv(PORTS_ENV, json.dumps([port]))
    calibration = {
        name: {"id": index, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
        for index, name in enumerate(SO101_MOTOR_NAMES, 1)
    }
    (tmp_path / "fixture.json").write_text(json.dumps(calibration))
    config = SO101FollowerConfig(
        port=port,
        id="fixture",
        calibration_dir=tmp_path,
        use_degrees=True,
        max_relative_target=dict.fromkeys(SO101_MOTOR_NAMES, 5.0),
    )
    original = SOFollower.__init__
    follower = None
    try:
        with integrity.feetech_reply_integrity(port, diagnostic_path=tmp_path / "fault.json"):
            install_guarded_followers()
            with stadia_follower_construction():
                follower = SO101Follower(config)
            handler = follower.bus.port_handler
            replies = []
            for motor_id in range(1, 7):
                payload = [32] if corrupt and motor_id == 1 else [0xD0, 0x07]
                packet = [255, 255, motor_id, len(payload) + 2, 0, *payload]
                replies.extend([*packet, ~sum(packet[2:]) & 255])
            writes = []
            handler.is_open = True
            handler.clearPort = lambda: None
            handler.setPacketTimeout = lambda size: None
            handler.isPacketTimeout = lambda: not replies

            def read_port(size):
                chunk = replies[:size]
                del replies[:size]
                return chunk

            def write_port(packet):
                writes.append(bytes(packet))
                return len(packet)

            handler.readPort = read_port
            handler.writePort = write_port
            follower._lelab_follower_guard.before_write = lambda: None
            action = {f"{name}.pos": 50.0 if name == "gripper" else 10.0 for name in SO101_MOTOR_NAMES}
            if corrupt:
                with pytest.raises(FollowerGuardError):
                    follower.send_action(action)
                assert [packet[4] for packet in writes] == [sdk.INST_SYNC_READ]
                assert integrity.feetech_reply_integrity_status(handler)["quarantined"]
                with pytest.raises(FollowerGuardError):
                    follower.send_action(action)
                assert len(writes) == 1
            else:
                returned = follower.send_action(action)
                assert returned["elbow_flex.pos"] == 5.0
                assert [packet[4] for packet in writes] == [sdk.INST_SYNC_READ, sdk.INST_SYNC_WRITE]
    finally:
        SOFollower.__init__ = original
        if follower is not None:
            follower.bus.port_handler.is_open = False


@pytest.mark.parametrize("recording", [False, True])
def test_production_stadia_factories_enter_supported_construction_context(monkeypatch, recording):
    import lelab.utils.follower_guard as guard_module
    import lerobot.robots.so_follower as so_module
    from lelab.stadia.recording_session import _default_recording_follower_factory
    from lelab.stadia.session import FollowerBuildSpec, _default_follower_factory

    installed = []
    monkeypatch.setattr(guard_module, "install_guarded_followers", lambda: installed.append(True))

    def construct(config):
        assert installed == [True]
        assert guard_module._supported_construction.get()
        assert config.port == "/fake/guarded"
        return "constructed-without-device"

    monkeypatch.setattr(so_module, "SO101Follower", construct)
    factory = _default_recording_follower_factory if recording else _default_follower_factory
    spec = FollowerBuildSpec(
        port="/fake/guarded",
        calibration_id="test",
        max_relative_target=dict.fromkeys(SO101_MOTOR_NAMES, 5.0),
        cameras={},
    )
    assert factory(spec) == "constructed-without-device"
    assert not guard_module._supported_construction.get()


def test_real_module_child_blocks_uncovered_bridge_constructor(monkeypatch, tmp_path):
    monkeypatch.setenv(PORTS_ENV, '["/fake/guarded"]')
    monkeypatch.delenv("LELAB_PYTHON_MODULE_WRAPPER", raising=False)
    (tmp_path / "bridge_child_fixture.py").write_text(
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from lerobot.robots.so_follower.so_follower import SOFollower\n"
        "from lelab.utils.follower_guard import FollowerGuardError\n"
        "assert sys.argv[1:] == ['--label=two words']\n"
        "try:\n"
        "    SOFollower.__init__(SimpleNamespace(), SimpleNamespace(port='/fake/guarded'))\n"
        "except FollowerGuardError:\n"
        "    print('blocked-before-constructor')\n"
        "else:\n"
        "    raise AssertionError('missing child guard')\n"
    )
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")])
    )
    result = subprocess.run(
        python_module_command("bridge_child_fixture", ["--label=two words"]),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "blocked-before-constructor" in result.stdout


def test_quarantined_teardown_preserves_errors_but_reports_unknown_torque():
    from lelab.control_session import TorqueOutcome
    from tests.test_stadia_session import make_harness

    harness = make_harness()
    harness.bus._lelab_follower_guard = SimpleNamespace(protocol_status=lambda: {"quarantined": True})

    def fault_at_end(_count):
        harness.bus.disable_errors = [RuntimeError("protocol quarantined")] * 3
        harness.bus.torque_readback = RuntimeError("verification prohibited")

    harness.follower.send_hook = fault_at_end
    result = harness.worker.run()
    assert result.torque.outcome is TorqueOutcome.UNKNOWN
    assert result.torque.disable_errors
    assert result.torque.missing_motors == SO101_MOTOR_NAMES
