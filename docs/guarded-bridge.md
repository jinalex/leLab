# Guarded SO-101 bridge release

This is a software remediation for the September 6 temperature-as-position
incident. It is **not deployed or physically validated**. Ethernet is not a
prerequisite. Keep the existing direct-USB robot record and calibration.

## Scope and setup

The first guarded release supports **Stadia teleoperation and Stadia recording**
on selected bridge ports. Their SO-101 bus receives pre-normalization feedback
checks, calibrated final-goal checks, and a current controller authorization
check immediately before a goal write. Recording still requires its own physical
and state/action/image timing validation before collecting training data.

Native leader teleop/recording, calibration, and policy inference on those ports
are deliberately rejected before the follower connects. Their constructors run
outside the supported command-owner context. This also applies to LeLab's module
subprocesses. Direct USB and other unselected ports retain their normal paths.
Calibration should be completed using the existing direct connection.

Use reviewed, compatible Tailbridge and LeLab revisions in the same Python
environment. The implementation is pinned to LeRobot 0.6.0 and requires
Tailbridge's selected-port Feetech SDK reply-integrity wrapper. No installed
vendor files are edited. A missing guard fails before device open.

For example, after separately creating the PTY and camera/serial tunnels:

```sh
export LELAB_GUARDED_FOLLOWER_PORTS='["/tmp/example-follower"]'
export LELAB_PYTHON_MODULE_WRAPPER='["-m","tailbridge","run-python","--pty","/tmp/example-follower","--feetech-timeout-ms","1000","--module"]'
python -m tailbridge run-python --pty /tmp/example-follower \
  --feetech-timeout-ms 1000 --module lelab.scripts.lelab
```

These are trusted operator environment variables, never HTTP inputs. Use the
same absolute PTY path in the wrapper and saved robot record. Production mode
is required; development reload would lose process-local SDK patches. This
example does not establish network tunnels or start teleoperation.

## Write contract

- Read decoded raw position before LeRobot normalization can clip it. Require
  integer encoder values within both native and loaded calibration limits.
- Validate the requested six-joint action and the **final** goal after LeRobot's
  relative limiting. Reject invalid measurements/goals, rather than clamping
  them into apparent validity. Keep the existing 5-unit per-joint relative cap
  and full calibrated travel; no smaller startup envelope is introduced.
- Bound final-goal continuity against both the requested goal and the last
  successfully issued goal, using the same relative cap. In-range but displaced
  feedback cannot make the relative limiter relocate the command arbitrarily.
  Issued goals are not claimed to be physically applied measurements.
- Require feedback from the current `send_action` call. No automatic read/write
  retries are issued by the guarded bus methods.
- Reject a command spending over 250 ms inside follower I/O. Also check the
  original controller sample against the existing 150 ms freshness limit, and
  re-read current RB, connection identity, stop, and hold state immediately
  before serialization/write. These are pre-write rejection bounds, not a
  promise of achievable control frequency or physical stop latency.
- Startup goal writes must exactly seed validated raw feedback, with current
  neutral/RB-up authorization before every seed and torque-enable write.
- A cancelled teleop step does not update the accepted integrator target or
  count as a sent command. Normal RB release can hold without reconnecting.
  A mid-transaction recording cancellation takes its existing send-failure /
  episode-invalidation path, so an unsent action is not recorded as applied.
- Returned actions are checked against calibrated bounds before adoption too.
  This is secondary protection; the bus boundary is the pre-write check.

Protocol corruption, timeout, or an expired follower transaction stops the
session. Bounded request/raw/normalized/final-goal context is added to the
Tailbridge diagnostic history; the LeLab guard also emits its bounded recent
events on a local validation fault, without logging every normal frame.

After protocol quarantine, further serial operations—including torque-off
verification—may be blocked. The existing cleanup must report **unknown** torque
state unless it has actual validated evidence. Do not interpret an error screen
as proof the physical motors are unpowered.

## Limits and remaining hardware gates

Raw Feetech replies have no register or transaction identifier. Correct-length,
correct-ID stale data can remain indistinguishable before an observable fault.
Native/calibration bounds reject impossible feedback, but do not prove physical
feedback continuity for a plausible in-range jump; a continuity envelope needs
measured actuator behavior, not an invented smaller travel range.

Likewise, a Mac-side pre-write deadline cannot expire bytes already handed to
TCP but delayed before reaching the Pi. Separate SSH connections remove one
source of camera/serial head-of-line blocking, not Wi-Fi airtime or CPU sharing.
This release does **not** establish arbitrary-WAN stale-command immunity or
unattended operation. If these raw-stream limits remain material, use the
planned Pi-local transaction/goal-expiry module before claiming that guarantee.

Before supervised motion:

1. Review and install pinned compatible revisions together; do not silently
   update the currently running app or bridge.
2. With no motion owner active, measure mixed position/temperature reads with
   camera off, capture-only, and preview-on. Run at least 30 minutes camera-on
   plus repeats; retain latency tails, failures, CPU, Wi-Fi and camera evidence.
3. Compare candidate camera budgets of 5–10 FPS and control rates of 10/15/20 Hz.
   Do not choose a permanent rate or change degrees-per-second semantics until
   measured. Never integrate catch-up movement after a stall.
4. Obtain separate approval for a short supervised physical test of movement,
   RB release, stop, and recovery. No live fault injection against a moving arm
   without its own supervised plan. Dataset/inference validation is separate.

## Software validation

`tests/test_follower_guard.py` uses actual LeRobot 0.6.0 `send_action`,
normalization and bus serialization with fake transport functions. No device is
opened. Tests cover incident-style corrupt raw values, gripper clamp masking,
invalid final goals, valid relative clipping, blocking-read expiry/cancellation,
startup seeding, returned-action poisoning, selected-port gates and subprocess
command propagation.

With the companion Tailbridge source available, the complete suite has 731 passes and 12
unchanged baseline failures: 11 existing shoulder-direction expectations and
one TorchCodec/FFmpeg video-loader failure. The unrelated controller mapping is
not modified to hide them. The unchanged baseline was rerun with the same
environment: 689 passes and the same 12 failures. All 42 targeted bridge
regressions pass, including real SDK + LeRobot integration and a real module
subprocess constructor gate. Without the optional Tailbridge dependency, the
two combined SDK integration cases are skipped explicitly.
