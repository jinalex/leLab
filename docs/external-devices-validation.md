# External-device integration validation

Baseline: fork main `f74f854` (merged Stadia changes). Python 3.12.12,
LeRobot 0.6.0. No controller mapping or robot motion limits changed.

## Software

- Focused generic camera/wrapper, recording, rollout, robot-record, and legacy
  leader tests: **87 passed**.
- Full suite with offline Hub mode, writable dataset cache, and localhost
  sockets permitted: **689 passed, 12 failed**. All twelve failures reproduce
  on the unchanged baseline: eleven Stadia shoulder-direction expectations,
  and one TorchCodec/FFmpeg shared-library failure. These are not a clean
  full-suite pass and remain separate follow-up work.
- TypeScript type-check and production frontend build pass.
- Python Ruff and `git diff --check` pass. Frontend lint reports no errors;
  existing hook warnings remain in Landing.
- Real installed camera plugin discovery and decoding succeed without directly
  importing that plugin from LeLab.
- A real child process launched through LeLab's generic command builder passed
  the native serial handshake and ten raw position-read cycles, confirming the
  subprocess retained the selected-PTY compatibility layer.
- Browser smoke check: saved bridge robot appears as Stadia Ready, preserves
  its calibration, and displays the manually configured PTY port. No teleop
  or calibration start request was issued.
- Remote camera settings remain visible/editable with local browser-camera
  access disabled; no browser-camera permission is needed to add a plugin.

## Read-only physical check

Pi 2 over SSH/Tailscale, serial follower plus 640x480 remote USB camera,
explicit Feetech packet timeout floor 1000 ms:

- Native LeRobot handshake passed; SDK and pySerial report 1,000,000 baud.
- 120 six-motor raw position reads, zero retries: mean 51.16 ms, p95 148.76 ms,
  max 573.08 ms.
- 60 fresh RGB frames: mean read 103.28 ms, p95 256.2 ms, max 635.61 ms.
- Camera cold start: 10.166 seconds.
- The launcher stopped its Pi agent and attachment after the probe; a subsequent
  clean start succeeded.

This proves transport/read-only access, not motion, stop/dropout behavior,
recorded-dataset quality, or policy inference. The first teleop run remains
user-supervised. Network jitter means a configured 30 Hz control loop is not a
guaranteed delivered rate. The bridge-specific host configuration and personal
launcher are kept outside this generic LeLab PR.
