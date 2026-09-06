# External serial devices and installed camera plugins

LeLab remains transport-agnostic. A serial adapter supplies a local port path;
an installed LeRobot camera plugin supplies frames. SSH/Tailscale and device
access policy belong to the bridge, not to LeLab. No new robot watchdog is added.

## Serial

Use Calibration's existing manual port field for the adapter's stable PTY path.
Keep the existing calibration for the same physical arm. A PTY is not a USB
device: it will not necessarily appear in the detected-USB dropdown.

On macOS, a high-baud Feetech consumer may need the bridge's explicit,
process-local PTY compatibility launcher. Start LeLab **through that launcher**,
in production mode. Starting plain `lelab` bypasses the compatibility layer.
The path configured in LeLab must exactly match the launcher's selected path.

## Camera

Install the camera plugin into the Python environment running LeLab (and its
inference subprocess). LeLab discovers installed LeRobot plugins; saved records
cannot request arbitrary Python module imports. Plugins are trusted executable
code, so only install packages you trust.

On Calibration, open **Add installed camera plugin (advanced)**. Set a unique
camera name, its registered backend type, and its plugin-specific JSON parameters.
Set resolution/FPS in the camera's Configuration section. Common fields `type`,
`width`, `height`, and `fps` cannot be overridden inside `parameters`.

Example saved camera for an externally configured Tailbridge tunnel:

```json
{
  "id": "remote-front",
  "name": "front",
  "type": "tailbridge",
  "width": 640,
  "height": 480,
  "fps": 10,
  "parameters": {
    "host": "127.0.0.1",
    "port": 17447,
    "resource_id": "front_camera",
    "timeout_s": 5,
    "startup_timeout_s": 30,
    "max_frame_age_ms": 1000
  }
}
```

These settings must match the camera stream configured on the bridge; LeLab
does not reconfigure a remote USB camera. Put authentication secrets in the
plugin's supported environment-variable mechanism, not in saved JSON.

Local OpenCV cameras retain their browser preview. Plugin cameras are opened
and disconnected by the teleop worker. Turn on Cameras in teleoperation to read
that worker's existing camera, without taking a second device claim. The preview
does not open hardware when no matching session is running, and errors clear
the displayed image. Preview is not a separate safety channel.

Recording uses the same saved camera configuration. Inference binds policy
camera names to saved camera IDs, so remote cameras do not need local indices.
Legacy local-index inference requests remain supported. Dimensions must match
the policy. Stadia recording's existing lack of live camera preview is unchanged.

## Inference subprocess wrapper

If your adapter modifies the current Python process, set the trusted operator
environment variable `LELAB_PYTHON_MODULE_WRAPPER` before starting LeLab.
It is a JSON array of arguments after the Python executable. The contract is:

```text
python <wrapper arguments> MODULE -- MODULE_ARGUMENTS
```

For example, after creating `/tmp/example-follower`:

```sh
export LELAB_PYTHON_MODULE_WRAPPER='["-m","tailbridge","run-python","--pty","/tmp/example-follower","--feetech-timeout-ms","1000","--module"]'
python -m tailbridge run-python --pty /tmp/example-follower \
  --feetech-timeout-ms 1000 --module lelab.scripts.lelab
```

LeLab does not accept wrapper commands over HTTP and never invokes a shell for
the wrapper. Without this variable, inference uses its original `python -m`
command. Development reload mode is rejected when the variable is set because
reload children do not inherit process-local patches.

## First supervised test

Start the bridge, verify native read-only bus/camera access, then start LeLab
through the wrapper. Select a separate bridge robot record to preserve your
direct-USB configuration. Start teleop only with the workspace clear and a
reachable power cutoff; existing Stadia neutral/RB controls are unchanged.
Begin slowly. This configuration does not promise 30 Hz across a WAN, nor does
a successful software/read-only test validate physical motion or dropout cleanup.
Do not use the first remote test as unattended operation or a training-quality
demonstration. Stop teleop before closing the bridge launcher.
