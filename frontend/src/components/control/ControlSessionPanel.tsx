import { AlertCircle, CheckCircle, Gamepad2, ShieldQuestion } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  SO101_MOTOR_NAMES,
  controlStatusError,
  controlStopReason,
  isTerminalControlState,
  type ControlStatus,
} from "@/lib/robotConfig";

interface ControlSessionPanelProps {
  status: ControlStatus | null;
  contractError?: string | null;
  compact?: boolean;
}

const yesNoUnknown = (value: boolean | null): string =>
  value === null ? "Unknown" : value ? "Yes" : "No";

const titleCase = (value: string): string =>
  value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

const stateColor = (status: ControlStatus): string => {
  if (status.state === "error") return "bg-red-600";
  if (status.state === "stopping") return "bg-orange-600";
  if (status.state === "running") return "bg-green-600";
  return "bg-slate-600";
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const torqueMessage = (
  status: ControlStatus,
  controllerMonitoringUnproven: boolean,
): string => {
  if (status.torque_outcome === "verified_off") {
    if (status.state === "error") {
      return "Torque is verified off across all six motors. That proves torque state only; this terminal error does not prove that every resource closed cleanly.";
    }
    if (controllerMonitoringUnproven) {
      return "Torque is verified off across all six motors. That proves torque state only; controller-reader teardown remains unproven.";
    }
    return "Torque is verified off across all six motors.";
  }
  if (status.torque_outcome === "failed") {
    return "Torque disable or readback failed. Do not assume the follower is safe to handle.";
  }
  if (status.torque_outcome === "unknown") {
    return "Torque state could not be verified. Do not assume the follower is safe to handle.";
  }
  return status.operation === "controller_check"
    ? "This controller-only operation did not instantiate, access, connect, calibrate, torque, or move robot hardware."
    : "No torque-disable attempt has been reported yet.";
};

const ControlSessionPanel = ({
  status,
  contractError = null,
  compact = false,
}: ControlSessionPanelProps) => {
  if (!status && !contractError) {
    return (
      <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-4 text-sm text-slate-400">
        Waiting for exact server-owned control status…
      </div>
    );
  }

  const stopReason = controlStopReason(status);
  const thermal = status?.thermal_snapshot ?? null;
  const controllerReady = status?.details.controller_ready;
  const controllerMonitoringActive =
    status?.details.controller_monitoring_active;
  const terminal = status ? isTerminalControlState(status.state) : false;
  const hasControllerEvidence = Boolean(
    status &&
      (status.operation === "stadia_teleoperation" ||
        status.operation === "stadia_recording" ||
        status.operation === "controller_check"),
  );
  const controllerMonitoringInactive = controllerMonitoringActive === false;
  const terminalControllerMonitoringUnproven = Boolean(
    terminal &&
      hasControllerEvidence &&
      controllerMonitoringActive !== false,
  );
  const lastObservedRaw = status?.details.controller_last_observed;
  const lastObserved = isObject(lastObservedRaw) ? lastObservedRaw : null;
  const lastObservedConnected =
    lastObserved &&
    (lastObserved.connected === null ||
      typeof lastObserved.connected === "boolean")
      ? lastObserved.connected
      : status?.controller_connected ?? null;
  const lastObservedRbHeld =
    lastObserved &&
    (lastObserved.rb_held === null ||
      typeof lastObserved.rb_held === "boolean")
      ? lastObserved.rb_held
      : status?.rb_held ?? null;
  const lastObservedNeutral =
    lastObserved &&
    (lastObserved.controls_neutral === null ||
      typeof lastObserved.controls_neutral === "boolean")
      ? lastObserved.controls_neutral
      : status?.controls_neutral ?? null;
  const lastObservedSampleAge =
    lastObserved &&
    typeof lastObserved.sample_age_s === "number" &&
    Number.isFinite(lastObserved.sample_age_s) &&
    lastObserved.sample_age_s >= 0
      ? lastObserved.sample_age_s
      : status?.sample_age_s ?? null;
  const displayedControllerSampleAge = controllerMonitoringInactive
    ? lastObservedSampleAge
    : status?.sample_age_s ?? null;
  const lastObservedMotion =
    lastObserved && typeof lastObserved.motion_state === "string"
      ? lastObserved.motion_state
      : status?.motion_state ?? "disarmed";
  const lastObservedError =
    lastObserved && typeof lastObserved.error === "string"
      ? lastObserved.error
      : null;
  const controllerError =
    controlStatusError(status) ??
    (controllerMonitoringInactive ? lastObservedError : null);
  const torqueOffCount = status
    ? SO101_MOTOR_NAMES.filter((motor) => status.torque.readback[motor] === false)
        .length
    : 0;
  const torqueUnknownCount = status
    ? SO101_MOTOR_NAMES.filter((motor) => status.torque.readback[motor] === null)
        .length
    : 0;

  return (
    <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-900/70 p-4 text-slate-100">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Server-owned control session
          </div>
          <div className="text-sm font-medium">
            {status ? titleCase(status.operation) : "Status unavailable"}
          </div>
        </div>
        {status && <Badge className={`${stateColor(status)} text-white`}>{status.state}</Badge>}
      </div>

      {contractError && (
        <Alert className="border-red-700 bg-red-950/60 text-red-100">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Status contract error:</strong> {contractError} Lease renewal is paused until an exact status is received.
          </AlertDescription>
        </Alert>
      )}

      {status && (
        <>
          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded bg-black/30 p-2">
              <div className="text-slate-500">Motion</div>
              <div>{status.motion_state}</div>
            </div>
            <div className="rounded bg-black/30 p-2">
              <div className="text-slate-500">Saturations</div>
              <div>{status.saturation_count}</div>
            </div>
            <div className="rounded bg-black/30 p-2">
              <div className="text-slate-500">Relative clips</div>
              <div>{status.relative_clipping_count}</div>
            </div>
            <div className="rounded bg-black/30 p-2">
              <div className="text-slate-500">Torque</div>
              <div>{status.torque_outcome}</div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 rounded border border-slate-800 bg-black/20 p-3 text-xs sm:grid-cols-3">
            <div>
              <span className="text-slate-500">Torque disable attempted:</span>{" "}
              {status.torque.disable_attempted ? "Yes" : "No"}
            </div>
            <div>
              <span className="text-slate-500">Readback supported:</span>{" "}
              {status.torque.verification_supported ? "Yes" : "No"}
            </div>
            <div>
              <span className="text-slate-500">Six-motor readback:</span>{" "}
              {torqueOffCount}/6 off
              {torqueUnknownCount > 0 ? ` · ${torqueUnknownCount} unknown` : ""}
            </div>
          </div>

          {hasControllerEvidence && (
            <div className="space-y-2 rounded border border-purple-900/70 bg-purple-950/20 p-3 text-xs">
              <div className="flex items-center gap-2 font-medium text-purple-200">
                <Gamepad2 className="h-4 w-4" /> Controller evidence
              </div>
              {controllerMonitoringInactive && (
                <div className="rounded border border-purple-900/60 bg-black/20 p-2 text-purple-200">
                  Controller monitoring has ended. Every value below is labeled and shown only as
                  last-observed evidence; none is a live controller reading.
                </div>
              )}
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed connected:"
                      : "Connected:"}
                  </span>{" "}
                  {yesNoUnknown(
                    controllerMonitoringInactive
                      ? lastObservedConnected
                      : status.controller_connected,
                  )}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed ready:"
                      : "Ready:"}
                  </span>{" "}
                  {typeof controllerReady === "boolean"
                    ? controllerReady
                      ? "Yes"
                      : "No"
                    : "Not reported"}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed RB held:"
                      : "RB held:"}
                  </span>{" "}
                  {yesNoUnknown(
                    controllerMonitoringInactive
                      ? lastObservedRbHeld
                      : status.rb_held,
                  )}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed release:"
                      : "Release observed:"}
                  </span>{" "}
                  {status.release_observed ? "Yes" : "No"}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed neutral:"
                      : "Neutral:"}
                  </span>{" "}
                  {yesNoUnknown(
                    controllerMonitoringInactive
                      ? lastObservedNeutral
                      : status.controls_neutral,
                  )}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed sample:"
                      : "Sample:"}
                  </span>{" "}
                  {status.sample_sequence === null ? "—" : `#${status.sample_sequence}`}
                  {displayedControllerSampleAge === null
                    ? ""
                    : ` · ${(displayedControllerSampleAge * 1000).toFixed(0)} ms`}
                </div>
                <div className="col-span-2">
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed identity:"
                      : "Identity:"}
                  </span>{" "}
                  {status.controller_product_name ?? "Not reported"}
                </div>
                <div className="col-span-2 break-all font-mono">
                  <span className="font-sans text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed GUID:"
                      : "GUID:"}
                  </span>{" "}
                  {status.controller_guid ?? "Not reported"}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed instance / generation:"
                      : "Instance / generation:"}
                  </span>{" "}
                  {status.controller_instance_id ?? "—"} / {status.controller_generation ?? "—"}
                </div>
                <div>
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed layout:"
                      : "Layout:"}
                  </span>{" "}
                  {status.controller_layout
                    ? `${status.controller_layout.axes}a · ${status.controller_layout.buttons}b · ${status.controller_layout.hats}h`
                    : "Not reported"}
                </div>
                <div className="col-span-2">
                  <span className="text-slate-500">
                    {controllerMonitoringInactive
                      ? "Last-observed motion:"
                      : "Controller motion:"}
                  </span>{" "}
                  {controllerMonitoringInactive
                    ? lastObservedMotion
                    : status.motion_state}
                </div>
              </div>
            </div>
          )}

          {terminalControllerMonitoringUnproven && (
            <Alert className="border-red-700 bg-red-950/60 text-red-100">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Controller teardown unproven:</strong>{" "}
                {controllerMonitoringActive === true
                  ? "The lifecycle is terminal, but controller monitoring is still marked active."
                  : "The terminal status did not prove that controller monitoring ended."}{" "}
                Do not infer that the reader or all related resources closed cleanly.
              </AlertDescription>
            </Alert>
          )}

          {!compact && status.joint_specs.length > 0 && (
            <div className="overflow-x-auto rounded border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-black/30 text-slate-400">
                  <tr>
                    <th className="px-2 py-2">Joint</th>
                    <th className="px-2 py-2">Unit</th>
                    <th className="px-2 py-2">Step / relative</th>
                    <th className="px-2 py-2">Calibrated bounds</th>
                  </tr>
                </thead>
                <tbody>
                  {status.joint_specs.map((spec) => (
                    <tr key={spec.action_key} className="border-t border-slate-800">
                      <td className="px-2 py-2 font-mono">{spec.action_key}</td>
                      <td className="px-2 py-2">{spec.unit}</td>
                      <td className="px-2 py-2">
                        {spec.max_step_per_tick} / {spec.max_relative_target}
                      </td>
                      <td className="px-2 py-2">
                        {spec.calibrated_min}…{spec.calibrated_max}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {compact && status.joint_specs.length > 0 && (
            <div className="rounded border border-slate-800 bg-black/20 p-3 text-xs">
              <div className="mb-1 font-medium text-slate-200">Joint units and limits</div>
              <div className="text-slate-400">
                Five arm joints: degrees · gripper: percentage points. Per-tick / relative limits:{" "}
                {status.joint_specs
                  .map(
                    (spec) =>
                      `${spec.action_key} ${spec.max_step_per_tick}/${spec.max_relative_target}`,
                  )
                  .join(" · ")}
              </div>
            </div>
          )}

          {thermal && (
            <div className="space-y-2 rounded border border-slate-800 bg-black/20 p-3 text-xs">
              <div className="font-medium text-slate-200">Thermal evidence (°C)</div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {SO101_MOTOR_NAMES.map((motor) => (
                  <div key={motor} className="rounded bg-black/30 p-2">
                    <div className="text-slate-500">{motor}</div>
                    <div>
                      {thermal.temperatures[motor] === null
                        ? "Unavailable"
                        : thermal.temperatures[motor].toFixed(1)}{" "}
                      current ·{" "}
                      {thermal.confirmed_peaks[motor] === null
                        ? "Unavailable"
                        : thermal.confirmed_peaks[motor].toFixed(1)}{" "}
                      confirmed peak
                    </div>
                    {(thermal.spike_counts[motor] > 0 ||
                      thermal.invalid_sample_counts[motor] > 0) && (
                      <div className="text-amber-300">
                        {thermal.spike_counts[motor]} spikes · {thermal.invalid_sample_counts[motor]} invalid
                        {thermal.invalid_sample_counts[motor] > 0 && (
                          <>
                            {" "}· last invalid {thermal.last_invalid_values[motor] === null
                              ? "non-finite/unavailable"
                              : `${thermal.last_invalid_values[motor].toFixed(1)}°C`}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              {thermal.warning_motors.length > 0 && (
                <div className="text-amber-300">
                  Warning motors: {thermal.warning_motors.join(", ")}
                </div>
              )}
            </div>
          )}

          {controllerError && (
            <Alert className="border-red-700 bg-red-950/60 text-red-100">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>
                  {controllerMonitoringInactive
                    ? "Last-observed controller error:"
                    : "Controller error:"}
                </strong>{" "}
                {controllerError}
              </AlertDescription>
            </Alert>
          )}

          {stopReason && (
            <Alert className="border-slate-700 bg-slate-950/70 text-slate-200">
              <ShieldQuestion className="h-4 w-4" />
              <AlertDescription>
                <strong>Stop reason:</strong> {stopReason}
              </AlertDescription>
            </Alert>
          )}

          {thermal?.stop_reason && thermal.stop_reason !== stopReason && (
            <Alert className="border-red-700 bg-red-950/60 text-red-100">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Thermal stop:</strong> {thermal.stop_reason}
              </AlertDescription>
            </Alert>
          )}

          {Object.keys(status.torque.disable_errors).length > 0 && (
            <Alert className="border-red-700 bg-red-950/60 text-red-100">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <strong>Torque-disable errors:</strong>{" "}
                {Object.entries(status.torque.disable_errors)
                  .map(([motor, message]) => `${motor}: ${message}`)
                  .join(" · ")}
              </AlertDescription>
            </Alert>
          )}

          {terminal && (
            <Alert
              className={
                status.torque_outcome === "verified_off" &&
                status.state !== "error" &&
                !terminalControllerMonitoringUnproven
                  ? "border-green-700 bg-green-950/50 text-green-100"
                  : status.state === "error" ||
                      status.torque_outcome === "failed"
                    ? "border-red-700 bg-red-950/60 text-red-100"
                    : "border-amber-800 bg-amber-950/40 text-amber-100"
              }
            >
              {status.torque_outcome === "verified_off" &&
              status.state !== "error" &&
              !terminalControllerMonitoringUnproven ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                <ShieldQuestion className="h-4 w-4" />
              )}
              <AlertDescription>
                {torqueMessage(status, terminalControllerMonitoringUnproven)}
              </AlertDescription>
            </Alert>
          )}

          <div className="break-all font-mono text-[10px] text-slate-600">
            session {status.session_id} · revision {status.revision}
          </div>
        </>
      )}
    </div>
  );
};

export default ControlSessionPanel;
