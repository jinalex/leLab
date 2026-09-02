export type TeleoperatorType = "leader_arm" | "stadia";

export const ROBOT_OPERATIONS = [
  "follower_calibration",
  "leader_calibration",
  "leader_teleoperation",
  "stadia_teleoperation",
  "leader_recording",
  "stadia_recording",
  "inference",
  "replay",
  "controller_check",
] as const;

export type RobotOperation = (typeof ROBOT_OPERATIONS)[number];

export interface RobotDeviceRecord {
  port: string;
  calibration: string;
}

export interface StadiaConfig {
  guid: string | null;
  deadzone: number;
  max_step_per_tick: number;
  arm_startup_travel_degrees: number;
  gripper_startup_travel_percentage_points: number;
}

export interface RobotCameraRecord {
  id: string;
  name: string;
  type: "opencv";
  camera_index?: number;
  device_id: string;
  width: number;
  height: number;
  fps?: number;
  fourcc?: string;
  backend?: string;
}

export interface ReadinessIssue {
  code: string;
  field: string | null;
  message: string;
}

export interface ReadinessResult {
  operation: RobotOperation;
  ready: boolean;
  issues: ReadinessIssue[];
}

export type RobotReadiness = Record<RobotOperation, ReadinessResult>;

/** Canonical browser representation of the backend RobotRecordV2. */
export interface RobotRecord {
  schema_version: 2;
  name: string;
  teleoperator_type: TeleoperatorType;
  follower: RobotDeviceRecord;
  leader: RobotDeviceRecord | null;
  stadia: StadiaConfig;
  cameras: RobotCameraRecord[];
  readiness: RobotReadiness;
  is_clean: boolean;
  // Read-only projections for screens that have not yet moved to nested data.
  leader_port: string;
  follower_port: string;
  leader_config: string;
  follower_config: string;
}

export type ControlState =
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "error";

export type TorqueOutcome =
  | "not_attempted"
  | "verified_off"
  | "failed"
  | "unknown";

export type MotionState = "disarmed" | "hold" | "enabled";
export type JointUnit = "degrees" | "gripper_percentage_points";

export const SO101_ACTION_KEYS = [
  "shoulder_pan.pos",
  "shoulder_lift.pos",
  "elbow_flex.pos",
  "wrist_flex.pos",
  "wrist_roll.pos",
  "gripper.pos",
] as const;

export const SO101_MOTOR_NAMES = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
] as const;

export type So101ActionKey = (typeof SO101_ACTION_KEYS)[number];
export type So101MotorName = (typeof SO101_MOTOR_NAMES)[number];

export interface ControllerLayout {
  axes: number;
  buttons: number;
  hats: number;
}

export interface JointLimit {
  max_step_per_tick: number;
  max_relative_target: number;
  startup_min: number;
  startup_max: number;
  calibrated_min: number;
  calibrated_max: number;
}

export interface JointSpec extends JointLimit {
  action_key: So101ActionKey;
  unit: JointUnit;
}

export interface ThermalStatus {
  temperatures: Record<So101MotorName, number | null>;
  reported_peaks: Record<So101MotorName, number | null>;
  confirmed_peaks: Record<So101MotorName, number | null>;
  spike_counts: Record<So101MotorName, number>;
  invalid_sample_counts: Record<So101MotorName, number>;
  last_invalid_values: Record<So101MotorName, number | null>;
  warning_motors: So101MotorName[];
  stop_reason: string | null;
}

export interface TorqueEvidence {
  outcome: TorqueOutcome;
  disable_attempted: boolean;
  verification_supported: boolean;
  readback: Record<So101MotorName, boolean | null>;
  missing_motors: So101MotorName[];
  invalid_motors: So101MotorName[];
  unexpected_motors: string[];
  disable_errors: Record<string, string>;
}

export interface ControlStatus {
  session_id: string;
  state: ControlState;
  operation: RobotOperation;
  resource_keys: ["control", RobotOperation];
  teleoperator_type: TeleoperatorType | null;
  claimed_at_utc: string;
  updated_at_utc: string;
  lease_deadline_monotonic: number;
  lease_ttl_s: number;
  lease_renew_interval_s: number;
  controller_connected: boolean | null;
  controller_error: string | null;
  controller_product_name: string | null;
  controller_guid: string | null;
  controller_instance_id: number | null;
  controller_generation: number | null;
  controller_layout: ControllerLayout | null;
  sample_sequence: number | null;
  sample_age_s: number | null;
  rb_held: boolean | null;
  release_observed: boolean;
  controls_neutral: boolean | null;
  motion_state: MotionState;
  joint_units: Partial<Record<So101ActionKey, JointUnit>>;
  joint_limits: Partial<Record<So101ActionKey, JointLimit>>;
  joint_specs: JointSpec[];
  saturation_count: number;
  relative_clipping_count: number;
  stop_reason: string | null;
  hold_requested: boolean;
  stop_requested: boolean;
  torque: TorqueEvidence;
  torque_outcome: TorqueOutcome;
  thermal_snapshot: ThermalStatus | null;
  teardown_completed_at_utc: string | null;
  revision: number;
  details: Record<string, unknown>;
}

export interface ControlEnvelopeRequirements {
  expectedSessionId?: string;
  expectedOperation: RobotOperation;
  expectedTeleoperatorType: TeleoperatorType | null;
  requireSuccess?: boolean;
  requireTopLevelSessionId?: boolean;
  requireStatusKey?: "status" | "control_status";
}

const CONTROL_STATES = new Set<ControlState>([
  "starting",
  "running",
  "stopping",
  "stopped",
  "error",
]);
const TORQUE_OUTCOMES = new Set<TorqueOutcome>([
  "not_attempted",
  "verified_off",
  "failed",
  "unknown",
]);
const MOTION_STATES = new Set<MotionState>(["disarmed", "hold", "enabled"]);
const JOINT_UNITS = new Set<JointUnit>([
  "degrees",
  "gripper_percentage_points",
]);
const OPERATIONS = new Set<RobotOperation>(ROBOT_OPERATIONS);
const ACTION_KEYS = new Set<string>(SO101_ACTION_KEYS);
const MOTOR_NAMES = new Set<string>(SO101_MOTOR_NAMES);
const CAMERA_BACKENDS = new Set([
  "ANY",
  "V4L2",
  "DSHOW",
  "PVAPI",
  "ANDROID",
  "AVFOUNDATION",
  "MSMF",
]);

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const exactKeys = (
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean => {
  const allowed = new Set([...required, ...optional]);
  return (
    required.every((key) => Object.prototype.hasOwnProperty.call(value, key)) &&
    Object.keys(value).every((key) => allowed.has(key))
  );
};

const nonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

const exactNonEmptyString = (value: unknown): value is string =>
  nonEmptyString(value) && value.trim() === value;

const nullableString = (value: unknown): value is string | null =>
  value === null || nonEmptyString(value);

const finiteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const nonnegativeInteger = (value: unknown): value is number =>
  Number.isInteger(value) && typeof value === "number" && value >= 0;

const nullableNonnegativeInteger = (value: unknown): value is number | null =>
  value === null || nonnegativeInteger(value);

const nullableNonnegativeNumber = (value: unknown): value is number | null =>
  value === null || (finiteNumber(value) && value >= 0);

const nullableBoolean = (value: unknown): value is boolean | null =>
  value === null || typeof value === "boolean";

const validCalibrationName = (value: string): boolean =>
  value === "" ||
  (value.trim() === value &&
    value.endsWith(".json") &&
    !value.includes("/") &&
    !value.includes("\\") &&
    !value.includes(".."));

const validRobotName = (value: unknown): value is string =>
  nonEmptyString(value) &&
  value.trim() === value &&
  !value.includes("/") &&
  !value.includes("\\") &&
  !value.includes("..");

const parseDevice = (value: unknown): RobotDeviceRecord | null => {
  if (!isObject(value) || !exactKeys(value, ["port", "calibration"])) return null;
  if (
    typeof value.port !== "string" ||
    value.port.includes("\0") ||
    typeof value.calibration !== "string" ||
    !validCalibrationName(value.calibration)
  ) {
    return null;
  }
  return { port: value.port, calibration: value.calibration };
};

const parseStadia = (value: unknown): StadiaConfig | null => {
  if (
    !isObject(value) ||
    !exactKeys(value, [
      "guid",
      "deadzone",
      "max_step_per_tick",
      "arm_startup_travel_degrees",
      "gripper_startup_travel_percentage_points",
    ])
  ) {
    return null;
  }
  const guid = value.guid;
  if (
    !(guid === null ||
      (nonEmptyString(guid) && guid.trim() === guid && !guid.includes("\0"))) ||
    !finiteNumber(value.deadzone) ||
    value.deadzone < 0 ||
    value.deadzone >= 1 ||
    !finiteNumber(value.max_step_per_tick) ||
    value.max_step_per_tick <= 0 ||
    value.max_step_per_tick > 0.35 ||
    !finiteNumber(value.arm_startup_travel_degrees) ||
    value.arm_startup_travel_degrees <= 0 ||
    value.arm_startup_travel_degrees > 45 ||
    !finiteNumber(value.gripper_startup_travel_percentage_points) ||
    value.gripper_startup_travel_percentage_points <= 0 ||
    value.gripper_startup_travel_percentage_points > 45
  ) {
    return null;
  }
  return {
    guid,
    deadzone: value.deadzone,
    max_step_per_tick: value.max_step_per_tick,
    arm_startup_travel_degrees: value.arm_startup_travel_degrees,
    gripper_startup_travel_percentage_points:
      value.gripper_startup_travel_percentage_points,
  };
};

const parseCamera = (value: unknown): RobotCameraRecord | null => {
  const required = ["id", "name", "type", "device_id", "width", "height"];
  const optional = ["camera_index", "fps", "fourcc", "backend"];
  if (!isObject(value) || !exactKeys(value, required, optional)) return null;
  if (
    !nonEmptyString(value.id) ||
    value.id.trim() !== value.id ||
    !nonEmptyString(value.name) ||
    value.name.trim() !== value.name ||
    value.type !== "opencv" ||
    typeof value.device_id !== "string" ||
    value.device_id.includes("\0") ||
    !nonnegativeInteger(value.camera_index ?? 0) ||
    !nonnegativeInteger(value.width) ||
    value.width < 1 ||
    value.width > 8192 ||
    !nonnegativeInteger(value.height) ||
    value.height < 1 ||
    value.height > 8192 ||
    !(value.fps === undefined || value.fps === null ||
      (nonnegativeInteger(value.fps) && value.fps >= 1 && value.fps <= 240)) ||
    !(value.fourcc === undefined || value.fourcc === null ||
      (typeof value.fourcc === "string" && value.fourcc.length === 4)) ||
    !(value.backend === undefined || value.backend === null ||
      (typeof value.backend === "string" && CAMERA_BACKENDS.has(value.backend)))
  ) {
    return null;
  }
  return {
    id: value.id,
    name: value.name,
    type: "opencv",
    camera_index: value.camera_index as number | undefined,
    device_id: value.device_id,
    width: value.width,
    height: value.height,
    fps: value.fps == null ? undefined : value.fps,
    fourcc: value.fourcc == null ? undefined : value.fourcc,
    backend: value.backend == null ? undefined : value.backend,
  };
};

const parseCameras = (value: unknown): RobotCameraRecord[] | null => {
  if (!Array.isArray(value)) return null;
  const cameras: RobotCameraRecord[] = [];
  const ids = new Set<string>();
  const names = new Set<string>();
  for (const raw of value) {
    const camera = parseCamera(raw);
    if (!camera || ids.has(camera.id) || names.has(camera.name)) return null;
    ids.add(camera.id);
    names.add(camera.name);
    cameras.push(camera);
  }
  return cameras;
};

const unavailable = (
  operation: RobotOperation,
  message = "Readiness is unavailable. Refresh the saved robot configuration.",
): ReadinessResult => ({
  operation,
  ready: false,
  issues: [{ code: "readiness_unavailable", field: null, message }],
});

const parseReadiness = (value: unknown): RobotReadiness | null => {
  if (!isObject(value) || !exactKeys(value, ROBOT_OPERATIONS)) return null;
  const parsed = {} as RobotReadiness;
  for (const operation of ROBOT_OPERATIONS) {
    const raw = value[operation];
    if (!isObject(raw) || !exactKeys(raw, ["operation", "ready", "issues"])) {
      return null;
    }
    if (raw.operation !== operation || typeof raw.ready !== "boolean" || !Array.isArray(raw.issues)) {
      return null;
    }
    const issues: ReadinessIssue[] = [];
    for (const issue of raw.issues) {
      if (
        !isObject(issue) ||
        !exactKeys(issue, ["code", "field", "message"]) ||
        !nonEmptyString(issue.code) ||
        !(issue.field === null || typeof issue.field === "string") ||
        !nonEmptyString(issue.message)
      ) {
        return null;
      }
      issues.push({ code: issue.code, field: issue.field, message: issue.message });
    }
    if (raw.ready !== (issues.length === 0)) return null;
    parsed[operation] = { operation, ready: raw.ready, issues };
  }
  return parsed;
};

export const normalizeRobotRecord = (value: unknown): RobotRecord | null => {
  if (
    !isObject(value) ||
    value.schema_version !== 2 ||
    !validRobotName(value.name) ||
    !exactKeys(value, [
      "schema_version",
      "name",
      "teleoperator_type",
      "follower",
      "leader",
      "stadia",
      "cameras",
      "readiness",
      "is_clean",
    ]) ||
    (value.teleoperator_type !== "leader_arm" &&
      value.teleoperator_type !== "stadia") ||
    typeof value.is_clean !== "boolean"
  ) {
    return null;
  }
  const follower = parseDevice(value.follower);
  const leader = value.leader === null ? null : parseDevice(value.leader);
  const stadia = parseStadia(value.stadia);
  const cameras = parseCameras(value.cameras);
  const readiness = parseReadiness(value.readiness);
  if (
    !follower ||
    (value.leader !== null && !leader) ||
    !stadia ||
    !cameras ||
    !readiness
  ) {
    return null;
  }
  const mode = value.teleoperator_type;
  const cleanOperation =
    mode === "stadia" ? "stadia_teleoperation" : "leader_teleoperation";
  if (value.is_clean !== readiness[cleanOperation].ready) return null;

  return {
    schema_version: 2,
    name: value.name,
    teleoperator_type: mode,
    follower,
    leader,
    stadia,
    cameras,
    readiness,
    is_clean: value.is_clean,
    leader_port: leader?.port ?? "",
    follower_port: follower.port,
    leader_config: leader?.calibration ?? "",
    follower_config: follower.calibration,
  };
};

export const readinessFor = (
  record: RobotRecord,
  operation: RobotOperation,
): ReadinessResult => record.readiness[operation] ?? unavailable(operation);

export const teleoperationOperation = (record: RobotRecord): RobotOperation =>
  record.teleoperator_type === "stadia"
    ? "stadia_teleoperation"
    : "leader_teleoperation";

export const recordingOperation = (record: RobotRecord): RobotOperation =>
  record.teleoperator_type === "stadia"
    ? "stadia_recording"
    : "leader_recording";

const parseMotorNumberRecord = (
  value: unknown,
  integers = false,
): Record<So101MotorName, number> | null => {
  if (!isObject(value) || !exactKeys(value, SO101_MOTOR_NAMES)) return null;
  const parsed = {} as Record<So101MotorName, number>;
  for (const motor of SO101_MOTOR_NAMES) {
    const entry = value[motor];
    if (!(integers ? nonnegativeInteger(entry) : finiteNumber(entry))) return null;
    parsed[motor] = entry as number;
  }
  return parsed;
};

const parseMotorBooleanRecord = (
  value: unknown,
): Record<So101MotorName, boolean | null> | null => {
  if (!isObject(value) || !exactKeys(value, SO101_MOTOR_NAMES)) return null;
  const parsed = {} as Record<So101MotorName, boolean | null>;
  for (const motor of SO101_MOTOR_NAMES) {
    const entry = value[motor];
    if (!nullableBoolean(entry)) return null;
    parsed[motor] = entry;
  }
  return parsed;
};

const parseMotorNullableNumberRecord = (
  value: unknown,
): Record<So101MotorName, number | null> | null => {
  if (!isObject(value) || !exactKeys(value, SO101_MOTOR_NAMES)) return null;
  const parsed = {} as Record<So101MotorName, number | null>;
  for (const motor of SO101_MOTOR_NAMES) {
    const entry = value[motor];
    if (entry !== null && !finiteNumber(entry)) return null;
    parsed[motor] = entry;
  }
  return parsed;
};

const parseStringArray = (value: unknown): string[] | null => {
  if (!Array.isArray(value) || !value.every((entry) => nonEmptyString(entry))) return null;
  if (new Set(value).size !== value.length) return null;
  return [...value];
};

const parseMotorArray = (value: unknown): So101MotorName[] | null => {
  const values = parseStringArray(value);
  if (!values || values.some((entry) => !MOTOR_NAMES.has(entry))) return null;
  if (new Set(values).size !== values.length) return null;
  return values as So101MotorName[];
};

const parseThermalStatus = (value: unknown): ThermalStatus | null => {
  if (value === null) return null;
  if (
    !isObject(value) ||
    !exactKeys(value, [
      "temperatures",
      "reported_peaks",
      "confirmed_peaks",
      "spike_counts",
      "invalid_sample_counts",
      "last_invalid_values",
      "warning_motors",
      "stop_reason",
    ])
  ) {
    return null;
  }
  const temperatures = parseMotorNullableNumberRecord(value.temperatures);
  const reportedPeaks = parseMotorNullableNumberRecord(value.reported_peaks);
  const confirmedPeaks = parseMotorNullableNumberRecord(value.confirmed_peaks);
  const spikeCounts = parseMotorNumberRecord(value.spike_counts, true);
  const invalidCounts = parseMotorNumberRecord(value.invalid_sample_counts, true);
  const lastInvalidValues = parseMotorNullableNumberRecord(value.last_invalid_values);
  const warnings = parseMotorArray(value.warning_motors);
  if (
    !temperatures ||
    !reportedPeaks ||
    !confirmedPeaks ||
    !spikeCounts ||
    !invalidCounts ||
    !lastInvalidValues ||
    !warnings ||
    !nullableString(value.stop_reason)
  ) {
    return null;
  }
  return {
    temperatures,
    reported_peaks: reportedPeaks,
    confirmed_peaks: confirmedPeaks,
    spike_counts: spikeCounts,
    invalid_sample_counts: invalidCounts,
    last_invalid_values: lastInvalidValues,
    warning_motors: warnings,
    stop_reason: value.stop_reason,
  };
};

const parseTorque = (value: unknown): TorqueEvidence | null => {
  if (
    !isObject(value) ||
    !exactKeys(value, [
      "outcome",
      "disable_attempted",
      "verification_supported",
      "readback",
      "missing_motors",
      "invalid_motors",
      "unexpected_motors",
      "disable_errors",
    ]) ||
    typeof value.outcome !== "string" ||
    !TORQUE_OUTCOMES.has(value.outcome as TorqueOutcome) ||
    typeof value.disable_attempted !== "boolean" ||
    typeof value.verification_supported !== "boolean"
  ) {
    return null;
  }
  const readback = parseMotorBooleanRecord(value.readback);
  const missing = parseMotorArray(value.missing_motors);
  const invalid = parseMotorArray(value.invalid_motors);
  const unexpected = parseStringArray(value.unexpected_motors);
  if (!readback || !missing || !invalid || !unexpected || !isObject(value.disable_errors)) {
    return null;
  }
  const disableErrors: Record<string, string> = {};
  for (const [key, message] of Object.entries(value.disable_errors)) {
    if (!nonEmptyString(key) || !nonEmptyString(message)) return null;
    disableErrors[key] = message;
  }
  const unknownMotors = SO101_MOTOR_NAMES.filter(
    (motor) => readback[motor] === null,
  );
  const unexplainedUnknowns = new Set([...missing, ...invalid]);
  if (
    missing.some((motor) => invalid.includes(motor)) ||
    unexplainedUnknowns.size !== unknownMotors.length ||
    unknownMotors.some((motor) => !unexplainedUnknowns.has(motor))
  ) {
    return null;
  }
  const noTorqueValues =
    missing.length === SO101_MOTOR_NAMES.length &&
    invalid.length === 0 &&
    unexpected.length === 0;
  const completeOff =
    value.verification_supported &&
    missing.length === 0 &&
    invalid.length === 0 &&
    SO101_MOTOR_NAMES.every((motor) => readback[motor] === false);
  const expectedOutcome: TorqueOutcome =
    Object.keys(disableErrors).length > 0 ||
    SO101_MOTOR_NAMES.some((motor) => readback[motor] === true)
      ? "failed"
      : completeOff
        ? "verified_off"
        : !value.disable_attempted && noTorqueValues
          ? "not_attempted"
          : "unknown";
  if (value.outcome !== expectedOutcome) return null;
  return {
    outcome: value.outcome as TorqueOutcome,
    disable_attempted: value.disable_attempted,
    verification_supported: value.verification_supported,
    readback,
    missing_motors: missing,
    invalid_motors: invalid,
    unexpected_motors: unexpected,
    disable_errors: disableErrors,
  };
};

const parseJointLimit = (value: unknown): JointLimit | null => {
  const keys = [
    "max_step_per_tick",
    "max_relative_target",
    "startup_min",
    "startup_max",
    "calibrated_min",
    "calibrated_max",
  ];
  if (!isObject(value) || !exactKeys(value, keys) || keys.some((key) => !finiteNumber(value[key]))) {
    return null;
  }
  if (
    (value.max_step_per_tick as number) <= 0 ||
    (value.max_relative_target as number) <= 0 ||
    (value.startup_min as number) > (value.startup_max as number) ||
    (value.calibrated_min as number) > (value.calibrated_max as number)
  ) {
    return null;
  }
  return {
    max_step_per_tick: value.max_step_per_tick as number,
    max_relative_target: value.max_relative_target as number,
    startup_min: value.startup_min as number,
    startup_max: value.startup_max as number,
    calibrated_min: value.calibrated_min as number,
    calibrated_max: value.calibrated_max as number,
  };
};

const parseJointEvidence = (
  rawUnits: unknown,
  rawLimits: unknown,
  rawSpecs: unknown,
): {
  units: Partial<Record<So101ActionKey, JointUnit>>;
  limits: Partial<Record<So101ActionKey, JointLimit>>;
  specs: JointSpec[];
} | null => {
  if (!isObject(rawUnits) || !isObject(rawLimits) || !Array.isArray(rawSpecs)) return null;
  if (
    Object.keys(rawUnits).some((key) => !ACTION_KEYS.has(key)) ||
    Object.keys(rawLimits).some((key) => !ACTION_KEYS.has(key)) ||
    ![0, SO101_ACTION_KEYS.length].includes(rawSpecs.length)
  ) {
    return null;
  }
  if (rawSpecs.length === 0) {
    return Object.keys(rawUnits).length === 0 && Object.keys(rawLimits).length === 0
      ? { units: {}, limits: {}, specs: [] }
      : null;
  }
  if (!exactKeys(rawUnits, SO101_ACTION_KEYS) || !exactKeys(rawLimits, SO101_ACTION_KEYS)) {
    return null;
  }
  const units: Partial<Record<So101ActionKey, JointUnit>> = {};
  const limits: Partial<Record<So101ActionKey, JointLimit>> = {};
  const specsByKey = new Map<So101ActionKey, JointSpec>();
  for (const actionKey of SO101_ACTION_KEYS) {
    const unit = rawUnits[actionKey];
    const limit = parseJointLimit(rawLimits[actionKey]);
    if (typeof unit !== "string" || !JOINT_UNITS.has(unit as JointUnit) || !limit) return null;
    const expectedUnit: JointUnit =
      actionKey === "gripper.pos" ? "gripper_percentage_points" : "degrees";
    if (unit !== expectedUnit) return null;
    units[actionKey] = unit as JointUnit;
    limits[actionKey] = limit;
  }
  for (const rawSpec of rawSpecs) {
    if (
      !isObject(rawSpec) ||
      !exactKeys(rawSpec, [
        "action_key",
        "unit",
        "max_step_per_tick",
        "max_relative_target",
        "startup_min",
        "startup_max",
        "calibrated_min",
        "calibrated_max",
      ]) ||
      typeof rawSpec.action_key !== "string" ||
      !ACTION_KEYS.has(rawSpec.action_key) ||
      typeof rawSpec.unit !== "string" ||
      !JOINT_UNITS.has(rawSpec.unit as JointUnit)
    ) {
      return null;
    }
    const actionKey = rawSpec.action_key as So101ActionKey;
    const limit = parseJointLimit({
      max_step_per_tick: rawSpec.max_step_per_tick,
      max_relative_target: rawSpec.max_relative_target,
      startup_min: rawSpec.startup_min,
      startup_max: rawSpec.startup_max,
      calibrated_min: rawSpec.calibrated_min,
      calibrated_max: rawSpec.calibrated_max,
    });
    if (!limit || specsByKey.has(actionKey)) return null;
    if (rawSpec.unit !== units[actionKey]) return null;
    const mappedLimit = limits[actionKey];
    if (!mappedLimit || JSON.stringify(limit) !== JSON.stringify(mappedLimit)) return null;
    specsByKey.set(actionKey, {
      action_key: actionKey,
      unit: rawSpec.unit as JointUnit,
      ...limit,
    });
  }
  if (specsByKey.size !== SO101_ACTION_KEYS.length) return null;
  return {
    units,
    limits,
    specs: SO101_ACTION_KEYS.map((key) => specsByKey.get(key) as JointSpec),
  };
};

const jsonSafe = (value: unknown): boolean => {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(jsonSafe);
  return isObject(value) && Object.entries(value).every(([key, entry]) => key.length > 0 && jsonSafe(entry));
};

export const parseControlStatus = (value: unknown): ControlStatus | null => {
  const keys = [
    "session_id", "state", "operation", "resource_keys", "teleoperator_type",
    "claimed_at_utc", "updated_at_utc", "lease_deadline_monotonic", "lease_ttl_s",
    "lease_renew_interval_s", "controller_connected", "controller_error",
    "controller_product_name", "controller_guid", "controller_instance_id",
    "controller_generation", "controller_layout", "sample_sequence", "sample_age_s",
    "rb_held", "release_observed", "controls_neutral", "motion_state", "joint_units",
    "joint_limits", "joint_specs", "saturation_count", "relative_clipping_count",
    "stop_reason", "hold_requested", "stop_requested", "torque", "torque_outcome",
    "thermal_snapshot", "teardown_completed_at_utc", "revision", "details",
  ];
  if (!isObject(value) || !exactKeys(value, keys)) return null;
  if (
    !exactNonEmptyString(value.session_id) ||
    typeof value.state !== "string" || !CONTROL_STATES.has(value.state as ControlState) ||
    typeof value.operation !== "string" || !OPERATIONS.has(value.operation as RobotOperation) ||
    !Array.isArray(value.resource_keys) || value.resource_keys.length !== 2 ||
    value.resource_keys[0] !== "control" || value.resource_keys[1] !== value.operation ||
    !(value.teleoperator_type === null || value.teleoperator_type === "leader_arm" || value.teleoperator_type === "stadia") ||
    !nonEmptyString(value.claimed_at_utc) || !nonEmptyString(value.updated_at_utc) ||
    !finiteNumber(value.lease_deadline_monotonic) || value.lease_deadline_monotonic < 0 ||
    !finiteNumber(value.lease_ttl_s) || value.lease_ttl_s <= 0 ||
    !finiteNumber(value.lease_renew_interval_s) || value.lease_renew_interval_s <= 0 ||
    value.lease_renew_interval_s >= value.lease_ttl_s ||
    !nullableBoolean(value.controller_connected) || !nullableString(value.controller_error) ||
    !nullableString(value.controller_product_name) || !nullableString(value.controller_guid) ||
    !nullableNonnegativeInteger(value.controller_instance_id) ||
    !nullableNonnegativeInteger(value.controller_generation) ||
    !nullableNonnegativeInteger(value.sample_sequence) ||
    !nullableNonnegativeNumber(value.sample_age_s) || !nullableBoolean(value.rb_held) ||
    typeof value.release_observed !== "boolean" || !nullableBoolean(value.controls_neutral) ||
    typeof value.motion_state !== "string" || !MOTION_STATES.has(value.motion_state as MotionState) ||
    !nonnegativeInteger(value.saturation_count) || !nonnegativeInteger(value.relative_clipping_count) ||
    !nullableString(value.stop_reason) || typeof value.hold_requested !== "boolean" ||
    typeof value.stop_requested !== "boolean" || !nullableString(value.teardown_completed_at_utc) ||
    !nonnegativeInteger(value.revision) || !isObject(value.details) || !jsonSafe(value.details)
  ) {
    return null;
  }
  if (
    (value.controller_connected === false && value.controller_error === null) ||
    (value.controller_connected === null && value.controller_error !== null)
  ) {
    return null;
  }
  let layout: ControllerLayout | null = null;
  if (value.controller_layout !== null) {
    if (
      !isObject(value.controller_layout) ||
      !exactKeys(value.controller_layout, ["axes", "buttons", "hats"]) ||
      !nonnegativeInteger(value.controller_layout.axes) ||
      !nonnegativeInteger(value.controller_layout.buttons) ||
      !nonnegativeInteger(value.controller_layout.hats)
    ) {
      return null;
    }
    layout = {
      axes: value.controller_layout.axes,
      buttons: value.controller_layout.buttons,
      hats: value.controller_layout.hats,
    };
  }
  const joints = parseJointEvidence(value.joint_units, value.joint_limits, value.joint_specs);
  const torque = parseTorque(value.torque);
  const thermal = parseThermalStatus(value.thermal_snapshot);
  if (!joints || !torque || (thermal === null && value.thermal_snapshot !== null)) return null;
  if (value.torque_outcome !== torque.outcome) return null;
  if (
    value.operation === "controller_check" &&
    (value.motion_state !== "disarmed" ||
      joints.specs.length !== 0 ||
      value.saturation_count !== 0 ||
      value.relative_clipping_count !== 0 ||
      thermal !== null ||
      torque.outcome !== "not_attempted")
  ) {
    return null;
  }
  if (value.state === "stopping" && !value.stop_requested) return null;
  if (
    (value.state === "stopped" || value.state === "error") !==
    (value.teardown_completed_at_utc !== null)
  ) {
    return null;
  }
  return {
    session_id: value.session_id,
    state: value.state as ControlState,
    operation: value.operation as RobotOperation,
    resource_keys: ["control", value.operation as RobotOperation],
    teleoperator_type: value.teleoperator_type as TeleoperatorType | null,
    claimed_at_utc: value.claimed_at_utc,
    updated_at_utc: value.updated_at_utc,
    lease_deadline_monotonic: value.lease_deadline_monotonic,
    lease_ttl_s: value.lease_ttl_s,
    lease_renew_interval_s: value.lease_renew_interval_s,
    controller_connected: value.controller_connected,
    controller_error: value.controller_error,
    controller_product_name: value.controller_product_name,
    controller_guid: value.controller_guid,
    controller_instance_id: value.controller_instance_id,
    controller_generation: value.controller_generation,
    controller_layout: layout,
    sample_sequence: value.sample_sequence,
    sample_age_s: value.sample_age_s,
    rb_held: value.rb_held,
    release_observed: value.release_observed,
    controls_neutral: value.controls_neutral,
    motion_state: value.motion_state as MotionState,
    joint_units: joints.units,
    joint_limits: joints.limits,
    joint_specs: joints.specs,
    saturation_count: value.saturation_count,
    relative_clipping_count: value.relative_clipping_count,
    stop_reason: value.stop_reason,
    hold_requested: value.hold_requested,
    stop_requested: value.stop_requested,
    torque,
    torque_outcome: torque.outcome,
    thermal_snapshot: thermal,
    teardown_completed_at_utc: value.teardown_completed_at_utc,
    revision: value.revision,
    details: { ...value.details },
  };
};

const statusEquivalent = (left: ControlStatus, right: ControlStatus): boolean =>
  JSON.stringify(left) === JSON.stringify(right);

const collectSessionIds = (
  value: unknown,
  result: string[],
  seen = new WeakSet<object>(),
): boolean => {
  if (typeof value !== "object" || value === null) return true;
  if (seen.has(value)) return true;
  seen.add(value);
  if (Array.isArray(value)) {
    return value.every((entry) => collectSessionIds(entry, result, seen));
  }
  for (const [key, entry] of Object.entries(value)) {
    if (key === "session_id") {
      if (!exactNonEmptyString(entry)) return false;
      result.push(entry);
    } else if (!collectSessionIds(entry, result, seen)) {
      return false;
    }
  }
  return true;
};

export const requireControlStatusEnvelope = (
  value: unknown,
  requirements: ControlEnvelopeRequirements,
): ControlStatus => {
  if (!isObject(value)) throw new Error("The backend returned a non-object control response.");
  if (requirements.requireSuccess && value.success !== true) {
    const message = typeof value.message === "string" ? value.message : null;
    const detail = typeof value.detail === "string" ? value.detail : null;
    throw new Error(message || detail || "The control request was not accepted.");
  }
  const topLevelSessionId = Object.prototype.hasOwnProperty.call(value, "session_id")
    ? value.session_id
    : undefined;
  if (
    topLevelSessionId !== undefined &&
    !exactNonEmptyString(topLevelSessionId)
  ) {
    throw new Error("The backend returned an invalid top-level control session ID.");
  }
  if (requirements.requireTopLevelSessionId && topLevelSessionId === undefined) {
    throw new Error("The backend omitted the top-level control session ID.");
  }
  const statuses: ControlStatus[] = [];
  for (const key of ["status", "control_status"] as const) {
    if (!Object.prototype.hasOwnProperty.call(value, key)) continue;
    if (key === "status" && !isObject(value[key])) {
      if (requirements.requireStatusKey === "status") {
        throw new Error("The backend returned an invalid status object.");
      }
      // Compatibility endpoints may already use a scalar `status` field for
      // their legacy phase. Their canonical lifecycle lives in control_status.
      continue;
    }
    const status = parseControlStatus(value[key]);
    if (!status) throw new Error(`The backend returned an invalid ${key} object.`);
    statuses.push(status);
  }
  if (
    requirements.requireStatusKey &&
    !Object.prototype.hasOwnProperty.call(value, requirements.requireStatusKey)
  ) {
    throw new Error(`The backend omitted ${requirements.requireStatusKey}.`);
  }
  if (statuses.length === 0) {
    const direct = parseControlStatus(value);
    if (direct) statuses.push(direct);
  }
  if (statuses.length === 0) throw new Error("The backend omitted the control status.");
  const status = statuses[0];
  if (statuses.some((candidate) => !statusEquivalent(candidate, status))) {
    throw new Error("The backend returned conflicting nested control statuses.");
  }
  const ids: string[] = [];
  if (!collectSessionIds(value, ids)) {
    throw new Error("The backend returned an invalid nested control session ID.");
  }
  if (new Set(ids).size !== 1) {
    throw new Error("The backend returned mismatched control session IDs.");
  }
  if (requirements.expectedSessionId !== undefined && status.session_id !== requirements.expectedSessionId) {
    throw new Error("The backend returned status for a different control session.");
  }
  if (status.operation !== requirements.expectedOperation) {
    throw new Error(`The backend returned ${status.operation} status for ${requirements.expectedOperation}.`);
  }
  if (status.teleoperator_type !== requirements.expectedTeleoperatorType) {
    throw new Error("The backend returned a mismatched control teleoperator type.");
  }
  return status;
};

export const extractControlStatus = (
  value: unknown,
  requirements?: ControlEnvelopeRequirements,
): ControlStatus | null => {
  if (requirements) {
    try {
      return requireControlStatusEnvelope(value, requirements);
    } catch {
      return null;
    }
  }
  return parseControlStatus(value);
};

/** Only controller transport/identity errors belong under a controller label. */
export const controlStatusError = (status: ControlStatus | null): string | null =>
  status?.controller_error ?? null;

export const controlStopReason = (status: ControlStatus | null): string | null =>
  status?.stop_reason ?? null;

export const isTerminalControlState = (state: ControlState): boolean =>
  state === "stopped" || state === "error";

const CONTROL_STATE_RANK: Record<ControlState, number> = {
  starting: 0,
  running: 1,
  stopping: 2,
  stopped: 3,
  error: 3,
};

/** Reconcile out-of-order status responses without allowing lifecycle rollback. */
export const reconcileControlStatus = (
  previous: ControlStatus | null,
  next: ControlStatus,
): ControlStatus => {
  if (!previous) return next;
  if (previous.session_id !== next.session_id) {
    throw new Error("A control status update changed session identity.");
  }
  if (next.revision < previous.revision) return previous;
  if (next.revision === previous.revision) {
    if (JSON.stringify(next) !== JSON.stringify(previous)) {
      throw new Error(
        "The backend returned contradictory control status at one revision.",
      );
    }
    return previous;
  }
  if (CONTROL_STATE_RANK[next.state] < CONTROL_STATE_RANK[previous.state]) {
    throw new Error("The backend regressed the control lifecycle.");
  }
  if (
    isTerminalControlState(previous.state) &&
    next.state !== previous.state
  ) {
    throw new Error("The backend changed an already-terminal control state.");
  }
  return next;
};
