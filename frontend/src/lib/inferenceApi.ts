import { Fetcher, apiRequest } from "./apiClient";
import {
  requireControlStatusEnvelope,
  type ControlStatus,
  type TeleoperatorType,
} from "./robotConfig";

export interface StartInferenceRequest {
  robot_name: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, {
    type: "opencv";
    camera_index: number;
    width: number;
    height: number;
    fps?: number;
  }>;
  duration_s: number;
}

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  exited?: boolean;
  exit_code?: number | null;
  outcome?: "idle" | "running" | "stopped" | "ok" | "ran_with_warning" | "failed";
  error?: string | null;
  hint?: string | null;
  cleanup_pending: boolean;
  stop_pending?: boolean;
  startup_failed?: boolean;
  stop_error?: string;
  session_id: string;
  control_status: unknown;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const hasExactKeys = (
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

const nullableFinite = (value: unknown): value is number | null =>
  value === null || (typeof value === "number" && Number.isFinite(value));

const nullableString = (value: unknown): value is string | null =>
  value === null || typeof value === "string";

export const parseInferenceStatus = (value: unknown): InferenceStatus | null => {
  const required = [
    "inference_active",
    "started_at",
    "rollout_started_at",
    "elapsed_s",
    "rollout_elapsed_s",
    "duration_s",
    "policy_ref",
    "log_path",
    "cleanup_pending",
    "session_id",
    "control_status",
  ];
  const optional = [
    "exited",
    "exit_code",
    "outcome",
    "error",
    "hint",
    "stop_pending",
    "startup_failed",
    "stop_error",
  ];
  if (
    !isObject(value) ||
    !hasExactKeys(value, required, optional) ||
    typeof value.inference_active !== "boolean" ||
    !nullableFinite(value.started_at) ||
    (value.started_at !== null && value.started_at < 0) ||
    !nullableFinite(value.rollout_started_at) ||
    (value.rollout_started_at !== null && value.rollout_started_at < 0) ||
    typeof value.elapsed_s !== "number" ||
    !Number.isFinite(value.elapsed_s) ||
    value.elapsed_s < 0 ||
    typeof value.rollout_elapsed_s !== "number" ||
    !Number.isFinite(value.rollout_elapsed_s) ||
    value.rollout_elapsed_s < 0 ||
    !nullableFinite(value.duration_s) ||
    (value.duration_s !== null && value.duration_s <= 0) ||
    !nullableString(value.policy_ref) ||
    !nullableString(value.log_path) ||
    typeof value.cleanup_pending !== "boolean" ||
    typeof value.session_id !== "string" ||
    !value.session_id.trim() ||
    value.session_id.trim() !== value.session_id ||
    !isObject(value.control_status) ||
    (value.exited !== undefined && typeof value.exited !== "boolean") ||
    (value.exit_code !== undefined &&
      value.exit_code !== null &&
      !Number.isInteger(value.exit_code)) ||
    (value.outcome !== undefined &&
      value.outcome !== "idle" &&
      value.outcome !== "running" &&
      value.outcome !== "stopped" &&
      value.outcome !== "ok" &&
      value.outcome !== "ran_with_warning" &&
      value.outcome !== "failed") ||
    (value.error !== undefined && !nullableString(value.error)) ||
    (value.hint !== undefined && !nullableString(value.hint)) ||
    (value.stop_pending !== undefined && typeof value.stop_pending !== "boolean") ||
    (value.startup_failed !== undefined && typeof value.startup_failed !== "boolean") ||
    (value.stop_error !== undefined && typeof value.stop_error !== "string")
  ) {
    return null;
  }
  return value as unknown as InferenceStatus;
};

export interface InferenceStartResult {
  message: string;
  log_path: string;
  session_id: string;
  status: ControlStatus;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
  expectedTeleoperatorType: TeleoperatorType,
): Promise<InferenceStartResult> {
  const data = await apiRequest<unknown>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
  const status = requireControlStatusEnvelope(data, {
    expectedOperation: "inference",
    expectedTeleoperatorType,
    requireSuccess: true,
    requireTopLevelSessionId: true,
    requireStatusKey: "status",
  });
  const raw = data as Record<string, unknown>;
  return {
    message: typeof raw.message === "string" ? raw.message : "Inference started.",
    log_path: typeof raw.log_path === "string" ? raw.log_path : "",
    session_id: status.session_id,
    status,
  };
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
  sessionId: string,
  expectedTeleoperatorType: TeleoperatorType,
): Promise<ControlStatus> {
  const data = await apiRequest<unknown>(baseUrl, fetcher, "/control-stop", {
    method: "POST",
    body: { session_id: sessionId },
    action: "Stop inference",
  });
  const status = requireControlStatusEnvelope(data, {
    expectedSessionId: sessionId,
    expectedOperation: "inference",
    expectedTeleoperatorType,
    requireSuccess: true,
    requireTopLevelSessionId: true,
    requireStatusKey: "status",
  });
  if (status.state === "starting" || status.state === "running") {
    throw new Error(
      "The backend accepted stop without entering stopping or a terminal state.",
    );
  }
  return status;
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  const data = await apiRequest<unknown>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
  const status = parseInferenceStatus(data);
  if (!status) {
    throw new Error("The backend returned malformed inference status.");
  }
  return status;
}
