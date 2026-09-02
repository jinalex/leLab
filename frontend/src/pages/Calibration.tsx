import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  ArrowLeft,
  Settings,
  Activity,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  Play,
  Square,
  Circle,
  Camera,
  Gamepad2,
  ShieldQuestion,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Logo from "@/components/Logo";
import PortDetectionButton from "@/components/ui/PortDetectionButton";
import PortDetectionModal from "@/components/ui/PortDetectionModal";
import { useApi } from "@/contexts/ApiContext";
import { isMotorRangeComplete } from "@/lib/calibrationTargets";
import CameraConfiguration, {
  type CameraConfig,
} from "@/components/recording/CameraConfiguration";
import ControlSessionPanel from "@/components/control/ControlSessionPanel";
import {
  useControlSession,
} from "@/hooks/useControlSession";
import {
  controlStatusError,
  controlStopReason,
  isTerminalControlState,
  normalizeRobotRecord,
  readinessFor,
  reconcileControlStatus,
  requireControlStatusEnvelope,
  SO101_MOTOR_NAMES,
  type ControlStatus,
  type RobotRecord,
  type StadiaConfig,
  type TeleoperatorType,
} from "@/lib/robotConfig";

const DISCONTINUITY_ERROR_PREFIX = "Motor discontinuity detected";

interface CalibrationStatus {
  calibration_active: boolean;
  status: string; // "idle", "connecting", "recording", "completed", "error", "stopping"
  device_type: string | null;
  error: string | null;
  message: string;
  step: number;
  total_steps: number;
  current_positions: Record<string, number> | null;
  recorded_ranges: Record<
    string,
    { min: number; max: number; current: number }
  > | null;
  cleanup_pending: boolean;
}

const CALIBRATION_STATES = new Set([
  "idle",
  "connecting",
  "recording",
  "completed",
  "error",
  "stopping",
]);

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

const requireRobotEnvelope = (
  value: unknown,
  expectedName: string,
): RobotRecord => {
  if (
    !isObject(value) ||
    !hasExactKeys(value, ["status", "robot"]) ||
    value.status !== "success"
  ) {
    throw new Error("The backend returned an invalid robot response envelope.");
  }
  if (value.robot === null) {
    throw new Error("The backend omitted the requested robot record.");
  }
  const record = normalizeRobotRecord(value.robot);
  if (!record) {
    throw new Error("The backend returned an invalid RobotRecordV2.");
  }
  if (record.name !== expectedName) {
    throw new Error("The backend returned a different robot record.");
  }
  return record;
};

const finiteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const parseCalibrationPositions = (
  value: unknown,
): Record<string, number> | null => {
  if (!isObject(value)) return null;
  if (
    Object.keys(value).length !== SO101_MOTOR_NAMES.length ||
    SO101_MOTOR_NAMES.some(
      (motor) =>
        !Object.prototype.hasOwnProperty.call(value, motor) ||
        !finiteNumber(value[motor]),
    )
  ) {
    return null;
  }
  return { ...(value as Record<string, number>) };
};

const parseCalibrationRanges = (
  value: unknown,
): CalibrationStatus["recorded_ranges"] | null => {
  if (!isObject(value)) return null;
  if (
    Object.keys(value).length !== SO101_MOTOR_NAMES.length ||
    SO101_MOTOR_NAMES.some(
      (motor) => !Object.prototype.hasOwnProperty.call(value, motor),
    )
  ) {
    return null;
  }
  const parsed: CalibrationStatus["recorded_ranges"] = {};
  for (const motor of SO101_MOTOR_NAMES) {
    const range = value[motor];
    if (
      !isObject(range) ||
      !hasExactKeys(range, ["min", "max", "current"]) ||
      !finiteNumber(range.min) ||
      !finiteNumber(range.max) ||
      !finiteNumber(range.current) ||
      range.min > range.current ||
      range.current > range.max
    ) {
      return null;
    }
    parsed[motor] = {
      min: range.min,
      max: range.max,
      current: range.current,
    };
  }
  return parsed;
};

const parseCalibrationStatus = (value: unknown): CalibrationStatus | null => {
  const required = [
    "calibration_active",
    "status",
    "device_type",
    "error",
    "message",
    "step",
    "total_steps",
    "current_positions",
    "recorded_ranges",
    "cleanup_pending",
  ];
  if (
    !isObject(value) ||
    !hasExactKeys(value, required, ["session_id", "control_status"]) ||
    typeof value.calibration_active !== "boolean" ||
    typeof value.status !== "string" ||
    !CALIBRATION_STATES.has(value.status) ||
    !(
      value.device_type === null ||
      value.device_type === "robot" ||
      value.device_type === "teleop"
    ) ||
    !(value.error === null || typeof value.error === "string") ||
    typeof value.message !== "string" ||
    typeof value.cleanup_pending !== "boolean" ||
    !Number.isInteger(value.step) ||
    (value.step as number) < 0 ||
    !Number.isInteger(value.total_steps) ||
    (value.total_steps as number) < 1 ||
    (value.session_id !== undefined &&
      (typeof value.session_id !== "string" ||
        !value.session_id.trim() ||
        value.session_id.trim() !== value.session_id)) ||
    (value.control_status !== undefined && !isObject(value.control_status))
  ) {
    return null;
  }
  const currentPositions =
    value.current_positions === null
      ? null
      : parseCalibrationPositions(value.current_positions);
  const recordedRanges =
    value.recorded_ranges === null
      ? null
      : parseCalibrationRanges(value.recorded_ranges);
  if (
    (value.current_positions !== null && currentPositions === null) ||
    (value.recorded_ranges !== null && recordedRanges === null) ||
    (value.cleanup_pending && value.status !== "error")
  ) {
    return null;
  }
  return {
    calibration_active: value.calibration_active,
    status: value.status,
    device_type: value.device_type,
    error: value.error,
    message: value.message,
    step: value.step as number,
    total_steps: value.total_steps as number,
    current_positions: currentPositions,
    recorded_ranges: recordedRanges,
    cleanup_pending: value.cleanup_pending,
  };
};

interface CalibrationRequest {
  device_type: string; // "robot" or "teleop"
  port: string;
  config_file: string;
  robot_name: string | null;
}

type RobotRecordPatch = {
  schema_version: 2;
  teleoperator_type?: TeleoperatorType;
  follower?: Partial<RobotRecord["follower"]>;
  leader?: Partial<NonNullable<RobotRecord["leader"]>> | null;
  stadia?: Partial<StadiaConfig>;
  cameras?: CameraConfig[];
};

interface CalibrationControlIdentity {
  sessionId: string;
  operation: "follower_calibration" | "leader_calibration";
  teleoperatorType: TeleoperatorType;
}

const validStadiaSettings = (settings: StadiaConfig): boolean =>
  (settings.guid === null ||
    (settings.guid.length > 0 &&
      settings.guid.trim() === settings.guid &&
      !settings.guid.includes("\0"))) &&
  Number.isFinite(settings.deadzone) &&
  settings.deadzone >= 0 &&
  settings.deadzone < 1 &&
  Number.isFinite(settings.max_step_per_tick) &&
  settings.max_step_per_tick > 0 &&
  settings.max_step_per_tick <= 0.35 &&
  Number.isFinite(settings.arm_startup_travel_degrees) &&
  settings.arm_startup_travel_degrees > 0 &&
  settings.arm_startup_travel_degrees <= 45 &&
  Number.isFinite(settings.gripper_startup_travel_percentage_points) &&
  settings.gripper_startup_travel_percentage_points > 0 &&
  settings.gripper_startup_travel_percentage_points <= 45;

const yesNoUnknown = (value: boolean | null | undefined): string =>
  value == null ? "Unknown" : value ? "Yes" : "No";

const nullableBooleanEvidence = (
  value: unknown,
): boolean | null | undefined =>
  value === null || typeof value === "boolean" ? value : undefined;

const followerCalibrationAvailable = (record: RobotRecord): boolean =>
  readinessFor(record, "inference").ready;

const leaderCalibrationAvailable = (record: RobotRecord): boolean => {
  const readiness = readinessFor(record, "leader_teleoperation");
  if (readiness.ready) return true;
  if (
    readiness.issues.some(
      (issue) =>
        issue.code === "readiness_unavailable" ||
        issue.code === "wrong_teleoperator_type"
    )
  ) {
    return false;
  }
  return !readiness.issues.some((issue) => issue.code.startsWith("leader_"));
};

const Calibration = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const robotName =
    (location.state as { robot_name?: string } | null)?.robot_name ?? null;
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  const consoleRef = useRef<HTMLDivElement>(null);
  const demoVideoRef = useRef<HTMLDivElement>(null);

  const [deviceType, setDeviceType] = useState<string>("teleop");
  const [port, setPort] = useState<string>("");
  const [robot, setRobot] = useState<RobotRecord | null>(null);
  const [isSavingRobot, setIsSavingRobot] = useState(false);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  // Off by default so merely opening the calibration page never grabs a camera.
  // The user explicitly starts a scan, which is when cameras are turned on,
  // enumerated, and the browser permission prompt is requested.
  const [camerasActive, setCamerasActive] = useState(false);
  const cameraSaveTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchRobot = useCallback(async (): Promise<RobotRecord | null> => {
    if (!robotName) return null;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`
      );
      const data: unknown = await res.json();
      if (!res.ok) return null;
      const r = requireRobotEnvelope(data, robotName);
      setRobot(r);
      return r;
    } catch (e) {
      console.error("Failed to load robot record:", e);
      return null;
    }
  }, [robotName, baseUrl, fetchWithHeaders]);

  // Initial fetch + form prefill on arrival.
  useEffect(() => {
    if (!robotName) return;
    let cancelled = false;
    (async () => {
      const r = await fetchRobot();
      if (!r || cancelled) return;
      // Default to the first incomplete side in the checklist (leader, then follower).
      const defaultDevice =
        r.teleoperator_type === "stadia"
          ? "robot"
          : !leaderCalibrationAvailable(r)
            ? "teleop"
            : !followerCalibrationAvailable(r)
              ? "robot"
              : "teleop";
      setDeviceType(defaultDevice);
      setPort(
        defaultDevice === "teleop"
          ? r.leader?.port || ""
          : r.follower.port || ""
      );
      setCameras(r.cameras ?? []);
    })();
    return () => {
      cancelled = true;
    };
  }, [robotName, fetchRobot]);

  const persistRobotPatch = useCallback(
    async (patch: RobotRecordPatch): Promise<RobotRecord | null> => {
      if (!robotName) return null;
      const response = await fetchWithHeaders(
        `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        }
      );
      const data: unknown = await response.json();
      if (!response.ok) {
        throw new Error(
          isObject(data) && typeof data.message === "string"
            ? data.message
            : "The robot configuration could not be saved.",
        );
      }
      const saved = requireRobotEnvelope(data, robotName);
      setRobot(saved);
      setCameras(saved.cameras);
      return saved;
    },
    [robotName, baseUrl, fetchWithHeaders]
  );

  // Persist camera changes back to the robot record (debounced).
  const handleCamerasChange = (next: CameraConfig[]) => {
    setCameras(next);
    if (!robotName) return;
    if (cameraSaveTimerRef.current) {
      clearTimeout(cameraSaveTimerRef.current);
    }
    cameraSaveTimerRef.current = setTimeout(async () => {
      try {
        await persistRobotPatch({ schema_version: 2, cameras: next });
      } catch (e) {
        console.error("Failed to save cameras to robot record:", e);
      }
    }, 500);
  };

  useEffect(() => {
    return () => {
      if (cameraSaveTimerRef.current) {
        clearTimeout(cameraSaveTimerRef.current);
      }
    };
  }, []);

  const [showPortDetection, setShowPortDetection] = useState(false);
  const [detectionRobotType, setDetectionRobotType] = useState<
    "leader" | "follower"
  >("leader");

  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>(
    {
      calibration_active: false,
      status: "idle",
      device_type: null,
      error: null,
      message: "",
      step: 0,
      total_steps: 1,
      current_positions: null,
      recorded_ranges: null,
      cleanup_pending: false,
    }
  );
  const [isPolling, setIsPolling] = useState(false);
  const [controllerSessionId, setControllerSessionId] = useState<string | null>(
    null
  );
  const [controllerStatus, setControllerStatus] =
    useState<ControlStatus | null>(null);
  const [controllerPending, setControllerPending] = useState(false);
  const [controllerLocalError, setControllerLocalError] = useState<
    string | null
  >(null);
  const [calibrationControlIdentity, setCalibrationControlIdentity] =
    useState<CalibrationControlIdentity | null>(null);
  const [calibrationStatusContractError, setCalibrationStatusContractError] =
    useState<string | null>(null);
  const controllerSessionRef = useRef<string | null>(null);
  const controllerStatusRef = useRef<ControlStatus | null>(null);

  const adoptControllerStatus = useCallback((next: ControlStatus) => {
    const reconciled = reconcileControlStatus(controllerStatusRef.current, next);
    controllerStatusRef.current = reconciled;
    setControllerStatus(reconciled);
    return {
      status: reconciled,
      stale: next.revision < reconciled.revision,
    };
  }, []);

  const calibrationControl = useControlSession({
    sessionId: calibrationControlIdentity?.sessionId ?? null,
    expectedOperation:
      calibrationControlIdentity?.operation ?? "follower_calibration",
    expectedTeleoperatorType:
      calibrationControlIdentity?.teleoperatorType ?? null,
    renewalBlocked: calibrationStatusContractError !== null,
  });

  useEffect(() => {
    controllerSessionRef.current =
      controllerStatus && isTerminalControlState(controllerStatus.state)
        ? null
        : controllerSessionId;
  }, [controllerSessionId, controllerStatus]);

  // The calibration lease uses useControlSession. Controller-check cleanup is
  // kept here because it has a separate session identity on this page.
  useEffect(() => {
    const stopControllerOnPageHide = () => {
      const sessionId = controllerSessionRef.current;
      if (!sessionId) return;
      fetchWithHeaders(`${baseUrl}/control-stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
        keepalive: true,
      }).catch(() => {
        // Best effort only. The server-side lease remains the backstop.
      });
    };
    window.addEventListener("pagehide", stopControllerOnPageHide);

    return () => {
      window.removeEventListener("pagehide", stopControllerOnPageHide);
      const sessionId = controllerSessionRef.current;
      if (sessionId) {
        fetchWithHeaders(`${baseUrl}/control-stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        }).catch((e) =>
          console.error("Failed to stop controller check on unmount:", e)
        );
      }
    };
  }, [baseUrl, fetchWithHeaders]);

  const pollStatus = async () => {
    try {
      const response = await fetchWithHeaders(`${baseUrl}/calibration-status`);
      const raw: unknown = await response.json();
      if (!response.ok) {
        throw new Error("Calibration status is unavailable.");
      }
      const status = parseCalibrationStatus(raw);
      if (!status) {
        throw new Error("The backend returned malformed calibration status.");
      }
      if (calibrationControlIdentity) {
        const controlUpdate = calibrationControl.ingestCompatibilityStatus(raw);
        if (controlUpdate.stale) return;
        const controlStatus = controlUpdate.status;
        const controlTerminal = isTerminalControlState(controlStatus.state);
        const calibrationOwnsResources =
          status.calibration_active && !status.cleanup_pending;
        if (
          (controlTerminal && calibrationOwnsResources) ||
          (!controlTerminal && !status.calibration_active)
        ) {
          throw new Error(
            "Calibration activity conflicts with the exact control lifecycle.",
          );
        }
      }
      setCalibrationStatusContractError(null);
      setCalibrationStatus(status);

      if (
        !status.calibration_active &&
        (status.status === "completed" ||
          status.status === "error" ||
          status.status === "idle")
      ) {
        setIsPolling(false);
      }
    } catch (error) {
      console.error("Error polling status:", error);
      setCalibrationStatusContractError(
        error instanceof Error
          ? error.message
          : "Calibration status polling failed.",
      );
    }
  };

  const handleStartCalibration = async () => {
    if (!robotName) {
      toast({
        title: "No robot selected",
        description: "Open Calibration from a robot's gear icon on the Landing page.",
        variant: "destructive",
      });
      return;
    }
    if (!port) {
      toast({
        title: "Missing port",
        description: "Set the device's serial port before starting.",
        variant: "destructive",
      });
      return;
    }
    if (robot?.teleoperator_type === "stadia" && deviceType !== "robot") {
      toast({
        title: "Follower calibration only",
        description: "Stadia mode never opens or calibrates a leader arm.",
        variant: "destructive",
      });
      return;
    }

    const savedRecord = await persistPort(port);
    if (!savedRecord) {
      toast({
        title: "Configuration was not saved",
        description: "Save the calibration destination before starting.",
        variant: "destructive",
      });
      return;
    }
    const calibrationOperation: CalibrationControlIdentity["operation"] =
      deviceType === "robot"
        ? "follower_calibration"
        : "leader_calibration";
    const readiness = readinessFor(savedRecord, calibrationOperation);
    if (!readiness.ready) {
      toast({
        title: "Calibration not ready",
        description:
          readiness.issues.map((issue) => issue.message).join(" ") ||
          "Complete the saved calibration configuration first.",
        variant: "destructive",
      });
      return;
    }

    const savedDevice =
      deviceType === "robot" ? savedRecord.follower : savedRecord.leader;
    if (!savedDevice || !savedDevice.calibration.endsWith(".json")) {
      toast({
        title: "Calibration destination unavailable",
        description: "The selected saved device has no valid calibration filename.",
        variant: "destructive",
      });
      return;
    }
    const request: CalibrationRequest = {
      device_type: deviceType,
      port: savedDevice.port,
      config_file: savedDevice.calibration.slice(0, -".json".length),
      robot_name: robotName,
    };

    try {
      const response = await fetchWithHeaders(`${baseUrl}/start-calibration`, {
        method: "POST",
        body: JSON.stringify(request),
      });

      const result = await response.json();

      if (response.ok && result.success === true) {
        const controlStatus = requireControlStatusEnvelope(result, {
          expectedOperation: calibrationOperation,
          expectedTeleoperatorType: savedRecord.teleoperator_type,
          requireSuccess: true,
          requireTopLevelSessionId: true,
          requireStatusKey: "status",
        });
        setCalibrationControlIdentity({
          sessionId: controlStatus.session_id,
          operation: calibrationOperation,
          teleoperatorType: savedRecord.teleoperator_type,
        });
        setCalibrationStatusContractError(null);
        toast({
          title: "Calibration Started",
          description: `Calibration started for ${deviceType}`,
        });
        setIsPolling(true);
      } else {
        toast({
          title: "Calibration Failed",
          description: result.message || "Failed to start calibration",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error starting calibration:", error);
      toast({
        title: "Calibration did not start",
        description:
          error instanceof Error
            ? error.message
            : "Failed to validate the calibration control session.",
        variant: "destructive",
      });
    }
  };

  const handleStopCalibration = async () => {
    try {
      if (!calibrationControlIdentity) {
        throw new Error("The exact calibration session ID is unavailable.");
      }
      await calibrationControl.requestStop();
      toast({
        title: "Calibration stop requested",
        description: "Waiting for the server-owned session to finish cleanup.",
      });
    } catch (error) {
      console.error("Error stopping calibration:", error);
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to stop calibration",
        variant: "destructive",
      });
    }
  };

  const handleCompleteStep = async () => {
    if (!calibrationStatus.calibration_active) return;

    try {
      if (!calibrationControlIdentity) {
        throw new Error("The exact calibration session ID is unavailable.");
      }
      const response = await fetchWithHeaders(
        `${baseUrl}/complete-calibration-step`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: calibrationControlIdentity.sessionId,
          }),
        }
      );

      const data = await response.json();

      if (response.ok && data.success === true) {
        calibrationControl.ingestStatusEnvelope(data);
        toast({
          title: "Step Completed",
          description: data.message,
        });
      } else {
        toast({
          title: "Step Failed",
          description: data.message || "Could not complete step",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error completing step:", error);
      toast({
        title: "Error",
        description: "Could not complete calibration step",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    if (
      calibrationStatus.status === "error" &&
      calibrationStatus.error?.startsWith(DISCONTINUITY_ERROR_PREFIX)
    ) {
      demoVideoRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [calibrationStatus.status, calibrationStatus.error]);

  useEffect(() => {
    if (!isPolling) return;
    // Single stable interval. Reads calibration_active from the ref each tick so
    // the interval doesn't tear down/recreate on every status change.
    pollStatus();
    const interval = setInterval(() => {
      pollStatus();
    }, 200);
    return () => clearInterval(interval);
    // pollStatus is stable enough — it only reads via fetchWithHeaders + setState.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPolling]);

  const pollControllerStatus = useCallback(async () => {
    if (!controllerSessionId) return;
    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/control-status?session_id=${encodeURIComponent(controllerSessionId)}`
      );
      const data = await response.json();
      if (!response.ok || data.success === false) {
        throw new Error(data.message || data.detail || "Controller status is unavailable.");
      }
      const next = requireControlStatusEnvelope(data, {
        expectedSessionId: controllerSessionId,
        expectedOperation: "controller_check",
        expectedTeleoperatorType: "stadia",
        requireSuccess: true,
        requireTopLevelSessionId: true,
        requireStatusKey: "status",
      });
      const update = adoptControllerStatus(next);
      if (!update.stale) setControllerLocalError(null);
    } catch (error) {
      setControllerLocalError(
        error instanceof Error ? error.message : "Controller status polling failed."
      );
    }
  }, [controllerSessionId, baseUrl, fetchWithHeaders, adoptControllerStatus]);

  const controllerLifecycleState = controllerStatus?.state ?? null;
  const controllerRenewIntervalS =
    controllerStatus?.lease_renew_interval_s ?? 1;

  useEffect(() => {
    if (
      !controllerSessionId ||
      (controllerLifecycleState &&
        isTerminalControlState(controllerLifecycleState))
    ) {
      return;
    }
    void pollControllerStatus();
    const interval = window.setInterval(() => {
      void pollControllerStatus();
    }, 300);
    return () => window.clearInterval(interval);
  }, [controllerSessionId, controllerLifecycleState, pollControllerStatus]);

  useEffect(() => {
    if (
      !controllerSessionId ||
      controllerLocalError ||
      !controllerLifecycleState ||
      !["starting", "running"].includes(controllerLifecycleState)
    ) {
      return;
    }

    const renew = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/control-lease/renew`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: controllerSessionId }),
        });
        const data = await response.json();
        if (!response.ok || data.success !== true) {
          throw new Error(data.message || data.detail || "Control lease renewal failed.");
        }
        const next = requireControlStatusEnvelope(data, {
          expectedSessionId: controllerSessionId,
          expectedOperation: "controller_check",
          expectedTeleoperatorType: "stadia",
          requireSuccess: true,
          requireTopLevelSessionId: true,
          requireStatusKey: "status",
        });
        adoptControllerStatus(next);
      } catch (error) {
        setControllerLocalError(
          error instanceof Error ? error.message : "Control lease renewal failed."
        );
      }
    };

    const configuredInterval = controllerRenewIntervalS * 1000;
    const intervalMs = Number.isFinite(configuredInterval)
      ? Math.max(250, Math.min(configuredInterval, 2000))
      : 1000;
    const interval = window.setInterval(() => {
      void renew();
    }, intervalMs);
    return () => window.clearInterval(interval);
  }, [
    controllerSessionId,
    controllerLocalError,
    controllerLifecycleState,
    controllerRenewIntervalS,
    baseUrl,
    fetchWithHeaders,
    adoptControllerStatus,
  ]);

  const handleStartControllerCheck = async () => {
    if (!robot || !robotName) return;
    const readiness = readinessFor(robot, "controller_check");
    if (!readiness.ready) {
      toast({
        title: "Controller check unavailable",
        description:
          readiness.issues.map((issue) => issue.message).join(" ") ||
          "Switch this robot to Stadia mode first.",
        variant: "destructive",
      });
      return;
    }

    setControllerPending(true);
    setControllerLocalError(null);
    controllerSessionRef.current = null;
    setControllerSessionId(null);
    controllerStatusRef.current = null;
    setControllerStatus(null);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/controller-check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ robot_name: robotName }),
      });
      const data = await response.json();
      if (!response.ok || data.success !== true) {
        throw new Error(data.message || data.detail || "Controller check could not start.");
      }
      const status = requireControlStatusEnvelope(data, {
        expectedOperation: "controller_check",
        expectedTeleoperatorType: "stadia",
        requireSuccess: true,
        requireTopLevelSessionId: true,
        requireStatusKey: "status",
      });
      const sessionId = status.session_id;
      controllerSessionRef.current = sessionId;
      setControllerSessionId(sessionId);
      adoptControllerStatus(status);
      toast({
        title: "Controller check started",
        description:
          "This operation checks the Stadia controller without instantiating, accessing, connecting, calibrating, torquing, or moving robot hardware.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Controller check could not start.";
      setControllerLocalError(message);
      toast({
        title: "Controller check failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setControllerPending(false);
    }
  };

  const handleStopControllerCheck = async () => {
    if (!controllerSessionId) return;
    setControllerPending(true);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/control-stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: controllerSessionId }),
      });
      const data = await response.json();
      if (!response.ok || data.success !== true) {
        throw new Error(data.message || data.detail || "Controller check could not stop.");
      }
      const status = requireControlStatusEnvelope(data, {
        expectedSessionId: controllerSessionId,
        expectedOperation: "controller_check",
        expectedTeleoperatorType: "stadia",
        requireSuccess: true,
        requireTopLevelSessionId: true,
        requireStatusKey: "status",
      });
      if (status.state === "starting" || status.state === "running") {
        throw new Error(
          "The backend accepted stop without entering stopping or a terminal state.",
        );
      }
      adoptControllerStatus(status);
      toast({
        title: "Stop requested",
        description: "Waiting for the controller-check session to finish.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Controller check could not stop.";
      setControllerLocalError(message);
      toast({
        title: "Stop request failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setControllerPending(false);
    }
  };

  const handleTeleoperatorTypeChange = async (next: string) => {
    if (next !== "leader_arm" && next !== "stadia") return;
    setIsSavingRobot(true);
    try {
      const saved = await persistRobotPatch({
        schema_version: 2,
        teleoperator_type: next,
      });
      if (!saved) throw new Error("The robot record no longer exists.");
      const nextDevice = saved.teleoperator_type === "stadia" ? "robot" : "teleop";
      setDeviceType(nextDevice);
      setPort(
        nextDevice === "robot"
          ? saved.follower.port
          : saved.leader?.port ?? ""
      );
    } catch (error) {
      toast({
        title: "Mode was not saved",
        description:
          error instanceof Error ? error.message : "Could not update teleoperator mode.",
        variant: "destructive",
      });
    } finally {
      setIsSavingRobot(false);
    }
  };

  const handleSaveStadiaSettings = async () => {
    if (!robot || !validStadiaSettings(robot.stadia)) return;
    setIsSavingRobot(true);
    try {
      const saved = await persistRobotPatch({
        schema_version: 2,
        stadia: robot.stadia,
      });
      if (!saved) throw new Error("The robot record no longer exists.");
      toast({
        title: "Stadia settings saved",
        description: "Units and safety bounds were saved to this robot.",
      });
    } catch (error) {
      toast({
        title: "Settings were not saved",
        description:
          error instanceof Error ? error.message : "Could not save Stadia settings.",
        variant: "destructive",
      });
    } finally {
      setIsSavingRobot(false);
    }
  };

  // Load default port when device type changes (skip when arriving from a tile —
  // the robot-record prefill above wins)
  useEffect(() => {
    const loadDefaultPort = async () => {
      if (!deviceType) return;
      if (robotName) return;

      try {
        const robotType = deviceType === "robot" ? "follower" : "leader";
        const response = await fetchWithHeaders(
          `${baseUrl}/robot-port/${robotType}`
        );
        const data = await response.json();
        if (data.status === "success") {
          const portToUse = data.saved_port || data.default_port;
          if (portToUse) {
            setPort(portToUse);
          }
        }
      } catch (error) {
        console.error("Error loading default port:", error);
      }
    };

    loadDefaultPort();
  }, [deviceType, robotName, baseUrl, fetchWithHeaders]);

  const handleDeviceTypeChange = (next: string) => {
    if (robot?.teleoperator_type === "stadia") return;
    setDeviceType(next);
    if (!robot) return;
    setPort(
      next === "teleop" ? robot.leader?.port || "" : robot.follower.port || ""
    );
  };

  // Refresh the robot record when a calibration completes so the checklist
  // flips to ✓ for the side that was just saved, and advance Device Type to
  // the next still-incomplete side (or stay on the current side if both done).
  useEffect(() => {
    if (calibrationStatus.status !== "completed") return;
    (async () => {
      const r = await fetchRobot();
      if (!r) return;
      const nextDevice =
        r.teleoperator_type === "stadia"
          ? "robot"
          : !leaderCalibrationAvailable(r)
            ? "teleop"
            : !followerCalibrationAvailable(r)
              ? "robot"
              : "teleop";
      setDeviceType(nextDevice);
      setPort(
        nextDevice === "teleop"
          ? r.leader?.port || ""
          : r.follower.port || ""
      );
    })();
  }, [calibrationStatus.status, fetchRobot]);

  const handlePortDetection = () => {
    const robotType = deviceType === "robot" ? "follower" : "leader";
    setDetectionRobotType(robotType);
    setShowPortDetection(true);
  };

  // Write the port for the current side straight into the robot record, so a
  // re-detected USB port (which shuffles on reboot/reconnect) sticks without
  // needing a full re-calibration. Mirrors the camera write-back above.
  const persistPort = useCallback(
    async (nextPort: string): Promise<RobotRecord | null> => {
      if (!robotName || !nextPort) return null;
      const side = deviceType === "robot" ? "follower" : "leader";
      const current = side === "follower" ? robot?.follower : robot?.leader;
      const calibration = current?.calibration || `${robotName}.json`;
      if (current?.port === nextPort && current.calibration === calibration) {
        return robot;
      }
      try {
        const patch: RobotRecordPatch =
          side === "follower"
            ? {
                schema_version: 2,
                follower: { port: nextPort, calibration },
              }
            : {
                schema_version: 2,
                leader: { port: nextPort, calibration },
              };
        return await persistRobotPatch(patch);
      } catch (e) {
        console.error("Failed to save port to robot record:", e);
        return null;
      }
    },
    [robotName, deviceType, robot, persistRobotPatch]
  );

  const handlePortDetected = (detectedPort: string) => {
    setPort(detectedPort);
    void persistPort(detectedPort);
  };

  const getStatusDisplay = () => {
    switch (calibrationStatus.status) {
      case "idle":
        return {
          color: "bg-slate-500",
          icon: <Settings className="w-4 h-4" />,
          text: "Idle",
        };
      case "connecting":
        return {
          color: "bg-yellow-500",
          icon: <Loader2 className="w-4 h-4 animate-spin" />,
          text: "Connecting",
        };
      case "recording":
        return {
          color: "bg-purple-500",
          icon: <Activity className="w-4 h-4" />,
          text: "Recording Ranges",
        };
      case "completed":
        return {
          color: "bg-green-500",
          icon: <CheckCircle className="w-4 h-4" />,
          text: "Completed",
        };
      case "error":
        return {
          color: "bg-red-500",
          icon: <XCircle className="w-4 h-4" />,
          text: "Error",
        };
      case "stopping":
        return {
          color: "bg-orange-500",
          icon: <Square className="w-4 h-4" />,
          text: "Stopping",
        };
      default:
        return {
          color: "bg-slate-500",
          icon: <Settings className="w-4 h-4" />,
          text: "Unknown",
        };
    }
  };

  const statusDisplay = getStatusDisplay();
  const controllerCheckActive =
    controllerSessionId !== null &&
    (!controllerStatus || !isTerminalControlState(controllerStatus.state));
  const displayedControllerStopReason = controlStopReason(controllerStatus);
  const controllerMonitoringActive =
    controllerStatus?.details.controller_monitoring_active;
  const controllerMonitoringInactive = controllerMonitoringActive === false;
  const controllerTerminalMonitoringUnproven = Boolean(
    controllerStatus &&
      isTerminalControlState(controllerStatus.state) &&
      controllerMonitoringActive !== false,
  );
  const controllerLastObservedRaw =
    controllerStatus?.details.controller_last_observed;
  const controllerLastObserved = isObject(controllerLastObservedRaw)
    ? controllerLastObservedRaw
    : null;
  const controllerConnectedEvidence = controllerMonitoringInactive
    ? nullableBooleanEvidence(controllerLastObserved?.connected)
    : controllerStatus?.controller_connected;
  const controllerRbEvidence = controllerMonitoringInactive
    ? nullableBooleanEvidence(controllerLastObserved?.rb_held)
    : controllerStatus?.rb_held;
  const controllerNeutralEvidence = controllerMonitoringInactive
    ? nullableBooleanEvidence(controllerLastObserved?.controls_neutral)
    : controllerStatus?.controls_neutral;
  const lastObservedSampleAge = controllerLastObserved?.sample_age_s;
  const controllerSampleAgeEvidence = controllerMonitoringInactive
    ? typeof lastObservedSampleAge === "number" &&
      Number.isFinite(lastObservedSampleAge) &&
      lastObservedSampleAge >= 0
      ? lastObservedSampleAge
      : null
    : controllerStatus?.sample_age_s ?? null;
  const controllerMotionEvidence =
    controllerMonitoringInactive &&
    typeof controllerLastObserved?.motion_state === "string"
      ? controllerLastObserved.motion_state
      : controllerStatus?.motion_state ?? "disarmed";
  const controllerLastObservedError =
    controllerMonitoringInactive &&
    typeof controllerLastObserved?.error === "string"
      ? controllerLastObserved.error
      : null;
  const displayedControllerError =
    controllerLocalError ??
    controlStatusError(controllerStatus) ??
    controllerLastObservedError;
  const controllerEvidencePrefix = controllerMonitoringInactive
    ? "Last-observed "
    : "";
  const stadiaSettingsAreValid =
    robot !== null && validStadiaSettings(robot.stadia);

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center gap-4 mb-6">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(-1)}
            className="text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div className="flex items-center gap-3">
            <Logo iconOnly />
            <h1 className="text-3xl font-bold">
              {robotName ? `Configure "${robotName}"` : "Device Configuration"}
            </h1>
          </div>
        </div>

        {!robotName && (
          <Alert className="mb-6 bg-amber-900/40 border-amber-700 text-amber-100">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Open Calibration from a robot's gear icon on the Landing page.
              Each robot has its own calibration; running this page directly is
              not supported.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Settings className="w-5 h-5 text-blue-400" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label
                  htmlFor="teleoperatorType"
                  className="text-sm font-medium text-slate-300"
                >
                  Teleoperator *
                </Label>
                <Select
                  value={robot?.teleoperator_type ?? "leader_arm"}
                  onValueChange={handleTeleoperatorTypeChange}
                  disabled={
                    !robot ||
                    isSavingRobot ||
                    calibrationStatus.calibration_active ||
                    controllerCheckActive
                  }
                >
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white rounded-md">
                    <SelectValue placeholder="Select teleoperator" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700 text-white">
                    <SelectItem
                      value="leader_arm"
                      className="hover:bg-slate-700"
                    >
                      Leader arm
                    </SelectItem>
                    <SelectItem value="stadia" className="hover:bg-slate-700">
                      Google Stadia Controller
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-400">
                  Switching modes preserves the saved leader configuration.
                  Stadia mode never opens it.
                </p>
              </div>

              {robot?.teleoperator_type === "stadia" && (
                <div className="space-y-4 rounded-lg border border-slate-700 bg-slate-900/40 p-4">
                  <div>
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
                      <Gamepad2 className="h-4 w-4 text-purple-400" />
                      Stadia control settings
                    </div>
                    <p className="mt-1 text-xs text-slate-400">
                      Control runs at a fixed 30 Hz. Arm joints use degrees;
                      the gripper uses percentage points.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="stadiaGuid" className="text-xs text-slate-300">
                      Controller GUID (optional)
                    </Label>
                    <Input
                      id="stadiaGuid"
                      value={robot.stadia.guid ?? ""}
                      onChange={(event) =>
                        setRobot((current) =>
                          current
                            ? {
                                ...current,
                                stadia: {
                                  ...current.stadia,
                                  guid: event.target.value || null,
                                },
                              }
                            : current
                        )
                      }
                      placeholder="Auto-select one exact Stadia controller"
                      className="bg-slate-700 border-slate-600 text-white"
                    />
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label htmlFor="stadiaDeadzone" className="text-xs text-slate-300">
                        Stick deadzone
                      </Label>
                      <NumberInput
                        id="stadiaDeadzone"
                        integer={false}
                        min="0"
                        max="0.99"
                        step="0.01"
                        value={robot.stadia.deadzone}
                        onChange={(value) => {
                          if (value === undefined) return;
                          setRobot((current) =>
                            current
                              ? {
                                  ...current,
                                  stadia: { ...current.stadia, deadzone: value },
                                }
                              : current
                          );
                        }}
                        className="bg-slate-700 border-slate-600 text-white"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="stadiaStep" className="text-xs text-slate-300">
                        Max step (degrees or pp / tick)
                      </Label>
                      <NumberInput
                        id="stadiaStep"
                        integer={false}
                        min="0.01"
                        max="0.35"
                        step="0.05"
                        value={robot.stadia.max_step_per_tick}
                        onChange={(value) => {
                          if (value === undefined) return;
                          setRobot((current) =>
                            current
                              ? {
                                  ...current,
                                  stadia: {
                                    ...current.stadia,
                                    max_step_per_tick: value,
                                  },
                                }
                              : current
                          );
                        }}
                        className="bg-slate-700 border-slate-600 text-white"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="stadiaArmTravel" className="text-xs text-slate-300">
                        Arm startup travel (degrees)
                      </Label>
                      <NumberInput
                        id="stadiaArmTravel"
                        integer={false}
                        min="0.1"
                        max="45"
                        step="0.5"
                        value={robot.stadia.arm_startup_travel_degrees}
                        onChange={(value) => {
                          if (value === undefined) return;
                          setRobot((current) =>
                            current
                              ? {
                                  ...current,
                                  stadia: {
                                    ...current.stadia,
                                    arm_startup_travel_degrees: value,
                                  },
                                }
                              : current
                          );
                        }}
                        className="bg-slate-700 border-slate-600 text-white"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="stadiaGripTravel" className="text-xs text-slate-300">
                        Gripper startup travel (pp)
                      </Label>
                      <NumberInput
                        id="stadiaGripTravel"
                        integer={false}
                        min="0.1"
                        max="45"
                        step="0.5"
                        value={
                          robot.stadia.gripper_startup_travel_percentage_points
                        }
                        onChange={(value) => {
                          if (value === undefined) return;
                          setRobot((current) =>
                            current
                              ? {
                                  ...current,
                                  stadia: {
                                    ...current.stadia,
                                    gripper_startup_travel_percentage_points:
                                      value,
                                  },
                                }
                              : current
                          );
                        }}
                        className="bg-slate-700 border-slate-600 text-white"
                      />
                    </div>
                  </div>
                  <p className="text-xs text-slate-500">
                    The follower also applies a fixed 5 degree / 5 percentage-point
                    relative-target limit per command.
                  </p>
                  {!stadiaSettingsAreValid && (
                    <Alert className="bg-red-900/40 border-red-700 text-red-100">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        Use finite values within the displayed bounds before saving.
                      </AlertDescription>
                    </Alert>
                  )}
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleSaveStadiaSettings}
                    disabled={!stadiaSettingsAreValid || isSavingRobot}
                    className="w-full border-purple-500 text-purple-200 hover:bg-purple-900/30"
                  >
                    {isSavingRobot && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Save controller settings
                  </Button>
                </div>
              )}

              {robot?.teleoperator_type === "leader_arm" ? (
                <div className="space-y-2">
                  <Label
                    htmlFor="deviceType"
                    className="text-sm font-medium text-slate-300"
                  >
                    Device Type *
                  </Label>
                  <Select
                    value={deviceType}
                    onValueChange={handleDeviceTypeChange}
                  >
                    <SelectTrigger className="bg-slate-700 border-slate-600 text-white rounded-md">
                      <SelectValue placeholder="Select device type" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700 text-white">
                      <SelectItem value="teleop" className="hover:bg-slate-700">
                        Teleoperator (Leader)
                      </SelectItem>
                      <SelectItem value="robot" className="hover:bg-slate-700">
                        Robot (Follower)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <Alert className="bg-purple-900/30 border-purple-700 text-purple-100">
                  <Gamepad2 className="h-4 w-4" />
                  <AlertDescription>
                    Stadia mode needs only the follower calibration. Leader-arm
                    fields remain saved but are not opened or used.
                  </AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label
                  htmlFor="port"
                  className="text-sm font-medium text-slate-300"
                >
                  {deviceType === "robot" ? "Follower port" : "Leader port"} *
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="port"
                    value={port}
                    onChange={(e) => setPort(e.target.value)}
                    onBlur={(e) => void persistPort(e.target.value)}
                    placeholder="/dev/tty.usbmodem..."
                    className="bg-slate-700 border-slate-600 text-white rounded-md flex-1"
                  />
                  <PortDetectionButton
                    onClick={handlePortDetection}
                    robotType={deviceType === "robot" ? "follower" : "leader"}
                    className="border-slate-600 hover:border-blue-500 text-slate-400 hover:text-blue-400 bg-slate-700 hover:bg-slate-600"
                  />
                </div>
              </div>

              <Separator className="bg-slate-700" />

              <div className="flex flex-col gap-3">
                {!calibrationStatus.calibration_active ? (
                  <Button
                    onClick={handleStartCalibration}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-full py-6 text-lg"
                    disabled={
                      !robotName ||
                      !deviceType ||
                      !port ||
                      controllerCheckActive
                    }
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Start Calibration
                  </Button>
                ) : (
                  <Button
                    onClick={handleStopCalibration}
                    variant="destructive"
                    className="w-full rounded-full py-6 text-lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Cancel Calibration
                  </Button>
                )}
              </div>

              {robot && (
                <div className="space-y-2 pt-2">
                  <div className="text-sm font-medium text-slate-300">
                    Robot calibration
                  </div>
                  {robot.teleoperator_type === "leader_arm" && (
                    <div className="flex items-center gap-2 text-sm">
                      {leaderCalibrationAvailable(robot) ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <Circle className="w-4 h-4 text-slate-500" />
                      )}
                      <span
                        className={
                          leaderCalibrationAvailable(robot)
                            ? "text-slate-200"
                            : "text-slate-400"
                        }
                      >
                        Leader (Teleoperator)
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-2 text-sm">
                    {followerCalibrationAvailable(robot) ? (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    ) : (
                      <Circle className="w-4 h-4 text-slate-500" />
                    )}
                    <span
                      className={
                        followerCalibrationAvailable(robot)
                          ? "text-slate-200"
                          : "text-slate-400"
                      }
                    >
                      Follower (Robot)
                    </span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Activity className="w-5 h-5 text-teal-400" />
                Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-slate-900/50 rounded-md">
                <span className="text-slate-300">Status:</span>
                <Badge
                  className={`${statusDisplay.color} text-white rounded-md`}
                >
                  {statusDisplay.icon}
                  <span className="ml-2">{statusDisplay.text}</span>
                </Badge>
              </div>

              {calibrationStatus.status === "recording" &&
                calibrationStatus.recorded_ranges && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-purple-400" />
                      <span className="text-sm font-medium text-slate-300">
                        Live Position Data
                      </span>
                    </div>
                    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                      <div className="space-y-3">
                        {Object.entries(calibrationStatus.recorded_ranges).map(
                          ([motor, range]) => {
                            const totalRange = range.max - range.min;
                            const currentOffset = range.current - range.min;
                            const progressPercent =
                              totalRange > 0
                                ? (currentOffset / totalRange) * 100
                                : 50;
                            const rangeComplete = isMotorRangeComplete(
                              calibrationStatus.device_type,
                              motor,
                              totalRange
                            );

                            return (
                              <div key={motor} className="space-y-2">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-2">
                                    <span className="text-white font-semibold text-sm">
                                      {motor}
                                    </span>
                                    {rangeComplete && (
                                      <CheckCircle
                                        className="w-4 h-4 text-green-400"
                                        aria-label="Range complete"
                                      />
                                    )}
                                  </div>
                                  <span className="text-slate-300 text-xs font-mono">
                                    {range.current}
                                  </span>
                                </div>
                                <div className="relative">
                                  <div className="w-full bg-slate-700 rounded-full h-3">
                                    <div
                                      className="bg-slate-600 h-3 rounded-full relative"
                                      style={{ width: "100%" }}
                                    >
                                      <div
                                        className={`absolute top-0 w-1 h-3 rounded-full transition-all duration-100 ${
                                          rangeComplete
                                            ? "bg-green-400"
                                            : "bg-yellow-400"
                                        }`}
                                        style={{
                                          left: `${Math.max(
                                            0,
                                            Math.min(100, progressPercent)
                                          )}%`,
                                          transform: "translateX(-50%)",
                                        }}
                                      />
                                    </div>
                                  </div>
                                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                                    <span>{range.min}</span>
                                    <span>{range.max}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          }
                        )}
                      </div>
                    </div>
                  </div>
                )}

              {calibrationStatus.status === "connecting" && (
                <Alert className="bg-yellow-900/50 border-yellow-700 text-yellow-200">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    Connecting to the device. Please ensure it's connected.
                  </AlertDescription>
                </Alert>
              )}

              {calibrationStatus.status === "recording" && (() => {
                const ranges = calibrationStatus.recorded_ranges ?? {};
                const motors = Object.entries(ranges);
                const allComplete =
                  motors.length > 0 &&
                  motors.every(([motor, range]) =>
                    isMotorRangeComplete(
                      calibrationStatus.device_type,
                      motor,
                      range.max - range.min
                    )
                  );
                return (
                  <div className="space-y-3">
                    <div className="flex justify-center">
                      <Button
                        onClick={handleCompleteStep}
                        disabled={!calibrationStatus.calibration_active}
                        className={`px-8 py-3 rounded-full transition-colors ${
                          allComplete
                            ? "bg-green-600 hover:bg-green-700"
                            : "bg-orange-500 hover:bg-orange-600"
                        }`}
                      >
                        {allComplete ? (
                          <CheckCircle className="w-4 h-4 mr-2" />
                        ) : (
                          <AlertCircle className="w-4 h-4 mr-2" />
                        )}
                        Save Calibration
                      </Button>
                    </div>
                    <Alert className="bg-purple-900/50 border-purple-700 text-purple-200">
                      <Activity className="h-4 w-4" />
                      <AlertDescription>
                        <strong>Important:</strong> Move EACH joint through its
                        full range. A check appears next to each joint once its
                        range is wide enough.
                      </AlertDescription>
                    </Alert>
                  </div>
                );
              })()}

              {calibrationStatus.status === "completed" && (
                <Alert className="bg-green-900/50 border-green-700 text-green-200">
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    Calibration completed successfully!
                  </AlertDescription>
                </Alert>
              )}

              {calibrationStatus.status === "error" &&
                calibrationStatus.error &&
                (calibrationStatus.error.startsWith(
                  DISCONTINUITY_ERROR_PREFIX
                ) ? (
                  <Alert className="bg-red-900/50 border-red-700 text-red-200">
                    <XCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold text-base mb-1">
                        Motor discontinuity detected
                      </div>
                      <div>
                        Make sure to start the calibration with the robot in a
                        middle position — all joints in the middle of their
                        ranges. See the calibration demo below for the correct
                        starting pose.
                      </div>
                    </AlertDescription>
                  </Alert>
                ) : (
                  <Alert className="bg-red-900/50 border-red-700 text-red-200">
                    <XCircle className="h-4 w-4" />
                    <AlertDescription>
                      <strong>Error:</strong> {calibrationStatus.error}
                    </AlertDescription>
                  </Alert>
                ))}

              {calibrationStatus.cleanup_pending && (
                <Alert className="bg-red-900/50 border-red-700 text-red-200">
                  <ShieldQuestion className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Resource cleanup is unproven.</strong> Do not start
                    another hardware operation; restart LeLab before retrying.
                  </AlertDescription>
                </Alert>
              )}

              <div
                ref={demoVideoRef}
                className="bg-slate-900/50 p-4 rounded-lg border border-slate-700"
              >
                <h4 className="font-semibold mb-3 text-slate-200">
                  Calibration Demo:
                </h4>
                <div className="relative rounded-lg overflow-hidden bg-slate-800">
                  <video
                    className="w-full h-auto rounded-md"
                    controls
                    preload="auto"
                    muted
                  >
                    <source
                      src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                      type="video/mp4"
                    />
                    <p className="text-slate-400 text-sm text-center py-4">
                      Your browser does not support the video tag.
                      <br />
                      <a
                        href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                        className="text-blue-400 hover:text-blue-300 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Click here to view the calibration video
                      </a>
                    </p>
                  </video>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {calibrationControlIdentity && (
          <Card className="mt-6 border-slate-700 bg-slate-800/60 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-slate-200">Calibration control lifecycle</CardTitle>
            </CardHeader>
            <CardContent>
              <ControlSessionPanel
                status={calibrationControl.status}
                contractError={
                  calibrationStatusContractError ?? calibrationControl.contractError
                }
              />
            </CardContent>
          </Card>
        )}

        {robot?.teleoperator_type === "stadia" && (
          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm mt-6">
            <CardHeader>
              <CardTitle className="flex items-center justify-between gap-3 text-slate-200">
                <span className="flex items-center gap-2">
                  <Gamepad2 className="w-5 h-5 text-purple-400" />
                  Stadia controller check
                </span>
                {controllerStatus && (
                  <Badge
                    className={`text-white ${
                      controllerStatus.state === "running"
                        ? "bg-green-600"
                        : controllerStatus.state === "error"
                          ? "bg-red-600"
                          : controllerStatus.state === "stopping"
                            ? "bg-orange-600"
                            : "bg-slate-600"
                    }`}
                  >
                    {controllerStatus.state}
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <Alert className="bg-purple-900/30 border-purple-700 text-purple-100">
                <ShieldQuestion className="h-4 w-4" />
                <AlertDescription>
                  This operation checks only the controller. It does not
                  instantiate, access, connect, calibrate, torque, or move robot
                  hardware.
                </AlertDescription>
              </Alert>

              {controllerStatus && controllerMonitoringInactive && (
                <Alert className="bg-purple-900/30 border-purple-700 text-purple-100">
                  <ShieldQuestion className="h-4 w-4" />
                  <AlertDescription>
                    Controller monitoring has ended. Every controller value
                    below is labeled and shown only as last-observed evidence;
                    none is a live reading.
                  </AlertDescription>
                </Alert>
              )}

              {controllerTerminalMonitoringUnproven && (
                <Alert className="bg-red-900/40 border-red-700 text-red-100">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Controller teardown unproven:</strong>{" "}
                    {controllerMonitoringActive === true
                      ? "The lifecycle is terminal, but controller monitoring is still marked active."
                      : "The terminal status did not prove that controller monitoring ended."}{" "}
                    Do not infer that the reader or all related resources closed
                    cleanly.
                  </AlertDescription>
                </Alert>
              )}

              <div className="flex flex-col sm:flex-row gap-3">
                {!controllerCheckActive ? (
                  <Button
                    type="button"
                    onClick={handleStartControllerCheck}
                    disabled={
                      controllerPending ||
                      calibrationStatus.calibration_active ||
                      !readinessFor(robot, "controller_check").ready
                    }
                    className="bg-purple-600 hover:bg-purple-700 text-white"
                  >
                    {controllerPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 mr-2" />
                    )}
                    Check controller
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="destructive"
                    onClick={handleStopControllerCheck}
                    disabled={
                      controllerPending || controllerStatus?.state === "stopping"
                    }
                  >
                    {controllerPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Square className="w-4 h-4 mr-2" />
                    )}
                    {controllerStatus?.state === "stopping"
                      ? "Stopping…"
                      : "Stop check"}
                  </Button>
                )}
                <p className="self-center text-xs text-slate-400">
                  The lease renews only while this check is starting or running.
                </p>
              </div>

              {controllerStatus && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">Lifecycle</div>
                    <div className="text-slate-100">{controllerStatus.state}</div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}connected
                    </div>
                    <div className="text-slate-100">
                      {yesNoUnknown(controllerConnectedEvidence)}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}controller ready
                    </div>
                    <div className="text-slate-100">
                      {typeof controllerStatus.details.controller_ready === "boolean"
                        ? controllerStatus.details.controller_ready
                          ? "Yes"
                          : "No"
                        : "Not reported"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}gate reason
                    </div>
                    <div className="text-slate-100">
                      {typeof controllerStatus.details.controller_gate_reason === "string"
                        ? controllerStatus.details.controller_gate_reason
                        : "Not reported"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerMonitoringInactive
                        ? "Last-observed controller identity"
                        : "Controller identity"}
                    </div>
                    <div className="text-slate-100">
                      {controllerStatus.controller_product_name ?? "Not reported"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3 sm:col-span-2">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}GUID
                    </div>
                    <div className="font-mono text-xs text-slate-100 break-all">
                      {controllerStatus.controller_guid ?? "Not reported"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}instance / generation
                    </div>
                    <div className="text-slate-100">
                      {controllerStatus.controller_instance_id ?? "—"} / {controllerStatus.controller_generation ?? "—"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}layout
                    </div>
                    <div className="text-slate-100">
                      {controllerStatus.controller_layout
                        ? `${controllerStatus.controller_layout.axes} axes · ${controllerStatus.controller_layout.buttons} buttons · ${controllerStatus.controller_layout.hats} hats`
                        : "Not reported"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerMonitoringInactive
                        ? "Last-observed sample"
                        : "Latest sample"}
                    </div>
                    <div className="text-slate-100">
                      {controllerStatus.sample_sequence == null
                        ? "Not reported"
                        : `#${controllerStatus.sample_sequence}`}
                      {controllerSampleAgeEvidence == null
                        ? ""
                        : ` · ${(controllerSampleAgeEvidence * 1000).toFixed(0)} ms old`}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}motion state
                    </div>
                    <div className="text-slate-100">
                      {controllerMotionEvidence}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}RB held
                    </div>
                    <div className="text-slate-100">
                      {yesNoUnknown(controllerRbEvidence)}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}release observed
                    </div>
                    <div className="text-slate-100">
                      {controllerStatus.release_observed ? "Yes" : "No"}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">
                      {controllerEvidencePrefix}controls neutral
                    </div>
                    <div className="text-slate-100">
                      {yesNoUnknown(controllerNeutralEvidence)}
                    </div>
                  </div>
                  <div className="rounded-md bg-slate-900/50 p-3">
                    <div className="text-xs text-slate-500">Session ID</div>
                    <div className="font-mono text-xs text-slate-100 break-all">
                      {controllerStatus.session_id}
                    </div>
                  </div>
                </div>
              )}

              {displayedControllerError && (
                <Alert className="bg-red-900/40 border-red-700 text-red-100">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>
                      {controllerMonitoringInactive && !controllerLocalError
                        ? "Last-observed controller error:"
                        : "Controller error:"}
                    </strong>{" "}
                    {displayedControllerError}
                  </AlertDescription>
                </Alert>
              )}

              {displayedControllerStopReason && (
                <Alert className="bg-slate-900/50 border-slate-700 text-slate-200">
                  <ShieldQuestion className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Stop reason:</strong> {displayedControllerStopReason}
                  </AlertDescription>
                </Alert>
              )}

              {controllerStatus &&
                isTerminalControlState(controllerStatus.state) &&
                <Alert className="bg-slate-900/50 border-slate-700 text-slate-200">
                  <ShieldQuestion className="h-4 w-4" />
                  <AlertDescription>
                    Controller check ended. This operation did not instantiate,
                    access, connect, calibrate, torque, or move robot hardware.
                  </AlertDescription>
                </Alert>}
            </CardContent>
          </Card>
        )}

        {robotName && (
          <Card className="bg-slate-800/60 border-slate-700 backdrop-blur-sm mt-6">
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="flex items-center gap-2 text-slate-200">
                <Settings className="w-5 h-5 text-blue-400" />
                Attached cameras
              </CardTitle>
              <div className="flex items-center gap-2">
                <Label
                  htmlFor="cameras-toggle"
                  className="text-sm text-slate-400 cursor-pointer"
                >
                  {camerasActive ? "On" : "Off"}
                </Label>
                <Switch
                  id="cameras-toggle"
                  checked={camerasActive}
                  onCheckedChange={setCamerasActive}
                  className="data-[state=checked]:bg-green-500"
                  aria-label="Turn cameras on or off"
                />
              </div>
            </CardHeader>
            <CardContent>
              {camerasActive ? (
                <CameraConfiguration
                  cameras={cameras}
                  onCamerasChange={handleCamerasChange}
                />
              ) : (
                <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-6 text-center space-y-3">
                  <Camera className="w-10 h-10 mx-auto text-slate-500" />
                  <div className="space-y-1">
                    <p className="text-slate-200 font-medium">Cameras are off</p>
                    <p className="text-sm text-slate-400 max-w-md mx-auto">
                      Turn cameras on to scan for connected devices and preview
                      them. The browser may briefly open a camera to read device
                      labels, and configured cameras stay active while previews
                      are visible; your browser will ask for camera permission.
                      Nothing is recorded.
                    </p>
                    {cameras.length > 0 && (
                      <p className="text-xs text-slate-500 pt-1">
                        {cameras.length} camera
                        {cameras.length === 1 ? "" : "s"} saved to this robot.
                      </p>
                    )}
                  </div>
                  <p className="flex items-center justify-center gap-1.5 text-xs text-slate-500">
                    <ShieldQuestion className="w-3.5 h-3.5" />
                    You'll be asked to grant camera access.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      <PortDetectionModal
        open={showPortDetection}
        onOpenChange={setShowPortDetection}
        robotType={detectionRobotType}
        onPortDetected={handlePortDetected}
      />
    </div>
  );
};

export default Calibration;
