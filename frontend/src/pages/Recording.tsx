import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";
import {
  ArrowLeft,
  MoreHorizontal,
  RotateCcw,
  Square,
  SkipForward,
  Play,
  Volume2,
  VolumeX,
} from "lucide-react";
import {
  getMuted,
  setMuted as persistMuted,
  playRecordingStartCue,
  playResetStartCue,
  playAutoAdvanceWarning,
} from "@/lib/recordingAudio";
import { useApi } from "@/contexts/ApiContext";
import ControlSessionPanel from "@/components/control/ControlSessionPanel";
import { useControlSession } from "@/hooks/useControlSession";
import {
  requireControlStatusEnvelope,
  type ControlStatus,
  type RobotOperation,
  type TeleoperatorType,
} from "@/lib/robotConfig";
import { isExactDatasetRepoId } from "@/lib/recordingContract";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface RecordingConfig {
  robot_name: string;
  dataset_repo_id: string;
  single_task: string;
  num_episodes: number;
  episode_time_s: number;
  reset_time_s: number;
  fps: number;
  video: boolean;
  push_to_hub: boolean;
  resume: boolean;
  streaming_encoding: boolean;
  cameras: Record<string, {
    type: string;
    camera_index?: number;
    width: number;
    height: number;
    fps?: number | null;
    fourcc?: string;
    backend?: string;
  }>;
}

interface RecordingNavigationState {
  recordingConfig: RecordingConfig;
  operation: "leader_recording" | "stadia_recording";
  teleoperator_type: TeleoperatorType;
}

type Phase = "preparing" | "recording" | "resetting" | "completed";
type BackendPhase = Phase | "recovery" | "stopping" | "error";
type LegacyRecordingOutcome = "idle" | "running" | "completed" | "failed";

interface BackendStatus {
  recording_active: boolean;
  current_phase: BackendPhase;
  current_episode?: number;
  total_episodes?: number;
  saved_episodes?: number;
  phase_elapsed_seconds?: number;
  phase_time_limit_s?: number | null;
  session_elapsed_seconds?: number;
  session_ended?: boolean;
  dataset_repo_id?: string;
  dataset_safe?: boolean;
  dataset_finalized?: boolean;
  dataset_uploaded?: boolean;
  upload_available?: boolean;
  dataset_error?: string | null;
  error?: string | null;
  exited?: boolean;
  outcome?: LegacyRecordingOutcome;
  cleanup_pending?: boolean;
  cameras?: string[]; // Names of the cameras configured for this session
  camera_feed_available?: boolean;
  session_start_time?: number;
  phase_start_time?: number;
  message: string;
  session_id: string;
  control_status: unknown;
  available_controls: {
    stop_recording: boolean;
    exit_early: boolean;
    rerecord_episode: boolean;
  };
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

const BACKEND_PHASES = new Set<BackendPhase>([
  "preparing",
  "recording",
  "resetting",
  "recovery",
  "stopping",
  "completed",
  "error",
]);

const LEGACY_RECORDING_OUTCOMES = new Set<LegacyRecordingOutcome>([
  "idle",
  "running",
  "completed",
  "failed",
]);

const parseRecordingNavigationState = (
  value: unknown,
): RecordingNavigationState | null => {
  if (
    !isObject(value) ||
    !hasExactKeys(value, ["recordingConfig", "operation", "teleoperator_type"]) ||
    !isObject(value.recordingConfig)
  ) {
    return null;
  }
  const config = value.recordingConfig;
  const configKeys = [
    "robot_name",
    "dataset_repo_id",
    "single_task",
    "num_episodes",
    "episode_time_s",
    "reset_time_s",
    "fps",
    "video",
    "push_to_hub",
    "resume",
    "streaming_encoding",
    "cameras",
  ];
  if (
    !hasExactKeys(config, configKeys) ||
    (value.operation !== "leader_recording" && value.operation !== "stadia_recording") ||
    (value.teleoperator_type !== "leader_arm" && value.teleoperator_type !== "stadia") ||
    (value.operation === "leader_recording") !==
      (value.teleoperator_type === "leader_arm") ||
    typeof config.robot_name !== "string" ||
    !config.robot_name.trim() ||
    config.robot_name.trim() !== config.robot_name ||
    typeof config.dataset_repo_id !== "string" ||
    !config.dataset_repo_id.trim() ||
    typeof config.single_task !== "string" ||
    !config.single_task.trim() ||
    !Number.isInteger(config.num_episodes) ||
    (config.num_episodes as number) < 1 ||
    !Number.isFinite(config.episode_time_s) ||
    (config.episode_time_s as number) <= 0 ||
    !Number.isFinite(config.reset_time_s) ||
    (config.reset_time_s as number) <= 0 ||
    !Number.isFinite(config.fps) ||
    (config.fps as number) <= 0 ||
    typeof config.video !== "boolean" ||
    typeof config.push_to_hub !== "boolean" ||
    typeof config.resume !== "boolean" ||
    typeof config.streaming_encoding !== "boolean" ||
    !isObject(config.cameras)
  ) {
    return null;
  }
  if (value.operation === "stadia_recording") {
    if (!isExactDatasetRepoId(config.dataset_repo_id)) {
      return null;
    }
  }
  for (const [name, camera] of Object.entries(config.cameras)) {
    if (
      !name.trim() ||
      !isObject(camera) ||
      !hasExactKeys(camera, ["type", "width", "height"], [
        "camera_index",
        "fps",
        "fourcc",
        "backend",
      ]) ||
      camera.type !== "opencv" ||
      !Number.isInteger(camera.width) ||
      (camera.width as number) < 1 ||
      !Number.isInteger(camera.height) ||
      (camera.height as number) < 1 ||
      (camera.camera_index !== undefined &&
        (!Number.isInteger(camera.camera_index) ||
          (camera.camera_index as number) < 0)) ||
      (camera.fps !== undefined &&
        camera.fps !== null &&
        (!Number.isInteger(camera.fps) || (camera.fps as number) < 1)) ||
      (camera.fourcc !== undefined &&
        (typeof camera.fourcc !== "string" || camera.fourcc.length !== 4)) ||
      (camera.backend !== undefined && typeof camera.backend !== "string")
    ) {
      return null;
    }
    if (
      value.operation === "stadia_recording" &&
      (!Object.prototype.hasOwnProperty.call(camera, "camera_index") ||
        !Number.isInteger(camera.camera_index) ||
        !Object.prototype.hasOwnProperty.call(camera, "fps") ||
        !(
          camera.fps === null ||
          (Number.isInteger(camera.fps) && (camera.fps as number) >= 1)
        ))
    ) {
      return null;
    }
  }
  return value as unknown as RecordingNavigationState;
};

const parseBackendStatus = (
  value: unknown,
  operation: RecordingNavigationState["operation"],
): BackendStatus | null => {
  const legacyEvidence = ["exited", "outcome", "cleanup_pending"];
  const required = [
    "recording_active",
    "current_phase",
    "session_ended",
    "available_controls",
    "message",
    "session_id",
    "control_status",
    ...(operation === "leader_recording" ? legacyEvidence : []),
  ];
  const optional = [
    "current_episode",
    "total_episodes",
    "saved_episodes",
    "phase_elapsed_seconds",
    "phase_time_limit_s",
    "session_elapsed_seconds",
    "dataset_repo_id",
    "dataset_safe",
    "dataset_finalized",
    "dataset_uploaded",
    "upload_available",
    "dataset_error",
    "error",
    "cameras",
    "camera_feed_available",
    "session_start_time",
    "phase_start_time",
    ...(operation === "stadia_recording" ? legacyEvidence : []),
  ];
  if (
    !isObject(value) ||
    !hasExactKeys(value, required, optional) ||
    !isObject(value.available_controls) ||
    !hasExactKeys(value.available_controls, [
      "stop_recording",
      "exit_early",
      "rerecord_episode",
    ])
  ) {
    return null;
  }
  const controls = value.available_controls;
  if (
    typeof value.recording_active !== "boolean" ||
    typeof value.current_phase !== "string" ||
    !BACKEND_PHASES.has(value.current_phase as BackendPhase) ||
    typeof value.session_ended !== "boolean" ||
    typeof value.message !== "string" ||
    typeof value.session_id !== "string" ||
    !value.session_id.trim() ||
    value.session_id.trim() !== value.session_id ||
    !isObject(value.control_status) ||
    typeof controls.stop_recording !== "boolean" ||
    typeof controls.exit_early !== "boolean" ||
    typeof controls.rerecord_episode !== "boolean"
  ) {
    return null;
  }
  const numeric = [
    "current_episode",
    "total_episodes",
    "saved_episodes",
    "phase_elapsed_seconds",
    "session_elapsed_seconds",
    "session_start_time",
    "phase_start_time",
  ];
  if (
    numeric.some(
      (key) =>
        value[key] !== undefined &&
        (typeof value[key] !== "number" || !Number.isFinite(value[key]) || (value[key] as number) < 0),
    ) ||
    (value.phase_time_limit_s !== undefined &&
      value.phase_time_limit_s !== null &&
      (typeof value.phase_time_limit_s !== "number" ||
        !Number.isFinite(value.phase_time_limit_s) ||
        value.phase_time_limit_s < 0)) ||
    (value.dataset_repo_id !== undefined && typeof value.dataset_repo_id !== "string") ||
    (value.dataset_safe !== undefined && typeof value.dataset_safe !== "boolean") ||
    (value.dataset_finalized !== undefined && typeof value.dataset_finalized !== "boolean") ||
    (value.dataset_uploaded !== undefined && typeof value.dataset_uploaded !== "boolean") ||
    (value.upload_available !== undefined && typeof value.upload_available !== "boolean") ||
    (value.dataset_error !== undefined &&
      value.dataset_error !== null &&
      typeof value.dataset_error !== "string") ||
    (value.error !== undefined && value.error !== null && typeof value.error !== "string") ||
    (value.exited !== undefined && typeof value.exited !== "boolean") ||
    (value.outcome !== undefined &&
      (typeof value.outcome !== "string" ||
        !LEGACY_RECORDING_OUTCOMES.has(
          value.outcome as LegacyRecordingOutcome,
        ))) ||
    (value.cleanup_pending !== undefined &&
      typeof value.cleanup_pending !== "boolean") ||
    (value.camera_feed_available !== undefined &&
      typeof value.camera_feed_available !== "boolean") ||
    (value.cameras !== undefined &&
      (!Array.isArray(value.cameras) ||
        !value.cameras.every(
          (camera) => typeof camera === "string" && camera.trim().length > 0,
        ) ||
        new Set(value.cameras).size !== value.cameras.length))
  ) {
    return null;
  }
  if (
    value.current_episode !== undefined &&
    (!Number.isInteger(value.current_episode) ||
      (value.current_episode as number) < 1)
  ) {
    return null;
  }
  if (
    value.total_episodes !== undefined &&
    (!Number.isInteger(value.total_episodes) ||
      (value.total_episodes as number) < 1)
  ) {
    return null;
  }
  if (
    value.saved_episodes !== undefined &&
    !Number.isInteger(value.saved_episodes)
  ) {
    return null;
  }
  if (
    (value.dataset_uploaded === true && value.upload_available !== false) ||
    (value.upload_available === true && value.dataset_uploaded !== false)
  ) {
    return null;
  }
  if (operation === "leader_recording") {
    const terminalFailure =
      value.current_phase === "error" &&
      value.exited === true &&
      value.outcome === "failed";
    const terminalSuccess =
      value.current_phase === "completed" &&
      value.exited === true &&
      value.outcome === "completed" &&
      value.cleanup_pending === false;
    const active =
      value.recording_active &&
      value.exited === false &&
      value.outcome === "running" &&
      value.cleanup_pending === false;
    const idle =
      !value.recording_active &&
      !value.session_ended &&
      value.exited === false &&
      value.outcome === "idle" &&
      value.cleanup_pending === false;
    if (
      !(terminalFailure || terminalSuccess || active || idle) ||
      (value.cleanup_pending === true && !terminalFailure)
    ) {
      return null;
    }
  }
  return value as unknown as BackendStatus;
};

type StadiaDatasetDisposition = "blocked" | "none" | "uploaded" | "manual";

const stadiaRecordingDatasetId = (
  status: ControlStatus,
): string | null => {
  const recording = status.details.recording;
  if (!isObject(recording) || !isExactDatasetRepoId(recording.dataset_repo_id)) {
    return null;
  }
  return recording.dataset_repo_id;
};

const requireStadiaTerminalDatasetEvidence = (status: BackendStatus): void => {
  if (
    status.saved_episodes === undefined ||
    status.dataset_safe === undefined ||
    status.dataset_finalized === undefined ||
    status.dataset_uploaded === undefined ||
    status.upload_available === undefined
  ) {
    throw new Error(
      "The terminal Stadia recording status omitted its dataset disposition evidence.",
    );
  }
};

const stadiaDatasetDisposition = (
  status: BackendStatus,
): StadiaDatasetDisposition => {
  if (
    status.saved_episodes === undefined ||
    status.dataset_safe !== true ||
    status.dataset_finalized === undefined ||
    status.dataset_uploaded === undefined ||
    status.upload_available === undefined ||
    Boolean(status.dataset_error)
  ) {
    return "blocked";
  }
  if (
    status.saved_episodes === 0 &&
    status.dataset_finalized === false &&
    status.dataset_uploaded === false &&
    status.upload_available === false
  ) {
    return "none";
  }
  if (status.dataset_finalized !== true) return "blocked";
  if (
    status.saved_episodes > 0 &&
    status.dataset_uploaded === true &&
    status.upload_available === false
  ) {
    return "uploaded";
  }
  if (
    status.saved_episodes > 0 &&
    status.dataset_uploaded === false &&
    status.upload_available === true
  ) {
    return "manual";
  }
  return "blocked";
};

const requireRecordingCommandSession = (
  value: unknown,
  sessionId: string,
  operation: RecordingNavigationState["operation"],
): void => {
  if (!isObject(value)) {
    throw new Error("The recording command returned a non-object response.");
  }
  if (
    value.session_id !== undefined &&
    value.session_id !== sessionId
  ) {
    throw new Error("The recording command returned a mismatched session ID.");
  }
  if (operation === "stadia_recording" && value.session_id !== sessionId) {
    throw new Error("The Stadia recording command omitted its exact session ID.");
  }
};

const Recording = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, wsBaseUrl, fetchWithHeaders } = useApi();

  const navigationState = parseRecordingNavigationState(location.state);
  const recordingConfig = navigationState?.recordingConfig ?? null;
  const [controlSessionId, setControlSessionId] = useState<string | null>(null);
  const [startedDatasetRepoId, setStartedDatasetRepoId] = useState<string | null>(
    null,
  );
  const [recordingContractError, setRecordingContractError] = useState<string | null>(null);
  const terminalToastRef = useRef(false);
  const control = useControlSession({
    sessionId: controlSessionId,
    expectedOperation:
      (navigationState?.operation ?? "leader_recording") as RobotOperation,
    expectedTeleoperatorType: navigationState?.teleoperator_type ?? null,
    renewalBlocked: recordingContractError !== null,
  });
  const ingestRecordingControlStatus = control.ingestCompatibilityStatus;
  const controlState = control.status?.state ?? null;

  // Backend status state - this is the single source of truth
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(
    null
  );
  const [recordingSessionStarted, setRecordingSessionStarted] = useState(false);

  const [optimisticPhase, setOptimisticPhase] = useState<Phase | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [muted, setMutedState] = useState<boolean>(() => getMuted());
  const prevRealPhaseRef = useRef<Phase | null>(null);
  // Bumps on each re-record so the auto-advance warning re-fires for the same episode number.
  const [rerecordTick, setRerecordTick] = useState(0);
  const warningFiredForPhaseRef = useRef<{ phase: Phase | null; episode: number | null; tick: number }>({ phase: null, episode: null, tick: 0 });
  // Guards against React StrictMode double-invocation of the start effect.
  const startInitiatedRef = useRef(false);

  // --- Camera preview layout -------------------------------------------------
  // Aspect ratio (width / height) of the configured cameras, used to size the
  // preview windows without letterboxing. Falls back to 4:3 if unknown.
  const cameraAspect = useMemo(() => {
    const cams = (recordingConfig as unknown as { cameras?: unknown })?.cameras;
    const list = Array.isArray(cams)
      ? cams
      : cams && typeof cams === "object"
      ? Object.values(cams as Record<string, unknown>)
      : [];
    const first = list[0] as { width?: number; height?: number } | undefined;
    if (first?.width && first?.height) return first.width / first.height;
    return 4 / 3;
  }, [recordingConfig]);

  // Measure the space left for the camera windows (via ResizeObserver, so it
  // re-fits on any viewport change) to size them as large as possible while
  // keeping the whole page within one screen (no scroll).
  const [cameraArea, setCameraArea] = useState({ w: 0, h: 0 });
  const cameraAreaObserver = useRef<ResizeObserver | null>(null);
  const cameraAreaRef = useCallback((node: HTMLDivElement | null) => {
    cameraAreaObserver.current?.disconnect();
    cameraAreaObserver.current = null;
    if (node) {
      const ro = new ResizeObserver((entries) => {
        const r = entries[0].contentRect;
        setCameraArea({ w: r.width, h: r.height });
      });
      ro.observe(node);
      cameraAreaObserver.current = ro;
    }
  }, []);

  // Pick the column count and per-window pixel size that maximizes the video
  // area within the measured space, given the camera count and aspect ratio.
  const cameraCount = backendStatus?.cameras?.length ?? 0;
  const cameraWindow = useMemo(() => {
    const { w, h } = cameraArea;
    if (!cameraCount || w <= 0 || h <= 0) return { width: 0, height: 0 };
    const gap = 12; // matches the grid's gap-3
    let best = { width: 0, height: 0, area: -1 };
    for (let cols = 1; cols <= cameraCount; cols++) {
      const rows = Math.ceil(cameraCount / cols);
      const cellW = (w - gap * (cols - 1)) / cols;
      const cellH = (h - gap * (rows - 1)) / rows;
      if (cellW <= 0 || cellH <= 0) continue;
      const width = Math.min(cellW, cellH * cameraAspect);
      const height = width / cameraAspect;
      const area = width * height;
      if (area > best.area) best = { width, height, area };
    }
    return { width: Math.floor(best.width), height: Math.floor(best.height) };
  }, [cameraArea, cameraCount, cameraAspect]);

  const toggleMute = useCallback(() => {
    setMutedState((prev) => {
      const next = !prev;
      persistMuted(next);
      return next;
    });
  }, []);

  // Redirect if no config provided
  useEffect(() => {
    if (!recordingConfig) {
      toast({
        title: "No Configuration",
        description: "Please start recording from the main page.",
        variant: "destructive",
      });
      navigate("/");
    }
  }, [recordingConfig, navigate, toast]);

  // Start recording session when component loads. The ref guard prevents
  // React StrictMode (and any future re-renders) from firing /start-recording
  // twice — the second call returns 409 and bounces the user home.
  useEffect(() => {
    if (recordingConfig && !startInitiatedRef.current) {
      startInitiatedRef.current = true;
      startRecordingSession();
    }
    // startRecordingSession is intentionally omitted: re-running this effect
    // on its identity change would re-fire /start-recording.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingConfig]);

  // Refs so the poll interval below stays stable and reads the latest values
  // without tearing itself down on every state change.
  const optimisticPhaseRef = useRef(optimisticPhase);
  optimisticPhaseRef.current = optimisticPhase;
  const rerecordTickRef = useRef(rerecordTick);
  rerecordTickRef.current = rerecordTick;

  // Poll backend status continuously to stay in sync
  useEffect(() => {
    if (!recordingSessionStarted || !controlSessionId || !navigationState) return;

    const pollStatus = async () => {
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/recording-status`
        );
        const raw: unknown = await response.json();
        if (!response.ok) {
          throw new Error("Recording status is unavailable.");
        }
        const controlUpdate = ingestRecordingControlStatus(raw);
        if (controlUpdate.stale) return;
        const controlStatus = controlUpdate.status;
        const status = parseBackendStatus(raw, navigationState.operation);
        if (!status) throw new Error("The backend returned malformed recording status.");
        if (navigationState.operation === "stadia_recording") {
          const controlDatasetRepoId =
            stadiaRecordingDatasetId(controlStatus);
          if (
            !startedDatasetRepoId ||
            !isExactDatasetRepoId(status.dataset_repo_id) ||
            status.dataset_repo_id !== startedDatasetRepoId ||
            controlDatasetRepoId !== startedDatasetRepoId
          ) {
            throw new Error(
              "The Stadia recording start, status, and control-status dataset IDs did not agree.",
            );
          }
          if (status.camera_feed_available === true) {
            throw new Error(
              "The backend unexpectedly advertised a camera preview for Stadia recording.",
            );
          }
        }
        const activeRecordingTuple =
          status.recording_active &&
          !status.session_ended &&
          status.available_controls.stop_recording;
        const endedRecordingTuple =
          !status.recording_active &&
          status.session_ended &&
          !status.available_controls.stop_recording;
        const controlTerminal =
          controlStatus.state === "stopped" || controlStatus.state === "error";
        const lifecycleConsistent = controlTerminal
          ? endedRecordingTuple
          : controlStatus.state === "stopping"
            ? activeRecordingTuple || endedRecordingTuple
            : activeRecordingTuple;
        if (!lifecycleConsistent) {
          throw new Error(
            "Recording activity conflicts with the exact control lifecycle.",
          );
        }
        setRecordingContractError(null);
        setBackendStatus(status);

        const currentOptimistic = optimisticPhaseRef.current;
        if (currentOptimistic && status.current_phase === currentOptimistic) {
          setOptimisticPhase(null);
        }

        const real = status.current_phase;
        const prev = prevRealPhaseRef.current;
        if (
          (real === "preparing" ||
            real === "recording" ||
            real === "resetting" ||
            real === "completed") &&
          prev !== real
        ) {
          if (real === "recording" && prev !== null) {
            playRecordingStartCue();
          } else if (real === "resetting") {
            playResetStartCue();
          }
          prevRealPhaseRef.current = real;
          warningFiredForPhaseRef.current = { phase: null, episode: null, tick: 0 };
        }

        const elapsed = status.phase_elapsed_seconds || 0;
        const limit = status.phase_time_limit_s || 0;
        const inFinalThreeSeconds = limit > 3 && elapsed >= limit - 3;
        const ep = status.current_episode ?? null;
        const tick = rerecordTickRef.current;
        const warned = warningFiredForPhaseRef.current;
        if (
          (real === "recording" || real === "resetting") &&
          inFinalThreeSeconds &&
          currentOptimistic === null &&
          (warned.phase !== real ||
            warned.episode !== ep ||
            warned.tick !== tick)
        ) {
          playAutoAdvanceWarning();
          warningFiredForPhaseRef.current = { phase: real, episode: ep, tick };
        }

        if (
          !status.recording_active &&
          status.session_ended &&
          controlTerminal
        ) {
          const terminalFailed =
            controlStatus.state === "error" ||
            status.current_phase === "error" ||
            status.outcome === "failed" ||
            status.cleanup_pending === true ||
            Boolean(status.error) ||
            Boolean(status.dataset_error);
          const terminalEvidence = [
            `Control state: ${controlStatus.state}.`,
            controlStatus.stop_reason
              ? `Stop reason: ${controlStatus.stop_reason}.`
              : "Stop reason: not reported.",
            status.error ? `Recording error: ${status.error}.` : null,
            status.dataset_error ? `Dataset error: ${status.dataset_error}.` : null,
            status.outcome ? `Legacy recording outcome: ${status.outcome}.` : null,
            status.cleanup_pending === true
              ? "Legacy resource cleanup remains unproven."
              : null,
          ]
            .filter((entry): entry is string => entry !== null)
            .join(" ");
          const torqueEvidence =
            controlStatus.torque_outcome === "verified_off"
              ? controlStatus.state === "error" ||
                status.cleanup_pending === true
                ? "Torque is verified off across all six motors. That proves torque state only; this error outcome does not prove that every resource closed cleanly."
                : "Torque is verified off across all six motors."
              : `Torque outcome is ${controlStatus.torque_outcome}; do not assume the follower is safe to handle.`;

          if (navigationState.operation === "stadia_recording") {
            requireStadiaTerminalDatasetEvidence(status);
            const disposition = stadiaDatasetDisposition(status);
            if (disposition === "blocked") {
              const message =
                status.dataset_error ||
                status.error ||
                "The Stadia dataset was unsafe, unfinalized despite saved data, or had no valid upload disposition.";
              setRecordingContractError(
                `${message} Upload is blocked; inspect or discard the local dataset before continuing.`,
              );
              if (!terminalToastRef.current) {
                terminalToastRef.current = true;
                toast({
                  title: terminalFailed
                    ? "Recording ended with an error"
                    : "Dataset upload blocked",
                  description: `${terminalEvidence} ${torqueEvidence} This Stadia recording is unsafe, poisoned, unfinalized despite saved data, or has an unproven upload disposition. Earlier saved episode counts do not override that result.`,
                  variant: "destructive",
                });
              }
              return;
            }
            if (disposition === "none") {
              terminalToastRef.current = true;
              toast({
                title: terminalFailed
                  ? "Recording ended with an error"
                  : "Recording stopped with no saved dataset",
                description: `${terminalEvidence} No episode was saved. ${torqueEvidence}`,
                variant:
                  terminalFailed || controlStatus.torque_outcome === "failed"
                    ? "destructive"
                    : "default",
              });
              navigate("/");
              return;
            }
            if (disposition === "uploaded") {
              terminalToastRef.current = true;
              toast({
                title: terminalFailed
                  ? "Recording ended with an error"
                  : "Dataset uploaded",
                description: `${terminalEvidence} The finalized dataset was uploaded. ${torqueEvidence}`,
                variant:
                  terminalFailed || controlStatus.torque_outcome === "failed"
                    ? "destructive"
                    : "default",
              });
              navigate("/");
              return;
            }
          }
          if (!terminalToastRef.current) {
            terminalToastRef.current = true;
            toast({
              title: terminalFailed
                ? "Recording ended with an error"
                : "Recording session ended",
              description: `${terminalEvidence} ${torqueEvidence}${
                navigationState.operation === "stadia_recording"
                  ? " The finalized dataset is available for manual upload."
                  : ""
              }`,
              variant:
                terminalFailed || controlStatus.torque_outcome === "failed"
                  ? "destructive"
                  : "default",
            });
          }
          // Earlier saved episodes can remain usable after the current attempt
          // fails, but the lifecycle failure above is always surfaced first.
          if (terminalFailed && (status.saved_episodes ?? 0) === 0) {
            navigate("/");
            return;
          }
          const datasetInfo = {
            dataset_repo_id:
              navigationState.operation === "stadia_recording"
                ? startedDatasetRepoId
                : status.dataset_repo_id || recordingConfig!.dataset_repo_id,
            single_task: recordingConfig!.single_task,
            num_episodes: recordingConfig!.num_episodes,
            saved_episodes: status.saved_episodes || 0,
            session_elapsed_seconds: status.session_elapsed_seconds || 0,
          };
          navigate("/upload", { state: { datasetInfo } });
        }
      } catch (error) {
        console.error("Error polling recording status:", error);
        setRecordingContractError(
          error instanceof Error ? error.message : "Recording status polling failed.",
        );
      }
    };

    pollStatus();
    const statusInterval = setInterval(pollStatus, 1000);
    return () => clearInterval(statusInterval);
  }, [
    recordingSessionStarted,
    controlSessionId,
    startedDatasetRepoId,
    navigationState,
    recordingConfig,
    navigate,
    baseUrl,
    fetchWithHeaders,
    toast,
    ingestRecordingControlStatus,
  ]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const startRecordingSession = async () => {
    if (!recordingConfig || !navigationState) return;
    try {
      const response = await fetchWithHeaders(`${baseUrl}/start-recording`, {
        method: "POST",
        body: JSON.stringify(recordingConfig),
      });

      const data: unknown = await response.json();

      if (response.ok && isObject(data) && data.success === true) {
        const status = requireControlStatusEnvelope(data, {
          expectedOperation: navigationState.operation,
          expectedTeleoperatorType: navigationState.teleoperator_type,
          requireSuccess: true,
          requireTopLevelSessionId: true,
          requireStatusKey: "status",
        });
        const datasetId = data.dataset_id;
        const controlDatasetRepoId =
          navigationState.operation === "stadia_recording"
            ? stadiaRecordingDatasetId(status)
            : null;
        if (
          typeof datasetId !== "string" ||
          !datasetId.trim() ||
          datasetId.trim() !== datasetId ||
          (navigationState.operation === "stadia_recording" &&
            (!isExactDatasetRepoId(datasetId) ||
              controlDatasetRepoId !== datasetId))
        ) {
          fetchWithHeaders(`${baseUrl}/control-stop`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: status.session_id }),
            keepalive: true,
          }).catch(() => {
            // The bounded server lease remains the backstop if this cleanup
            // request cannot survive navigation.
          });
          throw new Error(
            "The recording start response and control status did not agree on a valid stamped dataset ID.",
          );
        }
        setControlSessionId(status.session_id);
        setStartedDatasetRepoId(datasetId);
        setRecordingContractError(null);
        setRecordingSessionStarted(true);
        toast({
          title: "Recording Started",
          description: `Started recording ${recordingConfig.num_episodes} episodes`,
        });
      } else {
        toast({
          title: "Error Starting Recording",
          description:
            isObject(data) && typeof data.message === "string"
              ? data.message
              : "Failed to start recording session.",
          variant: "destructive",
        });
        navigate("/");
      }
    } catch (error) {
      toast({
        title: "Recording did not start",
        description:
          error instanceof Error
            ? error.message
            : "Could not validate the backend recording session.",
        variant: "destructive",
      });
      navigate("/");
    }
  };

  const handleExitEarly = useCallback(async () => {
    if (!backendStatus?.available_controls.exit_early) return;
    if (controlState !== "running") return;
    if (optimisticPhase !== null) return;

    const realPhase = backendStatus.current_phase;
    const next: Phase | null =
      realPhase === "recording" ? "resetting" :
      realPhase === "resetting" ? "recording" : null;

    if (!next) return;

    setOptimisticPhase(next);

    try {
      if (!controlSessionId || !navigationState) {
        throw new Error("The exact recording session ID is unavailable.");
      }
      const response = await fetchWithHeaders(
        `${baseUrl}/recording-exit-early`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: controlSessionId }),
        }
      );
      const data = await response.json();
      if (!response.ok || data.success !== true) {
        setOptimisticPhase(null);
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
        return;
      }
      requireRecordingCommandSession(
        data,
        controlSessionId,
        navigationState.operation,
      );
    } catch (error) {
      setOptimisticPhase(null);
      toast({
        title: "Episode command failed",
        description:
          error instanceof Error
            ? error.message
            : "Could not validate the recording command.",
        variant: "destructive",
      });
    }
  }, [
    backendStatus,
    controlState,
    optimisticPhase,
    controlSessionId,
    navigationState,
    baseUrl,
    fetchWithHeaders,
    toast,
  ]);

  const handleRerecordEpisode = useCallback(async () => {
    if (!backendStatus?.available_controls.rerecord_episode) return;
    if (controlState !== "running") return;

    try {
      if (!controlSessionId || !navigationState) {
        throw new Error("The exact recording session ID is unavailable.");
      }
      const response = await fetchWithHeaders(
        `${baseUrl}/recording-rerecord-episode`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: controlSessionId }),
        }
      );
      const data = await response.json();

      if (response.ok && data.success === true) {
        requireRecordingCommandSession(
          data,
          controlSessionId,
          navigationState.operation,
        );
        setRerecordTick((t) => t + 1);
        toast({
          title: "Re-recording Episode",
          description: `Episode ${backendStatus.current_episode} will be re-recorded.`,
        });
      } else {
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Re-record command failed",
        description:
          error instanceof Error
            ? error.message
            : "Could not validate the recording command.",
        variant: "destructive",
      });
    }
  }, [
    backendStatus,
    controlState,
    controlSessionId,
    navigationState,
    baseUrl,
    fetchWithHeaders,
    toast,
  ]);

  const handleStopRecording = useCallback(async () => {
    if (!backendStatus?.available_controls.stop_recording) return;
    if (controlState !== "starting" && controlState !== "running") return;
    try {
      await control.requestStop();

      toast({
        title: "Stopping recording",
        description: "Waiting for dataset cleanup, teardown, and torque evidence…",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to stop recording.",
        variant: "destructive",
      });
    }
  }, [backendStatus, controlState, control, toast]);

  const requestStopRecording = useCallback(() => {
    if (!backendStatus?.available_controls.stop_recording) return;
    if (controlState !== "starting" && controlState !== "running") return;
    setShowStopConfirm(true);
  }, [backendStatus, controlState]);

  const confirmStopRecording = useCallback(async () => {
    setShowStopConfirm(false);
    await handleStopRecording();
  }, [handleStopRecording]);

  const handlersRef = useRef({
    handleExitEarly,
    handleRerecordEpisode,
    requestStopRecording,
    showStopConfirm,
  });
  useEffect(() => {
    handlersRef.current = {
      handleExitEarly,
      handleRerecordEpisode,
      requestStopRecording,
      showStopConfirm,
    };
  });

  const sessionReady = recordingSessionStarted && backendStatus !== null;

  useEffect(() => {
    if (!sessionReady) return;

    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === " " || e.code === "Space" || e.key === "ArrowRight") {
        e.preventDefault();
        handlersRef.current.handleExitEarly();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handlersRef.current.handleRerecordEpisode();
      } else if (e.key === "Escape") {
        if (handlersRef.current.showStopConfirm) return;
        handlersRef.current.requestStopRecording();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sessionReady]);

  if (!recordingConfig) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg">No recording configuration found.</p>
          <Button onClick={() => navigate("/")} className="mt-4">
            Return to Home
          </Button>
        </div>
      </div>
    );
  }

  // Show loading state while waiting for backend status
  if (!backendStatus) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center p-6">
        <div className="w-full max-w-2xl space-y-5 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-500 mx-auto mb-4"></div>
          <p className="text-lg">Connecting to recording session...</p>
          {controlSessionId && (
            <ControlSessionPanel
              status={control.status}
              contractError={recordingContractError ?? control.contractError}
              compact
            />
          )}
        </div>
      </div>
    );
  }

  const realPhase = backendStatus.current_phase;
  const currentPhase: BackendPhase = optimisticPhase ?? realPhase;
  const currentEpisode = backendStatus.current_episode ?? 1;
  const totalEpisodes =
    backendStatus.total_episodes ?? recordingConfig.num_episodes;

  const phaseElapsedTime = optimisticPhase
    ? 0
    : backendStatus.phase_elapsed_seconds || 0;
  const phaseTimeLimit =
    currentPhase === "recording"
      ? recordingConfig.episode_time_s
      : currentPhase === "resetting"
      ? recordingConfig.reset_time_s
      : backendStatus.phase_time_limit_s || 0;

  const sessionElapsedTime = backendStatus.session_elapsed_seconds || 0;

  const getStatusText = () => {
    if (currentPhase === "recording") return `RECORDING EPISODE ${currentEpisode}`;
    if (currentPhase === "resetting") return "RESET — GET READY";
    if (currentPhase === "recovery") return "CONTROLLER RECOVERY — HOLD";
    if (currentPhase === "preparing") return "PREPARING SESSION";
    if (currentPhase === "stopping") return "STOPPING — CLEANING UP";
    if (currentPhase === "error") return "SESSION ERROR";
    return "SESSION COMPLETE";
  };

  const phaseColor =
    currentPhase === "recording"
      ? { dot: "bg-red-500", pill: "bg-red-500/15 text-red-300", timer: "text-green-400", bar: "bg-green-500", button: "bg-green-500 hover:bg-green-600" }
      : currentPhase === "resetting"
      ? { dot: "bg-orange-500", pill: "bg-orange-500/15 text-orange-300", timer: "text-orange-400", bar: "bg-orange-500", button: "bg-orange-500 hover:bg-orange-600" }
      : currentPhase === "recovery"
      ? { dot: "bg-amber-600", pill: "bg-amber-500/15 text-amber-300", timer: "text-amber-400", bar: "bg-amber-600", button: "bg-amber-700" }
      : currentPhase === "error"
      ? { dot: "bg-red-600", pill: "bg-red-500/15 text-red-300", timer: "text-red-400", bar: "bg-red-600", button: "bg-red-700" }
      : currentPhase === "stopping"
      ? { dot: "bg-orange-600", pill: "bg-orange-500/15 text-orange-300", timer: "text-orange-400", bar: "bg-orange-600", button: "bg-orange-700" }
      : { dot: "bg-gray-500", pill: "bg-gray-500/15 text-gray-300", timer: "text-gray-400", bar: "bg-gray-500", button: "bg-gray-500" };

  const primaryLabel =
    currentPhase === "recording"
      ? "End Episode"
      : currentPhase === "resetting"
      ? "Start Next Episode"
      : currentPhase === "stopping"
      ? "Stopping"
      : currentPhase === "recovery"
      ? "Controller recovery"
      : currentPhase === "error"
      ? "Unavailable"
      : currentPhase === "completed"
      ? "Complete"
      : "Preparing";

  const PrimaryIcon = currentPhase === "recording" ? SkipForward : Play;
  const stadiaDatasetBlocked =
    navigationState?.operation === "stadia_recording" &&
    backendStatus.session_ended &&
    stadiaDatasetDisposition(backendStatus) === "blocked";
  const cameraPreviewAllowed =
    navigationState?.operation === "leader_recording";

  return (
    <div className="h-screen bg-black text-white p-6 flex flex-col overflow-hidden">
      <div className="max-w-7xl w-full mx-auto flex-1 min-h-0 flex flex-col">
        <div className="mb-3 flex-shrink-0">
          <Button
            onClick={() =>
              backendStatus.recording_active
                ? requestStopRecording()
                : navigate("/")
            }
            variant="outline"
            className="border-gray-500 hover:border-gray-200 text-gray-300 hover:text-white"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </div>

        <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 flex-1 min-h-0 flex flex-col justify-center">
          <div className="mb-3 max-h-64 flex-shrink-0 overflow-y-auto">
            <ControlSessionPanel
              status={control.status}
              contractError={recordingContractError ?? control.contractError}
            />
          </div>
          <div className="flex justify-end items-center gap-4 mb-3 flex-shrink-0 text-sm text-gray-400">
            <span aria-label={`Episode ${currentEpisode} of ${totalEpisodes}`}>
              Episode <span className="text-white font-semibold">{currentEpisode}</span> / {totalEpisodes}
            </span>
            <span className="font-mono" aria-label={`Total session time ${formatTime(sessionElapsedTime)}`}>
              {formatTime(sessionElapsedTime)}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleMute}
              aria-label={muted ? "Unmute" : "Mute"}
              className="h-8 w-8 text-gray-400 hover:text-white hover:bg-gray-800"
            >
              {muted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white hover:bg-gray-800"
                  aria-label="More actions"
                >
                  <MoreHorizontal className="w-5 h-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                onCloseAutoFocus={(e) => e.preventDefault()}
                className="bg-gray-900 border-gray-700 text-white"
              >
                <DropdownMenuItem
                  onClick={handleRerecordEpisode}
                  disabled={
                    controlState !== "running" ||
                    !backendStatus.available_controls.rerecord_episode
                  }
                  className="focus:bg-gray-800 focus:text-white"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Re-record episode
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={requestStopRecording}
                  disabled={
                    (controlState !== "starting" && controlState !== "running") ||
                    !backendStatus.available_controls.stop_recording
                  }
                  className="text-red-400 focus:bg-gray-800 focus:text-red-300"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop recording
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="text-center mb-3 flex-shrink-0">
            <div
              role="status"
              aria-live="polite"
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${phaseColor.pill}`}
            >
              <span
                className={`w-2 h-2 rounded-full ${phaseColor.dot} ${
                  backendStatus.recording_active ? "animate-pulse" : ""
                }`}
              />
              {getStatusText()}
            </div>
          </div>

          {/* Camera windows for the cameras configured for this session. Shown
              from the preparing phase so the slots are laid out immediately;
              each fills with its live feed once the robot is connected and
              recording/resetting. This is the flexible region: it absorbs the
              space left after the fixed chrome, and cameraWindow sizes each feed
              as large as fits so the whole page stays within one screen. */}
          {backendStatus.cameras &&
            backendStatus.cameras.length > 0 &&
            cameraPreviewAllowed &&
            !backendStatus.session_ended && (
              <div
                ref={cameraAreaRef}
                className="flex-1 min-h-0 flex flex-wrap gap-3 justify-center content-center overflow-hidden mb-3"
              >
                {backendStatus.cameras.map((name) => (
                  <CameraFeed
                    key={name}
                    baseUrl={baseUrl}
                    name={name}
                    live={
                      currentPhase === "recording" ||
                      currentPhase === "resetting"
                    }
                    width={cameraWindow.width}
                    height={cameraWindow.height}
                  />
                ))}
              </div>
            )}

          <div className="text-center mb-3 flex-shrink-0">
            <div className={`text-7xl font-mono font-bold leading-none ${phaseColor.timer}`}>
              {formatTime(phaseElapsedTime)}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              / {formatTime(phaseTimeLimit)}
            </div>
          </div>

          <div className="w-full bg-gray-800 rounded-full h-1.5 mb-4 flex-shrink-0">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${phaseColor.bar}`}
              style={{
                width: `${
                  phaseTimeLimit > 0
                    ? Math.min((phaseElapsedTime / phaseTimeLimit) * 100, 100)
                    : 0
                }%`,
              }}
            />
          </div>

          <Button
            onClick={handleExitEarly}
            disabled={
              !backendStatus.available_controls.exit_early ||
              controlState !== "running" ||
              optimisticPhase !== null ||
              (currentPhase !== "recording" && currentPhase !== "resetting")
            }
            className={`w-full flex-shrink-0 text-white font-semibold py-6 text-lg disabled:opacity-50 ${phaseColor.button}`}
          >
            <PrimaryIcon className="w-5 h-5 mr-2" />
            {primaryLabel}
            {currentPhase !== "completed" && (
              <span className="ml-3 px-2 py-0.5 rounded text-xs font-mono bg-black/30 text-white/70">SPACE / →</span>
            )}
          </Button>

          {backendStatus.session_ended && (
            <p className="text-center text-sm text-gray-400 mt-6">
              {stadiaDatasetBlocked
                ? "Recording ended — upload is blocked because dataset safety, finalization, or upload disposition was not proven."
                : currentPhase === "error"
                ? "Recording ended with an error. The terminal dataset disposition is being applied."
                : "Recording complete — applying the terminal dataset disposition…"}
            </p>
          )}
        </div>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop recording?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              The server will end the current attempt, finalize only a proven-safe
              dataset, and complete teardown. Upload remains blocked if a Stadia
              dataset is poisoned, unsafe, or unfinalized.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              Keep recording
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmStopRecording}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              Stop
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

interface CameraFeedProps {
  baseUrl: string;
  name: string;
  // False during the preparing phase: show an empty slot until the camera is
  // connected and streaming. True once recording/resetting, when frames flow.
  live: boolean;
  // Pixel size computed by the parent to fit the available space. The box is
  // sized to the camera's aspect ratio, so the video fills it without letterbox.
  width: number;
  height: number;
}

// Renders one recording camera's window at an explicit size. During preparing it
// shows an empty placeholder; once live it plays the backend MJPEG stream. The
// browser renders a `multipart/x-mixed-replace` response natively in an <img>,
// so we just point it at /camera-feed/{name}. If the stream errors before frames
// flow (camera still warming up), retry with a cache-busting key after a delay.
const CameraFeed: React.FC<CameraFeedProps> = ({
  baseUrl,
  name,
  live,
  width,
  height,
}) => {
  const [reloadKey, setReloadKey] = useState(0);
  const [hasError, setHasError] = useState(false);
  const retryRef = useRef<number | null>(null);

  const src = `${baseUrl}/camera-feed/${encodeURIComponent(name)}?k=${reloadKey}`;

  useEffect(() => {
    return () => {
      if (retryRef.current) window.clearTimeout(retryRef.current);
    };
  }, []);

  const handleError = useCallback(() => {
    setHasError(true);
    if (retryRef.current) window.clearTimeout(retryRef.current);
    retryRef.current = window.setTimeout(() => {
      setHasError(false);
      setReloadKey((k) => k + 1);
    }, 1500);
  }, []);

  // 0 before the first measurement; skip rendering a zero-size box.
  if (width <= 0 || height <= 0) return null;

  return (
    <div
      style={{ width, height }}
      className="relative bg-gray-900 rounded-lg border border-gray-700 overflow-hidden flex items-center justify-center"
    >
      {!live ? (
        <span className="text-gray-500 text-sm">Getting ready…</span>
      ) : hasError ? (
        <span className="text-gray-500 text-sm">Connecting feed…</span>
      ) : (
        <img
          src={src}
          alt={`${name} live feed`}
          onError={handleError}
          className="w-full h-full object-cover"
        />
      )}
      <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-sm text-gray-200">
        {name}
      </span>
    </div>
  );
};

export default Recording;
