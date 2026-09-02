import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Loader2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import Logo from "@/components/Logo";
import ControlSessionPanel from "@/components/control/ControlSessionPanel";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useControlSession } from "@/hooks/useControlSession";
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
import {
  InferenceStatus,
  getInferenceStatus,
} from "@/lib/inferenceApi";
import type { TeleoperatorType } from "@/lib/robotConfig";

const POLL_MS = 1000;

interface InferenceNavigationState {
  session_id: string;
  teleoperator_type: TeleoperatorType;
}

const parseNavigationState = (value: unknown): InferenceNavigationState | null => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  if (
    Object.keys(raw).length !== 2 ||
    !Object.prototype.hasOwnProperty.call(raw, "session_id") ||
    !Object.prototype.hasOwnProperty.call(raw, "teleoperator_type") ||
    typeof raw.session_id !== "string" ||
    !raw.session_id.trim() ||
    raw.session_id.trim() !== raw.session_id ||
    (raw.teleoperator_type !== "leader_arm" && raw.teleoperator_type !== "stadia")
  ) {
    return null;
  }
  return raw as unknown as InferenceNavigationState;
};

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

const Inference: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigation = parseNavigationState(location.state);
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [compatibilityError, setCompatibilityError] = useState<string | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const navigatedAwayRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip navigatedAwayRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);
  const connectionErrorShownRef = useRef(false);
  const control = useControlSession({
    sessionId: navigation?.session_id ?? null,
    expectedOperation: "inference",
    expectedTeleoperatorType: navigation?.teleoperator_type ?? null,
    renewalBlocked: compatibilityError !== null,
  });
  const requestControlStop = control.requestStop;
  const ingestInferenceControlStatus = control.ingestCompatibilityStatus;

  useEffect(() => {
    if (navigation) return;
    toast({
      title: "Inference session unavailable",
      description: "Start inference from a job with a selected robot so its exact session can be verified.",
      variant: "destructive",
    });
    navigate("/", { replace: true });
  }, [navigation, navigate, toast]);

  useEffect(() => {
    if (!navigation) return;
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await requestControlStop();
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        const controlUpdate = ingestInferenceControlStatus(next);
        if (controlUpdate.stale) return;
        const controlStatus = controlUpdate.status;
        const controlTerminal =
          controlStatus.state === "stopped" || controlStatus.state === "error";
        if (next.inference_active !== !controlTerminal) {
          throw new Error(
            "Inference activity conflicts with the exact control lifecycle.",
          );
        }
        setCompatibilityError(null);
        connectionErrorShownRef.current = false;
        setStatus(next);
        // Auto-bounce home once the run is done.
        if (
          !next.inference_active &&
          controlTerminal &&
          !navigatedAwayRef.current
        ) {
          navigatedAwayRef.current = true;
          const inferenceFailed =
            controlStatus.state === "error" ||
            next.outcome === "failed" ||
            Boolean(next.error) ||
            (next.exited === true && next.exit_code !== 0);
          const lifecycleEvidence = [
            `Control state: ${controlStatus.state}.`,
            controlStatus.stop_reason
              ? `Stop reason: ${controlStatus.stop_reason}.`
              : "Stop reason: not reported.",
            next.outcome ? `Inference outcome: ${next.outcome}.` : null,
            next.error ? `Inference error: ${next.error}.` : null,
            next.exited
              ? `Process exit code: ${next.exit_code ?? "not reported"}.`
              : "Process exit was not reported.",
            next.log_path ? `Log: ${next.log_path}.` : null,
          ]
            .filter((entry): entry is string => entry !== null)
            .join(" ");
          toast({
            title: inferenceFailed
              ? "Inference ended with an error"
              : next.outcome === "ran_with_warning"
                ? "Inference finished with a warning"
                : "Inference finished",
            description: lifecycleEvidence,
            variant: inferenceFailed ? "destructive" : "default",
          });
          toast({
            title: "Inference session ended",
            description:
              controlStatus.torque_outcome === "verified_off"
                ? inferenceFailed
                  ? "Torque is verified off across all six motors. That proves torque state only; this error outcome does not prove that every resource closed cleanly."
                  : "Torque is verified off across all six motors."
                : `Torque outcome is ${controlStatus.torque_outcome}; do not assume the follower is safe to handle.`,
            variant:
              inferenceFailed || controlStatus.torque_outcome === "failed"
                ? "destructive"
                : "default",
          });
          navigate("/");
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        if (
          next.inference_active &&
          controlStatus.state === "running" &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          const message = e instanceof Error ? e.message : String(e);
          setCompatibilityError(message);
          if (!connectionErrorShownRef.current) {
            connectionErrorShownRef.current = true;
            toast({
              title: "Inference status unavailable",
              description: message,
              variant: "destructive",
            });
          }
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [
    navigation,
    baseUrl,
    fetchWithHeaders,
    navigate,
    toast,
    requestControlStop,
    ingestInferenceControlStatus,
  ]);

  const handleStop = async () => {
    setShowStopConfirm(false);
    try {
      if (!navigation) throw new Error("The exact inference session ID is unavailable.");
      await requestControlStop();
      // Status poll will catch the inactive state and navigate home.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  if (!navigation) return null;

  if (!status) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center p-6">
        <div className="w-full max-w-2xl space-y-4">
          <div className="flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to inference…
          </div>
          <ControlSessionPanel
            status={control.status}
            contractError={compatibilityError ?? control.contractError}
            compact
          />
        </div>
      </div>
    );
  }

  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isStopping = control.status?.state === "stopping";
  const isSettingUp =
    status.inference_active &&
    !isStopping &&
    status.rollout_started_at == null;
  const isRunning =
    status.inference_active &&
    !isStopping &&
    status.rollout_started_at != null;
  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillLabel = isSettingUp
    ? "SETTING UP"
    : isStopping
    ? "STOPPING"
    : isRunning
    ? "RUNNING"
    : "FINISHED";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  return (
    <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
      <div className="flex items-center gap-4 mb-8">
        <Button
          variant="ghost"
          size="icon"
          onClick={() =>
            status.inference_active
              ? setShowStopConfirm(true)
              : navigate("/")
          }
          disabled={
            status.inference_active && control.status?.state === "stopping"
          }
          className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <Logo />
        <h1 className="font-bold text-white text-2xl">Inference</h1>
      </div>

      <div className="flex-1 flex items-center justify-center">
        <div className="bg-gray-900 rounded-lg border border-gray-700 p-8 w-full max-w-xl">
          <div className="mb-6 max-h-80 overflow-y-auto text-left">
            <ControlSessionPanel
              status={control.status}
              contractError={compatibilityError ?? control.contractError}
            />
          </div>
          <div className="text-center mb-6">
            <div
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${
                isSettingUp
                  ? "bg-amber-500/15 text-amber-300"
                  : isStopping
                  ? "bg-orange-500/15 text-orange-300"
                  : "bg-green-500/15 text-green-300"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isSettingUp
                    ? "bg-amber-500"
                    : isStopping
                    ? "bg-orange-500"
                    : "bg-green-500"
                } ${status.inference_active ? "animate-pulse" : ""}`}
              />
              {pillLabel}
            </div>
          </div>

          <div className="text-center mb-4">
            <div
              className={`text-7xl font-mono font-bold leading-none ${
                isSettingUp
                  ? "text-amber-400"
                  : isStopping
                  ? "text-orange-400"
                  : "text-green-400"
              }`}
            >
              {formatTime(timerSeconds)}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              {isSettingUp
                ? "Loading policy & connecting hardware…"
                : isStopping
                ? "Waiting for teardown and torque evidence…"
                : `/ ${formatTime(duration)}`}
            </div>
          </div>

          <div className="w-full bg-gray-800 rounded-full h-1.5 mb-8">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${
                isSettingUp
                  ? "bg-amber-500/40 animate-pulse w-full"
                  : isStopping
                  ? "bg-orange-500/40 animate-pulse w-full"
                  : "bg-green-500"
              }`}
              style={
                isSettingUp || isStopping ? undefined : { width: `${pct}%` }
              }
            />
          </div>

          <div className="text-xs text-slate-500 break-all mb-6">
            policy: {status.policy_ref ?? "(unknown)"}
          </div>

          <Button
            onClick={() => setShowStopConfirm(true)}
            disabled={
              !status.inference_active ||
              (control.status?.state !== "starting" &&
                control.status?.state !== "running")
            }
            className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-6 text-lg disabled:opacity-50"
          >
            <Square className="w-5 h-5 mr-2" />
            Stop
          </Button>
        </div>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop inference?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              The server will request hold and teardown. This page stays active
              until terminal torque evidence is available.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              Keep running
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleStop}
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

export default Inference;
