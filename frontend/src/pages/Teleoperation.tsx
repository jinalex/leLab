import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import ControlSessionPanel from "@/components/control/ControlSessionPanel";
import { useToast } from "@/hooks/use-toast";
import { useControlSession } from "@/hooks/useControlSession";
import { useApi } from "@/contexts/ApiContext";
import { Button } from "@/components/ui/button";
import type { ControlStatus, RobotOperation, TeleoperatorType } from "@/lib/robotConfig";

const MIN_STADIA_SPEED = 0.25;
const MAX_STADIA_SPEED = 2;
const STADIA_SPEED_STEP = 0.25;

const statusSpeed = (status: ControlStatus | null): number | null => {
  const value = status?.details.stadia_speed_multiplier;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

const statusEffectiveStep = (status: ControlStatus | null): number | null => {
  const value = status?.details.stadia_effective_max_step_per_tick;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
};

interface TeleoperationNavigationState {
  session_id: string;
  operation: "leader_teleoperation" | "stadia_teleoperation";
  teleoperator_type: TeleoperatorType;
}

const parseNavigationState = (value: unknown): TeleoperationNavigationState | null => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  if (
    Object.keys(raw).length !== 3 ||
    !Object.prototype.hasOwnProperty.call(raw, "session_id") ||
    !Object.prototype.hasOwnProperty.call(raw, "operation") ||
    !Object.prototype.hasOwnProperty.call(raw, "teleoperator_type") ||
    typeof raw.session_id !== "string" ||
    !raw.session_id.trim() ||
    raw.session_id.trim() !== raw.session_id ||
    (raw.operation !== "leader_teleoperation" && raw.operation !== "stadia_teleoperation") ||
    (raw.teleoperator_type !== "leader_arm" && raw.teleoperator_type !== "stadia")
  ) {
    return null;
  }
  if (
    (raw.operation === "leader_teleoperation") !==
    (raw.teleoperator_type === "leader_arm")
  ) {
    return null;
  }
  return raw as unknown as TeleoperationNavigationState;
};

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const navigation = parseNavigationState(location.state);
  const [leaveAfterStop, setLeaveAfterStop] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [speedPending, setSpeedPending] = useState(false);
  const [speedEditing, setSpeedEditing] = useState(false);
  const terminalToastRef = useRef(false);

  const control = useControlSession({
    sessionId: navigation?.session_id ?? null,
    expectedOperation:
      (navigation?.operation ?? "leader_teleoperation") as RobotOperation,
    expectedTeleoperatorType: navigation?.teleoperator_type ?? null,
  });

  useEffect(() => {
    const reported = statusSpeed(control.status);
    if (reported !== null && !speedPending && !speedEditing) setSpeedMultiplier(reported);
  }, [control.status, speedPending, speedEditing]);

  useEffect(() => {
    if (navigation) return;
    toast({
      title: "Control session unavailable",
      description: "Start teleoperation from a configured robot so its exact session can be verified.",
      variant: "destructive",
    });
    navigate("/", { replace: true });
  }, [navigation, navigate, toast]);

  useEffect(() => {
    if (!control.status || !control.terminal || terminalToastRef.current) return;
    terminalToastRef.current = true;
    const terminalError = control.status.state === "error";
    const lifecycleEvidence = `Control state: ${control.status.state}. ${
      control.status.stop_reason
        ? `Stop reason: ${control.status.stop_reason}.`
        : "Stop reason: not reported."
    }`;
    const torqueEvidence =
      control.status.torque_outcome === "verified_off"
        ? terminalError
          ? "Torque is verified off across all six motors. That proves torque state only; this error outcome does not prove that every resource closed cleanly."
          : "Torque is verified off across all six motors."
        : `The session is terminal, but torque outcome is ${control.status.torque_outcome}. Do not assume the follower is safe to handle or that every resource closed cleanly.`;
    toast({
      title: terminalError
        ? "Teleoperation ended with an error"
        : "Teleoperation stopped",
      description: `${lifecycleEvidence} ${torqueEvidence}`,
      variant:
        terminalError || control.status.torque_outcome === "failed"
          ? "destructive"
          : "default",
    });
    if (leaveAfterStop) navigate("/");
  }, [control.status, control.terminal, leaveAfterStop, navigate, toast]);

  const handleGoBack = useCallback(async () => {
    if (!navigation) {
      navigate("/");
      return;
    }
    if (control.terminal) {
      navigate("/");
      return;
    }
    if (control.status?.state === "stopping") {
      setLeaveAfterStop(true);
      toast({
        title: "Teleoperation is stopping",
        description: "Waiting for terminal teardown and torque evidence.",
      });
      return;
    }
    setLeaveAfterStop(true);
    try {
      const next = await control.requestStop();
      toast({
        title: "Stopping teleoperation",
        description:
          next.state === "stopping"
            ? "Waiting for teardown and torque evidence."
            : `Server reported ${next.state}; waiting for terminal evidence.`,
      });
    } catch (error) {
      toast({
        title: "Stop request failed",
        description:
          error instanceof Error ? error.message : "The exact control session could not be stopped.",
        variant: "destructive",
      });
      setLeaveAfterStop(false);
    }
  }, [navigation, control, navigate, toast]);

  const applySpeed = useCallback(
    async (requested: number) => {
      if (!navigation || navigation.teleoperator_type !== "stadia") return;
      const multiplier = Math.max(
        MIN_STADIA_SPEED,
        Math.min(MAX_STADIA_SPEED, Math.round(requested / STADIA_SPEED_STEP) * STADIA_SPEED_STEP),
      );
      const confirmed = statusSpeed(control.status);
      if (confirmed === multiplier) {
        setSpeedMultiplier(multiplier);
        return;
      }
      setSpeedPending(true);
      try {
        const response = await fetchWithHeaders(`${baseUrl}/control-speed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: navigation.session_id,
            multiplier,
          }),
        });
        const data: unknown = await response.json();
        if (!response.ok) {
          const message =
            typeof (data as { message?: unknown })?.message === "string"
              ? (data as { message: string }).message
              : "Stadia speed could not be changed.";
          throw new Error(message);
        }
        const adopted = control.ingestStatusEnvelope(data).status;
        const reported = statusSpeed(adopted);
        if (reported === null || reported !== multiplier) {
          throw new Error("The backend did not confirm the requested Stadia speed.");
        }
        setSpeedMultiplier(reported);
      } catch (error) {
        setSpeedMultiplier(statusSpeed(control.status) ?? 1);
        toast({
          title: "Speed unchanged",
          description:
            error instanceof Error ? error.message : "Stadia speed could not be changed.",
          variant: "destructive",
        });
      } finally {
        setSpeedPending(false);
      }
    },
    [navigation, control, fetchWithHeaders, baseUrl, toast],
  );

  if (!navigation) return null;

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      <div className="w-full h-[95vh] flex">
        <VisualizerPanel
          onGoBack={() => void handleGoBack()}
          className="lg:w-full"
          rightSlot={
            <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
              {navigation.teleoperator_type === "stadia" && (
                <div className="rounded-lg border border-slate-700 bg-slate-900/80 p-4 text-white">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-slate-100">Stadia speed</div>
                      <div className="text-xs text-slate-400">
                        Release RB before changing speed.
                      </div>
                    </div>
                    <div className="font-mono text-lg text-purple-300">
                      {speedMultiplier.toFixed(2)}×
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-3">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      aria-label="Decrease Stadia speed"
                      disabled={
                        speedPending ||
                        control.status?.state !== "running" ||
                        control.status.motion_state === "enabled" ||
                        speedMultiplier <= MIN_STADIA_SPEED
                      }
                      onClick={() => void applySpeed(speedMultiplier - STADIA_SPEED_STEP)}
                      className="border-slate-600 bg-slate-800"
                    >
                      −
                    </Button>
                    <input
                      aria-label="Stadia speed multiplier"
                      type="range"
                      min={MIN_STADIA_SPEED}
                      max={MAX_STADIA_SPEED}
                      step={STADIA_SPEED_STEP}
                      value={speedMultiplier}
                      disabled={
                        speedPending ||
                        control.status?.state !== "running" ||
                        control.status.motion_state === "enabled"
                      }
                      onPointerDown={() => setSpeedEditing(true)}
                      onChange={(event) => setSpeedMultiplier(Number(event.target.value))}
                      onPointerUp={(event) => {
                        setSpeedEditing(false);
                        void applySpeed(Number(event.currentTarget.value));
                      }}
                      onKeyDown={() => setSpeedEditing(true)}
                      onKeyUp={(event) => {
                        setSpeedEditing(false);
                        void applySpeed(Number(event.currentTarget.value));
                      }}
                      className="h-2 flex-1 cursor-pointer accent-purple-500 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      aria-label="Increase Stadia speed"
                      disabled={
                        speedPending ||
                        control.status?.state !== "running" ||
                        control.status.motion_state === "enabled" ||
                        speedMultiplier >= MAX_STADIA_SPEED
                      }
                      onClick={() => void applySpeed(speedMultiplier + STADIA_SPEED_STEP)}
                      className="border-slate-600 bg-slate-800"
                    >
                      +
                    </Button>
                  </div>
                  <div className="mt-2 flex justify-between text-[11px] text-slate-500">
                    <span>{MIN_STADIA_SPEED.toFixed(2)}×</span>
                    <span>
                      {statusEffectiveStep(control.status)?.toFixed(2) ?? "—"}° / pp per tick
                    </span>
                    <span>{MAX_STADIA_SPEED.toFixed(2)}×</span>
                  </div>
                </div>
              )}
              <ControlSessionPanel
                status={control.status}
                contractError={control.contractError}
                compact
              />
              <div className="min-h-64 flex-1">
                <TeleopCameraPanel />
              </div>
            </div>
          }
        />
      </div>
    </div>
  );
};

export default TeleoperationPage;
