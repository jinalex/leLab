import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import ControlSessionPanel from "@/components/control/ControlSessionPanel";
import { useToast } from "@/hooks/use-toast";
import { useControlSession } from "@/hooks/useControlSession";
import type { RobotOperation, TeleoperatorType } from "@/lib/robotConfig";

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
  const navigation = parseNavigationState(location.state);
  const [leaveAfterStop, setLeaveAfterStop] = useState(false);
  const terminalToastRef = useRef(false);

  const control = useControlSession({
    sessionId: navigation?.session_id ?? null,
    expectedOperation:
      (navigation?.operation ?? "leader_teleoperation") as RobotOperation,
    expectedTeleoperatorType: navigation?.teleoperator_type ?? null,
  });

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

  if (!navigation) return null;

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      <div className="w-full h-[95vh] flex">
        <VisualizerPanel
          onGoBack={() => void handleGoBack()}
          className="lg:w-full"
          rightSlot={
            <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
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
