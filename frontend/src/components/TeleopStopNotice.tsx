import { useEffect } from "react";
import { useToast } from "@/hooks/use-toast";

const FLAG = "lelab:teleop-stopped";

/**
 * One-time confirmation that teleoperation was stopped during the previous
 * page's unload. Historical builds set this flag before knowing whether
 * teardown completed, so it can only confirm that a stop was requested.
 */
const TeleopStopNotice = () => {
  const { toast } = useToast();

  useEffect(() => {
    let stopped = false;
    try {
      stopped = sessionStorage.getItem(FLAG) === "1";
      if (stopped) sessionStorage.removeItem(FLAG);
    } catch {
      /* sessionStorage unavailable — nothing to show */
    }
    if (stopped) {
      toast({
        title: "Teleoperation stop was requested",
        description:
          "A page-unload request is best effort. Re-check server status before assuming torque is off.",
      });
    }
  }, [toast]);

  return null;
};

export default TeleopStopNotice;
