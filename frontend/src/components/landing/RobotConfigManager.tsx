import React from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import type { RobotRecord } from "@/hooks/useRobots";
import {
  readinessFor,
  requireControlStatusEnvelope,
  teleoperationOperation,
} from "@/lib/robotConfig";
import RobotTile from "./RobotTile";

interface RobotConfigManagerProps {
  selectedName: string | null;
  selectedRecord: RobotRecord | null;
  availableNames: string[];
  isLoading: boolean;
  loadError: string | null;
  selectRobot: (name: string) => void;
  createRobot: (name: string) => Promise<boolean>;
  deleteRobot: (name: string) => Promise<boolean>;
}

const RobotConfigManager: React.FC<RobotConfigManagerProps> = ({
  selectedName,
  selectedRecord,
  availableNames,
  isLoading,
  loadError,
  selectRobot,
  createRobot,
  deleteRobot,
}) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const handleConfigure = (name: string) => {
    navigate("/calibration", { state: { robot_name: name } });
  };

  const handleTeleop = async (robot: RobotRecord) => {
    const operation = teleoperationOperation(robot);
    const readiness = readinessFor(robot, operation);
    if (!readiness.ready) {
      toast({
        title: "Robot not ready",
        description:
          readiness.issues.map((issue) => issue.message).join(" ") ||
          "Complete the saved robot configuration before starting.",
        variant: "destructive",
      });
      return;
    }
    try {
      const res = await fetchWithHeaders(`${baseUrl}/move-arm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ robot_name: robot.name }),
      });
      const data = await res.json();
      if (res.ok && data.success === true) {
        const status = requireControlStatusEnvelope(data, {
          expectedOperation: operation,
          expectedTeleoperatorType: robot.teleoperator_type,
          requireSuccess: true,
          requireTopLevelSessionId: true,
          requireStatusKey: "status",
        });
        toast({
          title: "Teleoperation Started",
          description: data.message || `Started teleoperation for ${robot.name}.`,
        });
        navigate("/teleoperation", {
          state: {
            session_id: status.session_id,
            operation,
            teleoperator_type: robot.teleoperator_type,
          },
        });
      } else {
        toast({
          title: "Error Starting Teleoperation",
          description: data.message || "Failed to start.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Teleoperation did not start",
        description:
          error instanceof Error
            ? error.message
            : "Could not validate the backend control session.",
        variant: "destructive",
      });
    }
  };

  return (
    <RobotTile
      robot={selectedRecord}
      selectedName={selectedName}
      availableNames={availableNames}
      isLoading={isLoading}
      loadError={loadError}
      onSelect={selectRobot}
      onCreateNew={createRobot}
      onConfigure={handleConfigure}
      onTeleop={handleTeleop}
      onDelete={deleteRobot}
    />
  );
};

export default RobotConfigManager;
