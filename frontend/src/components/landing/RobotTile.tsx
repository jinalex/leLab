import React, { useState } from "react";
import { Settings, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { RobotRecord } from "@/hooks/useRobots";
import {
  readinessFor,
  teleoperationOperation,
} from "@/lib/robotConfig";
import RobotSelector from "./RobotSelector";
import StadiaStartupInstructions from "@/components/control/StadiaStartupInstructions";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface RobotTileProps {
  robot: RobotRecord | null;
  selectedName: string | null;
  availableNames: string[];
  isLoading: boolean;
  loadError: string | null;
  onSelect: (name: string) => void;
  onCreateNew: (name: string) => Promise<boolean>;
  onConfigure: (name: string) => void;
  onTeleop: (robot: RobotRecord) => void;
  onDelete: (name: string) => void;
  onStadiaSpeedChange: (name: string, speedMultiplier: number) => Promise<boolean>;
}

const RobotTile: React.FC<RobotTileProps> = ({
  robot,
  selectedName,
  availableNames,
  isLoading,
  loadError,
  onSelect,
  onCreateNew,
  onConfigure,
  onTeleop,
  onDelete,
  onStadiaSpeedChange,
}) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmStadiaTeleop, setConfirmStadiaTeleop] = useState(false);
  const [speedPending, setSpeedPending] = useState(false);
  const teleopReadiness = robot
    ? readinessFor(robot, teleoperationOperation(robot))
    : null;
  const teleopDisabled = !teleopReadiness?.ready;
  const status = robot
    ? teleopReadiness?.ready
      ? "Ready"
      : "Needs configuration"
    : null;
  const modeLabel =
    robot?.teleoperator_type === "stadia" ? "Stadia" : "Leader arm";
  const unavailableReason = teleopReadiness?.issues
    .map((issue) => issue.message)
    .join(" ");
  const globalSpeedOptions = Array.from(
    new Set([0.5, 1, 2, 3, 4, 5, robot?.stadia.speed_multiplier]),
  )
    .filter((speed): speed is number => typeof speed === "number")
    .sort((left, right) => left - right);

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 flex flex-col gap-2 relative">
      {loadError && (
        <Alert className="border-red-700 bg-red-950/50 text-red-100">
          <AlertDescription>
            Robot list unavailable: {loadError}
          </AlertDescription>
        </Alert>
      )}
      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <RobotSelector
            selectedName={selectedName}
            availableNames={availableNames}
            onSelect={onSelect}
            onCreateNew={onCreateNew}
            isLoading={isLoading}
          />
        </div>
        {status && (
          <p
            className={`text-xs truncate shrink-0 ${
              teleopReadiness?.ready ? "text-green-400" : "text-amber-400"
            }`}
          >
            {modeLabel} · {status}
          </p>
        )}
        {robot && (
          <div className="flex items-center gap-1 shrink-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-gray-300 hover:text-white"
                  onClick={() => onConfigure(robot.name)}
                  aria-label="Configure"
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Configure (calibrate)</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-red-400 hover:text-red-300 hover:bg-red-900/20"
                  onClick={() => setConfirmDelete(true)}
                  aria-label="Delete robot"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete robot config</TooltipContent>
            </Tooltip>
          </div>
        )}
      </div>

      {robot && (
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="w-full">
              <Button
                onClick={() => {
                  if (robot.teleoperator_type === "stadia") {
                    setConfirmStadiaTeleop(true);
                    return;
                  }
                  onTeleop(robot);
                }}
                disabled={teleopDisabled || speedPending}
                className={`w-full ${
                  teleopDisabled
                    ? "bg-red-500/30 hover:bg-red-500/30 text-red-200 cursor-not-allowed"
                    : "bg-yellow-500 hover:bg-yellow-600 text-white"
                }`}
              >
                Teleoperation
              </Button>
            </div>
          </TooltipTrigger>
          {teleopDisabled && (
            <TooltipContent>
              {unavailableReason || "Configure the robot first."}
            </TooltipContent>
          )}
        </Tooltip>
      )}

      {robot?.teleoperator_type === "stadia" && (
        <div className="flex items-center justify-between gap-3 rounded-md border border-purple-800/60 bg-purple-950/30 px-3 py-2">
          <div>
            <div className="text-xs font-medium text-purple-100">Global Stadia speed</div>
            <div className="text-[11px] text-purple-300">Teleoperation + recording · next session</div>
          </div>
          <Select
            value={String(robot.stadia.speed_multiplier)}
            disabled={speedPending}
            onValueChange={async (value) => {
              setSpeedPending(true);
              await onStadiaSpeedChange(robot.name, Number(value));
              setSpeedPending(false);
            }}
          >
            <SelectTrigger
              aria-label="Global Stadia speed"
              className="h-8 w-24 border-purple-700 bg-slate-900 text-purple-100"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {globalSpeedOptions.map((speed) => (
                <SelectItem key={speed} value={String(speed)}>
                  {speed}×
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {robot?.teleoperator_type === "stadia" && (
        <Dialog open={confirmStadiaTeleop} onOpenChange={setConfirmStadiaTeleop}>
          <DialogContent className="bg-gray-900 border-gray-800 text-white">
            <DialogHeader>
              <DialogTitle>Start Stadia teleoperation?</DialogTitle>
              <DialogDescription className="text-gray-400">
                Complete this quick trigger setup immediately after starting.
              </DialogDescription>
            </DialogHeader>
            <StadiaStartupInstructions beforeStart />
            <DialogFooter className="flex gap-2 justify-end">
              <Button
                variant="outline"
                className="border-gray-600 text-gray-300"
                onClick={() => setConfirmStadiaTeleop(false)}
              >
                Cancel
              </Button>
              <Button
                className="bg-yellow-500 hover:bg-yellow-600 text-white"
                onClick={() => {
                  setConfirmStadiaTeleop(false);
                  onTeleop(robot);
                }}
              >
                Start Stadia Teleoperation
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {robot && (
        <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
          <DialogContent className="bg-gray-900 border-gray-800 text-white">
            <DialogHeader>
              <DialogTitle>Delete robot config?</DialogTitle>
              <DialogDescription className="text-gray-400">
                This deletes the robot config file from disk. Calibration files
                are not removed. This cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="flex gap-2 justify-end">
              <Button
                variant="outline"
                className="border-gray-600 text-gray-300"
                onClick={() => setConfirmDelete(false)}
              >
                Cancel
              </Button>
              <Button
                className="bg-red-500 hover:bg-red-600 text-white"
                onClick={async () => {
                  setConfirmDelete(false);
                  await onDelete(robot.name);
                }}
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default RobotTile;
