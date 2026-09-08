import { Gamepad2 } from "lucide-react";

interface StadiaStartupInstructionsProps {
  beforeStart?: boolean;
}

const StadiaStartupInstructions = ({
  beforeStart = false,
}: StadiaStartupInstructionsProps) => (
  <div
    role="status"
    className="rounded-lg border border-purple-700/70 bg-purple-950/40 p-3 text-left text-sm text-purple-100"
  >
    <div className="flex items-center gap-2 font-semibold">
      <Gamepad2 className="h-4 w-4" /> Stadia trigger setup
    </div>
    <p className="mt-1 text-purple-200">
      {beforeStart ? "After clicking Start, " : "Now: "}
      release RB, center both sticks, fully squeeze LT and RT once, then release
      both. Keep all controls released until the session starts.
    </p>
    {beforeStart && (
      <p className="mt-1 text-xs text-purple-300">
        You will have 15 seconds. This lets Bluetooth establish the triggers'
        neutral range before LeLab accesses the robot.
      </p>
    )}
  </div>
);

export default StadiaStartupInstructions;
