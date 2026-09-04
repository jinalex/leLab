import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import {
  isTerminalControlState,
  reconcileControlStatus,
  requireControlStatusEnvelope,
  type ControlStatus,
  type RobotOperation,
  type TeleoperatorType,
} from "@/lib/robotConfig";

interface UseControlSessionOptions {
  sessionId: string | null;
  expectedOperation: RobotOperation;
  expectedTeleoperatorType: TeleoperatorType | null;
  pollIntervalMs?: number;
  stopOnUnmount?: boolean;
  renewalBlocked?: boolean;
}

interface ControlSessionHandle {
  status: ControlStatus | null;
  contractError: string | null;
  stopPending: boolean;
  terminal: boolean;
  requestStop: () => Promise<ControlStatus>;
  ingestCompatibilityStatus: (value: unknown) => {
    status: ControlStatus;
    stale: boolean;
  };
  ingestStatusEnvelope: (value: unknown) => {
    status: ControlStatus;
    stale: boolean;
  };
}

const errorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? error.message : fallback;

/**
 * Owns the browser half of one exact server control lease.
 *
 * A malformed or mismatched response is never adopted and pauses renewal until
 * an exact status poll succeeds. Page-unload cleanup remains best effort; the
 * server lease is the authoritative orphan backstop.
 */
export const useControlSession = ({
  sessionId,
  expectedOperation,
  expectedTeleoperatorType,
  pollIntervalMs = 400,
  stopOnUnmount = true,
  renewalBlocked = false,
}: UseControlSessionOptions): ControlSessionHandle => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<ControlStatus | null>(null);
  const [contractError, setContractError] = useState<string | null>(null);
  const [stopPending, setStopPending] = useState(false);
  const statusRef = useRef<ControlStatus | null>(null);
  const stopDispatchedRef = useRef(false);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    statusRef.current = null;
    stopDispatchedRef.current = false;
    setStatus(null);
    setContractError(null);
    setStopPending(false);
  }, [sessionId, expectedOperation, expectedTeleoperatorType]);

  const requirements = useCallback(
    (expectedSessionId: string) => ({
      expectedSessionId,
      expectedOperation,
      expectedTeleoperatorType,
      requireTopLevelSessionId: true,
    }),
    [expectedOperation, expectedTeleoperatorType],
  );

  const adopt = useCallback((next: ControlStatus) => {
    const previous = statusRef.current;
    const reconciled = reconcileControlStatus(previous, next);
    if (reconciled === previous) {
      if (previous && next.revision === previous.revision) {
        setContractError(null);
      }
      return reconciled;
    }
    statusRef.current = reconciled;
    setStatus(reconciled);
    setContractError(null);
    if (isTerminalControlState(reconciled.state)) {
      stopDispatchedRef.current = true;
      setStopPending(false);
    }
    return reconciled;
  }, []);

  const ingestCompatibilityStatus = useCallback(
    (
      value: unknown,
    ): { status: ControlStatus; stale: boolean } => {
      try {
        if (!sessionId) {
          throw new Error("There is no control session to validate.");
        }
        const next = requireControlStatusEnvelope(value, {
          ...requirements(sessionId),
          requireStatusKey: "control_status",
        });
        const reconciled = adopt(next);
        return {
          status: reconciled,
          stale: next.revision < reconciled.revision,
        };
      } catch (error) {
        const message = errorMessage(
          error,
          "The compatibility response violated the control status contract.",
        );
        setContractError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [sessionId, requirements, adopt],
  );

  const ingestStatusEnvelope = useCallback(
    (value: unknown): { status: ControlStatus; stale: boolean } => {
      try {
        if (!sessionId) {
          throw new Error("There is no control session to validate.");
        }
        const next = requireControlStatusEnvelope(value, {
          ...requirements(sessionId),
          requireSuccess: true,
          requireStatusKey: "status",
        });
        const reconciled = adopt(next);
        return {
          status: reconciled,
          stale: next.revision < reconciled.revision,
        };
      } catch (error) {
        const message = errorMessage(
          error,
          "The control response violated the status contract.",
        );
        setContractError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [sessionId, requirements, adopt],
  );

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/control-status?session_id=${encodeURIComponent(sessionId)}`,
        );
        const data: unknown = await response.json();
        if (!response.ok) {
          throw new Error(
            typeof (data as { message?: unknown })?.message === "string"
              ? (data as { message: string }).message
              : "Control status is unavailable.",
          );
        }
        const next = requireControlStatusEnvelope(data, {
          ...requirements(sessionId),
          requireSuccess: true,
          requireStatusKey: "status",
        });
        if (!cancelled) adopt(next);
      } catch (error) {
        if (!cancelled) {
          setContractError(errorMessage(error, "Control status polling failed."));
        }
      }
    };

    void poll();
    const interval = window.setInterval(() => {
      if (!statusRef.current || !isTerminalControlState(statusRef.current.state)) {
        void poll();
      }
    }, Math.max(250, pollIntervalMs));
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [
    sessionId,
    expectedOperation,
    expectedTeleoperatorType,
    pollIntervalMs,
    baseUrl,
    fetchWithHeaders,
    requirements,
    adopt,
  ]);

  const lifecycleState = status?.state ?? null;
  const renewIntervalS = status?.lease_renew_interval_s ?? null;

  useEffect(() => {
    if (
      !sessionId ||
      contractError ||
      renewalBlocked ||
      stopPending ||
      (lifecycleState !== "starting" && lifecycleState !== "running") ||
      renewIntervalS === null
    ) {
      return;
    }
    let cancelled = false;
    const renew = async () => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/control-lease/renew`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
        const data: unknown = await response.json();
        if (!response.ok) {
          throw new Error(
            typeof (data as { message?: unknown })?.message === "string"
              ? (data as { message: string }).message
              : "Control lease renewal failed.",
          );
        }
        const next = requireControlStatusEnvelope(data, {
          ...requirements(sessionId),
          requireSuccess: true,
          requireStatusKey: "status",
        });
        if (!cancelled) adopt(next);
      } catch (error) {
        if (!cancelled) {
          setContractError(errorMessage(error, "Control lease renewal failed."));
        }
      }
    };
    const intervalMs = Math.max(250, Math.min(renewIntervalS * 1000, 2000));
    const interval = window.setInterval(() => void renew(), intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [
    sessionId,
    lifecycleState,
    renewIntervalS,
    contractError,
    renewalBlocked,
    stopPending,
    baseUrl,
    fetchWithHeaders,
    requirements,
    adopt,
  ]);

  const requestStop = useCallback(async (): Promise<ControlStatus> => {
    if (!sessionId) throw new Error("There is no exact control session to stop.");
    setStopPending(true);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/control-stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      const data: unknown = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof (data as { message?: unknown })?.message === "string"
            ? (data as { message: string }).message
            : "Control stop request failed.",
        );
      }
      const next = requireControlStatusEnvelope(data, {
        ...requirements(sessionId),
        requireSuccess: true,
        requireStatusKey: "status",
      });
      if (next.state === "starting" || next.state === "running") {
        throw new Error(
          "The backend accepted stop without entering stopping or a terminal state.",
        );
      }
      const adopted = adopt(next);
      stopDispatchedRef.current = true;
      return adopted;
    } catch (error) {
      setContractError(errorMessage(error, "Control stop request failed."));
      throw error;
    } finally {
      setStopPending(false);
    }
  }, [sessionId, baseUrl, fetchWithHeaders, requirements, adopt]);

  useEffect(() => {
    if (!sessionId || !stopOnUnmount) return;
    const bestEffortStop = () => {
      const latest = statusRef.current;
      if (
        stopDispatchedRef.current ||
        (latest && isTerminalControlState(latest.state))
      ) {
        return;
      }
      // pagehide is often followed by unmount. Mark this exact-session stop as
      // dispatched before sending so those two browser paths cannot duplicate it.
      stopDispatchedRef.current = true;
      fetchWithHeaders(`${baseUrl}/control-stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
        keepalive: true,
      }).catch(() => {
        // The bounded server-side lease remains the authoritative backstop.
      });
    };
    window.addEventListener("pagehide", bestEffortStop);
    return () => {
      window.removeEventListener("pagehide", bestEffortStop);
      bestEffortStop();
    };
  }, [sessionId, stopOnUnmount, baseUrl, fetchWithHeaders]);

  return {
    status,
    contractError,
    stopPending,
    terminal: status ? isTerminalControlState(status.state) : false,
    requestStop,
    ingestCompatibilityStatus,
    ingestStatusEnvelope,
  };
};
