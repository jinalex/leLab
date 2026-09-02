import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  normalizeRobotRecord,
  type RobotRecord,
} from "@/lib/robotConfig";

export type { RobotRecord } from "@/lib/robotConfig";

const SELECTED_KEY = "lelab.selectedRobot";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const hasExactKeys = (
  value: Record<string, unknown>,
  keys: readonly string[],
): boolean =>
  Object.keys(value).length === keys.length &&
  keys.every((key) => Object.prototype.hasOwnProperty.call(value, key));

const readSelected = (): string | null => {
  try {
    const raw = localStorage.getItem(SELECTED_KEY);
    return raw && typeof raw === "string" ? raw : null;
  } catch {
    return null;
  }
};

const writeSelected = (name: string | null) => {
  try {
    if (name) localStorage.setItem(SELECTED_KEY, name);
    else localStorage.removeItem(SELECTED_KEY);
  } catch {
    // Storage may be unavailable (private mode, quota). Failures here are non-fatal.
  }
};

export const useRobots = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const location = useLocation();

  const [records, setRecords] = useState<Record<string, RobotRecord>>({});
  const [selectedName, setSelectedName] = useState<string | null>(() => readSelected());
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Re-fetch records when location changes (RobotConfigManager mounts only on Landing,
  // so this fires on initial mount and on back-navigation to Landing)
  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots`);
        const data: unknown = await res.json();
        if (cancelled) return;
        if (
          !res.ok ||
          !isObject(data) ||
          !hasExactKeys(data, ["status", "robots"]) ||
          data.status !== "success"
        ) {
          throw new Error(
            isObject(data) && typeof data.message === "string"
              ? data.message
              : "The robot list could not be loaded.",
          );
        }
        if (!Array.isArray(data.robots)) {
          throw new Error("The backend returned an invalid robot list.");
        }
        const next: Record<string, RobotRecord> = {};
        for (const [index, raw] of data.robots.entries()) {
          const record = normalizeRobotRecord(raw);
          if (!record) {
            throw new Error(
              `Robot entry ${index + 1} is not an exact RobotRecordV2. Nothing was loaded.`,
            );
          }
          if (record.name in next) {
            throw new Error(
              `The backend returned duplicate robot name "${record.name}". Nothing was loaded.`,
            );
          }
          next[record.name] = record;
        }
        setRecords(next);
        // Drop the selection if the underlying record vanished (deleted from another tab)
        setSelectedName((prev) => (prev && prev in next ? prev : null));
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to fetch robots:", e);
          setRecords({});
          setSelectedName(null);
          setLoadError(
            e instanceof Error ? e.message : "The robot list could not be loaded.",
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    fetchAll();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, location.key]);

  // Persist selection to localStorage
  useEffect(() => {
    writeSelected(selectedName);
  }, [selectedName]);

  const selectRobot = useCallback((name: string) => {
    setSelectedName(name);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedName(null);
  }, []);

  const createRobot = useCallback(
    async (rawName: string): Promise<boolean> => {
      const name = rawName.trim();
      if (!name) {
        toast({ title: "Missing name", description: "Robot name cannot be empty.", variant: "destructive" });
        return false;
      }
      if (/[/\\]|\.\./.test(name)) {
        toast({ title: "Invalid name", description: "Robot names cannot contain '/', '\\', or '..'", variant: "destructive" });
        return false;
      }
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots/${encodeURIComponent(name)}?create=true`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ schema_version: 2 }),
        });
        if (res.status === 409) {
          toast({
            title: "Already exists",
            description: `A robot named "${name}" already exists. Pick it from the dropdown or choose a different name.`,
            variant: "destructive",
          });
          return false;
        }
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to create", description: text, variant: "destructive" });
          return false;
        }
        const data: unknown = await res.json();
        if (
          !isObject(data) ||
          !hasExactKeys(data, ["status", "robot"]) ||
          data.status !== "success"
        ) {
          toast({
            title: "Invalid robot response",
            description: "The backend did not return an exact successful robot envelope.",
            variant: "destructive",
          });
          return false;
        }
        const record = normalizeRobotRecord(data.robot);
        if (record && record.name === name) {
          setRecords((prev) => ({ ...prev, [name]: record }));
          setSelectedName(name);
        } else {
          toast({
            title: "Invalid robot response",
            description: "The backend did not return a valid RobotRecordV2.",
            variant: "destructive",
          });
          return false;
        }
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const deleteRobot = useCallback(
    async (name: string): Promise<boolean> => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/robots/${encodeURIComponent(name)}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to delete", description: text, variant: "destructive" });
          return false;
        }
        const data: unknown = await res.json();
        if (
          !isObject(data) ||
          !hasExactKeys(data, ["status"]) ||
          data.status !== "success"
        ) {
          toast({
            title: "Invalid delete response",
            description: "The backend did not confirm the exact robot deletion envelope.",
            variant: "destructive",
          });
          return false;
        }
        setRecords((prev) => {
          const { [name]: _omit, ...rest } = prev;
          return rest;
        });
        setSelectedName((prev) => (prev === name ? null : prev));
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const selectedRecord = useMemo(
    () => (selectedName ? records[selectedName] ?? null : null),
    [selectedName, records]
  );

  const availableNames = useMemo(
    () => Object.keys(records).sort(),
    [records]
  );

  return {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading,
    loadError,
    selectRobot,
    clearSelection,
    createRobot,
    deleteRobot,
  };
};
