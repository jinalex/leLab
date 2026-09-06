import { useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

/** Poll only the active control owner's camera; never acquire a second device. */
export default function BackendCameraFeed({ name, sessionId }: { name: string; sessionId: string }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let previous: string | null = null;
    const clear = () => {
      if (previous) URL.revokeObjectURL(previous);
      previous = null;
    };
    const poll = async () => {
      try {
        const response = await fetchWithHeaders(
          `${baseUrl}/teleop-camera-frame/${encodeURIComponent(name)}?session_id=${encodeURIComponent(sessionId)}`,
          { signal: controller.signal, cache: "no-store" },
        );
        if (!response.ok) throw new Error("Frame unavailable");
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        clear();
        previous = URL.createObjectURL(blob);
        setSrc(previous);
      } catch {
        if (!controller.signal.aborted) {
          clear();
          setSrc(null);
        }
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 200);
    };
    setSrc(null);
    void poll();
    return () => { controller.abort(); clearTimeout(timer); clear(); };
  }, [baseUrl, fetchWithHeaders, name, sessionId]);
  return <div className="rounded-lg border border-gray-700 overflow-hidden">
    {src ? <img src={src} alt={name} className="w-full" /> :
      <div className="aspect-[4/3] flex items-center justify-center text-gray-400">Waiting for live camera frame</div>}
    <div className="p-2 text-sm text-gray-300">{name}</div>
  </div>;
}
