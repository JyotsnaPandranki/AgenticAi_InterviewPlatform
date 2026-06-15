import { useEffect, useMemo, useState } from "react";
import type { LiveInterviewState } from "../types/session";

export function useLiveInterviewState(initialSeconds = 90) {
  const [state, setState] = useState<LiveInterviewState>({
    isRunning: false,
    secondsLeft: initialSeconds,
    isListening: false,
    transcript: "",
  });

  useEffect(() => {
    if (!state.isRunning || state.secondsLeft <= 0) return;
    const t = setInterval(() => {
      setState((s) => ({ ...s, secondsLeft: Math.max(0, s.secondsLeft - 1) }));
    }, 1000);
    return () => clearInterval(t);
  }, [state.isRunning, state.secondsLeft]);

  const mmss = useMemo(() => {
    const mm = String(Math.floor(state.secondsLeft / 60)).padStart(2, "0");
    const ss = String(state.secondsLeft % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }, [state.secondsLeft]);

  return {
    state,
    mmss,
    start: () => setState((s) => ({ ...s, isRunning: true, isListening: true })),
    stop: () => setState((s) => ({ ...s, isRunning: false, isListening: false })),
    setTranscript: (transcript: string, confidence?: number) =>
      setState((s) => ({ ...s, transcript, confidence })),
    reset: () =>
      setState({
        isRunning: false,
        secondsLeft: initialSeconds,
        isListening: false,
        transcript: "",
      }),
  };
}
