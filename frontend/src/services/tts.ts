type TtsDebugState = {
  supported: boolean;
  voicesCount: number;
  selectedVoice: string;
  selectedLang: string;
  lastEvent: string;
  lastReason: string;
};

const ttsDebugState: TtsDebugState = {
  supported: typeof window !== 'undefined' ? 'speechSynthesis' in window : false,
  voicesCount: 0,
  selectedVoice: '',
  selectedLang: '',
  lastEvent: 'idle',
  lastReason: '',
};

function getSynth() {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null;
  return window.speechSynthesis;
}

async function loadVoices(timeoutMs = 1200): Promise<SpeechSynthesisVoice[]> {
  const synth = getSynth();
  if (!synth) return [];
  const now = synth.getVoices();
  ttsDebugState.voicesCount = now.length;
  if (now.length) return now;
  return await new Promise<SpeechSynthesisVoice[]>((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      const voices = synth.getVoices();
      ttsDebugState.voicesCount = voices.length;
      resolve(voices);
    };
    synth.onvoiceschanged = () => finish();
    window.setTimeout(() => finish(), timeoutMs);
  });
}

function pickStableVoice(voices: SpeechSynthesisVoice[]) {
  return (
    voices.find((v) => v.name === 'Samantha') ||
    voices.find((v) => v.name.toLowerCase().includes('samantha')) ||
    voices.find((v) => /en[-_]?us/i.test(v.lang)) ||
    voices.find((v) => /^en/i.test(v.lang)) ||
    voices[0]
  );
}

export async function speakQuestionText(
  text: string,
  opts?: { rate?: number; pitch?: number; volume?: number; lang?: string; timeoutMs?: number },
) {
  const synth = getSynth();
  if (!synth) {
    ttsDebugState.lastEvent = 'speak_failed';
    ttsDebugState.lastReason = 'speech_synthesis_unavailable';
    return { ok: false, reason: 'speech_synthesis_unavailable' };
  }

  const voices = await loadVoices();
  const voice = pickStableVoice(voices);
  ttsDebugState.selectedVoice = voice?.name || '(default)';
  ttsDebugState.selectedLang = opts?.lang || voice?.lang || 'en-US';
  ttsDebugState.lastEvent = 'speak_started';
  ttsDebugState.lastReason = '';

  return await new Promise<{ ok: boolean; reason?: string }>((resolve) => {
    let settled = false;
    let started = false;
    let startPoll: number | null = null;
    const done = (ok: boolean, reason?: string) => {
      if (settled) return;
      settled = true;
      try {
        synth.cancel();
      } catch {}
      if (startPoll !== null) {
        window.clearInterval(startPoll);
        startPoll = null;
      }
      ttsDebugState.lastEvent = ok ? 'speak_ok' : 'speak_failed';
      ttsDebugState.lastReason = reason || '';
      resolve({ ok, reason });
    };

    try {
      const u = new SpeechSynthesisUtterance(text);
      if (voice) u.voice = voice;
      u.lang = opts?.lang || voice?.lang || 'en-US';
      u.rate = opts?.rate ?? 0.95;
      u.pitch = opts?.pitch ?? 1.0;
      u.volume = opts?.volume ?? 1.0;

      u.onstart = () => {
        started = true;
        ttsDebugState.lastEvent = 'speak_onstart';
      };
      u.onend = () => done(true);
      u.onerror = () => {
        ttsDebugState.lastEvent = 'speak_onerror';
        done(false, 'utterance_error');
      };

      // Only cancel if something is actively queued/speaking.
      if (synth.speaking || synth.pending) synth.cancel();
      synth.resume();
      synth.speak(u);

      // Some browsers don't fire onstart reliably; detect actual start.
      startPoll = window.setInterval(() => {
        try {
          if (synth.speaking) {
            started = true;
            ttsDebugState.lastEvent = 'speak_detected_speaking';
          }
        } catch {}
      }, 120);

      const maxMs = opts?.timeoutMs ?? Math.min(22000, Math.max(3000, text.length * 55));
      window.setTimeout(() => {
        if (settled) return;
        if (started) {
          done(false, 'tts_stuck_after_start');
        } else {
          done(false, 'tts_never_started');
        }
      }, maxMs);
    } catch {
      ttsDebugState.lastEvent = 'speak_exception';
      done(false, 'tts_exception');
    }
  });
}

export function getTTSDebugState() {
  return { ...ttsDebugState };
}
