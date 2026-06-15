import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeVideoFrame, endInterview, submitAnswer, transcribeAudio } from '../services/api';
import { useAppState } from '../context/AppContext';
import { speakQuestionText } from '../services/tts';

type TurnState =
  | 'QUESTION_INITIALIZED'
  | 'TTS_SPEAKING'
  | 'LISTENING'
  | 'RECORDING'
  | 'SILENCE_DETECTED'
  | 'TRANSCRIBING'
  | 'EVALUATING'
  | 'TRANSITIONING'
  | 'INTERVIEW_COMPLETE';

export default function LiveInterviewPage() {
  const nav = useNavigate();
  const {
    state,
    setTranscript,
    setCurrentQuestion,
    setCurrentRound,
    setInterviewSummary,
    setAudioAnalysis,
    setCvAnalysis,
    setTranscriptionConfidence,
    setInterviewUiState,
    setLastEvaluation,
    setCoachingTip,
  } = useAppState();

  const [turnState, setTurnState] = useState<TurnState>('QUESTION_INITIALIZED');
  const [statusLine, setStatusLine] = useState('AI interviewer is speaking...');
  const [waveEnergy, setWaveEnergy] = useState(0.08);
  const [warningLines, setWarningLines] = useState<string[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);

  const turnIdRef = useRef(0);
  const submittedTurnRef = useRef<number | null>(null);
  const activeTurnKeyRef = useRef<string>('');
  const mountedRef = useRef(true);

  const audioStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const cvIntervalRef = useRef<number | null>(null);

  const audioChunksRef = useRef<Blob[]>([]);
  const speechDetectedRef = useRef(false);
  const lastSpeechAtRef = useRef(0);
  const startedAtRef = useRef(0);
  const responseSubmittedRef = useRef(false);

  const cv = state.cvAnalysis || {};
  const hasFace = Boolean(cv.face_detected) || Number(cv.eye_contact_score || 0) > 0.2;

  const computedWarnings = useMemo(() => {
    const warnings: string[] = [];
    if (!hasFace) warnings.push('Face not visible');
    if (Number(cv.eye_contact_score || 0) < 3.0 && hasFace) warnings.push('Looking away frequently');
    if (Number(cv.engagement_score || 0) < 3.8 && hasFace) warnings.push('Low engagement detected');
    if (Number(cv.distraction_score || 0) > 6.0 && hasFace) warnings.push('High distraction detected');
    if (Number(cv.perclos || 0) > 0.35 && hasFace) warnings.push('Warning: please look at the screen');
    return warnings;
  }, [cv, hasFace]);

  useEffect(() => {
    setWarningLines(computedWarnings);
  }, [computedWarnings]);

  useEffect(() => {
    mountedRef.current = true;
    void startCamera();
    return () => {
      mountedRef.current = false;
      stopCvStream();
      teardownTurnResources();
      cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
      window.speechSynthesis?.cancel();
    };
  }, []);

  useEffect(() => {
    if (!state.interviewId || !state.currentQuestion) return;
    const turnKey = `${state.currentRound}:${state.currentQuestion}`;
    if (activeTurnKeyRef.current === turnKey) return;
    activeTurnKeyRef.current = turnKey;
    void runTurn(state.currentQuestion);
  }, [state.interviewId, state.currentQuestion, state.currentRound]);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      cameraStreamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      startCvStream();
    } catch {
      // camera optional
    }
  }

  function startCvStream() {
    stopCvStream();
    cvIntervalRef.current = window.setInterval(() => {
      void captureCvFrame();
    }, 2000);
  }

  function stopCvStream() {
    if (cvIntervalRef.current) {
      window.clearInterval(cvIntervalRef.current);
      cvIntervalRef.current = null;
    }
  }

  function resetInterviewTurnState() {
    setTranscript('');
    setAudioAnalysis({});
    setTranscriptionConfidence(0);
    setWaveEnergy(0.08);
    speechDetectedRef.current = false;
    lastSpeechAtRef.current = 0;
    startedAtRef.current = 0;
    responseSubmittedRef.current = false;
    audioChunksRef.current = [];
    submittedTurnRef.current = null;
  }

  function teardownTurnResources() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch {}
    }
    mediaRecorderRef.current = null;

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((t) => t.stop());
      audioStreamRef.current = null;
    }
    analyserRef.current = null;
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  }

  function setState(stateName: TurnState, line: string) {
    setTurnState(stateName);
    setStatusLine(line);
    const map: Record<TurnState, any> = {
      QUESTION_INITIALIZED: 'idle',
      TTS_SPEAKING: 'dictating_question',
      LISTENING: 'listening',
      RECORDING: 'speech_detected',
      SILENCE_DETECTED: 'processing_answer',
      TRANSCRIBING: 'processing_answer',
      EVALUATING: 'processing_answer',
      TRANSITIONING: 'moving_next_question',
      INTERVIEW_COMPLETE: 'interview_complete',
    };
    setInterviewUiState(map[stateName]);
  }

  async function speakQuestion(turnId: number, question: string) {
    setState('TTS_SPEAKING', 'AI interviewer is speaking...');
    setWaveEnergy(0.2);
    const result = await speakQuestionText(question, {
      rate: 0.95,
      pitch: 1,
      volume: 1.0,
      lang: 'en-US',
      timeoutMs: 15000,
    });
    console.log('[TTS] result:', result);
    await new Promise((resolve) => window.setTimeout(resolve, 400));
    if (turnId !== turnIdRef.current) throw new Error('stale_turn');
    setState('LISTENING', 'Listening for your answer...');
  }

  async function captureAudioUntilSilence(turnId: number, maxSeconds = 90, silenceSeconds = 5): Promise<Blob> {
    teardownTurnResources();
    console.log('[TURN] starting mic...');
    await new Promise((resolve) => window.setTimeout(resolve, 300));
    const getMic = async () =>
      navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
    const stream = await Promise.race([
      getMic(),
      new Promise<never>((_, reject) =>
        window.setTimeout(() => reject(new Error('mic_timeout')), 8000),
      ),
    ]);
    console.log('[TURN] mic acquired');
    if (turnId !== turnIdRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error('stale_turn');
    }

    audioStreamRef.current = stream;
    const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorderRef.current = recorder;
    audioChunksRef.current = [];
    let recorderStarted = false;

    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    const src = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    src.connect(analyser);
    analyserRef.current = analyser;

    const pcm = new Uint8Array(analyser.fftSize);
    startedAtRef.current = performance.now();
    lastSpeechAtRef.current = performance.now();
    speechDetectedRef.current = false;
    let noiseFloor = 0;
    let noiseSamples = 0;
    let speechThreshold = 0.01;
    let frameCount = 0;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    setState('LISTENING', 'Listening...');

    const evaluate = () => {
      if (turnId !== turnIdRef.current) return;
      analyser.getByteTimeDomainData(pcm);
      let sum = 0;
      for (let i = 0; i < pcm.length; i += 1) {
        const v = (pcm[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / pcm.length);
      frameCount += 1;

      // Calibrate noise floor from initial ~1 second and keep a safe dynamic threshold.
      if (noiseSamples < 30) {
        noiseFloor += rms;
        noiseSamples += 1;
        const avgNoise = noiseFloor / noiseSamples;
        speechThreshold = Math.max(0.008, avgNoise * 2.2);
      }

      // UI bar should always react to mic energy.
      setWaveEnergy(Math.min(1, rms * 24 + 0.06));

      const now = performance.now();
      if (rms > speechThreshold) {
        lastSpeechAtRef.current = now;
        if (!speechDetectedRef.current) {
          speechDetectedRef.current = true;
          console.log('[TURN] speech detected', { rms, speechThreshold });
          // Start capturing answer only when actual speech begins.
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive') {
            mediaRecorderRef.current.start();
            recorderStarted = true;
          }
          setState('RECORDING', 'Recording response...');
        }
      }
      const hardTimeout = now - startedAtRef.current > maxSeconds * 1000;
      const silenceTimeout = speechDetectedRef.current && now - lastSpeechAtRef.current > silenceSeconds * 1000;
      if (frameCount % 50 === 0) {
        console.log('[TURN] rms', {
          rms: Number(rms.toFixed(4)),
          speechThreshold: Number(speechThreshold.toFixed(4)),
          speechDetected: speechDetectedRef.current,
        });
      }
      if (hardTimeout || silenceTimeout) {
        console.log('[TURN] silence detected, stopping');
        setState('SILENCE_DETECTED', 'Analyzing response...');
        if (mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop();
        } else {
          // Nothing captured (no speech), resolve immediately with empty blob.
          const empty = new Blob([], { type: 'audio/webm' });
          audioChunksRef.current = [empty];
          if (rafRef.current !== null) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
          }
        }
        return;
      }
      rafRef.current = requestAnimationFrame(evaluate);
    };
    rafRef.current = requestAnimationFrame(evaluate);

    return await new Promise<Blob>((resolve) => {
      let noSpeechPoll: number | null = null;
      recorder.onstop = () => {
        if (noSpeechPoll !== null) {
          window.clearInterval(noSpeechPoll);
          noSpeechPoll = null;
        }
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        resolve(blob);
      };
      // If recording never started (candidate stayed silent), finish on hard timeout path.
      noSpeechPoll = window.setInterval(() => {
        if (recorderStarted) {
          if (noSpeechPoll !== null) {
            window.clearInterval(noSpeechPoll);
            noSpeechPoll = null;
          }
          return;
        }
        const now = performance.now();
        if (now - startedAtRef.current > maxSeconds * 1000) {
          if (noSpeechPoll !== null) {
            window.clearInterval(noSpeechPoll);
            noSpeechPoll = null;
          }
          if (rafRef.current !== null) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
          }
          resolve(new Blob([], { type: 'audio/webm' }));
        }
      }, 250);
    });
  }

  async function captureCvFrame() {
    try {
      if (!videoRef.current) return {};
      const video = videoRef.current;
      if (!video.videoWidth || !video.videoHeight) return {};
      const c = document.createElement('canvas');
      c.width = video.videoWidth;
      c.height = video.videoHeight;
      const ctx = c.getContext('2d');
      if (!ctx) return {};
      ctx.drawImage(video, 0, 0, c.width, c.height);
      const image = c.toDataURL('image/jpeg', 0.72);
      const result = await analyzeVideoFrame(image, performance.now() / 1000);
      const metrics = {
        ...(result.cv_analysis || {}),
        face_detected: result.face_detected ? 1 : 0,
      } as Record<string, number>;
      setCvAnalysis(metrics);
      return metrics;
    } catch {
      // CV issues should never block interview progression.
      return {};
    }
  }

  async function runTurn(question: string) {
    console.log('[TURN] start, question:', question);
    turnIdRef.current += 1;
    const turnId = turnIdRef.current;
    resetInterviewTurnState();
    setState('QUESTION_INITIALIZED', 'Preparing question...');
    try {
      await speakQuestion(turnId, question);
      console.log('[TURN] TTS done, entering mic capture');
      const audioBlob = await captureAudioUntilSilence(turnId, 90, 5);
      if (turnId !== turnIdRef.current) return;
      setState('TRANSCRIBING', 'Transcribing response...');
      const [transcribed, cvMetrics] = await Promise.all([transcribeAudio(audioBlob), captureCvFrame()]);
      if (turnId !== turnIdRef.current) return;
      const transcript = (transcribed.transcript || '').trim();
      setTranscript(transcript);
      setAudioAnalysis((transcribed.audio_analysis || {}) as Record<string, number>);
      setTranscriptionConfidence(Number(transcribed.transcription_confidence || 0));

      if (responseSubmittedRef.current || submittedTurnRef.current === turnId) return;
      responseSubmittedRef.current = true;
      submittedTurnRef.current = turnId;
      setState('EVALUATING', 'Evaluating response...');
      const res = await submitAnswer({
        interview_id: state.interviewId,
        answer_text: transcript,
        transcript,
        audio_analysis: transcribed.audio_analysis || {},
        cv_analysis: (cvMetrics || {}) as Record<string, number>,
      });
      if (turnId !== turnIdRef.current) return;
      setLastEvaluation(res.evaluation || null);
      if (state.coachingMode && res?.evaluation?.coaching_feedback) {
        const c = res.evaluation.coaching_feedback;
        const tip = [c.technical_feedback, c.communication_feedback, c.behavioral_feedback]
          .filter(Boolean)
          .join(' ')
          .trim();
        setCoachingTip(tip);
      } else {
        setCoachingTip('');
      }

      if (res.done && res.summary) {
        setState('INTERVIEW_COMPLETE', 'Interview complete.');
        teardownTurnResources();
        setInterviewSummary(res.summary);
        nav('/summary');
        return;
      }

      setState('TRANSITIONING', 'Moving to next question...');
      teardownTurnResources();
      setCurrentQuestion(res.question || '');
      setCurrentRound(Number(res.round || state.currentRound + 1));
    } catch (e: any) {
      if (String(e?.message || '').includes('stale_turn')) return;
      teardownTurnResources();
      setStatusLine('Microphone/camera/network issue. Please retry.');
    }
  }

  async function endInterviewNow() {
    if (!state.interviewId) return;
    try {
      turnIdRef.current += 1;
      teardownTurnResources();
      setState('INTERVIEW_COMPLETE', 'Ending interview...');
      const res = await endInterview({ interview_id: state.interviewId, reason: 'candidate_ended' });
      if (res?.summary) {
        setInterviewSummary(res.summary);
        nav('/summary');
      }
    } catch {
      setStatusLine('Could not end interview. Please retry.');
    }
  }

  const showEvalSpinner = turnState === 'EVALUATING' || turnState === 'TRANSCRIBING' || turnState === 'SILENCE_DETECTED';

  return (
    <section className="mx-auto max-w-[1450px] py-8">
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        <aside className="rounded-2xl border border-white/10 bg-[#090d15]/70 p-4 lg:col-span-1">
          <div className="space-y-6 text-center text-white/55">
            <div>◫</div><div>▣</div><div>☷</div><div>◧</div><div>⚙</div>
          </div>
        </aside>

        <main className="lg:col-span-8">
          <div className="rounded-2xl border border-white/10 bg-[#0b111b]/70 p-8">
            <div className="flex items-center justify-between text-sm uppercase tracking-[0.18em] text-white/50">
              <div>Humanists-01 Engine</div>
              <div>Question {state.currentRound} / {state.totalQuestions}</div>
            </div>
            <div className="mt-5 rounded-xl border-l-2 border-tertiary/60 bg-white/[0.02] px-4 py-3 text-sm text-tertiary">Runtime state: {turnState}</div>
            {state.coachingMode && state.coachingTip ? (
              <div className="mt-4 rounded-xl border border-primary/30 bg-primary/10 px-4 py-3 text-sm text-primary">
                Coaching tip: {state.coachingTip}
              </div>
            ) : null}
            <h2 className="mt-7 text-6xl leading-[1.08] tracking-[-0.02em]">“{state.currentQuestion || 'Loading question...'}”</h2>

            <div className="mt-8 flex items-center gap-3 text-white/70">
              {showEvalSpinner ? (
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-tertiary/30 border-t-tertiary" />
              ) : (
                <div className="h-3 w-3 rounded-full bg-tertiary" style={{ boxShadow: `0 0 ${16 + waveEnergy * 40}px rgba(92,225,230,0.8)` }} />
              )}
              <div>{statusLine}</div>
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full bg-gradient-to-r from-primary to-tertiary transition-all duration-150" style={{ width: `${Math.max(8, Math.min(100, waveEnergy * 100))}%` }} />
            </div>

            <div className="mt-10 border-t border-white/10 pt-8">
              <div className="text-xs uppercase tracking-[0.2em] text-primary">Live Transcription</div>
              <div className="mt-3 min-h-20 text-2xl leading-relaxed text-white/80">{state.transcript ? `"${state.transcript}"` : 'Waiting for response...'}</div>
              <div className="mt-4 text-sm text-white/55">Transcription Confidence: {Number(state.transcriptionConfidence || 0).toFixed(2)}%</div>
            </div>
          </div>
        </main>

        <aside className="space-y-4 lg:col-span-3">
          <div className="rounded-2xl border border-white/10 bg-[#0b111b]/70 p-5">
            <div className="text-sm uppercase tracking-[0.18em] text-white/55">Live CV Feed</div>
            <div className="mt-4">
              <div className="mx-auto w-full max-w-[220px]">
                <Metric label="Eye Contact" value={Number(cv.eye_contact_score || 0).toFixed(2)} />
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Metric label="Engagement" value={Number(cv.engagement_score || 0).toFixed(2)} />
              <Metric label="Attention" value={Number(cv.attention_score || 0).toFixed(2)} />
              <Metric label="Blink/min" value={Number(cv.blink_rate || 0).toFixed(1)} />
              <Metric label="PERCLOS" value={Number(cv.perclos || 0).toFixed(2)} />
            </div>
            <div className="mt-4 space-y-2">
              {warningLines.length === 0 ? (
                <div className="rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2 text-xs text-white/55">No distraction warnings</div>
              ) : (
                warningLines.map((w, i) => (
                  <div key={`${w}-${i}`} className="rounded-lg border border-amber-400/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">{w}</div>
                ))
              )}
            </div>
          </div>
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0b111b]/70">
            <video ref={videoRef} autoPlay playsInline muted className="h-52 w-full object-cover opacity-90" />
            <div className="p-3 text-center text-xs uppercase tracking-[0.2em] text-white/50">Vision Analysis Active</div>
          </div>
          <button
            className="w-full rounded-full border border-red-400/40 bg-red-500/10 px-5 py-3 text-sm uppercase tracking-[0.18em] text-red-200"
            onClick={endInterviewNow}
          >
            End Interview
          </button>
        </aside>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-white/10 p-4 text-center">
      <div className="text-3xl">{value}</div>
      <div className="mt-1 text-xs tracking-[0.18em] text-white/50">{label}</div>
    </div>
  );
}
