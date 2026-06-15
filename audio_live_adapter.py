# =========================================================
# audio_live_adapter.py
# =========================================================
# PURPOSE:
# Live microphone adapter for interview rounds.
# Features:
# - ambient-noise calibration
# - speech start detection
# - auto-stop after 6s no speech (default)
# - Whisper transcription
# =========================================================

import tempfile
import time
from typing import Dict

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import noisereduce as nr
except Exception:
    nr = None


class LiveAudioResponseAdapter:
    def __init__(self, sample_rate: int = 16000, model_size: str = "small"):
        if sd is None or sf is None:
            raise ImportError("sounddevice and soundfile are required for audio_live mode")
        if WhisperModel is None:
            raise ImportError("faster-whisper is required for audio_live mode")

        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, compute_type="int8")
        # Keep a conservative default that worked better for transcript quality.
        self.energy_threshold = 0.008
        self.calibrated = False
        self.noise_reduction_strength = 0.4

    def calibrate(self, seconds: int = 3):
        """Calibrate ambient noise floor before interview starts."""
        print(f"[Audio] Calibration: stay silent for {seconds}s...")
        audio = sd.rec(int(seconds * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        arr = audio.squeeze()
        noise_rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size > 0 else 0.0
        # Previous aggressive multiplier caused missed speech on some mics.
        # Keep threshold close to noise floor and bounded to safe limits.
        self.energy_threshold = min(0.015, max(0.004, noise_rms * 1.2))
        self.calibrated = True
        print(f"[Audio] Calibration complete. energy_threshold={self.energy_threshold:.5f}")

    def _record_until_silence(
        self,
        max_seconds: int = 90,
        silence_seconds: float = 5.0,
        frame_size: int = 1024,
    ) -> np.ndarray:
        # Same proven logic as test_silence_autostop.py:
        # continuous InputStream + state machine + 5s sustained silence.
        print(f"\n[Audio] Listening... (max {max_seconds}s)")

        state = "WAITING_FOR_SPEECH"
        silence_start = None
        speech_frames = 0
        min_speech_frames = max(1, int((0.35 * self.sample_rate) / frame_size))  # ~350ms
        started_at = time.time()
        frames = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
        ) as stream:
            while True:
                if time.time() - started_at > max_seconds:
                    print("[Audio] Max window reached.")
                    break

                block, _overflowed = stream.read(frame_size)
                mono = block.squeeze().astype(np.float32)
                frames.append(mono)

                e = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0
                is_speech = e >= self.energy_threshold

                if state == "WAITING_FOR_SPEECH":
                    if is_speech:
                        speech_frames += 1
                        if speech_frames >= min_speech_frames:
                            state = "SPEAKING"
                            silence_start = None
                            print("[Audio] Speech detected...")
                    else:
                        speech_frames = 0

                elif state == "SPEAKING":
                    if is_speech:
                        silence_start = None
                    else:
                        state = "SILENCE_COUNTDOWN"
                        silence_start = time.time()
                        print("[Audio] Silence countdown started...")

                elif state == "SILENCE_COUNTDOWN":
                    if is_speech:
                        state = "SPEAKING"
                        silence_start = None
                        print("[Audio] Speech resumed.")
                    else:
                        elapsed = time.time() - silence_start
                        if elapsed >= silence_seconds:
                            print("[Audio] Silence timeout reached. Stopping.")
                            break

        if not frames:
            return np.zeros((1,), dtype=np.float32)

        audio = np.concatenate(frames).astype(np.float32)
        return audio

    def _clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Feature-path cleaning only.
        Do not use this as the default STT waveform source.
        """
        out = audio.astype(np.float32)
        if nr is not None:
            try:
                out = nr.reduce_noise(
                    y=out,
                    sr=self.sample_rate,
                    prop_decrease=self.noise_reduction_strength,
                ).astype(np.float32)
            except Exception:
                pass
        return out

    def _transcribe(self, wav_path: str, use_vad_filter: bool = False):
        """
        Low-level transcription call.
        Keep defaults close to teammate pipeline for best STT quality.
        """
        segments, info = self.model.transcribe(
            wav_path,
            beam_size=5,
            language="en",
            vad_filter=use_vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if use_vad_filter else None,
        )
        segs = list(segments)
        text = " ".join((s.text or "").strip() for s in segs).strip()

        logs = [getattr(s, "avg_logprob", None) for s in segs]
        logs = [x for x in logs if x is not None]
        if logs:
            avg_logprob = float(sum(logs) / len(logs))
            conf = max(0.0, min(100.0, (avg_logprob + 2.5) / 2.5 * 100.0))
        else:
            conf = 0.0

        no_speech_probs = [getattr(s, "no_speech_prob", None) for s in segs]
        no_speech_probs = [x for x in no_speech_probs if x is not None]
        if no_speech_probs:
            mean_no_speech_prob = float(sum(no_speech_probs) / len(no_speech_probs))
        else:
            mean_no_speech_prob = 0.0

        return text, conf, info, mean_no_speech_prob

    def _trim_trailing_silence_for_pause_metric(
        self,
        audio: np.ndarray,
        win_samples: int,
        max_trim_seconds: float = 4.0,
    ) -> np.ndarray:
        """
        Exclude trailing end-of-answer silence from pause_ratio computation.
        This prevents auto-stop silence from inflating hesitation metrics.
        """
        if audio.size == 0 or win_samples <= 0:
            return audio

        max_trim_windows = max(1, int((max_trim_seconds * self.sample_rate) / win_samples))

        # Build windowed RMS
        rms_vals = []
        for i in range(0, len(audio) - win_samples + 1, win_samples):
            seg = audio[i:i + win_samples]
            rms_vals.append(float(np.sqrt(np.mean(np.square(seg)))))

        if not rms_vals:
            return audio

        # Count trailing quiet windows (bounded by max_trim_windows)
        trailing_quiet = 0
        for v in reversed(rms_vals):
            if v < self.energy_threshold and trailing_quiet < max_trim_windows:
                trailing_quiet += 1
            else:
                break

        if trailing_quiet == 0:
            return audio

        trim_samples = trailing_quiet * win_samples
        if trim_samples >= len(audio):
            return audio
        return audio[:-trim_samples]

    def _estimate_features(self, audio: np.ndarray, duration_sec: float, text: str):
        words = text.split()
        wpm = (len(words) / max(1.0, duration_sec)) * 60.0

        # lightweight pause signal using low-energy windows
        win = int(0.2 * self.sample_rate)
        audio_for_pause = self._trim_trailing_silence_for_pause_metric(
            audio=audio,
            win_samples=win,
            max_trim_seconds=4.0,
        )

        if win <= 0 or len(audio_for_pause) < win:
            pause_ratio = 0.0
        else:
            vals = []
            for i in range(0, len(audio_for_pause) - win + 1, win):
                seg = audio_for_pause[i:i + win]
                rms = float(np.sqrt(np.mean(np.square(seg))))
                vals.append(rms)
            quiet = sum(1 for v in vals if v < self.energy_threshold)
            pause_ratio = quiet / max(1, len(vals))

        return {
            "pause_ratio": round(float(pause_ratio), 4),
            "wpm": round(float(wpm), 2),
            "filler_density": 0.0,
            "pitch_variation": 0.0,
            "speech_consistency": 0.0,
        }

    def _is_poor_transcript(self, text: str, conf: float, no_speech_prob: float) -> bool:
        words = text.split()
        if len(words) < 3:
            return True
        if conf < 38:
            return True
        if no_speech_prob > 0.75:
            return True
        # repeated-token gibberish heuristic
        if words:
            uniq_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
            if len(words) >= 8 and uniq_ratio < 0.35:
                return True
        return False

    # =====================================================
    # FLOW A: STT / TRANSCRIPTION (source of truth)
    # =====================================================
    def transcribe_audio_raw(self, raw_audio: np.ndarray) -> Dict:
        """
        Priority path for transcript quality.
        1) raw audio, no VAD filter (primary; restored behavior)
        2) raw audio with VAD filter (fallback)
        3) lightly cleaned audio, no VAD (last fallback)
        """
        # primary raw pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, raw_audio.astype(np.float32), self.sample_rate)
            text, conf, info, nsp = self._transcribe(tmp.name, use_vad_filter=False)

        if not self._is_poor_transcript(text, conf, nsp):
            return {
                "transcript": text,
                "transcription_confidence": round(float(conf), 2),
                "no_speech_prob": round(float(nsp), 4),
                "transcription_variant": "raw_no_vad",
            }

        # fallback 1: raw + VAD filter
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, raw_audio.astype(np.float32), self.sample_rate)
            t2, c2, i2, n2 = self._transcribe(tmp.name, use_vad_filter=True)
        if not self._is_poor_transcript(t2, c2, n2):
            return {
                "transcript": t2,
                "transcription_confidence": round(float(c2), 2),
                "no_speech_prob": round(float(n2), 4),
                "transcription_variant": "raw_vad",
            }

        # fallback 2: cleaned no-vad
        cleaned = self._clean_audio(raw_audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, cleaned.astype(np.float32), self.sample_rate)
            t3, c3, i3, n3 = self._transcribe(tmp.name, use_vad_filter=False)
        return {
            "transcript": t3,
            "transcription_confidence": round(float(c3), 2),
            "no_speech_prob": round(float(n3), 4),
            "transcription_variant": "clean_no_vad",
        }

    # =====================================================
    # FLOW B: FEATURE EXTRACTION / SCORING
    # =====================================================
    def extract_audio_features(self, raw_audio: np.ndarray, transcript: str) -> Dict:
        """
        Feature path can use cleaned audio. Must NOT alter transcript path.
        """
        processed = self._clean_audio(raw_audio)
        duration = len(processed) / float(self.sample_rate)
        return self._estimate_features(processed, duration_sec=duration, text=transcript)

    def get_response(self, round_index: int, duration_sec: int = 90, silence_seconds: float = 6.0) -> Dict:
        raw_audio = self._record_until_silence(max_seconds=duration_sec, silence_seconds=silence_seconds)

        stt = self.transcribe_audio_raw(raw_audio)
        text = stt["transcript"]
        conf = float(stt["transcription_confidence"])
        features = self.extract_audio_features(raw_audio, transcript=text)

        # communication proxy from ASR confidence + pacing + pauses
        pace = features["wpm"]
        pace_score = 1.0 - min(1.0, abs(pace - 135.0) / 135.0) if pace > 0 else 0.0
        pause_score = 1.0 - min(1.0, abs(features["pause_ratio"] - 0.22))
        comm = (0.55 * (conf / 100.0)) + (0.25 * pace_score) + (0.20 * pause_score)
        comm_score = max(0.0, min(100.0, comm * 100.0))

        return {
            "answer_text": text,
            "transcript": text,
            "communication_score": round(comm_score, 2),
            "confidence_score": round(conf, 2),
            "audio_features": features,
            "audio_analysis": {
                "transcription_confidence": round(conf, 2),
                "no_speech_prob": stt["no_speech_prob"],
                "transcription_variant": stt["transcription_variant"],
                "pause_ratio": features["pause_ratio"],
                "wpm": features["wpm"],
                "filler_density": features["filler_density"],
                "pitch_variation": features["pitch_variation"],
                "speech_consistency": features["speech_consistency"],
                "communication_score": round(comm_score, 2),
            },
            "transcription_confidence": round(conf, 2),
        }
