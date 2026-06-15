
# HIA (Humanists Interview Agent)

HIA is an AI-powered adaptive mock interview platform with resume intelligence, role recommendations, automatic voice interview flow, CV engagement signals, and persistent session memory.

---

# Prerequisites

- macOS/Linux (or equivalent commands for Windows)
- Recommended browsers: Brave or DuckDuckGo Browser
- Chrome may require additional permission configuration on some systems
- Python 3.10+
- Node.js 18+ and npm
- Microphone + webcam permissions enabled for localhost
- OpenRouter API key

---

# Quick Start

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
./run_local.sh
```

### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate
run_local.bat
```

---

# Setup

Run the following commands in order.

## 1) Backend environment

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install fastapi uvicorn python-multipart pydantic python-dotenv
pip install openai sentence-transformers scikit-learn numpy
pip install faster-whisper soundfile noisereduce sounddevice
pip install opencv-python mediapipe onnxruntime
pip install pdfplumber
```

### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip

pip install fastapi uvicorn python-multipart pydantic python-dotenv
pip install openai sentence-transformers scikit-learn numpy
pip install faster-whisper soundfile noisereduce sounddevice
pip install opencv-python mediapipe onnxruntime
pip install pdfplumber
```

Create/update `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
```

Example `.env.example`:

```env
OPENROUTER_API_KEY=
```

---

## 2) CV model download (required for computer vision features)

### macOS / Linux
```bash
mkdir -p CV/models

curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
-o CV/models/face_landmarker.task

curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
-o CV/models/pose_landmarker.task
```

### Windows (CMD)
```cmd
mkdir CV\models

curl.exe -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -o CV\models\face_landmarker.task

curl.exe -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -o CV\models\pose_landmarker.task
```

### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path CV\models

Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile CV\models\face_landmarker.task

Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile CV\models\pose_landmarker.task
```

Optional explicit exports/variables in case of an error:

**macOS/Linux:**
```bash
export MEDIAPIPE_FACE_MODEL="CV/models/face_landmarker.task"
export MEDIAPIPE_POSE_MODEL="CV/models/pose_landmarker.task"
```

**Windows (CMD):**
```cmd
set MEDIAPIPE_FACE_MODEL=CV\models\face_landmarker.task
set MEDIAPIPE_POSE_MODEL=CV\models\pose_landmarker.task
```

**Windows (PowerShell):**
```powershell
$env:MEDIAPIPE_FACE_MODEL="CV\models\face_landmarker.task"
$env:MEDIAPIPE_POSE_MODEL="CV\models\pose_landmarker.task"
```

---

## 3) Frontend install

```bash
cd frontend
npm install
```

---

# Run

## One-command run (recommended)

### macOS / Linux
```bash
./run_local.sh
```

### Windows
Double-click `run_local.bat` or run:
```cmd
run_local.bat
```

Starts:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

## Manual run

### macOS / Linux

#### Terminal A
```bash
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

#### Terminal B
```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

### Windows (CMD)

#### Terminal A
```cmd
venv\Scripts\activate
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

#### Terminal B
```cmd
cd frontend
set VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

---

# Demo Flow

1. Upload a resume
2. Wait for resume processing and role recommendations
3. Select a target role
4. Start the live interview
5. Grant microphone and webcam permissions
6. Answer adaptive interview questions
7. Review final evaluation and feedback summary

---

# Database

SQLite DB:

- `data/processed_data/interview_sessions.db`

Core behavior:

- Resume hash based deduplication
- One resume → multiple sessions
- Max 3 interviews per session
- Full interview summaries persisted per interview
- Session delete removes linked interviews + memory rows

---

# Main API Endpoints

- `POST /upload_resume`
- `GET /resume_list`
- `GET /sessions`
- `POST /start_interview`
- `POST /transcribe_audio`
- `POST /submit_answer`
- `POST /end_interview`
- `POST /analyze_video_frame`
- `DELETE /session/{session_id}`
- `GET /session_interviews`
- `GET /interview_record/{interview_id}`

---

# Project Structure

```text
HackathonSubmission/
├── api/
│   └── server.py
├── CV/
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── types.py
│   │   ├── landmarks.py
│   │   ├── features.py
│   │   ├── metrics.py
│   │   ├── emotion_model.py
│   │   ├── pipeline.py
│   │   └── cli.py
│   ├── models/
│   │   ├── face_landmarker.task
│   │   └── pose_landmarker.task
│   ├── emotion-ferplus-8.onnx
│   ├── requirements-cv.txt
│   ├── run_cv.py
│   ├── run_cv_summary.py
│   └── run_emotion_eval.py
├── data/
│   ├── processed_data/
│   │   ├── interview_embeddings.npy
│   │   ├── interview_metadata.json
│   │   ├── job_embeddings.npy
│   │   ├── jobs_metadata.json
│   │   ├── interview_sessions.db
│   │   └── other generated processed artifacts
│   └── uploaded_resumes/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── context/
│   │   ├── hooks/
│   │   ├── layouts/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── styles/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── adaptive_difficulty.py
├── answer_evaluator.py
├── audio_adapter.py
├── audio_live_adapter.py
├── cv_adapter.py
├── database_manager.py
├── embed_jobs.py
├── interview_indexer.py
├── interview_memory_manager.py
├── interview_orchestrator.py
├── job_matcher.py
├── multimodal_evaluator.py
├── process_resumes.py
├── question_retriever.py
├── recommend_jobs.py
├── resume_agent.py
├── resume_registry.py
├── run_interview_pipeline.py
├── session_manager.py
├── skill_normalizer.py
├── run_local.sh
└── ARCHITECTURE_OVERVIEW.pdf
```

---

# File Responsibilities

- `api/server.py` — FastAPI entrypoint; frontend-facing APIs, interview runtime, CV/audio endpoints
- `interview_orchestrator.py` — question generation flow, adaptive sequence, session-aware question memory
- `question_retriever.py` — retrieval over embedded interview question bank
- `answer_evaluator.py` — LLM-side scoring and structured feedback
- `multimodal_evaluator.py` — technical + audio + CV fusion and final per-round evaluation
- `audio_live_adapter.py` — live microphone capture/transcription feature extraction helpers
- `cv_adapter.py` — converts CV frame metrics into interview-compatible analytics
- `database_manager.py` — SQLite access and schema operations
- `session_manager.py` — resume/session/interview lifecycle management
- `interview_memory_manager.py` — compressed longitudinal memory (topics/strengths/weaknesses)
- `resume_registry.py` — upload registry, hash match, resume-file mapping
- `resume_agent.py`, `process_resumes.py`, `skill_normalizer.py` — resume extraction and profile normalization
- `recommend_jobs.py`, `job_matcher.py`, `embed_jobs.py` — job recommendation pipeline
- `interview_indexer.py` — interview dataset indexing and embedding artifacts build/update
- `run_interview_pipeline.py` — terminal pipeline runner (non-UI test/ops run)
- `frontend/src/pages/*` — user flow pages (home → upload → processing → roles → live → summary)
- `frontend/src/context/AppContext.tsx` — central frontend app state
- `frontend/src/services/*` — API + TTS service layer

---

# Troubleshooting

### `uvicorn: command not found`

Activate the virtual environment first:

```bash
source venv/bin/activate
```

---

### CV reports no face detected

Confirm `CV/models/*.task` files exist and backend startup logs show non-`None` `vision_pipeline`.

---

### No browser voice

Check browser speech synthesis support and microphone permissions.

The platform was tested primarily on Brave and DuckDuckGo Browser.

---

### Audio decode errors

Verify frontend sends the recorded blob correctly and backend receives a supported audio format.

---

# Notes

- Do not commit `.env` files or API keys
- Add `.env` to `.gitignore`
- All generated embeddings and interview artifacts are stored in `data/processed_data/`
- Resume uploads are stored in `data/uploaded_resumes/`
