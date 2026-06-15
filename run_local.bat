@echo off
:: run_local.bat - Windows Command Prompt launcher for HIA
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist venv (
    echo [HIA] Virtual environment 'venv' not found.
    echo [HIA] Creating python virtual environment 'venv'...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [HIA] Failed to create venv automatically. Make sure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo [HIA] Activating virtual environment and verifying dependencies...
call venv\Scripts\activate

echo [HIA] Starting Backend API Server (Port 8000) in a new window...
start "HIA Backend Server" cmd /k "call venv\Scripts\activate && uvicorn api.server:app --host 0.0.0.0 --port 8000"

echo [HIA] Starting Frontend App (Port 5173) in a new window...
cd frontend
set VITE_API_BASE_URL=http://localhost:8000
start "HIA Frontend Dev" cmd /k "npm run dev"

echo [HIA] Services started. Open http://localhost:5173 in your browser to start.
pause
