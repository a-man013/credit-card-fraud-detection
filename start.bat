@echo off
echo ========================================================
echo   Starting FraudShield AI (Backend + Frontend)
echo ========================================================
echo.
echo [1/2] Launching FastAPI Backend on http://localhost:8000...
start "FraudShield Backend (Port 8000)" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --reload --port 8000"

echo [2/2] Launching Vite React Frontend on http://localhost:5173...
start "FraudShield Frontend (Port 5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers have been launched in separate terminal windows!
echo - Frontend: http://localhost:5173
echo - Backend Docs: http://localhost:8000/docs
echo.
pause
