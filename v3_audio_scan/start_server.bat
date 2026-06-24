@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0backend"
echo ========================================
echo   v3 Audio Scan Server
echo ========================================
echo.
echo Server: http://127.0.0.1:8002
echo.
echo Press Ctrl+C to stop.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8002
pause