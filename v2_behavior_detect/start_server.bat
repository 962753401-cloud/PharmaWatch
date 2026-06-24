@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0backend"
echo ========================================
echo   v2 Behavior Detection Server
echo ========================================
echo.
echo Server: http://127.0.0.1:8001
echo.
echo Press Ctrl+C to stop.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8001
pause