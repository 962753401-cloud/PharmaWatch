@echo off
chcp 65001 >nul 2>&1
echo ================================================================
echo   Pharmacy Monitoring AI - All-in-One Launcher
echo   药店监控AI - 一键启动菜单
echo ================================================================
echo.
echo   [1] v1 Smoke Test  (port 8001) - Qwen3-VL video Q&A
echo   [2] v2 Behavior Detect (port 8001) - YOLO-World trigger
echo   [3] v3 Audio Scan (port 8002) - Audio-first keyword scan
echo   [q] Quit
echo.
set /p choice="Select version: "

if "%choice%"=="1" (
    cd /d "%~dp0v1_smoke_test\backend"
    python -m uvicorn main:app --host 0.0.0.0 --port 8001
) else if "%choice%"=="2" (
    cd /d "%~dp0v2_behavior_detect\backend"
    python -m uvicorn main:app --host 0.0.0.0 --port 8001
) else if "%choice%"=="3" (
    cd /d "%~dp0v3_audio_scan\backend"
    python -m uvicorn main:app --host 0.0.0.0 --port 8002
) else if "%choice%"=="q" (
    exit /b 0
) else (
    echo Invalid choice.
)
pause