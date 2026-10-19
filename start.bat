@echo off
setlocal
cd /d "%~dp0"
title MDAC2 - Local Chart Studio
if not exist "runtime\python\python.exe" (
  echo MDAC2 portable CPU Python runtime is missing.
  pause
  exit /b 1
)
set MDAC2_DEVICE=cpu
set MDAC2_WORKER_PYTHON=%CD%\runtime\python313\python.exe
set PYTHONPATH=%CD%\src
echo.
echo MDAC2: open the local URL below. No password required. Projects are saved in chart.
echo Keep this window open while using the studio. Press Ctrl+C to stop.
echo.
"runtime\python\python.exe" -m mdac2.server --root "%CD%" --runtime "%CD%\chart" %*
pause
