@echo off
setlocal
cd /d "%~dp0"
title MDAC2 - Local Chart Studio
set "MDAC2_PYTHON="
set "MDAC2_WORKER="
if exist "runtime\python\python.exe" (
  rem Portable package: use the bundled CPU-only runtimes.
  set "MDAC2_PYTHON=runtime\python\python.exe"
  set "MDAC2_DEVICE=cpu"
  if not exist "runtime\python313\python.exe" (
    echo MDAC2 portable CPU worker runtime is missing.
    pause
    exit /b 1
  )
  set "MDAC2_WORKER=%CD%\runtime\python313\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  rem Source checkout: use the existing workstation environment.
  set "MDAC2_PYTHON=.venv\Scripts\python.exe"
  if exist "tools\.deps\pretraining-py313\Scripts\python.exe" set "MDAC2_WORKER=%CD%\tools\.deps\pretraining-py313\Scripts\python.exe"
) else (
  echo MDAC2 Python runtime is missing.
  echo Use the portable package with its runtime folder, or prepare .venv for the source checkout.
  pause
  exit /b 1
)
if defined MDAC2_WORKER set "MDAC2_WORKER_PYTHON=%MDAC2_WORKER%"
set PYTHONPATH=%CD%\src
echo.
echo MDAC2: open the local URL below. No password required. Projects are saved in chart.
echo Keep this window open while using the studio. Press Ctrl+C to stop.
echo.
"%MDAC2_PYTHON%" -m mdac2.server --root "%CD%" --runtime "%CD%\chart" %*
pause
