@echo off
setlocal
cd /d "%~dp0"
title MDAC2 - Local Chart Studio
if not exist ".venv\Scripts\python.exe" (
  echo MDAC2 Python environment is missing. See tools\start-workstation.md.
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo Node.js and npm are required. See web\README.md.
  pause
  exit /b 1
)
if not exist "web\node_modules" (
  call npm.cmd --prefix web ci
  if errorlevel 1 exit /b 1
)
call npm.cmd --prefix web run build
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
echo MDAC2: open the local URL below. No password required. Projects are saved in chart.
echo Keep this window open while using the studio. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m mdac2.server --root "%CD%" %*
pause
