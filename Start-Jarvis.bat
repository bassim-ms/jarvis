@echo off
rem Start Jarvis on this computer only (127.0.0.1:8000). Runs the readiness check first.
rem Exit code: jarvis_setup.py's own (1 = not ready, 2 = failed) or, once started, Jarvis's.
setlocal
cd /d "%~dp0"

set "PY="
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Run Setup-Jarvis.bat first.
  pause
  exit /b 2
)

%PY% installer\jarvis_setup.py launch
set "CODE=%errorlevel%"
if not "%CODE%"=="0" (
  echo.
  echo Jarvis did not start ^(exit code %CODE%^). If this is a new install, run Setup-Jarvis.bat first.
  pause
)
exit /b %CODE%
