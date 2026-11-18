@echo off
rem Jarvis first-run setup (portable installer foundation). Double-click, or run from a terminal.
rem Finds Python 3.11+, then: creates the environment and installs dependencies, guides API-key entry, and reports readiness.
rem Stops at the first step that fails. jarvis_setup.py exit codes: 0 done, 1 done but not ready yet, 2 failed, 130 Ctrl+C.
setlocal
cd /d "%~dp0"

if not exist "installer\jarvis_setup.py" (
  echo This folder is not a complete Jarvis package: installer\jarvis_setup.py is missing.
  echo Extract the whole Jarvis ZIP again, then run Setup-Jarvis.bat from inside the extracted jarvis folder.
  pause
  exit /b 2
)

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found on this computer.
  echo Install Python 3.11 or newer from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this again.
  pause
  exit /b 2
)

set "STEP=1 of 3 - environment and dependencies"
echo === Jarvis setup: step %STEP% ===
%PY% installer\jarvis_setup.py init
if errorlevel 1 goto failed

echo.
set "STEP=2 of 3 - API keys"
echo === Jarvis setup: step %STEP% ^(what you type is not shown^) ===
%PY% installer\jarvis_setup.py keys
if errorlevel 1 goto failed

echo.
set "STEP=3 of 3 - final check"
echo === Jarvis setup: step %STEP% ===
%PY% installer\jarvis_setup.py check
if errorlevel 2 goto failed
if errorlevel 1 goto notready
echo.
echo To bring your data from another Jarvis folder:  %PY% installer\jarvis_setup.py migrate --from "C:\path\to\old\jarvis"
echo To start Jarvis: double-click Start-Jarvis.bat
pause
exit /b 0

:notready
echo.
echo Setup finished, but Jarvis is not ready to start yet - see MISSING above, fix it, then run this again.
pause
exit /b 1

:failed
set "CODE=%errorlevel%"
echo.
echo Setup stopped at step %STEP% ^(exit code %CODE%^). Nothing after this step was run.
echo Read the message above, fix what it names, and run Setup-Jarvis.bat again - finished steps are kept.
pause
exit /b %CODE%
