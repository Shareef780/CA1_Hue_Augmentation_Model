@echo off
setlocal
title Luggage demo: first-time setup
cd /d "%~dp0"

echo ===== 1. Python =====
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.13 --version >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
%PY% --version

echo ===== 2. Virtual environment =====
if not exist venv\Scripts\python.exe %PY% -m venv venv
if not exist venv\Scripts\python.exe goto novenv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo ===== 3. Packages. This downloads a few hundred MB, so it takes a while =====
pip install ultralytics==8.4.120 gradio
if errorlevel 1 goto failed

echo ===== 4. Self test =====
python luggage_app.py --selftest
if errorlevel 1 goto failed

echo.
echo Setup finished. To start the demo run RUN_DEMO.bat from Command Prompt.
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.12 from python.org, tick Add python.exe to PATH, then run this again.
pause
exit /b 1

:novenv
echo Could not create the virtual environment.
pause
exit /b 1

:failed
echo Something failed. Send the last lines of this window to Claude.
pause
exit /b 1
