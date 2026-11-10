@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat goto nosetup
call venv\Scripts\activate.bat
python luggage_app.py --selftest
pause
exit /b 0
:nosetup
echo Run SETUP_FIRST_TIME.bat first.
pause
exit /b 1
