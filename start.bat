@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtuele omgeving ontbreekt. Maak eerst: py -3.12 -m venv .venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launcher.py
pause
