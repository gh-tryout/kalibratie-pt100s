@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtuele omgeving ontbreekt. Maak eerst: py -3.12 -m venv .venv
  pause
  exit /b 1
)
echo Start PT100-kalibratie als website (upload via browser, bereikbaar op het netwerk)...
set PT100_WEB=1
".venv\Scripts\python.exe" launcher.py --web
pause
