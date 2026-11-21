@echo off
cd /d "%~dp0"
wsl.exe --cd "%CD%" -e sh -lc "export PYTHONPATH=$HOME/.local/autobim/lib; exec python3 -m autobim_studio.desktop"
if errorlevel 1 pause
