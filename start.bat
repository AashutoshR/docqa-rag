@echo off
rem Starts DocQA and opens it in your browser. Close this window to stop it.
cd /d "%~dp0"
start "" http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn server:app --port 8000
pause
