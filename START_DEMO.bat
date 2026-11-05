@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Hay cai dat theo README.md truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
