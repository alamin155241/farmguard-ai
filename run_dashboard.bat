@echo off
setlocal
cd /d "%~dp0"
set "FARMGUARD_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%FARMGUARD_PY%" (
  echo Creating the FarmGuard Python environment...
  python -m venv --system-site-packages .venv
)
"%FARMGUARD_PY%" -c "import streamlit, pandas, paho.mqtt.client, numpy, sklearn, matplotlib, joblib, cv2, ultralytics" >nul 2>nul
if errorlevel 1 (
  echo Installing missing FarmGuard dashboard packages...
  "%FARMGUARD_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)
"%FARMGUARD_PY%" -m streamlit run dashboard.py
if errorlevel 1 pause
