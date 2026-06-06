@echo off
echo Football Foul Detection System - Backend Server
echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting Flask backend server...
echo Backend will be available at http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
