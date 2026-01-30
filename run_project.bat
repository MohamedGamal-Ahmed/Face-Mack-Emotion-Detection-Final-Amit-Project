@echo off
:: Change directory to the folder where this script is located
pushd "%~dp0"

echo ======================================================
echo [SYSTEM] Starting Face Mask ^& Emotion Detection Project
echo ======================================================

:: Activate local virtual environment
if exist .venv\Scripts\activate.bat (
    set VENV_PATH=.venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found in .venv/
    echo Please make sure the .venv folder exists in this directory.
    pause
    exit /b
)

echo [SYSTEM] Activating virtual environment: %VENV_PATH%
call "%VENV_PATH%"

echo [SYSTEM] Launching Flask Application (app.py)...
python app.py

:: Keep window open if the app closes or crashes
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] The application crashed with error code %ERRORLEVEL%.
) else (
    echo.
    echo [SYSTEM] Application stopped manually.
)

pause
popd
