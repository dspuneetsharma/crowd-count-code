@echo off
echo PCC-Net Comprehensive Testing
echo ============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing/checking requirements...
pip install -r requirements_test.txt

echo.
echo Starting comprehensive test...
echo.

REM Run the test
python run_test.py

echo.
echo Test completed! Check the test_results folder for outputs.
echo.
pause
