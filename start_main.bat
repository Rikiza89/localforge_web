@echo off
:: Navigate to the script's directory (optional but recommended)
cd /d "%~dp0"

:: 1. Activate the virtual environment
call venv\Scripts\activate

:: 2. Run your Python script
python main.py

:: 3. Keep the window open if the script crashes or finishes
pause