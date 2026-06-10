@echo off
cd /d "%~dp0"

if exist "I:\anaconda\python.exe" (
    "I:\anaconda\python.exe" -m streamlit run app.py
    goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -m streamlit run app.py
    goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
    python -m streamlit run app.py
    goto :eof
)

echo Python was not found. Please install Python 3.10+ and run:
echo python -m pip install -r requirements.txt
echo python scripts/generate_dataset.py --n 500 --seed 20260609
echo python scripts/run_ocr.py --engine auto
echo python scripts/evaluate_ocr.py
echo python scripts/generate_report.py
pause
