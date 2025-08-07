@echo off
echo Starting Credit Risk Analytics System...
echo Activating virtual environment...

cd /d "E:\Credit Prediction System"
call .venv\Scripts\activate.bat
streamlit run streamlit_app.py

pause
