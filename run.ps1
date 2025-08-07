#!/usr/bin/env pwsh

Write-Host "Starting Credit Risk Analytics System..." -ForegroundColor Green
Write-Host "Using virtual environment..." -ForegroundColor Yellow

# Change to the project directory
Set-Location "E:\Credit Prediction System"

# Activate virtual environment and run Streamlit
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py

Write-Host "Application stopped." -ForegroundColor Yellow
