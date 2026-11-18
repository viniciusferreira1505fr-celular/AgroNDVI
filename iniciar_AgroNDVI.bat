@echo off
cd /d "%~dp0"
echo Iniciando AgroNDVI 1.0.0...
py -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Se aparecer erro de dependencia, execute: py -m pip install -r requirements.txt
)
pause
