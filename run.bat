@echo off
setlocal
cd /d "%~dp0"
echo Démarrage du service IA Lunetterie...
if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
) else (
    python -m venv venv
    call venv\Scripts\activate.bat
)
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
pause
