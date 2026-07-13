@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python non trovato nel PATH. Installa Python 3.10+ da https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creo l'ambiente virtuale in .venv ...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo.
echo === Verifica dipendenze (dentro il venv) ===
python setup.py
if errorlevel 1 (
    echo.
    echo Ci sono errori nelle dipendenze - controlla i messaggi sopra.
    pause
    exit /b 1
)

start "" cmd /c "timeout /t 3 >nul & start "" http://localhost:5000"

echo.
echo === Avvio ExoVision - chiudi questa finestra per fermare il server ===
python src\app.py

pause
