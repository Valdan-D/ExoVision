@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem Su una VM/macchina lenta, Flask puo' impiegare piu' di qualche secondo ad
rem avviarsi (import di librerie pesanti): un ritardo fisso prima di aprire il
rem browser puo' scattare troppo presto, mostrando un errore di connessione
rem invece della pagina. Questo blocco (invocato ricorsivamente su se stesso,
rem vedi ":attendi_apri" in fondo al file) fa polling reale sul server finche'
rem non risponde, con un timeout massimo di sicurezza per non restare bloccato
rem all'infinito se qualcosa va storto.
if "%~1"=="--attendi-apri" goto :attendi_apri

rem Preferiamo Python 3.10-3.12: torch/tensorflow (da cui dipende deepface)
rem spesso non hanno ancora build compatibili con versioni di Python molto
rem recenti appena uscite, causando un ResolutionImpossible di pip che
rem prova invano ogni versione di deepface senza mai trovarne una installabile.
set "PYCMD="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "pass" >nul 2>&1
    if not errorlevel 1 set "PYCMD=py -3.12"
)

rem Python 3.12 non trovato: proviamo a installarlo automaticamente con
rem winget prima di ripiegare su altre versioni gia' presenti sulla macchina
rem (stesso approccio gia' usato per FFmpeg in setup.py). Il launcher "py"
rem legge le installazioni dal registro di Windows, non solo dal PATH: subito
rem dopo l'installazione "py -3.12" e' gia' utilizzabile nella stessa sessione,
rem senza dover riaprire il terminale.
if not defined PYCMD (
    where winget >nul 2>&1
    if not errorlevel 1 (
        echo Python 3.12 non trovato: provo a installarlo con winget...
        winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
        py -3.12 -c "pass" >nul 2>&1
        if not errorlevel 1 (
            set "PYCMD=py -3.12"
        ) else (
            echo Installazione automatica non riuscita ^(o serve riavviare il terminale^) - provo altre versioni gia' presenti.
        )
    )
)

if not defined PYCMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 -c "pass" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -3.11"
    )
)

if not defined PYCMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.10 -c "pass" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -3.10"
    )
)

rem Nessuna 3.10/3.11/3.12 trovata: prima di arrenderci proviamo comunque
rem "py -3" (qualunque Python 3.x il launcher trovi come default, es. 3.14 su
rem una macchina dove è installata solo quella) invece di uscire subito.
rem Meglio avviare con un avviso (vedi il controllo poco sotto) che bloccarsi
rem con "Python non trovato" quando Python in realta' c'e', solo troppo recente.
if not defined PYCMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -c "pass" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -3"
    )
)

if not defined PYCMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)

if not defined PYCMD (
    echo Python non trovato nel PATH. Installa Python da https://www.python.org/downloads/
    echo ^(idealmente 3.10-3.12: vedi l'avviso qui sotto se ne trovo una diversa^)
    pause
    exit /b 1
)

for /f "delims=" %%v in ('!PYCMD! -c "import sys;print(str(sys.version_info[0])+chr(46)+str(sys.version_info[1]))" 2^>nul') do set "PYVER=%%v"
echo Uso !PYCMD! ^(Python !PYVER!^)

if "!PYVER!" NEQ "3.10" if "!PYVER!" NEQ "3.11" if "!PYVER!" NEQ "3.12" (
    echo.
    echo Attenzione: stai usando Python !PYVER! - le librerie IA ^(torch/tensorflow/deepface^)
    echo spesso non hanno ancora build compatibili con versioni di Python cosi' recenti ^(o vecchie^).
    echo Se l'installazione delle dipendenze fallisce piu' sotto, installa Python 3.12
    echo da https://www.python.org/downloads/ e assicurati che 'py -3.12' funzioni.
    echo.
)

if not exist ".venv" (
    echo Creo l'ambiente virtuale in .venv ...
    !PYCMD! -m venv .venv
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

start "" cmd /c ""%~f0" --attendi-apri"

echo.
echo === Avvio ExoVision - chiudi questa finestra per fermare il server ===
python src\app.py

pause
exit /b 0

:attendi_apri
set "TENTATIVI=0"
:attendi_loop
curl -s -o nul http://localhost:5000 >nul 2>&1
if not errorlevel 1 (
    start "" http://localhost:5000
    exit /b 0
)
set /a TENTATIVI+=1
if !TENTATIVI! GEQ 30 (
    rem Fallback di sicurezza: apri comunque il browser dopo ~30s anche se il
    rem polling non ha mai avuto risposta, invece di restare bloccato in
    rem silenzio (l'utente vede almeno l'errore di connessione nel browser,
    rem con l'indicazione che qualcosa nel server non e' partito).
    start "" http://localhost:5000
    exit /b 0
)
timeout /t 1 >nul
goto :attendi_loop
