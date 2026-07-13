#!/usr/bin/env bash
# ExoVision (DeepSight) — launcher macOS/Linux
# Scritto e rivisto per correttezza sintattica, non ancora verificato su
# hardware reale macOS/Linux (sviluppato e testato solo su Windows).
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10+ non trovato. Installalo dal gestore pacchetti del tuo sistema."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creo l'ambiente virtuale in .venv ..."
    python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo
echo "=== Verifica dipendenze (dentro il venv) ==="
python setup.py

(
    sleep 3
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:5000
    elif command -v open >/dev/null 2>&1; then
        open http://localhost:5000
    fi
) &

echo
echo "=== Avvio ExoVision — premi Ctrl+C per fermare il server ==="
python src/app.py
