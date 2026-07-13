#!/usr/bin/env bash
# ExoVision (DeepSight) — launcher macOS/Linux
# Scritto e rivisto per correttezza sintattica, non ancora verificato su
# hardware reale macOS/Linux (sviluppato e testato solo su Windows).
set -e
cd "$(dirname "$0")"

# Preferiamo Python 3.10-3.12: torch/tensorflow (da cui dipende deepface)
# spesso non hanno ancora build compatibili con versioni di Python molto
# recenti (es. una distro rolling che porta gia' Python 3.14 di default),
# causando un ResolutionImpossible di pip che prova invano ogni versione
# di deepface senza mai trovarne una installabile.
PYTHON_BIN=""
for candidato in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidato" >/dev/null 2>&1; then
        PYTHON_BIN="$candidato"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.10-3.12 non trovato. Installalo dal gestore pacchetti del tuo sistema."
    exit 1
fi

VERSIONE=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$VERSIONE" in
    3.10|3.11|3.12) ;;
    *)
        echo "Attenzione: $PYTHON_BIN e' Python $VERSIONE — le librerie IA (torch/tensorflow/deepface)"
        echo "spesso non hanno ancora build compatibili con versioni di Python cosi' recenti (o vecchie)."
        echo "Se l'installazione delle dipendenze fallisce piu' sotto, ti serve una Python 3.10-3.12"
        echo "in piu' su questo sistema (su alcune distro molto recenti potrebbe non essere ancora nei"
        echo "repository predefiniti: prova 'sudo apt install python3.12 python3.12-venv', e se il"
        echo "pacchetto non si trova valuta la PPA deadsnakes: sudo add-apt-repository ppa:deadsnakes/ppa)."
        ;;
esac

if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
    echo "Trovato un ambiente virtuale incompleto in .venv, lo ricreo..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo "Creo l'ambiente virtuale in .venv (con $PYTHON_BIN, versione $VERSIONE) ..."
    "$PYTHON_BIN" -m venv .venv
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo
    echo "Errore: la creazione del venv non ha prodotto .venv/bin/activate."
    echo "Su Debian/Ubuntu il modulo venv e' un pacchetto di sistema separato da python3:"
    echo "  sudo apt install python3-venv"
    echo "(o python3.<versione>-venv, es. python3.10-venv, a seconda della tua versione di Python)"
    exit 1
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
