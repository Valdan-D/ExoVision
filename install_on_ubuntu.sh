#!/usr/bin/env bash
# ExoVision (DeepSight) — installazione Python idoneo su Ubuntu/Debian
#
# Problema reale riscontrato: su una VM Ubuntu con solo Python 3.14 di sistema
# (release rolling/molto recente), "pip install" va in ResolutionImpossible
# provando ogni versione di deepface senza mai trovarne una installabile,
# perché torch/tensorflow non hanno ancora build per Python cosi' recente.
# Questo script verifica se sul sistema e' gia' presente una versione idonea
# (3.10-3.12) e, se manca, la installa (prima dai repository apt di default,
# poi dalla PPA deadsnakes se il pacchetto non e' disponibile).
set -e

verde()  { printf '\033[92m%s\033[0m\n' "$1"; }
rosso()  { printf '\033[91m%s\033[0m\n' "$1"; }
giallo() { printf '\033[93m%s\033[0m\n' "$1"; }

VERSIONI_IDONEE="3.12 3.11 3.10"

# ─── Verifica distro ──────────────────────────────────────────────────────────

if ! command -v apt-get >/dev/null 2>&1; then
    rosso "Questo script usa apt-get e va eseguito su Ubuntu/Debian."
    echo "Su un'altra distribuzione installa manualmente Python 3.10, 3.11 o 3.12"
    echo "(e il relativo pacchetto venv) con il gestore pacchetti del tuo sistema."
    exit 1
fi

# ─── Python idoneo gia' presente? ─────────────────────────────────────────────

PYTHON_TROVATO=""
for v in $VERSIONI_IDONEE; do
    if command -v "python$v" >/dev/null 2>&1; then
        PYTHON_TROVATO="python$v"
        break
    fi
done

if [ -n "$PYTHON_TROVATO" ]; then
    verde "✓ $PYTHON_TROVATO trovato — nessuna installazione necessaria."
    "$PYTHON_TROVATO" --version
    echo
    echo "Puoi procedere con: ./launch.sh"
    exit 0
fi

giallo "Nessuna versione idonea di Python (3.10-3.12) trovata sul sistema."
if command -v python3 >/dev/null 2>&1; then
    VERSIONE_ATTUALE=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    echo "Versione di sistema attuale: python3 -> $VERSIONE_ATTUALE"
fi

# ─── Serve sudo per installare pacchetti ──────────────────────────────────────

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo >/dev/null 2>&1; then
        rosso "Servono permessi di root per installare pacchetti e 'sudo' non e' disponibile."
        echo "Rilancia questo script come root, oppure installa manualmente:"
        echo "  python3.12 python3.12-venv"
        exit 1
    fi
    SUDO="sudo"
fi

# ─── Prova a installare python3.12 dai repository di default ────────────────

TARGET_VERSIONE="3.12"
TARGET_PKG="python$TARGET_VERSIONE"

echo
echo "=== Provo a installare $TARGET_PKG dai repository apt di default ==="
$SUDO apt-get update -y
if $SUDO apt-get install -y "$TARGET_PKG" "${TARGET_PKG}-venv"; then
    INSTALLATO=1
else
    INSTALLATO=0
fi

# ─── Se non disponibile, aggiungi la PPA deadsnakes e riprova ────────────────

if [ "$INSTALLATO" -ne 1 ] || ! command -v "$TARGET_PKG" >/dev/null 2>&1; then
    giallo "$TARGET_PKG non disponibile nei repository di default."
    echo "Aggiungo la PPA deadsnakes (build non ufficiali ma ampiamente usate"
    echo "per avere versioni di Python più vecchie/nuove di quella di sistema)..."
    $SUDO apt-get install -y software-properties-common
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -y
    if ! $SUDO apt-get install -y "$TARGET_PKG" "${TARGET_PKG}-venv"; then
        rosso "Installazione di $TARGET_PKG fallita anche dalla PPA deadsnakes."
        echo "Prova a installarlo manualmente o scegli un'altra versione tra 3.10/3.11."
        exit 1
    fi
fi

if ! command -v "$TARGET_PKG" >/dev/null 2>&1; then
    rosso "$TARGET_PKG installato ma non trovato nel PATH — riapri il terminale e riprova."
    exit 1
fi

echo
verde "✓ $TARGET_PKG installato correttamente:"
"$TARGET_PKG" --version
echo
echo "Puoi procedere con: ./launch.sh"
