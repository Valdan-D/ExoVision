"""
ExoVision — Setup e verifica dipendenze
Esegui questo script prima di tutto il resto:
  python setup.py
"""

import sys
import os
import subprocess
import shutil
import platform

# Su Windows la console usa spesso una codepage (es. cp1252) che non sa
# codificare i caratteri Unicode usati qui sotto (box-drawing, emoji, ecc.),
# facendo crashare lo script con UnicodeEncodeError su una console "pulita"
# (es. lanciata dal launcher DeepSight.bat). Forziamo l'output a UTF-8,
# sostituendo i caratteri non rappresentabili invece di sollevare eccezioni.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


# ─── Colori terminale ─────────────────────────────────────────────────────────

def verde(t):  return f"\033[92m{t}\033[0m"
def rosso(t):  return f"\033[91m{t}\033[0m"
def giallo(t): return f"\033[93m{t}\033[0m"
def grassetto(t): return f"\033[1m{t}\033[0m"


# ─── Utility ──────────────────────────────────────────────────────────────────

def ok(msg):    print(f"  {verde('✓')} {msg}")
def errore(msg): print(f"  {rosso('✗')} {msg}")
def avviso(msg): print(f"  {giallo('!')} {msg}")
def titolo(msg): print(f"\n{grassetto(msg)}")


# ─── Controllo Python ─────────────────────────────────────────────────────────

def check_python():
    titolo("Python")
    versione = sys.version_info
    if versione >= (3, 10):
        ok(f"Python {versione.major}.{versione.minor}.{versione.micro}")
        return True
    else:
        errore(f"Python {versione.major}.{versione.minor} trovato — richiesta versione 3.10 o superiore")
        print(f"     Scarica da: https://www.python.org/downloads/")
        return False


# ─── Controllo FFmpeg ─────────────────────────────────────────────────────────

COMANDI_INSTALL_FFMPEG = {
    "Windows": ["winget", "install", "--id", "Gyan.FFmpeg", "-e"],
    "Darwin":  ["brew", "install", "ffmpeg"],
    "Linux":   ["sudo", "apt-get", "install", "-y", "ffmpeg"],
}


def check_ffmpeg():
    titolo("FFmpeg")
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True
            )
            prima_riga = result.stdout.split("\n")[0]
            ok(prima_riga)
            return True
        except Exception:
            ok("FFmpeg trovato")
            return True

    errore("FFmpeg non trovato")
    sistema = platform.system()
    comando = COMANDI_INSTALL_FFMPEG.get(sistema)

    if comando:
        avviso(f"Provo a installarlo con: {' '.join(comando)}")
        try:
            subprocess.run(comando, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            errore(f"Installazione automatica fallita ({e})")
        else:
            if shutil.which("ffmpeg"):
                ok("FFmpeg installato correttamente")
                return True
            avviso("Comando eseguito ma FFmpeg non risulta ancora nel PATH — potrebbe servire riaprire il terminale.")

    if sistema == "Windows":
        print(f"     Scarica da:  https://github.com/BtbN/FFmpeg-Builds/releases")
        print(f"     Estrai e aggiungi la cartella 'bin' al PATH di sistema.")
        print(f"     Guida:       https://www.wikihow.com/Install-FFmpeg-on-Windows")
    elif sistema == "Darwin":
        print(f"     Installa con: brew install ffmpeg")
    else:
        print(f"     Installa con: sudo apt install ffmpeg")
    return False


# ─── Installazione pip ────────────────────────────────────────────────────────

def installa_requirements():
    titolo("Dipendenze Python (requirements.txt)")

    try:
        from importlib.metadata import version, PackageNotFoundError

        with open("requirements.txt") as f:
            righe = [r.strip() for r in f if r.strip() and not r.startswith("#")]

        mancanti = []
        for pacchetto in righe:
            # NB: cerchiamo per nome di distribuzione pip (quello scritto in
            # requirements.txt), non per nome del modulo importabile — sono
            # diversi per Pillow (PIL), ffmpeg-python (ffmpeg) e faster-whisper
            # (faster_whisper). Confrontare gli import name con i nomi pip
            # (versione precedente) dava falsi negativi su questi tre pacchetti
            # ad ogni esecuzione, anche quando erano già installati.
            nome = pacchetto.split("==")[0].split(">=")[0].split("<=")[0].strip()
            try:
                version(nome)
                ok(nome)
            except PackageNotFoundError:
                avviso(f"{nome} — da installare")
                mancanti.append(pacchetto)

        if mancanti:
            print(f"\n  ⏳ Installazione pacchetti mancanti...")
            # Su alcune distro Linux /tmp e' un tmpfs piccolo (es. 4 GB): pip vi
            # scarica/scompatta temporaneamente i pacchetti prima di installarli,
            # e un'installazione combinata di librerie IA pesanti (torch,
            # tensorflow, cuda, ecc.) puo' saturarlo anche se il disco ha
            # ampio spazio libero, fallendo con un errore di quota/spazio
            # esaurito fuorviante. Usiamo una cartella temporanea dentro il
            # progetto stesso (stesso filesystem del venv, tipicamente molto
            # piu' capiente) per evitare il problema.
            tmp_pip = os.path.join(os.getcwd(), ".pip-tmp")
            os.makedirs(tmp_pip, exist_ok=True)
            env = os.environ.copy()
            for var in ("TMPDIR", "TEMP", "TMP"):
                env[var] = tmp_pip
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + mancanti,
                    capture_output=True, text=True, env=env
                )
            finally:
                shutil.rmtree(tmp_pip, ignore_errors=True)
            if result.returncode == 0:
                ok("Installazione completata")
                return True
            else:
                errore("Errore durante l'installazione:")
                print(result.stderr[-500:])
                return False
        else:
            ok("Tutti i pacchetti già installati")
            return True

    except FileNotFoundError:
        errore("requirements.txt non trovato — assicurati di essere nella root del progetto")
        return False
    except Exception as e:
        errore(f"Errore: {e}")
        return False


# ─── Verifica import ──────────────────────────────────────────────────────────

def verifica_import():
    titolo("Verifica import")

    moduli = [
        ("PIL",            "Pillow"),
        ("piexif",         "piexif"),
        ("ffmpeg",         "ffmpeg-python"),
        ("easyocr",        "easyocr"),
        ("chromadb",       "chromadb"),
        ("flask",          "flask"),
        ("ultralytics",    "ultralytics"),
        ("faster_whisper", "faster-whisper"),
        ("transformers",   "transformers"),
        ("torch",          "torch"),
        ("cv2",            "opencv-python"),
        ("deepface",       "deepface"),
        ("moviepy",        "moviepy"),
        ("tf_keras",       "tf-keras"),
        ("sentencepiece",  "sentencepiece"),
    ]

    tutti_ok = True
    for modulo, nome_pip in moduli:
        try:
            __import__(modulo)
            ok(modulo)
        except ImportError:
            errore(f"{modulo} — non importabile (pip install {nome_pip})")
            tutti_ok = False

    return tutti_ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(grassetto("\n╔══════════════════════════════╗"))
    print(grassetto(  "║     ExoVision — Setup        ║"))
    print(grassetto(  "╚══════════════════════════════╝"))

    risultati = {}

    risultati["python"]  = check_python()
    risultati["ffmpeg"]  = check_ffmpeg()
    risultati["pip"]     = installa_requirements()
    risultati["import"]  = verifica_import()

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    titolo("Riepilogo")

    tutto_ok = all(risultati.values())

    for nome, stato in risultati.items():
        if stato:
            ok(nome)
        else:
            errore(nome)

    if tutto_ok:
        print(f"\n{verde(grassetto('✓ ExoVision è pronto. Puoi iniziare!'))}")
        print(f"\n  Prossimo passo:")
        print(f"  python src/exovision_metadata.py ./tua-cartella-foto\n")
    else:
        print(f"\n{giallo(grassetto('! Risolvi gli errori segnalati e riesegui setup.py'))}\n")
        if not risultati["ffmpeg"]:
            print(f"  {giallo('FFmpeg è l\'unica dipendenza che non si installa con pip.')}")
            print(f"  {giallo('Segui le istruzioni sopra per il tuo sistema operativo.')}\n")

    return 0 if tutto_ok else 1


if __name__ == "__main__":
    sys.exit(main())
