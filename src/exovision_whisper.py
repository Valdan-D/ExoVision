"""
ExoVision — Trascrizione audio dei video con faster-whisper e inserimento in SQLite
Dipendenze: pip install faster-whisper
Nota: alla prima esecuzione scarica il modello scelto, poi lavora offline.
La decodifica audio è inclusa in faster-whisper, non serve FFmpeg a parte.
"""

import sqlite3
import sys
import json
import math
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    # NB: niente sys.exit qui — vedi commento equivalente in exovision_ocr.py
    # (importato anche da app.py per l'elaborazione in background).


# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()
_whisper_cfg = _cfg.get("whisper", {})

MODELLO           = _whisper_cfg.get("modello", "small")
LINGUA            = _whisper_cfg.get("lingua")  # None = rilevamento automatico
CONFIDENZA_MINIMA = _whisper_cfg.get("confidenza_minima", 0.4)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}


# ─── Database ─────────────────────────────────────────────────────────────────

def init_tabella_trascrizioni(conn: sqlite3.Connection):
    """Crea la tabella trascrizioni se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trascrizioni (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            confidenza       REAL,
            data_estrazione  TEXT
        )
    """)
    conn.commit()


def get_file_id(conn: sqlite3.Connection, path: str):
    """Recupera l'id del file dal database tramite il path."""
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


def file_gia_processato(conn: sqlite3.Connection, file_id: int) -> bool:
    """Controlla se il video è già stato trascritto."""
    row = conn.execute(
        "SELECT id FROM trascrizioni WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row is not None


def inserisci_trascrizione(conn: sqlite3.Connection, file_id: int, testo, lingua, confidenza):
    """
    Inserisce la trascrizione nel database.
    Viene inserita una riga anche quando non c'è testo (nessun parlato o errore),
    con campi NULL, per marcare il video come processato e non ritentarlo ad ogni run.
    """
    conn.execute("""
        INSERT INTO trascrizioni (file_id, testo, lingua, confidenza, data_estrazione)
        VALUES (?, ?, ?, ?, ?)
    """, (
        file_id,
        testo,
        lingua,
        confidenza,
        datetime.now().isoformat()
    ))
    conn.commit()


# ─── Trascrizione ─────────────────────────────────────────────────────────────

def estrai_testo(model: WhisperModel, path: str):
    """
    Trascrive l'audio di un video con faster-whisper.
    Restituisce (testo, lingua_rilevata, confidenza_media) oppure (None, lingua, None).
    """
    try:
        segments, info = model.transcribe(path, language=LINGUA, vad_filter=True)

        segmenti_validi = []
        for seg in segments:
            confidenza = round(math.exp(seg.avg_logprob), 3)
            if confidenza >= CONFIDENZA_MINIMA and seg.text.strip():
                segmenti_validi.append((seg.text.strip(), confidenza))

        if not segmenti_validi:
            return None, info.language, None

        testo_unito = " ".join(t for t, _ in segmenti_validi)
        confidenza_media = round(
            sum(c for _, c in segmenti_validi) / len(segmenti_validi), 3
        )

        return testo_unito, info.language, confidenza_media

    except Exception as e:
        print(f"\n  ⚠️  Errore trascrizione su {path}: {e}")
        return None, None, None


# ─── Scan cartella ────────────────────────────────────────────────────────────

def processa_cartella(cartella: str, db_path: str = None):
    """
    Scansiona una cartella, trascrive l'audio di ogni video
    e salva il risultato in SQLite.
    """
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = sqlite3.connect(db_path)
    init_tabella_trascrizioni(conn)

    cartella = Path(cartella)
    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        conn.close()
        return

    print(f"\n📂 Cartella:  {cartella}")
    print(f"🗄️  Database:  {db_path}")
    print(f"🌐 Lingua:    {LINGUA or 'rilevamento automatico'}")
    print(f"📊 Soglia:    {CONFIDENZA_MINIMA * 100:.0f}% confidenza minima")
    print(f"\n⏳ Caricamento modello faster-whisper '{MODELLO}' (solo la prima volta)...")

    # CPU + int8 per garantire compatibilità su qualsiasi PC senza GPU
    model = WhisperModel(MODELLO, device="cpu", compute_type="int8")
    print("✅ Modello pronto.\n")

    processati  = 0
    con_testo   = 0
    senza_testo = 0
    saltati     = 0

    for file_path in sorted(cartella.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXT:
            continue

        path = str(file_path)

        file_id = get_file_id(conn, path)
        if not file_id:
            print(f"  ⏭️  {file_path.name} — non nel database, esegui prima exovision_metadata.py")
            saltati += 1
            continue

        if file_gia_processato(conn, file_id):
            saltati += 1
            continue

        print(f"  🎙️  {file_path.name}", end=" ... ", flush=True)
        testo, lingua, confidenza = estrai_testo(model, path)
        inserisci_trascrizione(conn, file_id, testo, lingua, confidenza)

        if testo:
            anteprima = testo[:60] + "..." if len(testo) > 60 else testo
            print(f"✅ [{lingua}, {confidenza*100:.0f}%] \"{anteprima}\"")
            con_testo += 1
        else:
            motivo = "nessun parlato rilevato" if lingua else "errore trascrizione"
            print(f"○  {motivo}")
            senza_testo += 1

        processati += 1

    print(f"\n✅ Trascrizione completata:")
    print(f"   Processati:   {processati}")
    print(f"   Con testo:    {con_testo}")
    print(f"   Senza testo:  {senza_testo}")
    print(f"   Saltati:      {saltati}")
    conn.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def mostra_risultati(db_path: str = "exovision.db"):
    """Mostra un riepilogo delle trascrizioni nel database."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n📊 Riepilogo trascrizioni nel database:\n")

        print("── Totali ──────────────────────────────")
        for row in conn.execute("""
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN testo IS NOT NULL THEN 1 ELSE 0 END) as con_testo,
                SUM(CASE WHEN testo IS NULL THEN 1 ELSE 0 END) as senza_testo,
                ROUND(AVG(CASE WHEN confidenza IS NOT NULL THEN confidenza END) * 100, 1) as conf_media
            FROM trascrizioni
        """):
            print(f"  Totale processati: {row[0]}")
            print(f"  Con testo:         {row[1]}")
            print(f"  Senza testo:       {row[2]}")
            print(f"  Confidenza media:  {row[3]}%")

        print("\n── Prime 5 con testo ───────────────────")
        for row in conn.execute("""
            SELECT f.nome_file, t.testo, ROUND(t.confidenza * 100, 1)
            FROM trascrizioni t JOIN files f ON t.file_id = f.id
            WHERE t.testo IS NOT NULL
            ORDER BY t.confidenza DESC
            LIMIT 5
        """):
            anteprima = row[1][:70] + "..." if len(row[1]) > 70 else row[1]
            print(f"  [{row[2]}%] {row[0]}: \"{anteprima}\"")

    except Exception as e:
        print(f"⚠️  Errore: {e} — assicurati di aver eseguito prima lo script principale.")
    finally:
        conn.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not WHISPER_OK:
        print("⚠️  faster-whisper non trovato. Installa con: pip install faster-whisper")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python exovision_whisper.py <cartella>       — trascrive l'audio")
        print("  python exovision_whisper.py --report         — mostra riepilogo")
        print("  python exovision_whisper.py <cartella> <db>  — specifica il db")
        print()
        print("Es: python exovision_whisper.py ./archivio_flickr")
        sys.exit(1)

    if sys.argv[1] == "--report":
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        mostra_risultati(db)
    else:
        cartella = sys.argv[1]
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        processa_cartella(cartella, db)
