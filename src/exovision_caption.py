"""
ExoVision — Didascalia automatica delle foto (image captioning) con BLIP
Dipendenze: pip install transformers torch Pillow
Nota: alla prima esecuzione scarica il modello (~1GB), poi lavora offline.
La didascalia è generata in inglese (i modelli BLIP pubblici più affidabili
sono addestrati in inglese) ed è un campo distinto dalla descrizione manuale
scritta dall'utente in UI.
"""

import sqlite3
import shutil
import sys
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    CAPTION_OK = True
except ImportError:
    CAPTION_OK = False
    # NB: niente sys.exit qui — vedi commento equivalente in exovision_ocr.py
    # (importato anche da app.py per l'elaborazione in background).


# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
_CONFIG_EXAMPLE_PATH = Path(__file__).parent.parent / "config.example.json"

def _load_config():
    if not _CONFIG_PATH.exists() and _CONFIG_EXAMPLE_PATH.exists():
        shutil.copy(_CONFIG_EXAMPLE_PATH, _CONFIG_PATH)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()
_caption_cfg = _cfg.get("caption", {})

MODELLO = _caption_cfg.get("modello", "Salesforce/blip-image-captioning-base")

FOTO_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}


# ─── Database ─────────────────────────────────────────────────────────────────

def init_tabella_didascalie(conn: sqlite3.Connection):
    """Crea la tabella didascalie se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS didascalie (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            data_estrazione  TEXT
        )
    """)
    conn.commit()


def file_gia_processato(conn: sqlite3.Connection, file_id: int) -> bool:
    """Controlla se il file ha già una didascalia generata."""
    row = conn.execute(
        "SELECT id FROM didascalie WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row is not None


def get_file_id(conn: sqlite3.Connection, path: str):
    """Recupera l'id del file dal database tramite il path."""
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


def inserisci_didascalia(conn: sqlite3.Connection, file_id: int, testo, lingua="en"):
    """Inserisce la didascalia nel database."""
    conn.execute("""
        INSERT INTO didascalie (file_id, testo, lingua, data_estrazione)
        VALUES (?, ?, ?, ?)
    """, (
        file_id,
        testo,
        lingua if testo else None,
        datetime.now().isoformat()
    ))
    conn.commit()


# ─── Generazione didascalia ───────────────────────────────────────────────────

def genera_didascalia(model: "BlipForConditionalGeneration", processor: "BlipProcessor", path: str):
    """
    Genera una didascalia (in inglese) per un'immagine con BLIP.
    Restituisce il testo oppure None in caso di errore.
    """
    try:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            inputs = processor(img, return_tensors="pt")
            # num_beams + repetition_penalty/no_repeat_ngram_size: senza, il greedy
            # decoding di BLIP tende a incastrarsi in loop ripetitivi
            # (es. "the porsche 911 911 911 911...") invece di una frase sensata.
            out = model.generate(
                **inputs, max_new_tokens=30, num_beams=3,
                repetition_penalty=1.5, no_repeat_ngram_size=3
            )
            testo = processor.decode(out[0], skip_special_tokens=True).strip()

        return testo or None

    except Exception as e:
        print(f"\n  ⚠️  Errore didascalia su {path}: {e}")
        return None


# ─── Scan cartella ────────────────────────────────────────────────────────────

def processa_cartella(cartella: str, db_path: str = None):
    """
    Scansiona una cartella, genera una didascalia per ogni immagine
    e la inserisce in SQLite.
    """
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = sqlite3.connect(db_path)
    init_tabella_didascalie(conn)

    cartella = Path(cartella)
    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        conn.close()
        return

    print(f"\n📂 Cartella:  {cartella}")
    print(f"🗄️  Database:  {db_path}")
    print(f"🧠 Modello:   {MODELLO}")
    print(f"\n⏳ Caricamento modello BLIP (solo la prima volta)...")

    processor = BlipProcessor.from_pretrained(MODELLO)
    model = BlipForConditionalGeneration.from_pretrained(MODELLO)
    print("✅ Modello pronto.\n")

    processati = 0
    con_testo  = 0
    errori     = 0
    saltati    = 0

    for file_path in sorted(cartella.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in FOTO_EXT:
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

        print(f"  🖼️  {file_path.name}", end=" ... ", flush=True)
        testo = genera_didascalia(model, processor, path)

        if testo:
            inserisci_didascalia(conn, file_id, testo)
            print(f"✅ \"{testo}\"")
            con_testo += 1
        else:
            inserisci_didascalia(conn, file_id, None)
            print("○  nessuna didascalia generata")
            errori += 1

        processati += 1

    print(f"\n✅ Didascalie completate:")
    print(f"   Processati:   {processati}")
    print(f"   Con testo:    {con_testo}")
    print(f"   Errori:       {errori}")
    print(f"   Saltati:      {saltati}")
    conn.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def mostra_risultati(db_path: str = "exovision.db"):
    """Mostra un riepilogo delle didascalie nel database."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n📊 Riepilogo didascalie nel database:\n")

        print("── Totali ──────────────────────────────")
        for row in conn.execute("""
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN testo IS NOT NULL THEN 1 ELSE 0 END) as con_testo,
                SUM(CASE WHEN testo IS NULL THEN 1 ELSE 0 END) as senza_testo
            FROM didascalie
        """):
            print(f"  Totale processati: {row[0]}")
            print(f"  Con didascalia:    {row[1]}")
            print(f"  Senza didascalia:  {row[2]}")

        print("\n── Prime 5 ──────────────────────────────")
        for row in conn.execute("""
            SELECT f.nome_file, d.testo
            FROM didascalie d JOIN files f ON d.file_id = f.id
            WHERE d.testo IS NOT NULL
            ORDER BY d.id DESC
            LIMIT 5
        """):
            print(f"  {row[0]}: \"{row[1]}\"")

    except Exception as e:
        print(f"⚠️  Errore: {e} — assicurati di aver eseguito prima lo script principale.")
    finally:
        conn.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not CAPTION_OK:
        print("⚠️  transformers non trovato. Installa con: pip install transformers torch Pillow")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python exovision_caption.py <cartella>       — genera le didascalie")
        print("  python exovision_caption.py --report         — mostra riepilogo")
        print("  python exovision_caption.py <cartella> <db>  — specifica il db")
        print()
        print("Es: python exovision_caption.py ./archivio_flickr")
        sys.exit(1)

    if sys.argv[1] == "--report":
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        mostra_risultati(db)
    else:
        cartella = sys.argv[1]
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        processa_cartella(cartella, db)
