"""
ExoVision — Estrazione testo OCR da immagini con EasyOCR
Dipendenze: pip install easyocr Pillow
Niente installazioni di sistema — tutto gestito da pip.
Nota: alla prima esecuzione scarica i modelli (~500MB), poi lavora offline.
"""

import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import easyocr
    from PIL import Image
    import numpy as np
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False
    print("⚠️  easyocr non trovato. Installa con: pip install easyocr Pillow")
    sys.exit(1)


# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()

LINGUE            = _cfg["ocr"]["lingue"]
CONFIDENZA_MINIMA = _cfg["ocr"]["confidenza_minima"]

FOTO_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}


# ─── Database ─────────────────────────────────────────────────────────────────

def init_tabella_ocr(conn: sqlite3.Connection):
    """Crea la tabella ocr se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ocr (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            confidenza       REAL,
            data_estrazione  TEXT
        )
    """)
    conn.commit()


def file_gia_processato(conn: sqlite3.Connection, file_id: int) -> bool:
    """Controlla se il file è già stato processato con OCR."""
    row = conn.execute(
        "SELECT id FROM ocr WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row is not None


def get_file_id(conn: sqlite3.Connection, path: str):
    """Recupera l'id del file dal database tramite il path."""
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


def inserisci_ocr(conn: sqlite3.Connection, file_id: int, testo, lingua, confidenza):
    """Inserisce il risultato OCR nel database."""
    conn.execute("""
        INSERT INTO ocr (file_id, testo, lingua, confidenza, data_estrazione)
        VALUES (?, ?, ?, ?, ?)
    """, (
        file_id,
        testo,
        lingua,
        confidenza,
        datetime.now().isoformat()
    ))
    conn.commit()


# ─── Estrazione OCR ───────────────────────────────────────────────────────────

def estrai_testo(reader: easyocr.Reader, path: str):
    """
    Estrae testo da un'immagine con EasyOCR.
    Restituisce (testo, confidenza_media) oppure (None, None).
    """
    try:
        with Image.open(path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img_array = np.array(img)

        # EasyOCR restituisce lista di (bbox, testo, confidenza)
        risultati = reader.readtext(img_array)

        if not risultati:
            return None, None

        # Filtra per confidenza minima
        risultati_validi = [
            (testo, conf)
            for _, testo, conf in risultati
            if conf >= CONFIDENZA_MINIMA and testo.strip()
        ]

        if not risultati_validi:
            return None, None

        testo_unito = " ".join(t for t, _ in risultati_validi)
        confidenza_media = round(
            sum(c for _, c in risultati_validi) / len(risultati_validi), 3
        )

        return testo_unito, confidenza_media

    except Exception as e:
        print(f"\n  ⚠️  Errore OCR su {path}: {e}")
        return None, None


# ─── Scan cartella ────────────────────────────────────────────────────────────

def processa_cartella(cartella: str, db_path: str = None):
    """
    Scansiona una cartella, estrae testo OCR da ogni immagine
    e lo inserisce in SQLite.
    """
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = sqlite3.connect(db_path)
    init_tabella_ocr(conn)

    cartella = Path(cartella)
    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        conn.close()
        return

    print(f"\n📂 Cartella:  {cartella}")
    print(f"🗄️  Database:  {db_path}")
    print(f"🌐 Lingue:    {', '.join(LINGUE)}")
    print(f"📊 Soglia:    {CONFIDENZA_MINIMA * 100:.0f}% confidenza minima")
    print(f"\n⏳ Caricamento modelli EasyOCR (solo la prima volta)...")

    # GPU=False per garantire compatibilità su qualsiasi PC
    reader = easyocr.Reader(LINGUE, gpu=False)
    print("✅ Modelli pronti.\n")

    processati  = 0
    con_testo   = 0
    senza_testo = 0
    saltati     = 0

    for file_path in sorted(cartella.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in FOTO_EXT:
            continue

        path = str(file_path)

        # Controlla che il file sia nel DB
        file_id = get_file_id(conn, path)
        if not file_id:
            print(f"  ⏭️  {file_path.name} — non nel database, esegui prima exovision_metadata.py")
            saltati += 1
            continue

        # Salta se già processato
        if file_gia_processato(conn, file_id):
            saltati += 1
            continue

        print(f"  🔍 {file_path.name}", end=" ... ", flush=True)
        testo, confidenza = estrai_testo(reader, path)

        if testo:
            inserisci_ocr(conn, file_id, testo, "+".join(LINGUE), confidenza)
            anteprima = testo[:60] + "..." if len(testo) > 60 else testo
            print(f"✅ [{confidenza*100:.0f}%] \"{anteprima}\"")
            con_testo += 1
        else:
            inserisci_ocr(conn, file_id, None, None, confidenza)
            motivo = f"confidenza bassa ({confidenza*100:.0f}%)" if confidenza else "nessun testo"
            print(f"○  {motivo}")
            senza_testo += 1

        processati += 1

    print(f"\n✅ OCR completato:")
    print(f"   Processati:   {processati}")
    print(f"   Con testo:    {con_testo}")
    print(f"   Senza testo:  {senza_testo}")
    print(f"   Saltati:      {saltati}")
    conn.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def mostra_risultati(db_path: str = "exovision.db"):
    """Mostra un riepilogo dei risultati OCR nel database."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n📊 Riepilogo OCR nel database:\n")

        print("── Totali ──────────────────────────────")
        for row in conn.execute("""
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN testo IS NOT NULL THEN 1 ELSE 0 END) as con_testo,
                SUM(CASE WHEN testo IS NULL THEN 1 ELSE 0 END) as senza_testo,
                ROUND(AVG(CASE WHEN confidenza IS NOT NULL THEN confidenza END) * 100, 1) as conf_media
            FROM ocr
        """):
            print(f"  Totale processati: {row[0]}")
            print(f"  Con testo:         {row[1]}")
            print(f"  Senza testo:       {row[2]}")
            print(f"  Confidenza media:  {row[3]}%")

        print("\n── Prime 5 con testo ───────────────────")
        for row in conn.execute("""
            SELECT f.nome_file, o.testo, ROUND(o.confidenza * 100, 1)
            FROM ocr o JOIN files f ON o.file_id = f.id
            WHERE o.testo IS NOT NULL
            ORDER BY o.confidenza DESC
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
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python exovision_ocr.py <cartella>       — estrae OCR")
        print("  python exovision_ocr.py --report         — mostra riepilogo")
        print("  python exovision_ocr.py <cartella> <db>  — specifica il db")
        print()
        print("Es: python exovision_ocr.py ./archivio_flickr")
        sys.exit(1)

    if sys.argv[1] == "--report":
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        mostra_risultati(db)
    else:
        cartella = sys.argv[1]
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        processa_cartella(cartella, db)
