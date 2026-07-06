"""
ExoVision — Riconoscimento oggetti con YOLOv8 nano e inserimento in SQLite
Dipendenze: pip install ultralytics
Nota: alla prima esecuzione scarica il modello (~6MB), poi lavora offline.
"""

import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("⚠️  ultralytics non trovato. Installa con: pip install ultralytics")
    sys.exit(1)


# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()

MODELLO           = _cfg["yolo"]["modello"]
CONFIDENZA_MINIMA = _cfg["yolo"]["confidenza_minima"]

FOTO_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}


# ─── Database ─────────────────────────────────────────────────────────────────

def init_tabella_oggetti(conn: sqlite3.Connection):
    """Crea la tabella oggetti se non esiste."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS oggetti (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            oggetto          TEXT,
            confidenza       REAL,
            bbox_x1          REAL,
            bbox_y1          REAL,
            bbox_x2          REAL,
            bbox_y2          REAL,
            data_estrazione  TEXT
        )
    """)
    conn.commit()


def file_gia_processato(conn: sqlite3.Connection, file_id: int) -> bool:
    """Controlla se il file è già stato processato."""
    row = conn.execute(
        "SELECT id FROM oggetti WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row is not None


def get_file_id(conn: sqlite3.Connection, path: str):
    """Recupera l'id del file dal database tramite il path."""
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,)
    ).fetchone()
    return row[0] if row else None


def inserisci_oggetti(conn: sqlite3.Connection, file_id: int, oggetti: list):
    """
    Inserisce la lista di oggetti rilevati nel database.
    Se la lista è vuota, inserisce un record con oggetto NULL
    per segnare il file come processato.
    """
    now = datetime.now().isoformat()

    if not oggetti:
        conn.execute("""
            INSERT INTO oggetti (file_id, oggetto, confidenza, data_estrazione)
            VALUES (?, NULL, NULL, ?)
        """, (file_id, now))
    else:
        for o in oggetti:
            conn.execute("""
                INSERT INTO oggetti
                    (file_id, oggetto, confidenza, bbox_x1, bbox_y1, bbox_x2, bbox_y2, data_estrazione)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                o["oggetto"],
                o["confidenza"],
                o["bbox_x1"], o["bbox_y1"],
                o["bbox_x2"], o["bbox_y2"],
                now
            ))
    conn.commit()


# ─── Riconoscimento oggetti ───────────────────────────────────────────────────

def rileva_oggetti(model: YOLO, path: str) -> list:
    """
    Rileva oggetti in un'immagine con YOLOv8.
    Restituisce lista di dizionari con oggetto, confidenza e bounding box.
    """
    try:
        # verbose=False per non stampare output YOLO ad ogni immagine
        risultati = model(path, verbose=False, conf=CONFIDENZA_MINIMA)

        oggetti = []
        for r in risultati:
            for box in r.boxes:
                oggetti.append({
                    "oggetto":    model.names[int(box.cls)],
                    "confidenza": round(float(box.conf), 3),
                    "bbox_x1":    round(float(box.xyxy[0][0]), 1),
                    "bbox_y1":    round(float(box.xyxy[0][1]), 1),
                    "bbox_x2":    round(float(box.xyxy[0][2]), 1),
                    "bbox_y2":    round(float(box.xyxy[0][3]), 1),
                })

        return oggetti

    except Exception as e:
        print(f"\n  ⚠️  Errore su {path}: {e}")
        return []


# ─── Scan cartella ────────────────────────────────────────────────────────────

def processa_cartella(cartella: str, db_path: str = None):
    """
    Scansiona una cartella, rileva oggetti in ogni immagine
    e salva i risultati in SQLite.
    """
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = sqlite3.connect(db_path)
    init_tabella_oggetti(conn)

    cartella = Path(cartella)
    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        conn.close()
        return

    print(f"\n📂 Cartella:  {cartella}")
    print(f"🗄️  Database:  {db_path}")
    print(f"🤖 Modello:   {MODELLO}")
    print(f"📊 Soglia:    {CONFIDENZA_MINIMA * 100:.0f}% confidenza minima")
    print(f"\n⏳ Caricamento modello YOLOv8 (solo la prima volta)...")

    model = YOLO(MODELLO)
    print("✅ Modello pronto.\n")

    processati    = 0
    con_oggetti   = 0
    senza_oggetti = 0
    saltati       = 0

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
        oggetti = rileva_oggetti(model, path)
        inserisci_oggetti(conn, file_id, oggetti)

        if oggetti:
            nomi = ", ".join(set(o["oggetto"] for o in oggetti))
            print(f"✅ {len(oggetti)} oggetti: {nomi}")
            con_oggetti += 1
        else:
            print("○  nessun oggetto rilevato")
            senza_oggetti += 1

        processati += 1

    print(f"\n✅ Riconoscimento completato:")
    print(f"   Processati:      {processati}")
    print(f"   Con oggetti:     {con_oggetti}")
    print(f"   Senza oggetti:   {senza_oggetti}")
    print(f"   Saltati:         {saltati}")
    conn.close()


# ─── Report ───────────────────────────────────────────────────────────────────

def mostra_risultati(db_path: str = "exovision.db"):
    """Mostra un riepilogo degli oggetti rilevati nel database."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n📊 Riepilogo oggetti nel database:\n")

        print("── Totali ──────────────────────────────")
        for row in conn.execute("""
            SELECT
                COUNT(DISTINCT file_id) as file_processati,
                COUNT(CASE WHEN oggetto IS NOT NULL THEN 1 END) as totale_oggetti,
                COUNT(DISTINCT oggetto) as categorie_uniche
            FROM oggetti
        """):
            print(f"  File processati:   {row[0]}")
            print(f"  Oggetti rilevati:  {row[1]}")
            print(f"  Categorie uniche:  {row[2]}")

        print("\n── Top 10 oggetti più frequenti ────────")
        for row in conn.execute("""
            SELECT oggetto, COUNT(*) as conteggio, ROUND(AVG(confidenza) * 100, 1) as conf_media
            FROM oggetti
            WHERE oggetto IS NOT NULL
            GROUP BY oggetto
            ORDER BY conteggio DESC
            LIMIT 10
        """):
            print(f"  {row[0]:<20} {row[1]:>4} volte  (conf. media {row[2]}%)")

        print("\n── Prime 5 immagini con più oggetti ────")
        for row in conn.execute("""
            SELECT f.nome_file, COUNT(o.id) as n_oggetti,
                   GROUP_CONCAT(DISTINCT o.oggetto) as lista
            FROM oggetti o JOIN files f ON o.file_id = f.id
            WHERE o.oggetto IS NOT NULL
            GROUP BY o.file_id
            ORDER BY n_oggetti DESC
            LIMIT 5
        """):
            print(f"  {row[0]}: {row[1]} oggetti — {row[2]}")

    except Exception as e:
        print(f"⚠️  Errore: {e} — assicurati di aver eseguito prima lo script principale.")
    finally:
        conn.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python exovision_yolo.py <cartella>       — rileva oggetti")
        print("  python exovision_yolo.py --report         — mostra riepilogo")
        print("  python exovision_yolo.py <cartella> <db>  — specifica il db")
        print()
        print("Es: python exovision_yolo.py ./archivio_flickr")
        sys.exit(1)

    if sys.argv[1] == "--report":
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        mostra_risultati(db)
    else:
        cartella = sys.argv[1]
        db = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
        processa_cartella(cartella, db)
