"""
ExoVision — Estrazione keyframe dai video (scene detection) e inserimento in SQLite
Dipendenze: pip install ffmpeg-python
FFmpeg deve essere installato sul sistema: https://ffmpeg.org/download.html
"""

import sqlite3
import sys
import re
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import ffmpeg
    FFMPEG_OK = True
except ImportError:
    FFMPEG_OK = False
    # NB: niente sys.exit qui — vedi commento equivalente in exovision_ocr.py
    # (importato anche da app.py per l'elaborazione in background).


# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()

SOGLIA_SCENA   = _cfg["video"].get("soglia_scena", 0.4)
CARTELLA_FRAME = _cfg["video"].get("cartella_frame", "frame")

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

_PTS_RE = re.compile(rb"pts_time:([\d.]+)")


# ─── Database ─────────────────────────────────────────────────────────────────

def init_tabella_frame(conn: sqlite3.Connection):
    """Crea la tabella frame se non esiste (normalmente già creata da exovision_metadata.py)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS frame (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id         INTEGER NOT NULL REFERENCES files(id),
            timestamp_sec   REAL,
            path_frame      TEXT
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
    """Controlla se il video è già stato processato."""
    row = conn.execute(
        "SELECT id FROM frame WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row is not None


def inserisci_frame(conn: sqlite3.Connection, file_id: int, frames: list):
    """
    Inserisce i keyframe estratti nel database.
    Se la lista è vuota (errore ffmpeg o nessuna scena rilevata), inserisce
    comunque una riga segnaposto (timestamp/path NULL) per marcare il video
    come processato — altrimenti verrebbe ritentato ad ogni esecuzione.
    """
    if not frames:
        conn.execute("""
            INSERT INTO frame (file_id, timestamp_sec, path_frame)
            VALUES (?, NULL, NULL)
        """, (file_id,))
    else:
        for fr in frames:
            conn.execute("""
                INSERT INTO frame (file_id, timestamp_sec, path_frame)
                VALUES (?, ?, ?)
            """, (file_id, fr["timestamp_sec"], fr["path_frame"]))
    conn.commit()


# ─── Estrazione keyframe ──────────────────────────────────────────────────────

def _estrai_frame_singolo(path: str, cartella_out: Path, file_id: int, video_stem: str, timestamp: float = 1.0):
    """
    Estrae un singolo frame a un timestamp fisso — usato come fallback quando
    la scene detection non trova alcun taglio di scena (comunissimo nei video
    brevi a piano continuo, es. b-roll: senza questo fallback quei video non
    avrebbero mai un'anteprima statica in UI).
    """
    out_path = cartella_out / f"fr-{file_id}-0-{video_stem}.jpg"
    try:
        (
            ffmpeg
            .input(path, ss=timestamp)
            .output(str(out_path), vframes=1)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        print(f"  ⚠️  Errore estrazione frame singolo su {path}: {stderr.splitlines()[-1] if stderr else ''}")
        return None

    if not out_path.exists():
        return None
    return {"path_frame": str(out_path), "timestamp_sec": round(timestamp, 2)}


def estrai_keyframe(path: str, cartella_out: Path, file_id: int, video_stem: str) -> list:
    """
    Estrae i keyframe di un video con scene detection ffmpeg.
    Salva i frame come <cartella_out>/fr-<file_id>-<n>-<video_stem>.jpg
    (il file_id evita collisioni tra video con lo stesso nome in cartelle diverse)
    e restituisce la lista {timestamp_sec, path_frame} nell'ordine di estrazione.
    Se la scene detection non trova alcun taglio di scena (o fallisce), prova
    a estrarre un singolo frame fisso come anteprima di riserva.
    """
    prefisso = f"fr-{file_id}-"
    pattern  = str(cartella_out / f"{prefisso}%d-{video_stem}.jpg")
    num_re   = re.compile(re.escape(prefisso) + r"(\d+)-")

    try:
        _, err = (
            ffmpeg
            .input(path)
            .filter("select", f"gt(scene,{SOGLIA_SCENA})")
            .filter("showinfo")
            .output(pattern, vsync="vfr", start_number=1)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        print(f"  ⚠️  Errore ffmpeg su {path}: {stderr.splitlines()[-1] if stderr else ''}")
        fallback = _estrai_frame_singolo(path, cartella_out, file_id, video_stem)
        return [fallback] if fallback else []

    timestamp = [float(m) for m in _PTS_RE.findall(err)]

    frame_paths = sorted(
        cartella_out.glob(f"{prefisso}*-{video_stem}.jpg"),
        key=lambda p: int(num_re.search(p.name).group(1))
    )

    if len(frame_paths) != len(timestamp):
        print(
            f"  ⚠️  Mismatch su {path}: {len(frame_paths)} frame scritti "
            f"vs {len(timestamp)} timestamp letti da ffmpeg — scarto il risultato"
        )
        frame_paths = []

    if not frame_paths:
        fallback = _estrai_frame_singolo(path, cartella_out, file_id, video_stem)
        return [fallback] if fallback else []

    return [
        {"path_frame": str(fp), "timestamp_sec": round(ts, 2)}
        for fp, ts in zip(frame_paths, timestamp)
    ]


# ─── Scan cartella ────────────────────────────────────────────────────────────

def processa_cartella(cartella: str, db_path: str = None):
    """
    Scansiona una cartella, estrae i keyframe di ogni video (scene detection)
    e salva i risultati in SQLite.
    """
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = sqlite3.connect(db_path)
    init_tabella_frame(conn)

    cartella = Path(cartella)
    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        conn.close()
        return

    cartella_out = Path(__file__).parent.parent / CARTELLA_FRAME
    cartella_out.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Cartella:      {cartella}")
    print(f"🗄️  Database:      {db_path}")
    print(f"🎞️  Frame in:      {cartella_out}")
    print(f"📊 Soglia scena:  {SOGLIA_SCENA}\n")

    processati = 0
    con_frame  = 0
    senza_frame = 0
    saltati    = 0

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

        print(f"  🎬 {file_path.name}", end=" ... ", flush=True)
        frames = estrai_keyframe(path, cartella_out, file_id, file_path.stem)
        inserisci_frame(conn, file_id, frames)

        if frames:
            print(f"✅ {len(frames)} keyframe estratti")
            con_frame += 1
        else:
            print("○  nessun keyframe estratto")
            senza_frame += 1

        processati += 1

    print(f"\n✅ Estrazione completata:")
    print(f"   Processati:      {processati}")
    print(f"   Con keyframe:    {con_frame}")
    print(f"   Senza keyframe:  {senza_frame}")
    print(f"   Saltati:         {saltati}")
    conn.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FFMPEG_OK:
        print("⚠️  ffmpeg-python non trovato. Installa con: pip install ffmpeg-python")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso:")
        print("  python exovision_frames.py <cartella>       — estrae i keyframe")
        print("  python exovision_frames.py <cartella> <db>  — specifica il db")
        print()
        print("Es: python exovision_frames.py ./archivio_flickr")
        sys.exit(1)

    cartella = sys.argv[1]
    db       = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"
    processa_cartella(cartella, db)
