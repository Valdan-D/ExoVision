"""
ExoVision — Estrazione metadati foto/video e inserimento in SQLite
Dipendenze: pip install Pillow piexif ffmpeg-python
FFmpeg deve essere installato sul sistema: https://ffmpeg.org/download.html
"""

import sqlite3
import os
import json
from pathlib import Path
from datetime import datetime

# ─── Configurazione ───────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def _load_config():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

_cfg = _load_config()

# Pillow per immagini
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    import piexif
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False
    print("⚠️  Pillow o piexif non trovati. Installa con: pip install Pillow piexif")

# ffmpeg-python per video
try:
    import ffmpeg
    FFMPEG_OK = True
except ImportError:
    FFMPEG_OK = False
    print("⚠️  ffmpeg-python non trovato. Installa con: pip install ffmpeg-python")


# ─── Estensioni supportate ────────────────────────────────────────────────────
FOTO_EXT  = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}


# ─── DATABASE ─────────────────────────────────────────────────────────────────

def init_db(db_path: str = "exovision.db") -> sqlite3.Connection:
    """Crea il database e le tabelle se non esistono."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            path                TEXT UNIQUE NOT NULL,
            nome_file           TEXT,
            tipo                TEXT CHECK(tipo IN ('foto', 'video')),
            estensione          TEXT,
            dimensione_bytes    INTEGER,
            data_modifica       TEXT,
            data_indicizzazione TEXT,
            metadati_completi   INTEGER DEFAULT 0  -- 0=incompleti, 1=completi
        );

        CREATE TABLE IF NOT EXISTS metadati_foto (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id     INTEGER NOT NULL REFERENCES files(id),
            larghezza   INTEGER,
            altezza     INTEGER,
            modalita    TEXT,         -- RGB, RGBA, L, ecc.
            data_scatto TEXT,
            camera_make TEXT,
            camera_model TEXT,
            iso         INTEGER,
            apertura    TEXT,
            otturatore  TEXT,
            lunghezza_focale TEXT,
            gps_lat     REAL,
            gps_lon     REAL,
            extra_exif  TEXT          -- JSON con campi EXIF aggiuntivi
        );

        CREATE TABLE IF NOT EXISTS metadati_video (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id         INTEGER NOT NULL REFERENCES files(id),
            durata_secondi  REAL,
            larghezza       INTEGER,
            altezza         INTEGER,
            framerate       REAL,
            codec_video     TEXT,
            codec_audio     TEXT,
            bitrate         INTEGER,
            extra           TEXT      -- JSON con info aggiuntive
        );

        CREATE TABLE IF NOT EXISTS frame (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id         INTEGER NOT NULL REFERENCES files(id),
            timestamp_sec   REAL,
            path_frame      TEXT
        );
    """)

    conn.commit()
    return conn


# ─── ESTRAZIONE METADATI FOTO ─────────────────────────────────────────────────

def _decode_gps(gps_data: dict):
    """Converte i dati GPS EXIF in coordinate decimali."""
    try:
        def to_decimal(values):
            d, m, s = values
            return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

        lat = to_decimal(gps_data.get(2, [(0,1),(0,1),(0,1)]))
        lon = to_decimal(gps_data.get(4, [(0,1),(0,1),(0,1)]))

        if gps_data.get(1, b'N') in (b'S', 'S'):
            lat = -lat
        if gps_data.get(3, b'E') in (b'W', 'W'):
            lon = -lon

        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


def estrai_metadati_foto(path: str) -> dict:
    """Estrae metadati EXIF da una foto."""
    meta = {
        "larghezza": None, "altezza": None, "modalita": None,
        "data_scatto": None, "camera_make": None, "camera_model": None,
        "iso": None, "apertura": None, "otturatore": None,
        "lunghezza_focale": None, "gps_lat": None, "gps_lon": None,
        "extra_exif": {}
    }

    if not PILLOW_OK:
        return meta

    try:
        with Image.open(path) as img:
            meta["larghezza"] = img.width
            meta["altezza"]   = img.height
            meta["modalita"]  = img.mode

            exif_raw = img._getexif()
            if not exif_raw:
                return meta

            exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}

            meta["data_scatto"]      = str(exif.get("DateTimeOriginal", ""))
            meta["camera_make"]      = str(exif.get("Make", "")).strip()
            meta["camera_model"]     = str(exif.get("Model", "")).strip()
            meta["iso"]              = exif.get("ISOSpeedRatings")
            meta["otturatore"]       = str(exif.get("ExposureTime", ""))
            meta["lunghezza_focale"] = str(exif.get("FocalLength", ""))

            apertura = exif.get("FNumber")
            if apertura:
                meta["apertura"] = f"f/{apertura[0]/apertura[1]:.1f}" if isinstance(apertura, tuple) else str(apertura)

            gps_raw = exif.get("GPSInfo")
            if gps_raw:
                gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
                meta["gps_lat"], meta["gps_lon"] = _decode_gps(gps_raw)

            # Campi extra (esclude binari)
            skip = {"MakerNote", "UserComment", "GPSInfo", "PrintImageMatching"}
            extra = {k: str(v) for k, v in exif.items()
                     if k not in skip and not isinstance(v, bytes)}
            meta["extra_exif"] = extra

    except Exception as e:
        print(f"  ⚠️  Errore lettura EXIF {path}: {e}")

    return meta


# ─── ESTRAZIONE METADATI VIDEO ────────────────────────────────────────────────

def estrai_metadati_video(path: str) -> dict:
    """Estrae metadati da un video tramite ffprobe."""
    meta = {
        "durata_secondi": None, "larghezza": None, "altezza": None,
        "framerate": None, "codec_video": None, "codec_audio": None,
        "bitrate": None, "extra": {}
    }

    if not FFMPEG_OK:
        return meta

    try:
        probe = ffmpeg.probe(path)
        fmt   = probe.get("format", {})

        meta["durata_secondi"] = float(fmt.get("duration", 0)) or None
        meta["bitrate"]        = int(fmt.get("bit_rate", 0)) or None

        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video" and not meta["codec_video"]:
                meta["codec_video"] = stream.get("codec_name")
                meta["larghezza"]   = stream.get("width")
                meta["altezza"]     = stream.get("height")

                fps_raw = stream.get("r_frame_rate", "0/1")
                num, den = fps_raw.split("/")
                meta["framerate"] = round(int(num) / int(den), 2) if int(den) else None

            elif stream.get("codec_type") == "audio" and not meta["codec_audio"]:
                meta["codec_audio"] = stream.get("codec_name")

        meta["extra"] = {"format_name": fmt.get("format_name", "")}

    except Exception as e:
        print(f"  ⚠️  Errore lettura video {path}: {e}")

    return meta


# ─── INSERIMENTO NEL DB ───────────────────────────────────────────────────────

def inserisci_file(conn: sqlite3.Connection, path: str, tipo: str) -> int | None:
    """Inserisce un file nella tabella files, restituisce l'id."""
    c    = conn.cursor()
    stat = os.stat(path)
    nome = os.path.basename(path)
    ext  = Path(path).suffix.lower()

    try:
        c.execute("""
            INSERT OR IGNORE INTO files
                (path, nome_file, tipo, estensione, dimensione_bytes, data_modifica, data_indicizzazione)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            path, nome, tipo, ext,
            stat.st_size,
            datetime.fromtimestamp(stat.st_mtime).isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()

        c.execute("SELECT id FROM files WHERE path = ?", (path,))
        row = c.fetchone()
        return row[0] if row else None

    except Exception as e:
        print(f"  ⚠️  Errore inserimento {path}: {e}")
        return None


def inserisci_metadati_foto(conn: sqlite3.Connection, file_id: int, meta: dict):
    c = conn.cursor()
    c.execute("""
        INSERT INTO metadati_foto
            (file_id, larghezza, altezza, modalita, data_scatto, camera_make,
             camera_model, iso, apertura, otturatore, lunghezza_focale,
             gps_lat, gps_lon, extra_exif)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        file_id,
        meta["larghezza"], meta["altezza"], meta["modalita"],
        meta["data_scatto"], meta["camera_make"], meta["camera_model"],
        meta["iso"], meta["apertura"], meta["otturatore"],
        meta["lunghezza_focale"], meta["gps_lat"], meta["gps_lon"],
        json.dumps(meta["extra_exif"], ensure_ascii=False)
    ))

    # Aggiorna flag metadati_completi
    campi_chiave = ["larghezza", "altezza", "data_scatto", "camera_make"]
    completo = all(meta.get(k) for k in campi_chiave)
    conn.cursor().execute(
        "UPDATE files SET metadati_completi = ? WHERE id = ?",
        (1 if completo else 0, file_id)
    )
    conn.commit()


def inserisci_metadati_video(conn: sqlite3.Connection, file_id: int, meta: dict):
    c = conn.cursor()
    c.execute("""
        INSERT INTO metadati_video
            (file_id, durata_secondi, larghezza, altezza, framerate,
             codec_video, codec_audio, bitrate, extra)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        file_id,
        meta["durata_secondi"], meta["larghezza"], meta["altezza"],
        meta["framerate"], meta["codec_video"], meta["codec_audio"],
        meta["bitrate"], json.dumps(meta["extra"], ensure_ascii=False)
    ))

    campi_chiave = ["durata_secondi", "larghezza", "altezza", "codec_video"]
    completo = all(meta.get(k) for k in campi_chiave)
    conn.cursor().execute(
        "UPDATE files SET metadati_completi = ? WHERE id = ?",
        (1 if completo else 0, file_id)
    )
    conn.commit()


# ─── SCAN CARTELLA ────────────────────────────────────────────────────────────

def scansiona_cartella(cartella: str, db_path: str = None):
    """Scansiona una cartella e indicizza tutti i file multimediali."""
    if db_path is None:
        db_path = _cfg["archivio"]["db"]
    conn = init_db(db_path)
    cartella = Path(cartella)

    if not cartella.exists():
        print(f"❌ Cartella non trovata: {cartella}")
        return

    print(f"\n📂 Scansione: {cartella}")
    print(f"🗄️  Database:  {db_path}\n")

    foto_count  = 0
    video_count = 0
    errori      = 0

    for file_path in sorted(cartella.rglob("*")):
        if not file_path.is_file():
            continue

        ext  = file_path.suffix.lower()
        path = str(file_path)

        if ext in FOTO_EXT:
            print(f"  🖼️  {file_path.name}")
            file_id = inserisci_file(conn, path, "foto")
            if file_id:
                meta = estrai_metadati_foto(path)
                inserisci_metadati_foto(conn, file_id, meta)
                foto_count += 1

        elif ext in VIDEO_EXT:
            print(f"  🎬 {file_path.name}")
            file_id = inserisci_file(conn, path, "video")
            if file_id:
                meta = estrai_metadati_video(path)
                inserisci_metadati_video(conn, file_id, meta)
                video_count += 1

    print(f"\n✅ Indicizzazione completata:")
    print(f"   Foto:   {foto_count}")
    print(f"   Video:  {video_count}")
    print(f"   Errori: {errori}")
    conn.close()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python exovision_metadata.py <percorso_cartella> [percorso_db]")
        print("Es:  python exovision_metadata.py ./archivio exovision.db")
        sys.exit(1)

    cartella = sys.argv[1]
    db_path  = sys.argv[2] if len(sys.argv) > 2 else "exovision.db"

    scansiona_cartella(cartella, db_path)