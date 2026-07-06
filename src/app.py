"""
ExoVision — Server Flask
Collega l'interfaccia HTML agli script Python.
Avvio: python app.py
Poi apri il browser su: http://localhost:5000
"""

import sqlite3
import os
import json
from pathlib import Path
from uuid import uuid4
from flask import Flask, jsonify, request, send_from_directory, abort
from werkzeug.utils import secure_filename

import exovision_metadata as metadata_pipeline

app = Flask(__name__, static_folder="UI", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB per richiesta di import

# ── Configurazione ────────────────────────────────────────────────────────────

CONFIG_PATH         = Path(__file__).parent.parent / "config.json"
CONFIG_EXAMPLE_PATH = Path(__file__).parent.parent / "config.example.json"

def load_config():
    """
    Legge config.json. Se manca (primo avvio, mai creato a mano), lo genera
    copiando config.example.json invece di andare in crash.
    """
    if not CONFIG_PATH.exists():
        with open(CONFIG_EXAMPLE_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return cfg

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

DB_PATH = os.environ.get("EXOVISION_DB", load_config()["archivio"]["db"])


# ── Utility DB ────────────────────────────────────────────────────────────────

def db_ready():
    return Path(DB_PATH).exists()

def _ensure_schema(conn: sqlite3.Connection):
    """
    Crea le tabelle mancanti se non tutti gli script della pipeline sono stati
    ancora eseguiti (es. solo exovision_metadata.py, senza OCR/YOLO/frame/whisper).
    Evita 500 "no such table" nelle API quando l'archivio è processato a metà.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            path                TEXT UNIQUE NOT NULL,
            nome_file           TEXT,
            tipo                TEXT CHECK(tipo IN ('foto', 'video')),
            estensione          TEXT,
            dimensione_bytes    INTEGER,
            data_modifica       TEXT,
            data_indicizzazione TEXT,
            metadati_completi   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS metadati_foto (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id     INTEGER NOT NULL REFERENCES files(id),
            larghezza   INTEGER,
            altezza     INTEGER,
            modalita    TEXT,
            data_scatto TEXT,
            camera_make TEXT,
            camera_model TEXT,
            iso         INTEGER,
            apertura    TEXT,
            otturatore  TEXT,
            lunghezza_focale TEXT,
            gps_lat     REAL,
            gps_lon     REAL,
            extra_exif  TEXT
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
            extra           TEXT
        );

        CREATE TABLE IF NOT EXISTS ocr (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            confidenza       REAL,
            data_estrazione  TEXT
        );

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
        );

        CREATE TABLE IF NOT EXISTS frame (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id         INTEGER NOT NULL REFERENCES files(id),
            timestamp_sec   REAL,
            path_frame      TEXT
        );

        CREATE TABLE IF NOT EXISTS trascrizioni (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            confidenza       REAL,
            data_estrazione  TEXT
        );
    """)
    conn.commit()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


# ── Rotte statiche ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve l'interfaccia HTML principale."""
    return send_from_directory("UI", "exovision.html")


# ── API — Ricerca ─────────────────────────────────────────────────────────────

@app.route("/api/search")
def search():
    """
    Ricerca nell'archivio.
    Parametri GET:
      q     — testo della query (es. 'tramonto montagna')
      limit — numero massimo di risultati (default 20)

    Risposta JSON:
      { query, results: [ { id, nome_file, tipo, path, metadati_completi, anteprima } ] }

    NOTE PER SIMONE: questo endpoint usa una ricerca per parola chiave su OCR e oggetti.
    Quando search.py con ChromaDB sarà pronto, sostituisci il corpo di questa funzione
    con la chiamata al motore semantico vero.
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 20))

    if not query or not db_ready():
        return jsonify({"query": query, "results": []})

    conn = get_db()
    try:
        # Ricerca mock su testo OCR e oggetti rilevati
        # TODO Simone: sostituire con ricerca vettoriale ChromaDB
        results = conn.execute("""
            SELECT DISTINCT
                f.id,
                f.nome_file,
                f.tipo,
                f.path,
                f.metadati_completi,
                o.testo   AS testo_ocr,
                og.oggetto AS oggetto
            FROM files f
            LEFT JOIN ocr o         ON f.id = o.file_id
            LEFT JOIN oggetti og     ON f.id = og.file_id
            WHERE
                o.testo    LIKE :q
                OR og.oggetto LIKE :q
                OR f.nome_file LIKE :q
            LIMIT :limit
        """, {"q": f"%{query}%", "limit": limit}).fetchall()

        return jsonify({
            "query": query,
            "results": [
                {
                    "id":                row["id"],
                    "nome_file":         row["nome_file"],
                    "tipo":              row["tipo"],
                    "path":              row["path"],
                    "metadati_completi": bool(row["metadati_completi"]),
                    "anteprima":         f"/api/thumb/{row['id']}",
                }
                for row in results
            ]
        })
    finally:
        conn.close()


# ── API — Dettaglio file ───────────────────────────────────────────────────────

@app.route("/api/file/<int:file_id>")
def file_detail(file_id):
    """
    Dettaglio completo di un file: metadati, OCR, oggetti rilevati.
    Risposta JSON:
      { file, metadati, ocr, oggetti, simili }
    """
    if not db_ready():
        abort(404, description="Database non ancora inizializzato.")

    conn = get_db()
    try:
        # Dati base
        file_row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        ).fetchone()

        if not file_row:
            abort(404, description=f"File {file_id} non trovato nel database.")

        # Metadati specifici per tipo
        if file_row["tipo"] == "foto":
            meta = conn.execute(
                "SELECT * FROM metadati_foto WHERE file_id = ?", (file_id,)
            ).fetchone()
        else:
            meta = conn.execute(
                "SELECT * FROM metadati_video WHERE file_id = ?", (file_id,)
            ).fetchone()

        # Testo OCR
        ocr = conn.execute(
            "SELECT testo, confidenza FROM ocr WHERE file_id = ?", (file_id,)
        ).fetchone()

        # Oggetti rilevati
        oggetti = conn.execute("""
            SELECT oggetto, confidenza, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM oggetti
            WHERE file_id = ? AND oggetto IS NOT NULL
            ORDER BY confidenza DESC
        """, (file_id,)).fetchall()

        # Immagini simili — placeholder
        # TODO Alice/Simone: sostituire con query ChromaDB per embedding vicini
        simili = conn.execute("""
            SELECT DISTINCT f.id, f.nome_file, f.path
            FROM files f
            JOIN oggetti o ON f.id = o.file_id
            WHERE o.oggetto IN (
                SELECT oggetto FROM oggetti WHERE file_id = ? AND oggetto IS NOT NULL
            )
            AND f.id != ?
            LIMIT 6
        """, (file_id, file_id)).fetchall()

        return jsonify({
            "file": dict(file_row),
            "metadati": dict(meta) if meta else {},
            "ocr": {
                "testo":      ocr["testo"] if ocr else None,
                "confidenza": ocr["confidenza"] if ocr else None,
            },
            "oggetti": [dict(o) for o in oggetti],
            "simili": [
                {
                    "id":        s["id"],
                    "nome_file": s["nome_file"],
                    "anteprima": f"/api/thumb/{s['id']}",
                }
                for s in simili
            ]
        })
    finally:
        conn.close()


# ── API — Lista file ───────────────────────────────────────────────────────────

@app.route("/api/files")
def file_list():
    """
    Lista paginata di tutti i file nell'archivio.
    Parametri GET:
      page   — pagina (default 1)
      limit  — risultati per pagina (default 24)
      tipo   — filtra per 'foto' o 'video' (opzionale)
    """
    page  = max(1, int(request.args.get("page", 1)))
    limit = min(100, int(request.args.get("limit", 24)))
    tipo  = request.args.get("tipo", None)
    offset = (page - 1) * limit

    if not db_ready():
        return jsonify({"page": page, "limit": limit, "totale": 0, "pagine": 0, "results": []})

    conn = get_db()
    try:
        where = "WHERE f.tipo = :tipo" if tipo else ""
        params = {"limit": limit, "offset": offset, "tipo": tipo}

        totale = conn.execute(
            f"SELECT COUNT(*) FROM files f {where}", params
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT
                f.id, f.nome_file, f.tipo, f.path, f.metadati_completi,
                f.dimensione_bytes, f.data_modifica,
                mf.larghezza, mf.altezza, mf.data_scatto,
                GROUP_CONCAT(DISTINCT og.oggetto) AS tag_string
            FROM files f
            LEFT JOIN metadati_foto mf ON f.id = mf.file_id
            LEFT JOIN oggetti og        ON f.id = og.file_id AND og.confidenza >= 0.5
            {where}
            GROUP BY f.id
            ORDER BY f.data_indicizzazione DESC
            LIMIT :limit OFFSET :offset
        """, params).fetchall()

        return jsonify({
            "page":    page,
            "limit":   limit,
            "totale":  totale,
            "pagine":  (totale + limit - 1) // limit,
            "results": [
                {
                    "id":                row["id"],
                    "nome_file":         row["nome_file"],
                    "tipo":              row["tipo"],
                    "path":              row["path"],
                    "metadati_completi": bool(row["metadati_completi"]),
                    "anteprima":         f"/api/thumb/{row['id']}",
                    "dimensione_bytes":  row["dimensione_bytes"],
                    "data":              row["data_scatto"] or (row["data_modifica"] or "").split("T")[0] or None,
                    "larghezza":         row["larghezza"],
                    "altezza":           row["altezza"],
                    "tags":              row["tag_string"].split(",") if row["tag_string"] else [],
                }
                for row in rows
            ]
        })
    finally:
        conn.close()


# ── API — Statistiche ─────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    """
    Statistiche generali dell'archivio — utile per la scheda Log nell'interfaccia.
    """
    if not db_ready():
        return jsonify({
            "file":    {"totale": 0, "foto": 0, "video": 0, "incompleti": 0},
            "ocr":     {"file_con_testo": 0},
            "oggetti": {"rilevamenti_totali": 0}
        })

    conn = get_db()
    try:
        totali = conn.execute("""
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN tipo = 'foto' THEN 1 ELSE 0 END) as foto,
                SUM(CASE WHEN tipo = 'video' THEN 1 ELSE 0 END) as video,
                SUM(CASE WHEN metadati_completi = 0 THEN 1 ELSE 0 END) as incompleti
            FROM files
        """).fetchone()

        ocr_tot = conn.execute(
            "SELECT COUNT(*) FROM ocr WHERE testo IS NOT NULL"
        ).fetchone()[0]

        oggetti_tot = conn.execute(
            "SELECT COUNT(*) FROM oggetti WHERE oggetto IS NOT NULL"
        ).fetchone()[0]

        return jsonify({
            "file": {
                "totale":     totali["totale"],
                "foto":       totali["foto"],
                "video":      totali["video"],
                "incompleti": totali["incompleti"],
            },
            "ocr": {
                "file_con_testo": ocr_tot
            },
            "oggetti": {
                "rilevamenti_totali": oggetti_tot
            }
        })
    finally:
        conn.close()


# ── API — Thumbnail ───────────────────────────────────────────────────────────

@app.route("/api/thumb/<int:file_id>")
def thumb(file_id):
    """
    Serve il file immagine originale dato l'id nel DB.
    NOTE PER STEFANO: per ora serve il file originale direttamente.
    In futuro si può aggiungere ridimensionamento con Pillow.
    """
    if not db_ready():
        abort(404)

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT path, tipo FROM files WHERE id = ?", (file_id,)
        ).fetchone()

        if not row:
            abort(404)

        path = Path(row["path"])
        if not path.exists():
            abort(404, description=f"File non trovato su disco: {path}")

        return send_from_directory(str(path.parent), path.name)
    finally:
        conn.close()


# ── API — Importazione ────────────────────────────────────────────────────────

def _cartella_archivio() -> Path:
    """
    Cartella dove vengono salvati i file caricati dalla tab Importa.
    Usa archivio.percorso da config.json se impostato, altrimenti
    una cartella di default alla root del progetto.
    """
    percorso = load_config()["archivio"].get("percorso") or ""
    return Path(percorso) if percorso else Path(__file__).parent.parent / "archivio_importati"


@app.route("/api/import", methods=["POST"])
def import_files():
    """
    Riceve i file caricati dalla tab Importa (multipart/form-data, campo "files"),
    li salva su disco e li indicizza subito con la stessa logica di
    exovision_metadata.py (metadati EXIF/ffprobe → SQLite).
    OCR/YOLO/frame/trascrizione restano step separati, da lanciare a parte.
    """
    files = request.files.getlist("files")
    if not files:
        abort(400, description="Nessun file ricevuto.")

    cartella = _cartella_archivio()
    cartella.mkdir(parents=True, exist_ok=True)

    estensioni_valide = metadata_pipeline.FOTO_EXT | metadata_pipeline.VIDEO_EXT

    conn = get_db()
    indicizzati, saltati, errori = 0, 0, 0

    try:
        for f in files:
            nome = secure_filename(f.filename or "")
            ext  = Path(nome).suffix.lower()

            if not nome or ext not in estensioni_valide:
                saltati += 1
                continue

            dest = cartella / nome
            if dest.exists():
                dest = cartella / f"{dest.stem}_{uuid4().hex[:8]}{dest.suffix}"
            f.save(dest)

            tipo    = "foto" if ext in metadata_pipeline.FOTO_EXT else "video"
            file_id = metadata_pipeline.inserisci_file(conn, str(dest), tipo)

            if not file_id:
                errori += 1
                continue

            if tipo == "foto":
                meta = metadata_pipeline.estrai_metadati_foto(str(dest))
                metadata_pipeline.inserisci_metadati_foto(conn, file_id, meta)
            else:
                meta = metadata_pipeline.estrai_metadati_video(str(dest))
                metadata_pipeline.inserisci_metadati_video(conn, file_id, meta)

            indicizzati += 1
    finally:
        conn.close()

    return jsonify({"indicizzati": indicizzati, "saltati": saltati, "errori": errori})


# ── API — Configurazione ─────────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    """Restituisce il contenuto di config.json."""
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def save_config():
    """
    Salva le impostazioni ricevute in config.json.
    Il body JSON può contenere solo le chiavi da aggiornare (merge parziale).
    """
    updates = request.get_json(silent=True)
    if not updates:
        abort(400, description="Body JSON mancante o non valido.")

    cfg = load_config()

    # Merge un livello di profondità — sufficiente per la struttura attuale
    for section, values in updates.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return jsonify({"ok": True, "config": cfg})


# ── Modifica manuale metadati ──────────────────────────────────────────────────

@app.route("/api/file/<int:id>/metadata", methods=["POST"])
def update_file_metadata(id):
    """
    Riceve le modifiche manuali dal frontend e aggiorna il database SQLite nelle tabelle reali.
    """
    data = request.get_json(silent=True)
    if not data:
        abort(400, description="Dati JSON mancanti o non validi.")

    nome_file = data.get("nome_file")
    data_creazione = data.get("data_creazione")
    gps_lat = data.get("gps_lat")
    gps_lon = data.get("gps_lon")

    conn = get_db()
    cursor = conn.cursor()

    try:
        # 1. Aggiorna il nome del file nella tabella principale 'files'
        if nome_file is not None:
            cursor.execute(
                "UPDATE files SET nome_file = ? WHERE id = ?",
                (nome_file, id)
            )

        # 2. Aggiorna o inserisci i dettagli nella tabella corretta 'metadati_foto'
        # La colonna si chiama 'id', non 'id_file'
        cursor.execute("SELECT id FROM metadati_foto WHERE id = ?", (id,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute(
                """
                UPDATE metadati_foto 
                SET data_creazione = ?, gps_lat = ?, gps_lon = ? 
                WHERE id = ?
                """,
                (data_creazione, gps_lat, gps_lon, id)
            )
        else:
            cursor.execute(
                """
                INSERT INTO metadati_foto (id, data_creazione, gps_lat, gps_lon) 
                VALUES (?, ?, ?, ?)
                """,
                (id, data_creazione, gps_lat, gps_lon)
            )

        # Applica e salva definitivamente nel file del database
        conn.commit()
        return jsonify({"ok": True, "messaggio": "Metadati aggiornati correttamente!"}), 200

    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({"errore": f"Errore del database: {str(e)}"}), 500
    finally:
        conn.close()
# ── Gestione errori ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"errore": str(e)}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"errore": "File troppo grandi (limite 2 GB per richiesta di import)."}), 413


# ── Avvio ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== ExoVision - Server Flask ===")
    print(f"  Database : {DB_PATH} {'(trovato)' if db_ready() else '(non ancora creato)'}")
    print(f"  UI       : src/UI/exovision.html")
    print(f"\n  Apri il browser su: http://localhost:5000\n")

    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
