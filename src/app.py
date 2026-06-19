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
from flask import Flask, jsonify, request, send_from_directory, abort

app = Flask(__name__, static_folder="UI", static_url_path="")

# ── Configurazione ────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

DB_PATH = os.environ.get("EXOVISION_DB", load_config()["archivio"]["db"])


# ── Utility DB ────────────────────────────────────────────────────────────────

def db_ready():
    return Path(DB_PATH).exists()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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


# ── Gestione errori ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"errore": str(e)}), 404


# ── Avvio ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== ExoVision - Server Flask ===")
    print(f"  Database : {DB_PATH} {'(trovato)' if db_ready() else '(non ancora creato)'}")
    print(f"  UI       : src/UI/exovision.html")
    print(f"\n  Apri il browser su: http://localhost:5000\n")

    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
