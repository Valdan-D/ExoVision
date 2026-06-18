"""
ExoVision — Server Flask
Collega l'interfaccia HTML agli script Python.
Avvio: python app.py
Poi apri il browser su: http://localhost:5000
"""

import sqlite3
import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, abort

app = Flask(__name__, static_folder="ui", static_url_path="")

# ── Configurazione ────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("EXOVISION_DB", "exovision.db")


# ── Utility DB ────────────────────────────────────────────────────────────────

def get_db():
    """Apre una connessione al database SQLite."""
    if not Path(DB_PATH).exists():
        abort(503, description="Database non trovato. Esegui prima exovision_metadata.py.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # accesso per nome colonna
    return conn


# ── Rotte statiche ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve l'interfaccia HTML principale."""
    return send_from_directory("ui", "index.html")


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

    if not query:
        return jsonify({"query": "", "results": []})

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

    conn = get_db()
    try:
        where = "WHERE tipo = :tipo" if tipo else ""
        params = {"limit": limit, "offset": offset, "tipo": tipo}

        totale = conn.execute(
            f"SELECT COUNT(*) FROM files {where}", params
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT id, nome_file, tipo, path, metadati_completi
            FROM files {where}
            ORDER BY data_indicizzazione DESC
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
                    "metadati_completi": bool(row["metadati_completi"]),
                    "anteprima":         f"/api/thumb/{row['id']}",
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


# ── Gestione errori ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"errore": str(e)}), 404

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({"errore": str(e)}), 503


# ── Avvio ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════╗")
    print("║     ExoVision · Server Flask     ║")
    print("╚══════════════════════════════════╝")
    print(f"\n  Database: {DB_PATH}")
    print(f"  UI:       src/ui/index.html")
    print(f"\n  Apri il browser su: http://localhost:5000\n")

    app.run(debug=True, port=5000)
