"""
ExoVision — Server Flask
Collega l'interfaccia HTML agli script Python.
Avvio: python app.py
Poi apri il browser su: http://localhost:5000
"""

import sqlite3
import os
import json
import threading
import queue
from pathlib import Path
from uuid import uuid4
from flask import Flask, jsonify, request, send_from_directory, abort
from werkzeug.utils import secure_filename

import exovision_metadata as metadata_pipeline
import exovision_ocr as ocr_pipeline
import exovision_yolo as yolo_pipeline
import exovision_frames as frames_pipeline
import exovision_whisper as whisper_pipeline
import exovision_caption as caption_pipeline
from computer_vision.database import ExoVisionDB, CHROMA_OK
from computer_vision.models.embedding import SigLIPEmbedder, SIGLIP_OK
from computer_vision.models.face_rec import ExoFaceRecognizer, FACE_OK
from computer_vision.models.obj_detection import YoloDetector

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


# ── Elaborazione in background dopo /api/import (OCR/YOLO/frame/whisper) ──────
#
# /api/import fa solo lo step 1 (metadati) in modo sincrono e risponde subito.
# OCR/YOLO/frame/trascrizione vengono accodati (un solo worker li elabora in
# sequenza, vedi _coda_elaborazione più sotto) appena il file è caricato,
# così l'upload non resta bloccato per i tempi di
# caricamento modelli + inferenza (da ~1s a video lunghi con whisper).
# Se una libreria non è installata (es. faster-whisper), il relativo step
# viene semplicemente saltato: gli altri proseguono normalmente.

_model_lock  = threading.Lock()
_ocr_reader  = None
_yolo_model  = None
_whisper_model = None
_caption_model     = None
_caption_processor = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None and ocr_pipeline.EASYOCR_OK:
        with _model_lock:
            if _ocr_reader is None:
                _ocr_reader = ocr_pipeline.easyocr.Reader(ocr_pipeline.LINGUE, gpu=False)
    return _ocr_reader

def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None and yolo_pipeline.YOLO_OK:
        with _model_lock:
            if _yolo_model is None:
                _yolo_model = YoloDetector(yolo_pipeline.MODELLO, conf=yolo_pipeline.CONFIDENZA_MINIMA)
    return _yolo_model

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and whisper_pipeline.WHISPER_OK:
        with _model_lock:
            if _whisper_model is None:
                _whisper_model = whisper_pipeline.WhisperModel(
                    whisper_pipeline.MODELLO, device="cpu", compute_type="int8"
                )
    return _whisper_model

def _get_caption_model_e_processor():
    global _caption_model, _caption_processor
    if _caption_model is None and caption_pipeline.CAPTION_OK:
        with _model_lock:
            if _caption_model is None:
                _caption_processor = caption_pipeline.BlipProcessor.from_pretrained(caption_pipeline.MODELLO)
                _caption_model = caption_pipeline.BlipForConditionalGeneration.from_pretrained(caption_pipeline.MODELLO)
    return _caption_model, _caption_processor

_vector_db        = None
_siglip_embedder  = None
_face_recognizer  = None

def _get_vector_db():
    global _vector_db
    if _vector_db is None and CHROMA_OK:
        with _model_lock:
            if _vector_db is None:
                percorso = load_config().get("chroma", {}).get("percorso", "exovision_vector_db")
                _vector_db = ExoVisionDB(path=percorso)
    return _vector_db

def _get_siglip_embedder():
    global _siglip_embedder
    if _siglip_embedder is None and SIGLIP_OK:
        with _model_lock:
            if _siglip_embedder is None:
                _siglip_embedder = SigLIPEmbedder()
    return _siglip_embedder

def _get_face_recognizer():
    global _face_recognizer
    if _face_recognizer is None and FACE_OK:
        with _model_lock:
            if _face_recognizer is None:
                _face_recognizer = ExoFaceRecognizer()
    return _face_recognizer


_in_progress_lock = threading.Lock()
_in_progress_ids  = set()

def _file_in_elaborazione(file_id: int) -> bool:
    with _in_progress_lock:
        return file_id in _in_progress_ids


def _calcola_ia_stato(tipo: str, has_ocr: bool, has_oggetti: bool, has_frame: bool,
                       has_trascrizione: bool, has_didascalia: bool = False) -> str:
    """
    Riassume lo stato di OCR/YOLO/didascalia (foto) o frame/trascrizione/YOLO/
    didascalia sul primo frame (video) in un'unica etichetta, contando solo
    gli step per cui la libreria richiesta
    è disponibile su questa macchina — altrimenti un file resterebbe per
    sempre "parziale" se ad es. faster-whisper o transformers non sono
    installati, anche dopo aver rielaborato tutto il possibile.
    Valori: "non_disponibile" (nessuno step eseguibile qui), "non_elaborato",
    "parziale", "elaborato".
    """
    if tipo == "foto":
        richiesti = []
        if ocr_pipeline.EASYOCR_OK:      richiesti.append(has_ocr)
        if yolo_pipeline.YOLO_OK:        richiesti.append(has_oggetti)
        if caption_pipeline.CAPTION_OK:  richiesti.append(has_didascalia)
    else:
        richiesti = []
        if frames_pipeline.FFMPEG_OK:      richiesti.append(has_frame)
        if whisper_pipeline.WHISPER_OK:    richiesti.append(has_trascrizione)
        if yolo_pipeline.YOLO_OK:          richiesti.append(has_oggetti)
        if caption_pipeline.CAPTION_OK:    richiesti.append(has_didascalia)

    if not richiesti:
        return "non_disponibile"
    fatti = sum(1 for r in richiesti if r)
    if fatti == len(richiesti):
        return "elaborato"
    if fatti == 0:
        return "non_elaborato"
    return "parziale"


def _indicizza_semantico_e_volti(file_id: int, img_path: str, oggetti: list):
    """
    Calcola l'embedding SigLIP dell'immagine (o del frame rappresentativo per i
    video) e, se è presente una persona, gli embedding ArcFace dei volti
    rilevati — entrambi salvati in ChromaDB. Usa upsert quindi rielaborare lo
    stesso file non crea duplicati.
    """
    tag_rilevati = [o["oggetto"] for o in oggetti if o.get("oggetto")]

    if SIGLIP_OK and CHROMA_OK:
        try:
            from PIL import Image
            db = _get_vector_db()
            embedder = _get_siglip_embedder()
            img = Image.open(img_path).convert("RGB")
            vettore = embedder.istanzia_vettori_batch([img])[0]
            metadata = {"file_id": file_id}
            for t in set(tag_rilevati):
                metadata[f"yolo_has_{t.lower().strip()}"] = 1
            db.aggiungi_batch_semantica([f"file_{file_id}"], [vettore], [metadata])
        except Exception as e:
            print(f"⚠️  Errore embedding semantico in background su file {file_id}: {e}")

    # Senza YOLO non sappiamo se c'è una persona nell'inquadratura: tentiamo
    # comunque il rilevamento volti piuttosto che disabilitarlo silenziosamente.
    if FACE_OK and CHROMA_OK and (not yolo_pipeline.YOLO_OK or "person" in tag_rilevati):
        try:
            db = _get_vector_db()
            recognizer = _get_face_recognizer()
            vettori_volti = recognizer.estrai_vettore_volto(img_path)
            db.coll_volti.delete(where={"file_id": file_id})
            if vettori_volti:
                ids = [f"face_{file_id}_{i}" for i in range(len(vettori_volti))]
                metadatas = [{"file_id": file_id, "path": img_path} for _ in vettori_volti]
                db.aggiungi_batch_volti(ids, vettori_volti, metadatas)
        except Exception as e:
            print(f"⚠️  Errore embedding volti in background su file {file_id}: {e}")


def _post_process_file(file_id: int, path: str, tipo: str):
    """
    Esegue OCR+YOLO+didascalia (foto) o keyframe+trascrizione (video) su un
    singolo file. Chiamata dal worker della coda di elaborazione — non va
    invocata direttamente da più punti in parallelo (vedi _accoda_elaborazione).
    """
    with _in_progress_lock:
        _in_progress_ids.add(file_id)

    conn = sqlite3.connect(DB_PATH)
    try:
        if tipo == "foto":
            if ocr_pipeline.EASYOCR_OK:
                try:
                    reader = _get_ocr_reader()
                    testo, confidenza = ocr_pipeline.estrai_testo(reader, path)
                    ocr_pipeline.init_tabella_ocr(conn)
                    lingua = "+".join(ocr_pipeline.LINGUE) if testo else None
                    ocr_pipeline.inserisci_ocr(conn, file_id, testo, lingua, confidenza)
                except Exception as e:
                    print(f"⚠️  Errore OCR in background su file {file_id}: {e}")

            oggetti = []
            if yolo_pipeline.YOLO_OK:
                try:
                    model = _get_yolo_model()
                    oggetti = yolo_pipeline.rileva_oggetti(model, path)
                    yolo_pipeline.init_tabella_oggetti(conn)
                    yolo_pipeline.inserisci_oggetti(conn, file_id, oggetti)
                except Exception as e:
                    print(f"⚠️  Errore YOLO in background su file {file_id}: {e}")

            if caption_pipeline.CAPTION_OK:
                try:
                    model, processor = _get_caption_model_e_processor()
                    testo = caption_pipeline.genera_didascalia(model, processor, path)
                    caption_pipeline.init_tabella_didascalie(conn)
                    caption_pipeline.inserisci_didascalia(conn, file_id, testo)
                except Exception as e:
                    print(f"⚠️  Errore didascalia in background su file {file_id}: {e}")

            _indicizza_semantico_e_volti(file_id, path, oggetti)

        else:  # video
            frame_rappresentativo = None
            if frames_pipeline.FFMPEG_OK:
                try:
                    cartella_out = Path(__file__).parent.parent / frames_pipeline.CARTELLA_FRAME
                    cartella_out.mkdir(parents=True, exist_ok=True)
                    frames = frames_pipeline.estrai_keyframe(path, cartella_out, file_id, Path(path).stem)
                    frames_pipeline.init_tabella_frame(conn)
                    frames_pipeline.inserisci_frame(conn, file_id, frames)
                    if frames:
                        frame_rappresentativo = frames[0]["path_frame"]
                except Exception as e:
                    print(f"⚠️  Errore estrazione frame in background su file {file_id}: {e}")

            if whisper_pipeline.WHISPER_OK:
                try:
                    model = _get_whisper_model()
                    testo, lingua, confidenza = whisper_pipeline.estrai_testo(model, path)
                    whisper_pipeline.init_tabella_trascrizioni(conn)
                    whisper_pipeline.inserisci_trascrizione(conn, file_id, testo, lingua, confidenza)
                except Exception as e:
                    print(f"⚠️  Errore trascrizione in background su file {file_id}: {e}")

            # Tag/didascalia sul primo frame estratto — stessa logica delle foto,
            # applicata al keyframe rappresentativo (senza rianalizzare l'intero video).
            if frame_rappresentativo:
                oggetti = []
                if yolo_pipeline.YOLO_OK:
                    try:
                        model = _get_yolo_model()
                        oggetti = yolo_pipeline.rileva_oggetti(model, frame_rappresentativo)
                        yolo_pipeline.init_tabella_oggetti(conn)
                        yolo_pipeline.inserisci_oggetti(conn, file_id, oggetti)
                    except Exception as e:
                        print(f"⚠️  Errore YOLO (frame video) in background su file {file_id}: {e}")

                if caption_pipeline.CAPTION_OK:
                    try:
                        model, processor = _get_caption_model_e_processor()
                        testo = caption_pipeline.genera_didascalia(model, processor, frame_rappresentativo)
                        caption_pipeline.init_tabella_didascalie(conn)
                        caption_pipeline.inserisci_didascalia(conn, file_id, testo)
                    except Exception as e:
                        print(f"⚠️  Errore didascalia (frame video) in background su file {file_id}: {e}")

                _indicizza_semantico_e_volti(file_id, frame_rappresentativo, oggetti)
    finally:
        conn.close()
        with _in_progress_lock:
            _in_progress_ids.discard(file_id)


# ── Coda di elaborazione (un solo file alla volta) ─────────────────────────────
#
# Un thread separato per ogni file (versione precedente) fa sì che, quando più
# file vengono caricati/rielaborati insieme (stesso batch di /api/import,
# oppure i pulsanti "Rielabora tutti"/"Elabora tutti" in UI), più modelli
# pesanti (whisper, YOLO, BLIP) girino in parallelo sulla stessa CPU: in
# pratica una delle elaborazioni può fallire in modo silenzioso (nessuna riga
# scritta, solo un errore in console facile da perdere in mezzo agli altri).
# Una coda con un solo worker elabora i file in sequenza: più lento in totale,
# ma niente più elaborazioni "sparite" per contesa di risorse.

_coda_elaborazione = queue.Queue()

def _worker_elaborazione():
    while True:
        file_id, path, tipo = _coda_elaborazione.get()
        try:
            _post_process_file(file_id, path, tipo)
        except Exception as e:
            print(f"⚠️  Errore imprevisto in elaborazione background su file {file_id}: {e}")
        finally:
            _coda_elaborazione.task_done()

threading.Thread(target=_worker_elaborazione, daemon=True).start()

def _accoda_elaborazione(file_id: int, path: str, tipo: str):
    _coda_elaborazione.put((file_id, path, tipo))


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
            metadati_completi   INTEGER DEFAULT 0,
            descrizione         TEXT
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
            extra           TEXT,
            data_creazione  TEXT,
            gps_lat         REAL,
            gps_lon         REAL
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

        CREATE TABLE IF NOT EXISTS didascalie (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id          INTEGER NOT NULL REFERENCES files(id),
            testo            TEXT,
            lingua           TEXT,
            data_estrazione  TEXT
        );
    """)
    conn.commit()

# ── Aggiunta colonne mancanti sui DB creati con schema precedente ──
    for tabella, colonna, tipo in [
        ("files", "descrizione", "TEXT"),
        ("metadati_video", "data_creazione", "TEXT"),
        ("metadati_video", "gps_lat", "REAL"),
        ("metadati_video", "gps_lon", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {tabella} ADD COLUMN {colonna} {tipo};")
            conn.commit()
        except sqlite3.OperationalError:
            pass

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

    NOTE PER SIMONE: questo endpoint usa una ricerca per parola chiave su OCR,
    oggetti e trascrizioni audio. Quando search.py con ChromaDB sarà pronto,
    sostituisci il corpo di questa funzione con la chiamata al motore semantico vero.
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 20))

    if not query or not db_ready():
        return jsonify({"query": query, "results": []})

    conn = get_db()
    try:
        # Ricerca mock su testo OCR, oggetti rilevati e trascrizioni audio
        # TODO Simone: sostituire con ricerca vettoriale ChromaDB
        results = conn.execute("""
            SELECT DISTINCT
                f.id,
                f.nome_file,
                f.tipo,
                f.path,
                f.metadati_completi,
                o.testo   AS testo_ocr,
                og.oggetto AS oggetto,
                t.testo   AS testo_trascrizione
            FROM files f
            LEFT JOIN ocr o           ON f.id = o.file_id
            LEFT JOIN oggetti og      ON f.id = og.file_id
            LEFT JOIN trascrizioni t  ON f.id = t.file_id
            WHERE
                o.testo    LIKE :q
                OR og.oggetto LIKE :q
                OR f.nome_file LIKE :q
                OR t.testo LIKE :q
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
                    "anteprima":         f"/api/preview/{row['id']}",
                }
                for row in results
            ]
        })
    finally:
        conn.close()


@app.route("/api/search/semantic")
def search_semantic():
    """
    Ricerca semantica per concetto/descrizione (SigLIP + ChromaDB), affiancata
    alla ricerca per parola chiave di /api/search (non la sostituisce).
    Parametri GET:
      q     — descrizione testuale della query (es. "tramonto sul lago")
      limit — numero massimo di risultati (default 20)
      tag   — se presente, filtra sui file in cui YOLO ha rilevato quel tag

    Risposta JSON: { query, results: [ { id, nome_file, tipo, path, metadati_completi, anteprima } ] }
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 20))
    tag   = request.args.get("tag") or None

    if not SIGLIP_OK or not CHROMA_OK:
        return jsonify({
            "query": query, "results": [],
            "errore": "Ricerca semantica non disponibile: libreria mancante su questo server."
        }), 503

    if not query or not db_ready():
        return jsonify({"query": query, "results": []})

    embedder = _get_siglip_embedder()
    db       = _get_vector_db()
    vettore  = embedder.istanzia_vettore_testo(query)
    trovati  = db.cerca_ibrido(vettore, tag_filtro=tag, n_risultati=limit)

    id_ordine = []
    for doc_id in trovati.get("ids", [[]])[0]:
        if doc_id.startswith("file_"):
            try:
                id_ordine.append(int(doc_id.split("_", 1)[1]))
            except ValueError:
                continue

    if not id_ordine:
        return jsonify({"query": query, "results": []})

    conn = get_db()
    try:
        placeholders  = ",".join("?" * len(id_ordine))
        righe_per_id  = {
            r["id"]: r for r in conn.execute(
                f"SELECT id, nome_file, tipo, path, metadati_completi FROM files WHERE id IN ({placeholders})",
                id_ordine
            ).fetchall()
        }
        return jsonify({
            "query": query,
            "results": [
                {
                    "id":                rid,
                    "nome_file":         righe_per_id[rid]["nome_file"],
                    "tipo":              righe_per_id[rid]["tipo"],
                    "path":              righe_per_id[rid]["path"],
                    "metadati_completi": bool(righe_per_id[rid]["metadati_completi"]),
                    "anteprima":         f"/api/preview/{rid}",
                }
                for rid in id_ordine if rid in righe_per_id
            ]
        })
    finally:
        conn.close()


@app.route("/api/search/face", methods=["POST"])
def search_face():
    """
    Ricerca per volto simile: riceve una foto di riferimento (multipart/form-data,
    campo "file", un solo file) ed interroga ChromaDB (embedding ArcFace) per
    trovare i file dell'archivio con volti simili, sotto la soglia di distanza
    coseno configurata (config.volti.soglia_distanza, default 0.68).

    Risposta JSON: { results: [ { id, nome_file, tipo, anteprima, distanza } ] }
    """
    if not FACE_OK or not CHROMA_OK:
        return jsonify({
            "results": [],
            "errore": "Riconoscimento facciale non disponibile: libreria mancante su questo server."
        }), 503

    file = request.files.get("file")
    if not file or not file.filename:
        abort(400, description="Nessuna foto di riferimento ricevuta.")

    cartella_tmp = Path(__file__).parent.parent / "tmp_ricerca_volti"
    cartella_tmp.mkdir(parents=True, exist_ok=True)
    tmp_path = cartella_tmp / f"{uuid4().hex}_{secure_filename(file.filename)}"
    file.save(tmp_path)

    try:
        recognizer    = _get_face_recognizer()
        vettori_volti = recognizer.estrai_vettore_volto(str(tmp_path))
        if not vettori_volti:
            return jsonify({"results": [], "errore": "Nessun volto rilevato nella foto caricata."})

        soglia = load_config().get("volti", {}).get("soglia_distanza", 0.68)
        limit  = int(request.args.get("limit", 20))
        db     = _get_vector_db()

        migliore_distanza = {}
        for vettore in vettori_volti:
            trovati   = db.cerca_volto_simile(vettore, n_risultati=limit)
            distanze  = trovati.get("distances", [[]])[0]
            metadatas = trovati.get("metadatas", [[]])[0]
            for meta, distanza in zip(metadatas, distanze):
                fid = meta.get("file_id")
                if fid is None or distanza > soglia:
                    continue
                if fid not in migliore_distanza or distanza < migliore_distanza[fid]:
                    migliore_distanza[fid] = distanza

        if not migliore_distanza:
            return jsonify({"results": []})

        id_ordine = sorted(migliore_distanza, key=migliore_distanza.get)

        conn = get_db()
        try:
            placeholders = ",".join("?" * len(id_ordine))
            righe_per_id = {
                r["id"]: r for r in conn.execute(
                    f"SELECT id, nome_file, tipo, metadati_completi FROM files WHERE id IN ({placeholders})",
                    id_ordine
                ).fetchall()
            }
            return jsonify({
                "results": [
                    {
                        "id":                rid,
                        "nome_file":         righe_per_id[rid]["nome_file"],
                        "tipo":              righe_per_id[rid]["tipo"],
                        "metadati_completi": bool(righe_per_id[rid]["metadati_completi"]),
                        "anteprima":         f"/api/preview/{rid}",
                        "distanza":          round(migliore_distanza[rid], 4),
                    }
                    for rid in id_ordine if rid in righe_per_id
                ]
            })
        finally:
            conn.close()
    finally:
        tmp_path.unlink(missing_ok=True)


# ── API — Dettaglio file ───────────────────────────────────────────────────────

@app.route("/api/file/<int:file_id>")
def file_detail(file_id):
    """
    Dettaglio completo di un file: metadati, OCR, trascrizione audio, didascalia IA, oggetti rilevati.
    Risposta JSON:
      { file, metadati, ocr, trascrizione, didascalia, oggetti, simili }
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
        # NB: la UI legge/scrive la data come "data_creazione" per entrambi i tipi;
        # nella tabella foto la colonna storica si chiama "data_scatto" (EXIF), va aliasata.
        if file_row["tipo"] == "foto":
            meta = conn.execute(
                "SELECT *, data_scatto AS data_creazione FROM metadati_foto WHERE file_id = ?", (file_id,)
            ).fetchone()
        else:
            meta = conn.execute(
                "SELECT * FROM metadati_video WHERE file_id = ?", (file_id,)
            ).fetchone()

        # Testo OCR
        ocr = conn.execute(
            "SELECT testo, confidenza FROM ocr WHERE file_id = ?", (file_id,)
        ).fetchone()

        # Trascrizione audio (video)
        trascrizione = conn.execute(
            "SELECT testo, lingua, confidenza FROM trascrizioni WHERE file_id = ?", (file_id,)
        ).fetchone()

        # Didascalia generata dall'IA (foto) — distinta dalla descrizione manuale in files.descrizione
        didascalia = conn.execute(
            "SELECT testo, lingua FROM didascalie WHERE file_id = ?", (file_id,)
        ).fetchone()

        # Oggetti rilevati
        oggetti = conn.execute("""
            SELECT oggetto, confidenza, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM oggetti
            WHERE file_id = ? AND oggetto IS NOT NULL
            ORDER BY confidenza DESC
        """, (file_id,)).fetchall()

        # Immagini simili — via embedding SigLIP/ChromaDB se disponibile,
        # altrimenti fallback sulla vecchia query per tag YOLO condivisi.
        simili_ids = []
        if SIGLIP_OK and CHROMA_OK:
            try:
                db = _get_vector_db()
                proprio = db.coll_semantica.get(ids=[f"file_{file_id}"], include=["embeddings"])
                vettori_propri = proprio.get("embeddings")
                if vettori_propri is not None and len(vettori_propri) > 0:
                    trovati = db.cerca_ibrido(vettori_propri[0], n_risultati=7)
                    for doc_id in trovati.get("ids", [[]])[0]:
                        if doc_id.startswith("file_"):
                            sid = int(doc_id.split("_", 1)[1])
                            if sid != file_id:
                                simili_ids.append(sid)
            except Exception as e:
                print(f"⚠️  Errore ricerca simili (ChromaDB) su file {file_id}: {e}")

        if simili_ids:
            placeholders = ",".join("?" * len(simili_ids))
            righe_per_id = {
                r["id"]: r for r in conn.execute(
                    f"SELECT id, nome_file FROM files WHERE id IN ({placeholders})", simili_ids
                ).fetchall()
            }
            simili = [righe_per_id[sid] for sid in simili_ids[:6] if sid in righe_per_id]
        else:
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
            "trascrizione": {
                "testo":      trascrizione["testo"] if trascrizione else None,
                "lingua":     trascrizione["lingua"] if trascrizione else None,
                "confidenza": trascrizione["confidenza"] if trascrizione else None,
            },
            "didascalia": {
                "testo":  didascalia["testo"] if didascalia else None,
                "lingua": didascalia["lingua"] if didascalia else None,
            },
            "oggetti": [dict(o) for o in oggetti],
            "simili": [
                {
                    "id":        s["id"],
                    "nome_file": s["nome_file"],
                    "anteprima": f"/api/preview/{s['id']}",
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
                GROUP_CONCAT(DISTINCT og.oggetto) AS tag_string,
                EXISTS(SELECT 1 FROM ocr WHERE ocr.file_id = f.id)             AS has_ocr,
                EXISTS(SELECT 1 FROM oggetti WHERE oggetti.file_id = f.id)     AS has_oggetti,
                EXISTS(SELECT 1 FROM frame WHERE frame.file_id = f.id)        AS has_frame,
                EXISTS(SELECT 1 FROM trascrizioni WHERE trascrizioni.file_id = f.id) AS has_trascrizione,
                EXISTS(SELECT 1 FROM didascalie WHERE didascalie.file_id = f.id)   AS has_didascalia
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
                    "metadati_completi":    bool(row["metadati_completi"]),
                    "file_esistente":       Path(row["path"]).exists(),
                    "elaborazione_in_corso": _file_in_elaborazione(row["id"]),
                    "ia_stato":          _calcola_ia_stato(
                                             row["tipo"], bool(row["has_ocr"]), bool(row["has_oggetti"]),
                                             bool(row["has_frame"]), bool(row["has_trascrizione"]),
                                             bool(row["has_didascalia"])
                                         ),
                    "anteprima":         f"/api/preview/{row['id']}",
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
    Serve il file originale dato l'id nel DB (usato per la visualizzazione
    a schermo intero: <img> per le foto, <video src> per i video).
    NOTE PER STEFANO: per ora serve il file originale direttamente.
    In futuro si può aggiungere ridimensionamento con Pillow.
    """
    if not db_ready():
        abort(404)

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)
        ).fetchone()

        if not row:
            abort(404)

        path = Path(row["path"])
        if not path.exists():
            abort(404, description=f"File non trovato su disco: {path}")

        return send_from_directory(str(path.parent), path.name)
    finally:
        conn.close()


@app.route("/api/preview/<int:file_id>")
def preview(file_id):
    """
    Serve un'anteprima renderizzabile in <img> dato l'id nel DB.
    Per le foto è il file immagine originale; per i video è il primo
    keyframe estratto da exovision_frames.py (un <img> non può
    renderizzare direttamente un file video).
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

        if row["tipo"] == "video":
            frame = conn.execute(
                """
                SELECT path_frame FROM frame
                WHERE file_id = ? AND path_frame IS NOT NULL
                ORDER BY timestamp_sec ASC LIMIT 1
                """,
                (file_id,)
            ).fetchone()
            if not frame:
                abort(404, description="Nessun keyframe disponibile per questo video: esegui exovision_frames.py.")
            path = Path(frame["path_frame"])
        else:
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
    OCR/YOLO/frame/trascrizione partono subito dopo in background (un thread
    per file, vedi _post_process_file) e non ritardano la risposta di questo
    endpoint: /api/files espone "elaborazione_in_corso" per farlo sapere alla UI.
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

            _accoda_elaborazione(file_id, str(dest), tipo)

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


# ── Modifica manuale metadati e descrizione ─────────────────────────────────────

@app.route("/api/file/<int:id>/metadata", methods=["POST"])
def update_file_metadata(id):
    """
    Riceve le modifiche manuali dal frontend e aggiorna il database SQLite nelle tabelle reali.
    """
    data = request.get_json(silent=True)
    if not data:
        abort(400, description="Dati JSON mancanti o non validi.")

    nome_file = data.get("nome_file")
    descrizione = data.get("descrizione")
    data_creazione = data.get("data_creazione")
    gps_lat = data.get("gps_lat")
    gps_lon = data.get("gps_lon")

    conn = get_db()
    cursor = conn.cursor()

    try:
        # 1. Controlla se il file esiste e identifica se è una foto o un video
        file_row = cursor.execute("SELECT tipo FROM files WHERE id = ?", (id,)).fetchone()
        if not file_row:
            abort(404, description="File non trovato.")
        tipo = file_row["tipo"]

        # 2. Aggiorna nome_file e descrizione nella tabella principale 'files'
        cursor.execute(
            "UPDATE files SET nome_file = ?, descrizione = ? WHERE id = ?",
            (nome_file, descrizione, id)
        )

        # 3. Aggiorna i dettagli specifici (data e GPS) nella tabella corretta in base al tipo
        tabella = "metadati_foto" if tipo == "foto" else "metadati_video"
        colonna_data = "data_scatto" if tipo == "foto" else "data_creazione"

        cursor.execute(f"SELECT id FROM {tabella} WHERE file_id = ?", (id,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute(
                f"""
                UPDATE {tabella}
                SET {colonna_data} = ?, gps_lat = ?, gps_lon = ?
                WHERE file_id = ?
                """,
                (data_creazione, gps_lat, gps_lon, id)
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {tabella} (file_id, {colonna_data}, gps_lat, gps_lon)
                VALUES (?, ?, ?, ?)
                """,
                (id, data_creazione, gps_lat, gps_lon)
            )

        conn.commit()
        return jsonify({"ok": True, "messaggio": "Salvataggio completato con successo!"}), 200

    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({"errore": f"Errore del database: {str(e)}"}), 500
    finally:
        conn.close()


# ── API — File fantasma (record il cui file è sparito dal disco) ──────────────

@app.route("/api/ghost")
def ghost_files():
    """
    Scansiona l'archivio e segnala i record il cui file non esiste più su disco
    (es. cancellato manualmente dalla cartella). Non modifica il database:
    solo rilevamento, la rimozione è un'azione esplicita separata
    (DELETE /api/file/<id>).
    """
    if not db_ready():
        return jsonify({"totale": 0, "results": []})

    conn = get_db()
    try:
        rows = conn.execute("SELECT id, nome_file, tipo, path FROM files").fetchall()
        mancanti = [dict(row) for row in rows if not Path(row["path"]).exists()]
        return jsonify({"totale": len(mancanti), "results": mancanti})
    finally:
        conn.close()


@app.route("/api/file/<int:id>", methods=["DELETE"])
def delete_file(id):
    """
    Elimina definitivamente un record dal database (e tutte le righe collegate
    nelle tabelle metadati_foto/metadati_video/ocr/oggetti/frame/trascrizioni).
    Non tocca il file su disco: pensato per ripulire i "file fantasma" il cui
    file è già sparito, non per cancellare file ancora presenti in archivio.
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        file_row = cursor.execute("SELECT id FROM files WHERE id = ?", (id,)).fetchone()
        if not file_row:
            abort(404, description="File non trovato nel database.")

        for tabella in ("metadati_foto", "metadati_video", "ocr", "oggetti", "frame", "trascrizioni"):
            cursor.execute(f"DELETE FROM {tabella} WHERE file_id = ?", (id,))
        cursor.execute("DELETE FROM files WHERE id = ?", (id,))

        conn.commit()
        return jsonify({"ok": True, "messaggio": "Record eliminato dal database."}), 200
    except sqlite3.Error as e:
        conn.rollback()
        return jsonify({"errore": f"Errore del database: {str(e)}"}), 500
    finally:
        conn.close()


# ── API — Elaborazione IA arretrata / manuale (OCR/YOLO/frame/whisper) ────────

@app.route("/api/backlog")
def backlog_files():
    """
    Segnala i file mai passati da OCR/YOLO (foto) o frame/trascrizione (video)
    — tipicamente quelli indicizzati prima che /api/import agganciasse
    l'elaborazione in background, o importati mentre una libreria non era
    installata. Non modifica il database: solo rilevamento, l'elaborazione
    è un'azione esplicita separata (POST /api/file/<id>/reprocess).
    """
    if not db_ready():
        return jsonify({"totale": 0, "results": []})

    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT
                f.id, f.nome_file, f.tipo,
                EXISTS(SELECT 1 FROM ocr WHERE ocr.file_id = f.id)             AS has_ocr,
                EXISTS(SELECT 1 FROM oggetti WHERE oggetti.file_id = f.id)     AS has_oggetti,
                EXISTS(SELECT 1 FROM frame WHERE frame.file_id = f.id)        AS has_frame,
                EXISTS(SELECT 1 FROM trascrizioni WHERE trascrizioni.file_id = f.id) AS has_trascrizione,
                EXISTS(SELECT 1 FROM didascalie WHERE didascalie.file_id = f.id)   AS has_didascalia
            FROM files f
        """).fetchall()

        arretrati = [
            {"id": row["id"], "nome_file": row["nome_file"], "tipo": row["tipo"]}
            for row in rows
            if _calcola_ia_stato(
                row["tipo"], bool(row["has_ocr"]), bool(row["has_oggetti"]),
                bool(row["has_frame"]), bool(row["has_trascrizione"]),
                bool(row["has_didascalia"])
            ) == "non_elaborato"
        ]
        return jsonify({"totale": len(arretrati), "results": arretrati})
    finally:
        conn.close()


@app.route("/api/file/<int:id>/reprocess", methods=["POST"])
def reprocess_file(id):
    """
    (Ri)avvia OCR/YOLO/didascalia (foto) o frame/trascrizione (video) per un
    file specifico, in background — sia per i file mai elaborati (vedi
    /api/backlog) sia per rielaborare un file già processato in precedenza
    (es. dopo aver installato una libreria che prima mancava).
    Le righe già presenti per quel file nelle tabelle coinvolte vengono
    cancellate prima di rilanciare l'elaborazione, per non accumulare
    duplicati ad ogni rielaborazione.
    """
    conn = get_db()
    try:
        row = conn.execute("SELECT path, tipo FROM files WHERE id = ?", (id,)).fetchone()
        if not row:
            abort(404, description="File non trovato nel database.")
        path, tipo = row["path"], row["tipo"]

        if not Path(path).exists():
            abort(404, description="Il file non esiste più su disco.")

        cursor = conn.cursor()
        tabelle = ("ocr", "oggetti", "didascalie") if tipo == "foto" else ("frame", "trascrizioni", "oggetti", "didascalie")
        for tabella in tabelle:
            cursor.execute(f"DELETE FROM {tabella} WHERE file_id = ?", (id,))
        conn.commit()
    finally:
        conn.close()

    _accoda_elaborazione(id, path, tipo)
    return jsonify({"ok": True, "messaggio": "Elaborazione in coda."}), 202


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
