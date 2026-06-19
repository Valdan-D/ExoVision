# ExoVision — Guida per Claude Code

## Cos'è il progetto
Archivio intelligente di foto e video con indicizzazione automatica, OCR e riconoscimento oggetti. Interfaccia web Flask + HTML/JS vanilla. Tutto gira in locale.

## Struttura
```
ExoVision/
├── config.example.json       # Template configurazione (committato)
├── config.json               # Configurazione locale (gitignored — non toccare)
├── src/
│   ├── app.py                # Server Flask, espone tutte le API REST
│   ├── exovision_metadata.py # Step 1: scansiona cartella, estrae EXIF/ffprobe → SQLite
│   ├── exovision_ocr.py      # Step 2: OCR con EasyOCR sulle immagini già nel DB
│   ├── exovision_yolo.py     # Step 3: object detection YOLOv8n sulle immagini già nel DB
│   └── UI/
│       └── exovision.html    # Tutto il frontend (CSS + JS inline, no framework)
├── requirements.txt
└── yolov8n.pt                # Modello YOLO (gitignored, ~6MB, scaricato da ultralytics)
```

## Flusso d'uso corretto
1. `python src/exovision_metadata.py <cartella>` — crea DB e indicizza
2. `python src/exovision_ocr.py <cartella>` — aggiunge OCR
3. `python src/exovision_yolo.py <cartella>` — aggiunge object detection
4. `python src/app.py` — avvia server su http://localhost:5000

## API Flask (src/app.py)
| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/` | GET | Serve exovision.html |
| `/api/files` | GET | Lista paginata (`page`, `limit`, `tipo`) |
| `/api/search` | GET | Ricerca keyword (`q`, `limit`) su OCR + oggetti |
| `/api/file/<id>` | GET | Dettaglio: metadati, OCR, oggetti, simili |
| `/api/thumb/<id>` | GET | Serve il file immagine originale |
| `/api/stats` | GET | Statistiche generali archivio |
| `/api/config` | GET | Legge config.json |
| `/api/config` | POST | Salva config.json (merge parziale) |

## Schema SQLite (exovision.db)
- `files` — record principale per ogni file (path, nome, tipo, dimensione, data)
- `metadati_foto` — EXIF: dimensioni, data scatto, camera, ISO, apertura, GPS
- `metadati_video` — durata, risoluzione, codec, framerate
- `ocr` — testo estratto, lingua, confidenza
- `oggetti` — oggetti YOLO: nome, confidenza, bounding box
- `frame` — frame estratti dai video (tabella pronta, script in sviluppo)

## Configurazione (config.json)
Letta da tutti gli script Python all'avvio e dalla UI via `/api/config`.
```json
{
  "archivio": { "percorso": "", "db": "exovision.db" },
  "ocr":      { "lingue": ["it","en"], "confidenza_minima": 0.4 },
  "yolo":     { "modello": "yolov8n.pt", "confidenza_minima": 0.4 },
  "video":    { "frame_interval_secondi": 5 },
  "ricerca":  { "soglia_similarita": 0.75, "modalita": "locale" },
  "ui":       { "tema_scuro": false }
}
```

## Frontend (exovision.html)
JS vanilla puro, nessun framework. Tutto in un unico file HTML.
- **Ricerca/Grid**: `GET /api/files` (init) e `GET /api/search?q=...` (ricerca), debounce 350ms
- **Dettaglio**: `GET /api/file/<id>`, mostra EXIF/video, OCR, oggetti YOLO, simili
- **Impostazioni**: `GET /api/config` all'avvio, `POST /api/config` al salvataggio
- **Metadati tab**: ancora su dati mock — da collegare alle API
- **Importa tab**: ancora mock — dipende dagli script backend in sviluppo

## Lavori in corso / TODO
- [ ] Script analisi video (a cura di una collega) — estrazione frame + metadati avanzati
- [ ] Collegare la tab Metadati alle API reali (ora usa dati mock)
- [ ] Ricerca semantica con ChromaDB/CLIP (menzionta nei TODO di app.py)
- [ ] Thumbnail video: endpoint `/api/thumb/<id>` ora fallisce per i video (serve estrazione frame con ffmpeg)
- [ ] Collegare la tab Log a `/api/stats` invece del seed mock

## Dipendenze principali
```
flask, Pillow, piexif, ffmpeg-python, easyocr, ultralytics
```
FFmpeg deve essere installato a livello di sistema (richiesto da ffmpeg-python).

## Convenzioni
- Avviare sempre il server da dentro `src/`: `python src/app.py` dalla root
- Il DB viene cercato relativo alla CWD — usare sempre percorsi assoluti in produzione
- I modelli `.pt` YOLO non vanno in git (gitignored via `*.pt`)
