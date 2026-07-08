# 🔍 ExoVision

> Motore di ricerca semantica per archivi foto e video — locale, leggero, cross-platform.

ExoVision permette di cercare immagini e video usando il linguaggio naturale, senza dipendere da servizi cloud o hardware potente. Tutto gira in locale sul proprio PC, su qualsiasi sistema operativo.

---

## ✨ Funzionalità principali

- **Ricerca per parola chiave** — su testo OCR, oggetti rilevati, trascrizioni audio e nome file
- **Supporto foto e video** — indicizzati con la stessa pipeline; i video ereditano tag oggetti e didascalia IA dal primo keyframe estratto
- **OCR** — estrae il testo presente nelle immagini (EasyOCR)
- **Object detection** — riconosce oggetti nelle immagini e nei video (YOLOv8n)
- **Didascalia automatica (image captioning)** — genera una frase descrittiva in inglese per ogni foto/video con BLIP, distinta dalla descrizione manuale scritta dall'utente
- **Estrazione keyframe** — i video vengono analizzati con scene detection ffmpeg; se non viene rilevato alcun taglio di scena (video a piano continuo, molto comune), un frame fisso viene estratto come anteprima di riserva così ogni video ha sempre un'immagine statica
- **Trascrizione audio** — l'audio dei video viene convertito in testo con faster-whisper per arricchire la ricerca
- **Import da browser** — trascina foto/video nella tab Importa: vengono caricati, salvati e indicizzati subito (metadati, sincrono); OCR/YOLO/didascalia/frame/trascrizione partono subito dopo **in background** (coda seriale, un file alla volta) senza bloccare l'upload
- **Recupero archivio** — rileva e ripulisce i "file fantasma" (record il cui file è stato cancellato dal disco) e rielabora i file mai passati da OCR/YOLO/didascalia (es. indicizzati prima che l'elaborazione in background esistesse)
- **Ricerca semantica e riconoscimento facciale** *(modulo `src/computer_vision`, in sviluppo separato)* — embedding SigLIP su ChromaDB per ricerca concettuale e riconoscimento biometrico dei volti (ArcFace/RetinaFace); non ancora integrato con l'interfaccia web principale
- **Interfaccia web locale** — si apre nel browser, zero installazioni aggiuntive
- **Gestione metadati** — visualizzazione e segnalazione di file con metadati incompleti o non ancora elaborati dall'IA
- **Configurabile** — tutti i parametri modificabili dall'interfaccia (`config.json`, creato in automatico al primo avvio)

---

## 🏗️ Architettura

```
ExoVision/
├── setup.py                    # Verifica/installa dipendenze Python + FFmpeg
├── requirements.txt
├── config.example.json         # Template configurazione (committato)
├── config.json                 # Configurazione locale (creato in automatico al primo avvio)
├── src/
│   ├── app.py                  # Server web locale (Flask) — espone tutte le API REST
│   ├── exovision_metadata.py   # Step 1: metadati EXIF/ffprobe → SQLite
│   ├── exovision_ocr.py        # Step 2: OCR (EasyOCR) sulle immagini
│   ├── exovision_yolo.py       # Step 3: object detection (YOLOv8n)
│   ├── exovision_frames.py     # Step 4: keyframe video (scene detection ffmpeg + fallback frame singolo)
│   ├── exovision_whisper.py    # Step 5: trascrizione audio video (faster-whisper)
│   ├── exovision_caption.py    # Step 6: didascalia foto/video (image captioning con BLIP)
│   ├── computer_vision/        # Modulo separato: ricerca semantica ChromaDB (SigLIP) +
│   │   ├── main.py             #   riconoscimento facciale (ArcFace/RetinaFace via DeepFace).
│   │   ├── database.py         #   Standalone, non ancora agganciato alle API di app.py.
│   │   └── models/
│   └── UI/
│       └── exovision.html      # Interfaccia web (HTML/CSS/JS inline)
├── docs/
│   └── database_sqlite.md      # Documentazione database
├── frame/                   # Keyframe estratti dai video (ignorata da git)
├── tests/
├── data/                    # (ignorata da git — dati locali)
├── .gitignore
└── README.md
```

### Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Linguaggio | Python 3.10+ |
| Database metadati | SQLite (`exovision.db`) |
| Database vettoriale | ChromaDB — usato nel modulo separato `computer_vision`, non ancora collegato alle API principali |
| Estrazione metadati foto | Pillow + piexif |
| Estrazione metadati video | FFmpeg + ffmpeg-python |
| OCR | EasyOCR |
| Object detection | YOLOv8n (ultralytics) |
| Didascalia foto/video | BLIP (transformers + torch) |
| Estrazione keyframe | FFmpeg (scene detection + fallback frame singolo) |
| Trascrizione audio | faster-whisper (locale, CPU) |
| Ricerca semantica | SigLIP embeddings (modulo `computer_vision`) |
| Riconoscimento facciale | ArcFace + RetinaFace via DeepFace (modulo `computer_vision`) |
| Interfaccia | HTML/CSS/JS — servita via Flask |
| Cross-platform | Windows, macOS, Linux |

---

## 🚀 Installazione

### Prerequisiti

- Python 3.10 o superiore
- FFmpeg installato sul sistema → [ffmpeg.org](https://ffmpeg.org/download.html)

### Setup

```bash
# Clona il repository
git clone https://github.com/Valdan-D/ExoVision.git
cd ExoVision

# Verifica/installa dipendenze Python e FFmpeg
python setup.py
```

`config.json` non serve crearlo a mano se il primo comando che lanci è `python src/app.py`: viene generato automaticamente da `config.example.json`. Se invece parti da uno degli script di indicizzazione (`exovision_metadata.py` ecc.), copialo prima tu: `cp config.example.json config.json`.

### Prima esecuzione

```bash
# 1. Indicizza una cartella di foto/video (metadati EXIF/ffprobe)
python src/exovision_metadata.py ./tuo-archivio

# 2. OCR sulle immagini
python src/exovision_ocr.py ./tuo-archivio

# 3. Object detection sulle immagini
python src/exovision_yolo.py ./tuo-archivio

# 4. Keyframe dai video (scene detection)
python src/exovision_frames.py ./tuo-archivio

# 5. Trascrizione audio dai video
python src/exovision_whisper.py ./tuo-archivio

# 6. Didascalia automatica delle foto (image captioning)
python src/exovision_caption.py ./tuo-archivio

# Avvia l'interfaccia web
python src/app.py
```

Apri il browser su `http://localhost:5000`

> Gli script 2-6 servono solo per elaborare in blocco un archivio già indicizzato da riga di comando. Caricando i file dalla tab **Importa** dell'interfaccia web, OCR/YOLO/didascalia/frame/trascrizione partono automaticamente in background subito dopo l'upload — non serve lanciarli a mano. Se una libreria non è installata, il relativo step viene semplicemente saltato per i nuovi file (gli altri proseguono); la tab Metadati permette di rilevare e rielaborare in un secondo momento i file rimasti senza IA.

---

## 👥 Team

| Ruolo | Nome |
|---|---|
| Team Leader | Danilo |
| UI Designer | Stefano |
| Integration Developer | Simone |
| Computer Vision Developer | Alice |
| Database Designer | Felipe |

---

## 📁 Documentazione

- [Struttura database SQLite](docs/database_sqlite.md)

---

## 📄 Licenza

Progetto didattico — ITS ICT Torino, Learning by Projects 2026.
