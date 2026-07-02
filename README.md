# 🔍 ExoVision

> Motore di ricerca semantica per archivi foto e video — locale, leggero, cross-platform.

ExoVision permette di cercare immagini e video usando il linguaggio naturale, senza dipendere da servizi cloud o hardware potente. Tutto gira in locale sul proprio PC, su qualsiasi sistema operativo.

---

## ✨ Funzionalità principali

- **Ricerca per parola chiave** — su testo OCR, oggetti rilevati e nome file (ricerca semantica ChromaDB pianificata)
- **Supporto foto e video** — indicizza entrambi con la stessa pipeline
- **OCR** — estrae il testo presente nelle immagini (EasyOCR)
- **Object detection** — riconosce oggetti nelle immagini (YOLOv8n)
- **Estrazione keyframe** — i video vengono analizzati con scene detection ffmpeg, non a intervallo fisso, per catturare solo i cambi di scena significativi
- **Trascrizione audio** — l'audio dei video viene convertito in testo con faster-whisper per arricchire la ricerca
- **Import da browser** — trascina foto/video nella tab Importa: vengono caricati, salvati e indicizzati subito (step metadati; OCR/YOLO/frame/trascrizione restano script da lanciare a parte)
- **Interfaccia web locale** — si apre nel browser, zero installazioni aggiuntive
- **Gestione metadati** — visualizzazione e segnalazione di file con metadati incompleti
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
│   ├── app.py                  # Server web locale (Flask)
│   ├── exovision_metadata.py   # Step 1: metadati EXIF/ffprobe → SQLite
│   ├── exovision_ocr.py        # Step 2: OCR (EasyOCR) sulle immagini
│   ├── exovision_yolo.py       # Step 3: object detection (YOLOv8n)
│   ├── exovision_frames.py     # Step 4: keyframe video (scene detection ffmpeg)
│   ├── exovision_whisper.py    # Step 5: trascrizione audio video (faster-whisper)
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
| Database vettoriale | ChromaDB (pianificato, non ancora collegato) |
| Estrazione metadati foto | Pillow + piexif |
| Estrazione metadati video | FFmpeg + ffmpeg-python |
| OCR | EasyOCR |
| Object detection | YOLOv8n (ultralytics) |
| Estrazione keyframe | FFmpeg (scene detection) |
| Trascrizione audio | faster-whisper (locale, CPU) |
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

# Avvia l'interfaccia web
python src/app.py
```

Apri il browser su `http://localhost:5000`

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
