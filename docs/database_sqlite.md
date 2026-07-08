# ExoVision — Struttura Database SQLite

Il database SQLite (`exovision.db`) gestisce tutti i **metadati** di foto e video indicizzati dal sistema. È un file singolo, cross-platform, senza server. Viene creato automaticamente alla prima esecuzione dello script.

---

## Schema generale

```
exovision.db
├── files               → anagrafica di ogni file multimediale
├── metadati_foto       → dati EXIF estratti dalle foto
├── metadati_video      → dati tecnici estratti dai video
├── ocr                 → testo estratto dalle immagini (EasyOCR)
├── oggetti             → oggetti rilevati in immagini e video (YOLOv8n)
├── frame               → keyframe estratti dai video (scene detection ffmpeg, con fallback a frame singolo)
├── trascrizioni        → trascrizione audio dei video (faster-whisper)
└── didascalie          → didascalia IA di foto/video (image captioning con BLIP)
```

---

## Tabella `files`

Tabella centrale. Ogni riga rappresenta un file (foto o video) presente nell'archivio.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco, autoincrementale |
| `path` | TEXT UNIQUE | Percorso assoluto del file sul disco |
| `nome_file` | TEXT | Nome del file con estensione |
| `tipo` | TEXT | `foto` oppure `video` |
| `estensione` | TEXT | Estensione del file (`.jpg`, `.mp4`, ecc.) |
| `dimensione_bytes` | INTEGER | Peso del file in byte |
| `data_modifica` | TEXT | Data ultima modifica (ISO 8601) |
| `data_indicizzazione` | TEXT | Data in cui il file è stato indicizzato (ISO 8601) |
| `metadati_completi` | INTEGER | `1` = metadati completi, `0` = metadati mancanti o parziali |
| `descrizione` | TEXT | Descrizione manuale scritta dall'utente in UI — distinta dalla didascalia generata dall'IA (tabella `didascalie`) |

> Il flag `metadati_completi` permette all'interfaccia di segnalare all'utente i file che necessitano attenzione. Lo stato dell'elaborazione IA (OCR/YOLO/didascalia/trascrizione) è invece derivato a runtime dall'esistenza di righe nelle rispettive tabelle (`ia_stato` in `GET /api/files`), non da una colonna dedicata.

---

## Tabella `metadati_foto`

Contiene i dati EXIF estratti da ogni foto. Collegata a `files` tramite `file_id`.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento a `files.id` |
| `larghezza` | INTEGER | Larghezza in pixel |
| `altezza` | INTEGER | Altezza in pixel |
| `modalita` | TEXT | Modalità colore (`RGB`, `RGBA`, `L`, ecc.) |
| `data_scatto` | TEXT | Data e ora dello scatto (da EXIF) |
| `camera_make` | TEXT | Produttore della fotocamera |
| `camera_model` | TEXT | Modello della fotocamera |
| `iso` | INTEGER | Valore ISO |
| `apertura` | TEXT | Apertura del diaframma (es. `f/2.8`) |
| `otturatore` | TEXT | Tempo di esposizione |
| `lunghezza_focale` | TEXT | Lunghezza focale dell'obiettivo |
| `gps_lat` | REAL | Latitudine GPS in gradi decimali |
| `gps_lon` | REAL | Longitudine GPS in gradi decimali |
| `extra_exif` | TEXT | JSON con campi EXIF aggiuntivi non mappati |

> `metadati_completi` in `files` viene impostato a `1` solo se `larghezza`, `altezza`, `data_scatto` e `camera_make` sono tutti presenti.

---

## Tabella `metadati_video`

Contiene i dati tecnici estratti dai video tramite FFprobe. Collegata a `files` tramite `file_id`.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento a `files.id` |
| `durata_secondi` | REAL | Durata del video in secondi |
| `larghezza` | INTEGER | Larghezza in pixel |
| `altezza` | INTEGER | Altezza in pixel |
| `framerate` | REAL | Frame per secondo |
| `codec_video` | TEXT | Codec video (es. `h264`, `vp9`) |
| `codec_audio` | TEXT | Codec audio (es. `aac`, `mp3`) |
| `bitrate` | INTEGER | Bitrate totale in bit/s |
| `extra` | TEXT | JSON con informazioni aggiuntive sul formato |

> `metadati_completi` viene impostato a `1` solo se `durata_secondi`, `larghezza`, `altezza` e `codec_video` sono tutti presenti.

---

## Tabella `ocr`

Testo estratto dalle immagini con EasyOCR. Collegata a `files` tramite `file_id`.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento a `files.id` |
| `testo` | TEXT | Testo estratto (NULL se nessun testo trovato o sotto la soglia di confidenza) |
| `lingua` | TEXT | Lingue usate per il riconoscimento (es. `it+en`) |
| `confidenza` | REAL | Confidenza media (0–1) dei blocchi di testo accettati |
| `data_estrazione` | TEXT | Data di esecuzione dell'OCR (ISO 8601) |

> Viene sempre inserita una riga per ogni immagine processata (anche con `testo` NULL), così il file non viene ririprocessato ad ogni esecuzione.

---

## Tabella `oggetti`

Oggetti rilevati con YOLOv8n. Collegata a `files` tramite `file_id` — per le foto l'analisi gira sull'immagine originale, per i video sul primo keyframe rappresentativo estratto (tabella `frame`).

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento a `files.id` |
| `oggetto` | TEXT | Nome della classe rilevata (NULL se nessun oggetto rilevato) |
| `confidenza` | REAL | Confidenza della rilevazione (0–1) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | REAL | Bounding box dell'oggetto in pixel |
| `data_estrazione` | TEXT | Data di esecuzione della rilevazione (ISO 8601) |

> Come per `ocr`, se non viene rilevato nulla si inserisce comunque una riga con `oggetto` NULL per marcare il file come processato.

---

## Tabella `frame`

Registra i keyframe estratti dai video tramite scene detection ffmpeg (frame salvati solo in corrispondenza di un cambio di scena, non a intervallo fisso). Ogni frame è trattato come un'immagine indipendente ai fini dell'embedding visivo.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento al video in `files.id` |
| `timestamp_sec` | REAL | Posizione temporale del frame nel video (secondi) |
| `path_frame` | TEXT | Percorso del file immagine estratto su disco (`frame/fr-<file_id>-<n>-<nomevideo>.jpg`) |

> Gli embedding di ogni frame verranno salvati in ChromaDB, usando `frame.id` come chiave di collegamento. Se l'estrazione fallisce o non rileva cambi di scena, viene comunque inserita una riga segnaposto (`timestamp_sec`/`path_frame` NULL) per non ritentare il video ad ogni esecuzione.

---

## Tabella `trascrizioni`

Trascrizione dell'audio dei video con faster-whisper. Collegata a `files` tramite `file_id`.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento al video in `files.id` |
| `testo` | TEXT | Trascrizione (NULL se nessun parlato rilevato o errore) |
| `lingua` | TEXT | Lingua rilevata automaticamente da whisper (o forzata da config) |
| `confidenza` | REAL | Confidenza media (0–1), derivata da `exp(avg_logprob)` dei segmenti |
| `data_estrazione` | TEXT | Data di esecuzione della trascrizione (ISO 8601) |

> Stessa logica di segnaposto delle altre tabelle: una riga viene sempre inserita, anche a vuoto, per evitare ritentativi infiniti.

---

## Tabella `didascalie`

Didascalia generata dall'IA (BLIP, in inglese) — per le foto sull'immagine originale, per i video sul primo keyframe rappresentativo. Collegata a `files` tramite `file_id`, distinta dalla descrizione manuale (`files.descrizione`).

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento a `files.id` |
| `testo` | TEXT | Didascalia generata (NULL se non generata o errore) |
| `lingua` | TEXT | Lingua della didascalia (sempre `en`, i modelli BLIP pubblici sono addestrati in inglese) |
| `data_estrazione` | TEXT | Data di generazione della didascalia (ISO 8601) |

> Stessa logica di segnaposto delle altre tabelle: una riga viene sempre inserita, anche a vuoto, per evitare ritentativi infiniti.

---

## Relazioni

```
files (1) ──────────────── (1) metadati_foto
files (1) ──────────────── (1) metadati_video
files (1) ──────────────── (N) ocr
files (1) ──────────────── (N) oggetti
files (1) ──────────────── (N) frame
files (1) ──────────────── (N) trascrizioni
files (1) ──────────────── (N) didascalie
```

---

## Collegamento con ChromaDB

SQLite gestisce i **metadati strutturati**. ChromaDB gestisce gli **embedding vettoriali** per la ricerca semantica. I due database si parlano tramite l'`id` della tabella `files` (per foto) e l'`id` della tabella `frame` (per i frame video).

```
SQLite files.id  ←→  ChromaDB document_id  (foto)
SQLite frame.id  ←→  ChromaDB document_id  (frame video)
```

Quando l'utente fa una ricerca semantica, ChromaDB restituisce una lista di `id` — ExoVision usa quegli id per recuperare i metadati completi da SQLite e mostrarli nell'interfaccia.

---

## Query utili

```sql
-- File con metadati incompleti
SELECT nome_file, tipo FROM files WHERE metadati_completi = 0;

-- Conteggio per tipo
SELECT tipo, COUNT(*) FROM files GROUP BY tipo;

-- Foto con coordinate GPS
SELECT f.nome_file, m.gps_lat, m.gps_lon
FROM files f JOIN metadati_foto m ON f.id = m.file_id
WHERE m.gps_lat IS NOT NULL;

-- Frame di un video specifico
SELECT f.nome_file, fr.timestamp_sec, fr.path_frame
FROM files f JOIN frame fr ON f.id = fr.file_id
WHERE f.nome_file = 'video.mp4'
ORDER BY fr.timestamp_sec;

-- Immagini con testo OCR rilevato
SELECT f.nome_file, o.testo, o.confidenza
FROM files f JOIN ocr o ON f.id = o.file_id
WHERE o.testo IS NOT NULL;

-- Video con trascrizione audio disponibile
SELECT f.nome_file, t.lingua, t.testo
FROM files f JOIN trascrizioni t ON f.id = t.file_id
WHERE t.testo IS NOT NULL;
```
