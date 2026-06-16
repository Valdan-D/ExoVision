# ExoVision — Struttura Database SQLite

Il database SQLite (`exovision.db`) gestisce tutti i **metadati** di foto e video indicizzati dal sistema. È un file singolo, cross-platform, senza server. Viene creato automaticamente alla prima esecuzione dello script.

---

## Schema generale

```
exovision.db
├── files               → anagrafica di ogni file multimediale
├── metadati_foto       → dati EXIF estratti dalle foto
├── metadati_video      → dati tecnici estratti dai video
└── frame               → frame estratti dai video ogni N secondi
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

> Il flag `metadati_completi` permette all'interfaccia di segnalare all'utente i file che necessitano attenzione.

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

## Tabella `frame`

Registra i frame estratti dai video ogni N secondi. Ogni frame è trattato come un'immagine indipendente ai fini dell'embedding visivo.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `id` | INTEGER PK | Identificativo univoco |
| `file_id` | INTEGER FK | Riferimento al video in `files.id` |
| `timestamp_sec` | REAL | Posizione temporale del frame nel video (secondi) |
| `path_frame` | TEXT | Percorso del file immagine estratto su disco |

> Gli embedding di ogni frame vengono salvati in ChromaDB, usando `frame.id` come chiave di collegamento.

---

## Relazioni

```
files (1) ──────────────── (1) metadati_foto
files (1) ──────────────── (1) metadati_video
files (1) ──────────────── (N) frame
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
```
