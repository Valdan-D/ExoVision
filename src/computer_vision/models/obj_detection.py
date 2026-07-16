"""
Modulo per l'estrazione di metadati visivi tramite Object Detection con YOLO (Versione Batch).

Questo file contiene l'implementazione del motore di rilevamento oggetti di ExoVision,
sfruttando l'architettura leggera YOLO (Ultralytics) per identificare elementi
letterali nelle immagini e popolarne i metadati tradizionali su ChromaDB.

Motore di inferenza unico per tutto il progetto (usato sia da app.py/exovision_yolo.py
per il flusso principale, sia da computer_vision/main.py per l'ingestion standalone) —
non legge config.json e non tocca SQLite: chi lo chiama decide model_path/conf e cosa
farne del risultato.
"""

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    # NB: niente sys.exit qui — stesso pattern di ocr_pipeline.EASYOCR_OK. Questo
    # modulo è importato anche da app.py per l'elaborazione in background, quindi
    # deve poter essere importato anche senza ultralytics installato.


class YoloDetector:
    """
    Classe per l'estrazione di etichette (tag) e conteggi di oggetti discreti da immagini PIL usando YOLO.
    Incapsula l'inizializzazione del modello pre-addestrato (default: YOLO Nano)
    e la logica di parsing dei bounding box per estrarre classi univoche in modalità batch.
    """

    def __init__(self, model_path="yolov8m.pt", conf=0.20):
        """
        Inizializza il modello YOLO scaricando i pesi se non presenti localmente.
        Args:
            model_path (str): Nome o percorso del file dei pesi del modello.
            conf (float): soglia di confidenza minima di default, usata da
                estrai_tag_batch/rileva_oggetti quando non se ne passa una esplicita.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        print(f"[+] YOLO caricato correttamente ({model_path}).")

    def estrai_tag_batch(self, lista_immagini, conf=None):
        """
        Esegue l'object detection su un intero batch di immagini, estraendo tag univoci e conteggi per ciascuna.
        Questo metodo sfrutta la capacità nativa di YOLO di elaborare parallelamente una lista 
        di immagini per massimizzare il throughput hardware. Analizza la lista di oggetti `Results` 
        generata dall'inferenza di blocco, esegue il parsing dei bounding box di ogni singola foto, 
        mappa gli indici di classe nei rispettivi nomi testuale ed elimina i duplicati categoria per categoria.
        Args:
            lista_immagini (list): Una lista di oggetti immagine in formato PIL.Image da analizzare.
        Returns:
            tuple: Una tupla contenente due liste parallele e ordinate rispetto al batch di input:
                - list[list[str]]: Una lista di liste, dove ogni sotto-lista racchiude i tag univoci 
                  rilevati nella rispettiva immagine (es. [['person', 'dog'], ['traffic light']]). 
                  Restituisce ['none'] per le immagini prive di rilevamenti.
                - list[int]: Una lista di interi, in cui ogni elemento rappresenta il numero totale 
                  di oggetti fisici discreti individuati nella scena (es. [3, 1]).
        """
        # Passiamo l'intera lista a YOLO per l'inferenza parallela.
        # Se non viene passata una soglia esplicita usiamo self.conf (di default
        # 0.20, storicamente più permissiva della soglia "single-image" per
        # intercettare anche i tag più "difficili" nell'ingestion batch).
        soglia = conf if conf is not None else self.conf
        results_list = self.model(lista_immagini, conf=soglia, verbose=False)

        batch_tag_unici = []
        batch_conteggi = []

        # Iteriamo sui risultati di ciascuna immagine nel batch
        for r in results_list:
            oggetti = [self.model.names[int(box.cls[0])] for box in r.boxes]
            tag_unici = list(set(oggetti))

            batch_tag_unici.append(tag_unici if tag_unici else ["none"])
            batch_conteggi.append(len(oggetti))

        return batch_tag_unici, batch_conteggi

    def rileva_oggetti(self, path, conf=None):
        """
        Rileva oggetti in una singola immagine (da percorso su disco), restituendo
        una lista di dict {oggetto, confidenza} — un elemento per ogni oggetto
        individuato (non deduplicato, a differenza di estrai_tag_batch). Usata dal
        flusso principale (app.py/exovision_yolo.py) per popolare la tabella SQL
        `oggetti` e i chip in UI. Niente bounding box: non consumati da nessun
        client (verificato in exovision.html e nello schema DB), quindi omessi
        per restare aderenti a quello che serve davvero.
        """
        soglia = conf if conf is not None else self.conf
        try:
            risultati = self.model(path, conf=soglia, verbose=False)
            oggetti = []
            for r in risultati:
                for box in r.boxes:
                    oggetti.append({
                        "oggetto":    self.model.names[int(box.cls)],
                        "confidenza": round(float(box.conf), 3),
                    })
            return oggetti
        except Exception as e:
            print(f"⚠️  Errore YOLO su {path}: {e}")
            return []
