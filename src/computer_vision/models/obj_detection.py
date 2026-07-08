"""
Modulo per l'estrazione di metadati visivi tramite Object Detection con YOLO (Versione Batch).

Questo file contiene l'implementazione del motore di rilevamento oggetti di ExoVision,
sfruttando l'architettura leggera YOLO (Ultralytics) per identificare elementi
letterali nelle immagini e popolarne i metadati tradizionali su ChromaDB.
"""

from ultralytics import YOLO


class YoloDetector:
    """
    Classe per l'estrazione di etichette (tag) e conteggi di oggetti discreti da immagini PIL usando YOLO.    
    Incapsula l'inizializzazione del modello pre-addestrato (default: YOLO Nano)
    e la logica di parsing dei bounding box per estrarre classi univoche in modalità batch.
    """

    def __init__(self, model_path="yolov8m.pt"):
        """
        Inizializza il modello YOLO scaricando i pesi se non presenti localmente.
        Args:
            model_path (str): Nome o percorso del file dei pesi del modello.
        """
        self.model = YOLO(model_path)
        print(f"[+] YOLO Nano caricato correttamente!")

    def estrai_tag_batch(self, lista_immagini):
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
        # Passiamo l'intera lista a YOLO per l'inferenza parallela
        # Abbassiamo la soglia di confidenza (es. a 0.20 o 0.25) per intercettare anche i tag più "difficili"
        results_list = self.model(lista_immagini, conf=0.20, verbose=False)

        batch_tag_unici = []
        batch_conteggi = []

        # Iteriamo sui risultati di ciascuna immagine nel batch
        for r in results_list:
            oggetti = [self.model.names[int(box.cls[0])] for box in r.boxes]
            tag_unici = list(set(oggetti))

            batch_tag_unici.append(tag_unici if tag_unici else ["none"])
            batch_conteggi.append(len(oggetti))

        return batch_tag_unici, batch_conteggi


''''
# Se usi 'yolov8n.pt', cioè il nano, sostituiscilo con uno di questi:
self.model = YOLO("yolov8s.pt")  # Più preciso del nano, comunque veloce 
# oppure
self.model = YOLO("yolov8m.pt")  # Medium, molto robusto sui dettagli complessi
'''
