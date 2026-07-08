import os
from deepface import DeepFace


class ExoFaceRecognizer:
    """
    Estrattore biometrico di caratteristiche facciali (ArcFace + RetinaFace).    
    In questa nuova architettura a doppia collezione, la classe non memorizza più
    i volti noti in locale, ma si limita ad estrarre i vettori grezzi (embeddings)
    da salvare o cercare direttamente all'interno di ChromaDB.
    """

    def __init__(self):
        self.modello_target = "ArcFace"
        self.detector_backend = "retinaface"
        print(
            f"[-] Inizializzazione Estrattore Volti ({self.modello_target} + {self.detector_backend})...")

    def estrai_vettore_volto(self, percorso_immagine):
        """
        Analizza l'immagine, rileva i volti presenti ed estrae i loro vettori matematici.
        Args:
            percorso_immagine (str): Percorso del file immagine da analizzare.
        Returns:
            list[list[float]]: Una lista contenente i vettori (embeddings) di tutti i volti trovati.
                               Restituisce una lista vuota se non viene rilevato alcun volto umano.
        """
        try:
            # Estragga le feature biometriche pure tramite DeepFace
            embedding_objs = DeepFace.represent(
                img_path=os.path.abspath(percorso_immagine),
                model_name=self.modello_target,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )

            # Estraiamo solo il vettore float (l'embedding) per ogni volto rilevato nella foto
            vettori_volti = [obj["embedding"] for obj in embedding_objs]
            return vettori_volti

        except ValueError:
            # Comportamento standard di DeepFace (RetinaFace) quando non trova volti umani nell'immagine
            return []
        except Exception as e:
            print(
                f"[Warning Face] Errore durante l'estrazione facciale su {os.path.basename(percorso_immagine)}: {e}")
            return []
