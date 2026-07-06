"""
Modulo per la generazione di embedding multimodali tramite SigLIP.
Questo file contiene l'implementazione del motore di embedding di ExoVision,
sfruttando il modello di Google (SigLIP) per mappare immagini e testo nello
stesso spazio vettoriale geometrico a 768 dimensioni.
"""

"""
Modulo per la generazione di embedding multimodali tramite SigLIP (Versione Batch Corretta).
"""

import torch
from transformers import AutoProcessor, AutoModel


class SigLIPEmbedder:
    def __init__(self, model_path="google/siglip-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        print(f"[+] SigLIP caricato correttamente su: {self.device}")

    def istanzia_vettori_batch(self, lista_immagini):
        """
        Trasforma una LISTA di immagini PIL in una lista di vettori (768 float ciascuno).
        """
        inputs = self.processor(images=lista_immagini,
                                return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        if not isinstance(outputs, torch.Tensor):
            features = outputs.pooler_output if hasattr(
                outputs, 'pooler_output') else outputs[0]
        else:
            features = outputs

        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().tolist()

    def istanzia_vettore_testo(self, testo_query):
        """
        Trasforma una singola stringa di testo in un unico vettore normalizzato di 768 float.
        """
        inputs = self.processor(
            text=[testo_query],
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        # Estraiamo in modo pulito il primo e unico elemento del batch testuale
        if not isinstance(outputs, torch.Tensor):
            features = outputs.pooler_output if hasattr(
                outputs, 'pooler_output') else outputs[0]
        else:
            features = outputs

        # Normalizzazione L2
        features = features / features.norm(dim=-1, keepdim=True)

        # .flatten().tolist() garantisce una lista piatta monodimensionale di float
        return features.cpu().numpy().flatten().tolist()
