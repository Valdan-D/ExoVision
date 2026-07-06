import torch
from transformers import AutoProcessor, AutoModel


class SigLIPEmbedder:
    
    def __init__(self, model_path="google/siglip-base-patch16-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        print(f"[+] SigLIP caricato correttamente su: {self.device}")

    def istanzia_vettore_immagine(self, immagine_pil):
        """
            Trasforma un'immagine PIL in una lista di 768 numeri decimali.
        """
        inputs = self.processor(images=immagine_pil,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        if not isinstance(outputs, torch.Tensor):
            features = outputs.pooler_output if hasattr(
                outputs, 'pooler_output') else outputs[0]
        else:
            features = outputs

        # Normalizzazione L2
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def istanzia_vettore_testo(self, testo_query):
        """
        Trasforma una stringa di testo in una lista di 768 numeri decimali.
        """
        inputs = self.processor(
            text=[testo_query], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        if not isinstance(outputs, torch.Tensor):
            features = outputs.pooler_output if hasattr(
                outputs, 'pooler_output') else outputs[0]
        else:
            features = outputs

        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()
