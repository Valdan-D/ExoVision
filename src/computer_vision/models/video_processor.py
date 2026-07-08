from database import ExoVisionDB
import os
import sys
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class VideoProcessor:

    def __init__(self, video_filename, folder_video="video_test", output_root="data"):
        self.video_filename = video_filename
        self.video_path = os.path.join(folder_video, video_filename)
        self.video_name = os.path.splitext(video_filename)[0]

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Il video {self.video_path} non esiste.")

        # Cartelle di output
        self.dir_keyframes = os.path.join(
            output_root, "output_keyframes", self.video_name)
        self.dir_audio = os.path.join(output_root, "output_audio")
        os.makedirs(self.dir_keyframes, exist_ok=True)
        os.makedirs(self.dir_audio, exist_ok=True)

        self.audio_path = os.path.join(
            self.dir_audio, f"{self.video_name}.wav")
        self.db_en = ExoVisionDB()

    def _estrai_e_trascrivi_audio(self, model_size="base"):
        """Estrae la traccia audio e la passa a Whisper per ottenere i segmenti temporali."""
        print(f"[AUDIO] Verifica traccia in {self.video_filename}...")
        video = VideoFileClip(self.video_path)

        # CONTROLLO SE IL VIDEO HA L'AUDIO
        if video.audio is None:
            print(
                "[AUDIO] Nessuna traccia audio trovata in questo video. Salto la trascrizione Whisper.")
            video.close()
            return []  # Ritorna una lista vuota di testi parlati

        print("[AUDIO] Traccia trovata. Estrazione in corso...")
        video.audio.write_audiofile(self.audio_path, logger=None)
        video.close()

        print(f"[AUDIO] Trascrizione con Whisper ({model_size})...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(self.audio_path, beam_size=5)

        scenari_audio = []
        for segment in segments:
            scenari_audio.append({
                "start": segment.start,
                "end": segment.end,
                "testo": segment.text.strip()
            })
            print(
                f"  [{segment.start:.1f}s -> {segment.end:.1f}s]: '{segment.text.strip()}'")
        return scenari_audio

    def _trova_testo_per_timestamp(self, secondi, segmenti_audio):
        """Trova la frase pronunciata nel secondo esatto del frame."""
        for seg in segmenti_audio:
            if seg["start"] <= secondi <= seg["end"]:
                return seg["testo"]
        return ""

    def elabora_video(self, yolo_model, siglip_model, face_model, threshold=0.7):
        """Esegue l'intera pipeline: audio, scene detection, modelli IA e scrittura DB."""
        print(f"\n=== AVVIO ELABORAZIONE VIDEO: {self.video_filename} ===")

        # 1. Fase Audio
        segmenti_audio = self._estrai_e_trascrivi_audio()

        # 2. Fase Video (Scene Detection)
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        prev_hist = None
        frame_idx = 0

        # Variabili per il resoconto finale
        frame_salvati = 0
        tutti_i_tag_trovati = set()

        print(f"\n[VIDEO] Avvio Scene Detection ({fps:.2f} FPS)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calcolo istogramma
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_frame], [0, 1], None,
                                [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            is_keyframe = False
            similarieta_stamp = 1.0  # Default

            # Se è il primissimo frame, lo salviamo a prescindere!
            if prev_hist is None:
                is_keyframe = True
                similarieta_stamp = 0.0  # Valore fittizio per "Nuova Scena Iniziale"
            else:
                # Confrontiamo con l'ULTIMO KEYFRAME SALVATO
                similarieta = cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_CORREL)
                similarieta_stamp = similarieta
                if similarieta < threshold:
                    is_keyframe = True

            # Se abbiamo rilevato un cambio scena (o è il primo frame)
            if is_keyframe:
                frame_salvati += 1
                secondi = frame_idx / fps
                minuti = int(secondi // 60)
                sec = int(secondi % 60)
                timestamp_str = f"{minuti:02d}:{sec:02d}"

                # Salva il frame su disco
                filename = f"frame_{frame_idx}_{minuti:02d}{sec:02d}.jpg"
                filepath = os.path.join(self.dir_keyframes, filename)
                cv2.imwrite(filepath, frame)

                # --- INTERROGAZIONE MODELLI IA REALI ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                # 1. YOLO
                tag_yolo_batch, _ = yolo_model.estrai_tag_batch([img_pil])
                lista_tag = [str(t).lower().strip() for t in tag_yolo_batch[0]] if isinstance(
                    tag_yolo_batch[0], list) else [str(tag_yolo_batch[0]).lower().strip()]
                if "none" in lista_tag:
                    lista_tag.remove("none")
                if not lista_tag:
                    lista_tag = ["none"]

                for t in lista_tag:
                    if t != "none":
                        tutti_i_tag_trovati.add(t)

                # 2. SigLIP
                vettori_siglip_batch = siglip_model.istanzia_vettori_batch([
                                                                           img_pil])
                vettore = vettori_siglip_batch[0]

                # Se è un array/tensor usiamo tolist(), se è già una lista la lasciamo così
                embedding_vettore = vettore.tolist() if hasattr(
                    vettore, "tolist") else list(vettore)

                # 3. Volti
                ids_volti_frame = []
                embeddings_volti_frame = []
                metadati_volti_frame = []

                if "person" in lista_tag:
                    vettori_facciali = face_model.estrai_vettore_volto(
                        filepath)
                    for v_idx, vettore_volto in enumerate(vettori_facciali):
                        ids_volti_frame.append(f"face_{filename}_{v_idx}")
                        embeddings_volti_frame.append(vettore_volto)
                        metadati_volti_frame.append(
                            {"parent_img_path": filepath, "timestamp": timestamp_str})

                volti_rilevati = f"{len(embeddings_volti_frame)} volto/i" if embeddings_volti_frame else "Nessuno"

                # Audio
                testo_parlato = self._trova_testo_per_timestamp(
                    secondi, segmenti_audio)

                # Metadati e Scrittura DB
                id_univoco = f"VID_{self.video_name}_FR_{frame_idx}"
                tag_stringa = ", ".join(lista_tag)

                metadati = {
                    "video_sorgente": self.video_filename,
                    "timestamp": timestamp_str,
                    "yolo_tags": tag_stringa,
                    "volti_rilevati": volti_rilevati,
                    "audio_trascrizione": testo_parlato,
                    "caption_ricerca": f"Visual: {tag_stringa}. Audio: {testo_parlato}"
                }

                self.db_en.coll_semantica.add(
                    ids=[id_univoco],
                    embeddings=[embedding_vettore],
                    metadatas=[metadati]
                )

                if embeddings_volti_frame:
                    self.db_en.aggiungi_batch_volti(
                        ids_volti_frame, embeddings_volti_frame, metadati_volti_frame)

                print(
                    f" -> [FRAME {frame_idx}] Scena a {timestamp_str} | Similarità: {similarieta_stamp:.2f} | YOLO: [{tag_stringa}] | Volti: {volti_rilevati}")

                # IMPORTANTE: Aggiorniamo il frame di riferimento SOLO quando salviamo un keyframe!
                prev_hist = hist

            frame_idx += 1

            cap.release()

            # --- STAMPA RESOCONTO FINALE ---
            print(f"\n=== RESOCONTO ELABORAZIONE {self.video_filename} ===")
            print(f" - Keyframe estratti e salvati nel DB: {frame_salvati}")
            if tutti_i_tag_trovati:
                print(
                    f" - Tag YOLO unici individuati: {', '.join(tutti_i_tag_trovati)}")
            else:
                print(" - Tag YOLO unici individuati: Nessuno oggetto rilevato")
            print("========================================================\n")
