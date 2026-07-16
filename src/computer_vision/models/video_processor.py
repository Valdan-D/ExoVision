import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import guardati con try/except: questo modulo è importato (per la sola
# scene detection, vedi rileva_scene_e_frame più sotto) anche da
# exovision_frames.py/app.py nel flusso principale, quindi non deve far
# crashare l'import se una di queste librerie manca (stesso pattern di
# ocr_pipeline.EASYOCR_OK). elabora_video()/_estrai_e_trascrivi_audio()
# restano invariati per l'uso standalone di computer_vision/main.py.
from computer_vision.database import ExoVisionDB

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except ImportError:
    MOVIEPY_OK = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_OK = True
except ImportError:
    FASTER_WHISPER_OK = False

try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_OK = True
except ImportError:
    SCENEDETECT_OK = False


def rileva_scene_e_frame(video_path: str, cartella_out, file_id: int, video_stem: str, soglia: float = 27.0) -> list:
    """
    Scene detection con PySceneDetect: estrae il primo frame di ogni scena
    rilevata e lo salva come <cartella_out>/fr-<file_id>-<n>-<video_stem>.jpg
    — stessa convenzione di naming di exovision_frames.py::estrai_keyframe(),
    così è un sostituto diretto (stesso formato di ritorno) usabile da
    exovision_frames.py senza toccare nessun chiamante a valle (tabella SQL
    `frame`, /api/preview/<id>, ecc.).

    Se PySceneDetect non rileva alcuna scena (video a piano fisso/continuo,
    comune nei video brevi), estrae un singolo frame fisso a 1s come
    anteprima di riserva — stessa garanzia del fallback storico in
    exovision_frames.py._estrai_frame_singolo, per non perdere la copertura
    di quel caso reale già risolto in passato.

    Restituisce lista di {"path_frame": str, "timestamp_sec": float} in
    ordine crescente di timestamp, oppure [] in caso di errore.
    """
    cartella_out = Path(cartella_out)
    prefisso = f"fr-{file_id}-"

    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        lista_scene = detect(video_path, ContentDetector(threshold=soglia))
        indici_frame = [scena[0].get_frames() for scena in lista_scene]

        if not indici_frame:
            # Nessun taglio di scena: forziamo almeno un frame a 1s (o il
            # primo disponibile su video più corti di 1s).
            indici_frame = [min(int(fps * 1.0), max(int(total_frames) - 1, 0))]

        risultati = []
        for n, f_idx in enumerate(indici_frame, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = cartella_out / f"{prefisso}{n}-{video_stem}.jpg"
            cv2.imwrite(str(out_path), frame)
            risultati.append({
                "path_frame": str(out_path),
                "timestamp_sec": round(f_idx / fps, 2),
            })

        return risultati
    except Exception as e:
        print(f"  ⚠️  Errore PySceneDetect su {video_path}: {e}")
        return []
    finally:
        cap.release()


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
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        prev_hist = None
        frame_idx = 0

        # Variabili per il resoconto finale
        frame_salvati = 0
        tutti_i_tag_trovati = set()

        print(f"\n[VIDEO] Avvio Scene Detection ({fps:.2f} FPS)...")

        print(f"\n[VIDEO] Analisi avanzata dei cambi scena con PySceneDetect...")
        from scenedetect import detect, ContentDetector

        # 1. Rileva automaticamente le scene reali nel video usando l'algoritmo Content
        lista_scene = detect(self.video_path, ContentDetector(threshold=27.0))

        # Prendiamo il frame iniziale di ogni scena rilevata
        frame_da_estrarre = []
        for i, scena in enumerate(lista_scene):
            # Ogni 'scena' contiene (tempo_inizio, tempo_fine) in formato FrameTimecode
            frame_inizio = scena[0].get_frames()
            frame_da_estrarre.append(frame_inizio)

        # Se non trova scene (improbabile), forziamo almeno il frame 0 e frame a metà
        if not frame_da_estrarre:
            frame_da_estrarre = [0, int(total_frames / 2)]

        print(
            f"[+] PySceneDetect ha individuato {len(frame_da_estrarre)} scene reali nel video!")

        # 2. Estrazione mirata dei frame identificati
        for f_idx in frame_da_estrarre:
            # Posizioniamo OpenCV esattamente sul frame della scena
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_salvati += 1
            secondi = f_idx / fps
            minuti = int(secondi // 60)
            sec = int(secondi % 60)
            timestamp_str = f"{minuti:02d}:{sec:02d}"

            # Salva il frame su disco
            filename = f"frame_{f_idx}_{minuti:02d}{sec:02d}.jpg"
            filepath = os.path.join(self.dir_keyframes, filename)
            cv2.imwrite(filepath, frame)

            # Interrogazione modelli IA
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
            embedding_vettore = vettore.tolist() if hasattr(
                vettore, "tolist") else list(vettore)

            # 3. Volti
            ids_volti_frame = []
            embeddings_volti_frame = []
            metadati_volti_frame = []

            if "person" in lista_tag:
                vettori_facciali = face_model.estrai_vettore_volto(filepath)
                for v_idx, vettore_volto in enumerate(vettori_facciali):
                    ids_volti_frame.append(f"face_{filename}_{v_idx}")
                    embeddings_volti_frame.append(vettore_volto)
                    metadati_volti_frame.append(
                        {"parent_img_path": filepath, "timestamp": timestamp_str})

            volti_rilevati = f"{len(embeddings_volti_frame)} volto/i" if embeddings_volti_frame else "Nessuno"

            # 4. Sincronizzazione AUDIO-VIDEO
            testo_parlato = self._trova_testo_per_timestamp(
                secondi, segmenti_audio)

            # Costruzione metadati complessi
            id_univoco = f"VID_{self.video_name}_FR_{f_idx}"
            tag_stringa = ", ".join(lista_tag)

            metadati = {
                "video_sorgente": self.video_filename,
                "timestamp": timestamp_str,
                "yolo_tags": tag_stringa,
                "volti_rilevati": volti_rilevati,
                "audio_trascrizione": testo_parlato,
                "caption_ricerca": f"Visual: {tag_stringa}. Audio: {testo_parlato}"
            }

            # Scrittura DB
            self.db_en.coll_semantica.add(
                ids=[id_univoco],
                embeddings=[embedding_vettore],
                metadatas=[metadati]
            )

            if embeddings_volti_frame:
                self.db_en.aggiungi_batch_volti(
                    ids_volti_frame, embeddings_volti_frame, metadati_volti_frame)

            print(
                f" -> [SCENA FRAME {f_idx}] Rilevata a {timestamp_str} | YOLO: [{tag_stringa}] | Audio sinc: '{testo_parlato}' | Volti: {volti_rilevati}")

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
