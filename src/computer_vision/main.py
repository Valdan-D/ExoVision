"""
Modulo principale della pipeline ExoVision.
Supporta l'ingestion ibrida sia dal dataset remoto (Flickr30k) 
sia da una cartella locale di immagini personalizzate, strutturato in funzioni modulari.
"""
# fmt: skip
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from PIL import Image
from database import ExoVisionDB
from models.embedding import SigLIPEmbedder
from models.obj_detection import YoloDetector
from computer_vision.models.face_rec import ExoFaceRecognizer
from models.video_processor import VideoProcessor
import numpy as np


# CONFIGURAZIONE SORGENTE DATI E PARAMETRI GLOBAL
USA_CARTELLA_LOCALE = True
CARTELLA_FOTO_TEST = "foto_test"
BATCH_SIZE = 32
LIMITE_IMMAGINI = 100
SOGLIA_SEMANTICA = 0.25
SOGLIA_CONFIDENZA_YOLO = 0.25


def carica_immagini_locali(cartella):
    """Carica le immagini da una cartella locale restituendo una lista strutturata."""
    lista_risultati = []
    estensioni_valide = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    if not os.path.exists(cartella):
        print(f"[!] La cartella '{cartella}' non esiste!")
        return lista_risultati

    for nome_file in os.listdir(cartella):
        if nome_file.lower().endswith(estensioni_valide):
            percorso_completo = f"{cartella}/{nome_file}"
            try:
                img_pil = Image.open(percorso_completo)
                img_pil.load()  # Forza il caricamento in memoria

                lista_risultati.append({
                    'image': img_pil,
                    'path': percorso_completo,
                    'caption': f"Immagine locale: {nome_file}"
                })
            except Exception as e:
                print(f"Errore nel caricamento del file {nome_file}: {e}")

    return lista_risultati


# ==========================================
# 1. CONNESSIONE E RESET DATABASE (PROTETTO)
# ==========================================
def connessione_db(reset=False):
    """
    Inizializza il database vettoriale ExoVisionDB e gestisce 
    lo svuotamento delle collezioni SOLO se reset=True.
    """
    db_en = ExoVisionDB()

    if reset:
        print("\n[-] Svuotamento e reset completo delle collezioni di ChromaDB...")
        # Reset Collezione Semantica
        try:
            record_semantica = db_en.coll_semantica.get()
            if record_semantica and record_semantica['ids']:
                db_en.coll_semantica.delete(ids=record_semantica['ids'])
                print("[+] Collezione Semantica resettata con successo!")
            else:
                print("[~] Collezione Semantica già vuota.")
        except Exception as e:
            print(
                f"[!] Errore durante il reset della collezione semantica: {e}")

        # Reset Collezione Volti
        try:
            record_volti = db_en.coll_volti.get()
            if record_volti and record_volti['ids']:
                db_en.coll_volti.delete(ids=record_volti['ids'])
                print("[+] Collezione Vettori Volti resettata con successo!")
            else:
                print("[~] Collezione Vettori Volti già vuota.")
        except Exception as e:
            print(f"[!] Errore durante il reset della collezione volti: {e}")
    else:
        print(
            "\n[+] Connessione a ChromaDB stabilita (Dati esistenti preservati con successo).")

    return db_en


# ==========================================
# 2. CARICAMENTO DATI NEL DB (INGESTION BATCH)
# ==========================================
def caricamento_dati_db(db_en, detector_en, embedder_en, face_en, dataset_pronto, limite_effettivo, vettori_classi):
    """Esegue la pipeline di inferenza e inserisce i dati in ChromaDB (Saltata in sola ricerca)."""
    print("\n[-] Inizio elaborazione e ingestion immagini...")
    print("-" * 60)

    for i in range(0, limite_effettivo, BATCH_SIZE):
        blocco_dataset = dataset_pronto[i: min(
            i + BATCH_SIZE, limite_effettivo)]
        batch_immagini = []
        batch_metadati = []
        batch_ids = []

        for idx, riga in enumerate(blocco_dataset):
            global_idx = i + idx
            batch_immagini.append(riga['image'])
            prefisso = "LOC" if USA_CARTELLA_LOCALE else "500"
            batch_ids.append(f"{prefisso}{global_idx}")
            batch_metadati.append({
                "caption_originale": str(riga['caption']),
                "is_deleted": False
            })

        try:
            tag_yolo_batch, _ = detector_en.estrai_tag_batch(batch_immagini)
            vettori_siglip_batch = embedder_en.istanzia_vettori_batch(
                batch_immagini)

            ids_volti = []
            embeddings_volti = []
            metadati_volti = []

            for idx in range(len(blocco_dataset)):
                percorso_foto_corrente = blocco_dataset[idx]['path']
                nome_file_corrente = os.path.basename(percorso_foto_corrente)

                vettore_immagine = vettori_siglip_batch[idx]
                img_norm = vettore_immagine / np.linalg.norm(vettore_immagine)

                lista_tag = [str(t).lower().strip() for t in tag_yolo_batch[idx]] if isinstance(
                    tag_yolo_batch[idx], list) else [str(tag_yolo_batch[idx]).lower().strip()]

                if "none" in lista_tag:
                    lista_tag.remove("none")

                if "person" in lista_tag:
                    vettori_facciali_trovati = face_en.estrai_vettore_volto(
                        percorso_foto_corrente)
                    for v_idx, vettore_volto in enumerate(vettori_facciali_trovati):
                        ids_volti.append(f"face_{nome_file_corrente}_{v_idx}")
                        embeddings_volti.append(vettore_volto)
                        metadati_volti.append(
                            {"parent_img_path": percorso_foto_corrente})

                tag_recuperati = []
                punteggi_classi = {}
                for classe, vettore_testo in vettori_classi.items():
                    text_norm = vettore_testo / np.linalg.norm(vettore_testo)
                    similarity = float(np.dot(img_norm, text_norm))
                    punteggi_classi[classe] = round(similarity, 4)

                    if not lista_tag and similarity > SOGLIA_SEMANTICA:
                        tag_recuperati.append(classe)

                lista_tag_finale = list(set(lista_tag + tag_recuperati))
                if not lista_tag_finale:
                    lista_tag_finale = ["none"]

                batch_metadati[idx]["yolo_tags"] = ", ".join(lista_tag_finale)
                batch_metadati[idx]["yolo_object_count"] = len(
                    lista_tag_finale) if lista_tag_finale != ["none"] else 0
                batch_metadati[idx]["identita_volti"] = "gestito_vettorialmente"

                for classe, score in punteggi_classi.items():
                    batch_metadati[idx][f"sim_{classe}"] = score

                for tag in lista_tag_finale:
                    if tag and tag != "none":
                        batch_metadati[idx][f"yolo_has_{tag}"] = 1

            db_en.aggiungi_batch_semantica(
                batch_ids, vettori_siglip_batch, batch_metadati)
            if embeddings_volti:
                db_en.aggiungi_batch_volti(
                    ids_volti, embeddings_volti, metadati_volti)

        except Exception as e:
            print(f"[!] Errore durante l'elaborazione del batch: {e}")

# ==========================================
# 2.5 CARICAMENTO SINGOLE FOTO AGGIUNTIVE
# ==========================================


def aggiungi_nuove_foto(percorsi_foto, db_en, detector_en, embedder_en, face_en, vettori_classi):
    """Estrae gli attributi e salva in ChromaDB una lista specifica di foto senza sovrascrivere i vecchi ID."""
    print("\n[-] Inizio elaborazione nuove foto specifiche...")
    from PIL import Image
    import time
    import os
    import numpy as np

    batch_immagini = []
    batch_metadati = []
    batch_ids = []
    foto_valide = []

    # 1. Caricamento immagini dal disco
    for percorso in percorsi_foto:
        if os.path.exists(percorso):
            try:
                img = Image.open(percorso).convert("RGB")
                batch_immagini.append(img)
                foto_valide.append(percorso)

                # Generiamo un ID univoco per evitare sovrascritture!
                nome_file = os.path.basename(percorso)
                timestamp = int(time.time())
                batch_ids.append(f"NEW_{timestamp}_{nome_file}")

                batch_metadati.append({
                    "caption_originale": percorso,
                    "is_deleted": False
                })
            except Exception as e:
                print(f"[!] Errore apertura {percorso}: {e}")
        else:
            print(f"[!] File non trovato: {percorso}")

    if not batch_immagini:
        print("[!] Nessuna foto valida trovata. Interruzione.")
        return

    # 2. Inferenza Multi-Modello
    try:
        print(
            f" -> Estrazione tag (YOLO) e vettori (SigLIP) per {len(batch_immagini)} foto...")
        tag_yolo_batch, _ = detector_en.estrai_tag_batch(batch_immagini)
        vettori_siglip_batch = embedder_en.istanzia_vettori_batch(
            batch_immagini)

        ids_volti = []
        embeddings_volti = []
        metadati_volti = []

        for idx, percorso in enumerate(foto_valide):
            vettore_immagine = vettori_siglip_batch[idx]
            img_norm = vettore_immagine / np.linalg.norm(vettore_immagine)
            nome_file_corrente = os.path.basename(percorso)

            lista_tag = [str(t).lower().strip() for t in tag_yolo_batch[idx]] if isinstance(
                tag_yolo_batch[idx], list) else [str(tag_yolo_batch[idx]).lower().strip()]

            if "none" in lista_tag:
                lista_tag.remove("none")

            # Estrazione Volti (se c'è una persona)
            if "person" in lista_tag:
                vettori_facciali_trovati = face_en.estrai_vettore_volto(
                    percorso)
                for v_idx, vettore_volto in enumerate(vettori_facciali_trovati):
                    ids_volti.append(
                        f"face_new_{nome_file_corrente}_{int(time.time())}_{v_idx}")
                    embeddings_volti.append(vettore_volto)
                    metadati_volti.append({"parent_img_path": percorso})

            # Controllo Semantico
            tag_recuperati = []
            punteggi_classi = {}
            for classe, vettore_testo in vettori_classi.items():
                text_norm = vettore_testo / np.linalg.norm(vettore_testo)
                similarity = float(np.dot(img_norm, text_norm))
                punteggi_classi[classe] = round(similarity, 4)

                # Se YOLO ha fallito ma SigLIP riconosce la classe
                if not lista_tag and similarity > 0.25:  # Usa la tua SOGLIA_SEMANTICA se l'hai dichiarata globale
                    tag_recuperati.append(classe)

            lista_tag_finale = list(set(lista_tag + tag_recuperati))
            if not lista_tag_finale:
                lista_tag_finale = ["none"]

            batch_metadati[idx]["yolo_tags"] = ", ".join(lista_tag_finale)
            batch_metadati[idx]["yolo_object_count"] = len(
                lista_tag_finale) if lista_tag_finale != ["none"] else 0
            batch_metadati[idx]["identita_volti"] = "gestito_vettorialmente"

            for classe, score in punteggi_classi.items():
                batch_metadati[idx][f"sim_{classe}"] = score
            for tag in lista_tag_finale:
                if tag and tag != "none":
                    batch_metadati[idx][f"yolo_has_{tag}"] = 1

        # 3. Scrittura in ChromaDB
        print(" -> Scrittura nel database in corso...")

        # Rendiamo la lista compatibile con ChromaDB a prescindere dal tipo (tensor/array)
        vettori_da_salvare = [v.tolist() if hasattr(
            v, 'tolist') else list(v) for v in vettori_siglip_batch]

        db_en.aggiungi_batch_semantica(
            batch_ids, vettori_da_salvare, batch_metadati)
        if embeddings_volti:
            db_en.aggiungi_batch_volti(
                ids_volti, embeddings_volti, metadati_volti)
            print(
                f" -> Salvati {len(embeddings_volti)} volti estratti dalle nuove foto.")

        print("[+] Nuove foto inserite con successo!")

    except Exception as e:
        print(f"[!] Errore durante l'inferenza o il salvataggio: {e}")


# ==========================================
# 3. INTERROGAZIONE: TAG E SEMANTICA
# ==========================================
def interrogazione_tag(db_en, embedder_en, query, tag_filtro=None):
    """
    Ricerca multimodale unificata: sfrutta la potenza visiva del database vettoriale
    e usa Python per validare incrociando i metadati testuali (Whisper e YOLO).
    """
    print(f"\n=================== RICERCA MULTIMODALE UNIFICATA ===================")
    print(f" -> Termine cercato: '{query}'")
    if tag_filtro:
        print(f" -> Filtro YOLO attivo:  '{tag_filtro}'")

    # 1. Interroghiamo il DB vettoriale per avere i 10 migliori match concettuali
    vettore_query = embedder_en.istanzia_vettore_testo(query)
    v_query = vettore_query.tolist() if hasattr(
        vettore_query, 'tolist') else vettore_query

    res = db_en.coll_semantica.query(
        query_embeddings=[v_query],
        n_results=10  # Ne peschiamo 10 per avere un buon margine
    )

    graduatoria = []

    # 2. Python analizza i metadati per creare la classifica perfetta
    if res and res['ids'] and len(res['ids'][0]) > 0:
        for i in range(len(res['ids'][0])):
            id_record = res['ids'][0][i]
            meta = res['metadatas'][0][i]
            dist = res['distances'][0][i]

            audio_text = meta.get('audio_trascrizione', '').lower()
            yolo_text = meta.get('yolo_tags', '').lower()

            # Verifichiamo se la parola è presente nel testo parlato
            match_audio = query.lower() in audio_text

            # Verifichiamo il filtro YOLO (se richiesto)
            if tag_filtro:
                if tag_filtro.lower() not in yolo_text:
                    continue  # Scarta il frame se non ha il tag YOLO richiesto

            # Assegniamo l'etichetta e la priorità (Combinato vince su tutto)
            if match_audio:
                tipo_match = "Combinato (Audio + Visivo)"
                priorita = 0
            else:
                tipo_match = "Visivo (SigLIP)"
                priorita = 1

            graduatoria.append({
                "id": id_record, "meta": meta, "dist": dist,
                "tipo": tipo_match, "priorita": priorita
            })

    # 3. Ordiniamo la classifica: prima i Combinati, poi i Visivi. A parità, vince la distanza minore.
    graduatoria = sorted(graduatoria, key=lambda x: (x['priorita'], x['dist']))
    graduatoria = graduatoria[:5]  # Teniamo solo i Top 5 assoluti

    # 4. Stampa finale
    print(
        f"\n--- GRADUATORIA FINALE DEI MATCH CORRISPONDENTI ({len(graduatoria)}) ---")
    if not graduatoria:
        print("[!] Nessun elemento risponde ai criteri di ricerca impostati.")
    else:
        for pos, dati in enumerate(graduatoria, 1):
            meta = dati['meta']
            sorgente = meta.get('caption_originale',
                                meta.get('video_sorgente', 'N/D'))
            timestamp_str = f" | Scena al secondo [{meta.get('timestamp')}]" if 'timestamp' in meta else ""

            print(f" Pos #{pos} | {dati['tipo']} | ID: {dati['id']}")
            print(f"    File Sorgente: {sorgente}{timestamp_str}")
            print(
                f"    Testo Audio:   '{meta.get('audio_trascrizione', 'Nessun parlato rilevato')}'")
            print(f"    Oggetti YOLO:  [{meta.get('yolo_tags', '')}]")
            print(f"    Distanza Vet:  {dati['dist']:.4f}")
            print("-" * 75)
    print("=====================================================================\n")


# ==========================================
# 4. INTERROGAZIONE: BIOMETRICA VOLTI
# ==========================================
def interrogazione_volti(db_en, face_en, percorso_foto_target, nome_persona="Target"):
    """
    Esegue la query geometrica biometrica sulla collezione dei volti.
    """
    print(
        f"\nINTERROGAZIONE BIOMETRICA (Ricerca volto di: '{nome_persona}')")
    print(f"[-] Analisi del file sorgente: {percorso_foto_target}")

    if not os.path.exists(percorso_foto_target):
        print(
            f"[!] Impossibile eseguire il test: File '{percorso_foto_target}' non trovato.")
        print("=" * 70)
        return

    try:
        vettori_trovati = face_en.estrai_vettore_volto(percorso_foto_target)

        if not vettori_trovati:
            print(
                f"[!] Nessun volto rilevato nella foto target '{percorso_foto_target}'.")
            print("=" * 70)
            return

        vettore_target = vettori_trovati[0]
        risultati_volti = db_en.cerca_volto_simile(
            vettore_target, n_risultati=5)

        print(
            f"--- Match Rilevati per {nome_persona} via Confronto Vettoriale ---")
        if risultati_volti and risultati_volti['metadatas'] and risultati_volti['metadatas'][0]:
            for i, metadata in enumerate(risultati_volti['metadatas'][0]):
                distanza = risultati_volti['distances'][0][i]
                nome_file_sorgente = os.path.basename(
                    metadata['parent_img_path'])

                if distanza <= 0.68:
                    print(
                        f"   [Face Match] '{nome_persona}' identificato/a in: {nome_file_sorgente} (Distanza: {distanza:.4f})")
                else:
                    print(
                        f"   [Debug] Volto simile ma scartato in: {nome_file_sorgente} (Distanza: {distanza:.4f})")
        else:
            print("[!] Nessun volto presente nel database per il confronto.")

    except Exception as e:
        print(f"[Errore durante la ricerca del volto]: {e}")

    print("=" * 70)


# ==========================================
# 5. ISPEZIONE DB PER NOME FOTO
# ==========================================
# def mostra_tag_foto(nome_foto="18.jpeg"):
#     try:
#         db_en = ExoVisionDB()
#         tutti_i_dati = db_en.coll_semantica.get(include=["metadatas"])

#         foto_trovata = False
#         if tutti_i_dati and tutti_i_dati['metadatas']:
#             for idx, metadati in enumerate(tutti_i_dati['metadatas']):
#                 caption = metadati.get('caption_originale', '')

#                 if nome_foto in caption:
#                     foto_trovata = True
#                     id_logico = tutti_i_dati['ids'][idx]
#                     tag_yolo = metadati.get('yolo_tags', 'Nessun tag rilevato')
#                     conteggio = metadati.get('yolo_object_count', 0)

#                     print(f"File: {nome_foto} (ID: {id_logico})")
#                     print(f"Tag YOLO: [{tag_yolo}]")
#                     print(f"Conteggio oggetti: {conteggio}")
#                     break

#         if not foto_trovata:
#             print(f"La foto '{nome_foto}' non è stata trovata nel database.")

#     except Exception as e:
#         print(f"Errore durante la lettura dei tag: {e}")


# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
def main():
    print("=" * 60)
    print("AVVIO FLUSSO EXOVISION")
    print("=" * 60)

    # ---------------------------------------------------------
    # INTERRUTTORI LOGICI PRINCIPALI (Imposta a True cosa vuoi eseguire)
    ESEGUI_INGESTION_FOTO = False
    ESEGUI_INGESTION_NUOVE_FOTO = False
    ESEGUI_RICERCA_TESTO = True
    ESEGUI_RICERCA_VOLTI = False
    ESEGUI_VIDEO_PIPELINE = False

    # query
    TESTO_RICERCA = "scoioattolo"

    # path per pescare foto/video
    NUOVE_FOTO_DA_AGGIUNGERE = [
        r"C:\Users\alice.amato\Documents\learning\git\exovision\foto_test\Friese.jpg",
        r"C:\Users\alice.amato\Documents\learning\git\exovision\foto_test\images.jpg"
    ]
    NOME_VIDEO_TEST = "YTDown.com_Shorts_Tour-di-una-TORINO-INSOLITA_Media_wMaxXl2nRKI_001_1080p60 (1).mp4"

    # ---------------------------------------------------------
    print("[~] Inizializzazione modelli IA in corso (attendere)...")
    embedder_en = SigLIPEmbedder()
    detector_en = YoloDetector()
    face_en = ExoFaceRecognizer()
    print("[+] Modelli caricati in RAM.")

    db_en = connessione_db(reset=ESEGUI_INGESTION_FOTO)

    # 1. INGESTION FOTO
    # Generiamo i vettori una volta sola per l'intero script, così sono pronti a prescindere
    CLASSI_YOLO = list(detector_en.model.names.values())
    vettori_classi = {classe: embedder_en.istanzia_vettore_testo(
        classe) for classe in CLASSI_YOLO}

    # 1. INGESTION FOTO BATCH
    if ESEGUI_INGESTION_FOTO:
        dataset_pronto = []
        if USA_CARTELLA_LOCALE:
            print(
                f"\n[-] Caricamento immagini locali da '{CARTELLA_FOTO_TEST}'...")
            dataset_pronto = carica_immagini_locali(CARTELLA_FOTO_TEST)
            limite_effettivo = len(dataset_pronto)
        else:
            print("\n[-] Caricamento dati da Hugging Face...")
            limite_effettivo = 0

        if limite_effettivo > 0:
            caricamento_dati_db(db_en, detector_en, embedder_en, face_en,
                                dataset_pronto, limite_effettivo, vettori_classi)

    # 1.2 INGESTION FOTO STATICHE
    if ESEGUI_INGESTION_NUOVE_FOTO:
        aggiungi_nuove_foto(NUOVE_FOTO_DA_AGGIUNGERE, db_en,
                            detector_en, embedder_en, face_en, vettori_classi)

    # 2. RICERCA IBRIDA
    if ESEGUI_RICERCA_TESTO:
        print("\n--- TEST RICERCA IBRIDA ---")
        interrogazione_tag(db_en, embedder_en,
                           query=TESTO_RICERCA, tag_filtro=None)

    # 3. RICERCA VOLTI NOTI
    if ESEGUI_RICERCA_VOLTI:
        print("\n--- TEST RICERCA VOLTI ---")
        interrogazione_volti(
            db_en, face_en, percorso_foto_target="known_faces/jim.jpg", nome_persona="jim carrey")

    # =========================================================
    # print("\n[-] Svuotamento dati video precedenti dal database...")
    # try:
    #     # 1. Elimina i frame video usando l'ID (cancella tutto ciò che inizia con VID_)
    #     # Nota: Usiamo un filtro sui metadati se non possiamo fare un reset completo,
    #     # oppure una cancellazione mirata. Il modo più sicuro in Chroma è filtrare per sorgente.
    #     db_en.coll_semantica.delete(
    #         where={
    #             "video_sorgente": "YTDown.com_Shorts_Tour-di-una-TORINO-INSOLITA_Media_wMaxXl2nRKI_001_1080p60 (1).mp4"}
    #     )
    #     print("[+] Rimossi i vecchi frame del video da 'galleria_semantica'.")

    #     # 2. Elimina i volti associati al video (filtrando per la presenza del metadato timestamp)
    #     if hasattr(db_en, 'coll_volti'):
    #         db_en.coll_volti.delete(
    #             where={"timestamp": {"$exists": True}}
    #         )
    #         print("[+] Rimossi i vecchi volti del video da 'vettori_volti'.")

    # except Exception as e:
    #     print(
    #         f"[!] Nota durante la pulizia: {e} (Se il database era già vuoto, ignora questo messaggio)")
    # print("-" * 60)
    # =========================================================

    # 4. PIPELINE MULTIMODALE VIDEO
    if ESEGUI_VIDEO_PIPELINE:
        print("\n--- PIPELINE MULTIMODALE VIDEO ---")
        try:
            processor = VideoProcessor(NOME_VIDEO_TEST,
                                       folder_video=r"C:\Users\alice.amato\Documents\learning\git\exovision\video_test"
                                       )

            # ORA USIAMO QUESTA! Passiamo i modelli reali.
            processor.elabora_video(
                yolo_model=detector_en,
                siglip_model=embedder_en,
                face_model=face_en,
                threshold=0.92
            )

        except FileNotFoundError as e:
            print(f"\n[ERRORE] {e}")
        except Exception as e:
            print(f"\n[ERRORE] Impossibile completare la pipeline video: {e}")

    print("\n=== PIPELINE EXOVISION TERMINATA ===")


if __name__ == "__main__":
    main()
