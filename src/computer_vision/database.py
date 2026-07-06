"""
Modulo di gestione del database vettoriale ChromaDB.

Astrae la persistenza degli embedding SigLIP su disco, con collezioni separate
per foto e frame video, e la ricerca ibrida (similarità vettoriale + filtro
sui tag di object detection rilevati da YOLO).
"""

from pathlib import Path

import chromadb

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"

COLLEZIONI = {
    "foto": "foto_embeddings",
    "video": "video_embeddings",
}


class ExoVisionDB:
    """
    Gestisce la connessione persistente a ChromaDB e le operazioni di
    indicizzazione e ricerca ibrida in modalità batch.
    """

    def __init__(self, path=None):
        """
        Inizializza la connessione persistente a ChromaDB e recupera o crea
        le collezioni per foto e video.

        Args:
            path (str, optional): Percorso locale dei file del database
                vettoriale. Default: <root progetto>/data/chroma_db.
        """
        path = path or str(_DATA_PATH)
        print("\n[-] Connessione a ChromaDB locale...")
        self.client_chroma = chromadb.PersistentClient(path=path)

        # Uso della similarità coseno ottimale per SigLIP
        self.collezioni = {
            tipo: self.client_chroma.get_or_create_collection(
                name=nome, metadata={"hnsw:space": "cosine"}
            )
            for tipo, nome in COLLEZIONI.items()
        }
        for tipo, collezione in self.collezioni.items():
            print(f"[+] Collegato alla collezione '{collezione.name}' ({tipo}) | Record attuali: {collezione.count()}")

    def aggiungi_batch(self, tipo, lista_ids, lista_vettori, lista_metadatas):
        """
        Salva un intero pacchetto di record (batch) in un colpo solo dentro
        la collezione indicata.

        Args:
            tipo (str): "foto" o "video" — determina la collezione di destinazione.
            lista_ids (list[str]): Lista di identificativi univoci.
            lista_vettori (list[list[float]]): Lista di vettori generati da SigLIP.
            lista_metadatas (list[dict]): Lista di dizionari coi metadati arricchiti (didascalie + tag YOLO).
        """
        self.collezioni[tipo].add(
            ids=lista_ids,
            embeddings=lista_vettori,
            metadatas=lista_metadatas
        )

    def cerca_ibrido(self, tipo, vettore_query, tag_filtro=None, n_risultati=3):
        """
        Esegue una ricerca semantica combinando la vicinanza vettoriale con un filtro logico.

        Args:
            tipo (str): "foto" o "video" — collezione su cui cercare.
            vettore_query (list): Il vettore del testo cercato.
            tag_filtro (str, optional): Tag di object detection da imporre come vincolo (es. "person").
            n_risultati (int): Numero massimo di corrispondenze da restituire.
        Returns:
            dict: Dizionario standard di ChromaDB contenente i risultati della query.
        """
        condizione_filtro = None
        if tag_filtro:
            tag_pulito = tag_filtro.lower().strip()
            condizione_filtro = {f"yolo_has_{tag_pulito}": 1}

        print(f"[Ricerca DB] Filtro applicato su ChromaDB ({tipo}): {condizione_filtro}")

        return self.collezioni[tipo].query(
            query_embeddings=[vettore_query],
            n_results=n_risultati,
            where=condizione_filtro
        )
