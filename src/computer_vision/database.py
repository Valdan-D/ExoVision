"""
Modulo di gestione del database vettoriale ChromaDB (Versione Batch).

Questo file astrae l'interfaccia di ChromaDB, gestendo la persistenza su disco,
l'indicizzazione dei record ibridi e le operazioni di recupero (retrieval)
avanzate tramite combinazione di filtri geometrici e metadati relazionali.
"""

import chromadb


class ExoVisionDB:
    """
    Classe per la gestione e l'interfacciamento con il database vettoriale ChromaDB.

    Incapsula la logica di connessione persistente, inserimento multi-layer 
    e ricerca ibrida (vettoriale + filtraggio booleano) in modalità batch.
    """

    def __init__(self, path="./exovision_vector_db", collection_name="exovision_embeddings"):
        """
        Inizializza la connessione persistente a ChromaDB e recupera o crea la collezione.

        Args:
            path (str): Percorso locale in cui salvare i file del database vettoriale.
            collection_name (str): Nome della collezione per l'indicizzazione delle immagini.
        """
        print("\n[-] Connessione a ChromaDB locale...")
        self.client_chroma = chromadb.PersistentClient(path=path)

        # Uso della similarità coseno ottimale per SigLIP
        self.collezione = self.client_chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(
            f"[+] Collegato alla collezione: '{self.collezione.name}' | Record attuali: {self.collezione.count()}")

    def aggiungi_batch(self, lista_ids, lista_vettori, lista_metadatas):
        """
        Salva un intero pacchetto di record (batch) in un colpo solo dentro il database.

        Args:
            lista_ids (list[str]): Lista di identificativi univoci.
            lista_vettori (list[list[float]]): Lista di vettori generati da SigLIP.
            lista_metadatas (list[dict]): Lista di dizionari contenenti i metadati arricchiti (didascalie + tag YOLO).
        """
        self.collezione.add(
            ids=lista_ids,
            embeddings=lista_vettori,
            metadatas=lista_metadatas
        )

    def cerca_ibrido(self, vettore_query, tag_filtro=None, n_risultati=3):
        """
        Esegue una ricerca semantica avanzata combinando la vicinanza vettoriale con un filtro logico.
        Args:
            vettore_query (list): Il vettore del testo cercato.
            tag_filtro (str, optional): Tag di object detection da imporre come vincolo (es. "person").
            n_risultati (int): Numero massimo di corrispondenze da restituire.
        Returns:
            dict: Dizionario standard di ChromaDB contenente i risultati della query.
        """

        # Log di debug utilissimo:        print(f"Filtro applicato su ChromaDB: {condizione_filtro}")

        # RICEVE IL TAG "dog" E LO TRASFORMA NEL FLAG BOOLEANO CHE ABBIAMO INIETTATO
        # if tag_filtro and tag_filtro.lower().strip() == "dog":
        #     condizione_filtro = {"ha_cane": 1}
        # else:
        #     condizione_filtro = None

        # Se viene passato un filtro, generiamo dinamicamente la chiave da cercare
        if tag_filtro:
            tag_pulito = tag_filtro.lower().strip()
            condizione_filtro = {f"yolo_has_{tag_pulito}": 1}
        else:
            condizione_filtro = None

        print(
            f"[Ricerca DB] Filtro applicato su ChromaDB: {condizione_filtro}")

        return self.collezione.query(
            query_embeddings=[vettore_query],
            n_results=n_risultati,
            where=condizione_filtro
        )
