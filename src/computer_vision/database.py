"""
Modulo di gestione del database vettoriale ChromaDB (Versione Batch a Doppia Collezione).

Questo file astrae l'interfaccia di ChromaDB, gestendo due collezioni distinte:
1. Galleria Semantica (SigLIP + Tag YOLO) per la ricerca concettuale e testuale.
2. Vettori Volti (ArcFace) per il riconoscimento biometrico istantaneo e dinamico.
"""

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False


class ExoVisionDB:
    """
    Classe per la gestione e l'interfacciamento con il database vettoriale ChromaDB.

    Gestisce l'inserimento multi-layer e la ricerca su due collezioni separate
    per garantire la massima flessibilità senza rieseguire l'ingestion.
    """

    def __init__(self, path="./exovision_vector_db"):
        """
        Inizializza la connessione a ChromaDB e istanzia le due collezioni separate.

        Args:
            path (str): Percorso locale in cui salvare i file del database vettoriale.
        """
        print("\n[-] Connessione a ChromaDB locale (Doppia Collezione)...")
        self.client_chroma = chromadb.PersistentClient(path=path)

        # 1. COLLEZIONE SEMANTICA (SigLIP)
        self.coll_semantica = self.client_chroma.get_or_create_collection(
            name="galleria_semantica",
            metadata={"hnsw:space": "cosine"}
        )

        # 2. COLLEZIONE VOLTI (ArcFace)
        self.coll_volti = self.client_chroma.get_or_create_collection(
            name="vettori_volti",
            # ArcFace lavora divinamente con la similarità coseno
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"[+] Collegato a 'galleria_semantica' | Record attuali: {self.coll_semantica.count()}")
        print(
            f"[+] Collegato a 'vettori_volti'      | Record attuali: {self.coll_volti.count()}")

    def aggiungi_batch_semantica(self, lista_ids, lista_vettori, lista_metadatas):
        """Salva i vettori globali SigLIP e i tag YOLO nella collezione semantica"""
        if not lista_ids:
            return
        self.coll_semantica.upsert(
            ids=lista_ids,
            embeddings=lista_vettori,
            metadatas=lista_metadatas
        )

    def aggiungi_batch_volti(self, lista_ids, lista_vettori, lista_metadatas):
        """Salva i vettori facciali estratti da RetinaFace/ArcFace nella collezione volti"""
        if not lista_ids:
            return
        self.coll_volti.upsert(
            ids=lista_ids,
            embeddings=lista_vettori,
            metadatas=lista_metadatas
        )

    def cerca_ibrido(self, vettore_query, tag_filtro=None, n_risultati=3):
        """
        Esegue una ricerca semantica combinando la vicinanza vettoriale con un filtro logico.
        (Cerca all'interno della collezione galleria_semantica)
        """
        if tag_filtro:
            tag_pulito = tag_filtro.lower().strip()
            condizione_filtro = {f"yolo_has_{tag_pulito}": 1}
        else:
            condizione_filtro = None

        print(
            f"[Ricerca DB Semantica] Filtro applicato su ChromaDB: {condizione_filtro}")

        return self.coll_semantica.query(
            query_embeddings=[vettore_query],
            n_results=n_risultati,
            where=condizione_filtro
        )

    def cerca_volto_simile(self, vettore_volto_query, n_risultati=5):
        """
        Interroga direttamente la collezione dei volti per trovare corrispondenze biometriche.
        Questo metodo rende istantanea la ricerca di NUOVE persone senza rifare l'ingestion.

        Args:
            vettore_volto_query (list): L'embedding ArcFace del volto da cercare.
            n_risultati (int): Numero di volti simili da restituire.
        """
        print(f"[Ricerca DB Volti] Confronte biometrico vettoriale in corso...")
        return self.coll_volti.query(
            query_embeddings=[vettore_volto_query],
            n_results=n_risultati
        )
