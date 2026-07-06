import chromadb
from chromadb.config import Settings
import os

# Definiamo la cartella in cui ChromaDB salverà i suoi dati vettoriali
CHROMA_DATA_PATH = os.path.join("data", "chroma_db")

def ottieni_client_chroma():
    """Inizializza e restituisce il client ChromaDB persistente su disco."""
    os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def inizializza_collezioni():
    client = ottieni_client_chroma()
    
    # Creiamo o recuperiamo la collezione per le foto
    collezione_foto = client.get_or_create_collection(
        name="foto_embeddings",
        metadata={"hnsw:space": "cosine"} # Usa la distanza coseno per il confronto semantico
    )
    
    # Creiamo o recuperiamo la collezione per i frame video (Soluzione Zero-Disk)
    collezione_video = client.get_or_create_collection(
        name="video_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    
    print("🤖 Collezioni ChromaDB (foto e video) pronte per la ricerca semantica!")
    return collezione_foto, collezione_video

def aggiungi_vettore(collezione_name, documento_id, vettore, metadati):
    """
    Aggiunge un vettore a ChromaDB.
    documento_id corrisponderà a files.id (per foto) o frame.id (per video) di SQLite.
    """
    client = ottieni_client_chroma()
    collezione = client.get_collection(name=collezione_name)
    
    collezione.add(
        ids=[str(documento_id)],
        embeddings=[vettore],
        metadatas=[metadati]
    )
    print(f"✅ Vettore salvato in ChromaDB per ID: {documento_id}")

if __name__ == "__main__":
    inizializza_collezioni()