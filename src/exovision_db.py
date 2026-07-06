import sqlite3
import os

DB_NAME = "exovision.db"

def get_connection():
    """Ritorna una connessione attiva al database abilitando le chiavi esterne (Foreign Keys)."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def inizializza_banco():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. TABELLA FILES (Con Macchina a Stati nella colonna metadati_completi)
    # Stati: 0 = In coda, 1 = In elaborazione, 2 = Completato, 3 = Errore
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE NOT NULL,
        nome_file TEXT NOT NULL,
        tipo TEXT CHECK(tipo IN ('foto', 'video')) NOT NULL,
        estensione TEXT NOT NULL,
        dimensione_bytes INTEGER NOT NULL,
        data_modifica TEXT NOT NULL,
        data_indicizzazione TEXT NOT NULL,
        metadati_completi INTEGER DEFAULT 0 CHECK(metadati_completi IN (0, 1, 2, 3))
    )
    """)
    
    # 2. TABELLA METADATI_FOTO
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadati_foto (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        larghezza INTEGER,
        altezza INTEGER,
        modalita TEXT,
        data_scatto TEXT,
        camera_make TEXT,
        camera_model TEXT,
        iso INTEGER,
        apertura TEXT,
        otturatore TEXT,
        lunghezza_focale TEXT,
        gps_lat REAL,
        gps_lon REAL,
        extra_exif TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    # 3. TABELLA METADATI_VIDEO
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadati_video (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        durata_secondi REAL,
        larghezza INTEGER,
        altezza INTEGER,
        framerate REAL,
        codec_video TEXT,
        codec_audio TEXT,
        bitrate INTEGER,
        extra TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    # 4. TABELLA OCR
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ocr (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        testo TEXT,
        lingua TEXT,
        confidenza REAL,
        data_estrazione TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    # 5. TABELLA OGGETTI
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS oggetti (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        oggetto TEXT,
        confidenza REAL,
        bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
        data_estrazione TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    # 6. TABELLA FRAME (Soluzione Zero-Disk: path_frame rimosso/opzionale)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS frame (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        timestamp_sec REAL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    # 7. TABELLA TRASCRIZIONI
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trascrizioni (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        testo TEXT,
        lingua TEXT,
        confidenza REAL,
        data_estrazione TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    conn.close()
    print("🚀 Database SQLite di ExoVision configurato con successo con le nuove regole!")

# --- FUNZIONI AUSILIARI UTILI PER IL PRODUTTORE/CONSUMATORE (BATCHING) ---

def ottieni_file_in_coda(limit=32):
    """Recupera un lotto (batch) di file in attesa di elaborazione (Stato 0)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, path, tipo FROM files WHERE metadati_completi = 0 LIMIT ?", (limit,))
    record = cursor.fetchall()
    conn.close()
    return record

def aggiorna_stato_file(file_id, stato):
    """Aggiorna lo stato del file nella macchina a stati (0, 1, 2 o 3)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE files SET metadati_completi = ? WHERE id = ?", (stato, file_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    inizializza_banco()