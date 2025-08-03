import os
import faiss
import pickle
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

FAQ_CHUNK_SIZE = int(os.getenv("FAQ_CHUNK_SIZE", "40"))

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)

def create_index(
    faq_file_path="assets/faq.txt",
    index_path="vector_db/faiss_index.bin",
    chunks_path="vector_db/chunks.pkl"
):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("[INFO] Generando nuevo índice FAISS desde FAQ...")

        FAQ_FILE = Path(faq_file_path)
        if not FAQ_FILE.exists():
            raise FileNotFoundError(f"No se encontró el archivo FAQ en: {faq_file_path}")

        with FAQ_FILE.open("r", encoding="utf-8") as f:
            text = f.read()

        # Divide por preguntas que empiezan con número y punto
        chunks = re.split(r'\n(?=\d+\.)', text)
        chunks = [chunk.replace('\n', ' ').strip() for chunk in chunks if len(chunk.strip()) > FAQ_CHUNK_SIZE]

        if not chunks:
            raise ValueError("[ERROR] El archivo FAQ está vacío o mal formateado.")

        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = model.encode(chunks, show_progress_bar=True)
        embeddings = normalize_embeddings(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Similitud coseno con embeddings normalizados
        index.add(embeddings)

        Path("vector_db").mkdir(exist_ok=True)

        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        print(f"[INFO] Ingesta completada: {len(chunks)} trozos indexados.")
    else:
        print("[INFO] Índice y chunks ya existen. No se regeneran.")

if __name__ == "__main__":
    create_index()
