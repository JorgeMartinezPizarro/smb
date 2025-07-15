# vector_db/faq_ingester.py
import os
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

def create_index(
	faq_file_path="data/faq.txt",
	index_path="vector_db/faiss_index.bin",
	chunks_path="vector_db/chunks.pkl"
):
	if not os.path.exists(index_path) or not os.path.exists(chunks_path):
		print("Generando nuevo índice FAISS desde FAQ...")
		
		FAQ_FILE = Path(faq_file_path)
		if not FAQ_FILE.exists():
			raise FileNotFoundError(f"No se encontró el archivo FAQ en: {faq_file_path}")

		with FAQ_FILE.open("r", encoding="utf-8") as f:
			text = f.read()

		# Trocea en párrafos/chunks
		chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
		if not chunks:
			raise ValueError("El archivo FAQ está vacío o mal formateado.")

		model = SentenceTransformer("all-MiniLM-L6-v2")
		embeddings = model.encode(chunks, show_progress_bar=True)
		dimension = embeddings.shape[1]

		# Crea índice FAISS
		index = faiss.IndexFlatL2(dimension)
		index.add(embeddings)

		Path("vector_db").mkdir(exist_ok=True)

		faiss.write_index(index, index_path)
		with open(chunks_path, "wb") as f:
			pickle.dump(chunks, f)

		print(f"Ingesta completada: {len(chunks)} trozos indexados.")
	else:
		print("Índice y chunks ya existen. No se regeneran.")
