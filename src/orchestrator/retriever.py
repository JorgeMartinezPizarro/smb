# retriever.py
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from faq_ingest  import create_index

class FAQRetriever:
	def __init__(self, index_path="vector_db/faiss_index.bin", chunks_path="vector_db/chunks.pkl"):
		self.index_path = index_path
		self.chunks_path = chunks_path
		self.model = SentenceTransformer("all-MiniLM-L6-v2")

		# Si falta alguno, crea el índice
		if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
			print("Índice o chunks no encontrados. Generando automáticamente con faq_ingester.py...\n")
			create_index()

		# Carga el índice FAISS
		self.index = faiss.read_index(self.index_path)

		# Carga los textos originales
		with open(self.chunks_path, "rb") as f:
			self.chunks = pickle.load(f)
	@staticmethod
	def clean_text(text):
		return ' '.join(text.lower().strip().split())

	def query(self, query_text, top_k=5):
		query_vec = self.model.encode([self.clean_text(query_text)])
		distances, indices = self.index.search(query_vec, top_k)
		print("Distancias:", distances)
		print("Indices:", indices)
		results = []
		seen = set()
		for idx in indices[0]:
			chunk = self.chunks[idx]
			if chunk not in seen:
				results.append(chunk)
				seen.add(chunk)
		return results

if __name__ == "__main__":
	retriever = FAQRetriever()
	pregunta = "¿Cuál es el horario de atención al cliente?"
	respuestas = retriever.query(pregunta)
	for i, r in enumerate(respuestas, 1):
		print(f"{i}. {r}\n")
