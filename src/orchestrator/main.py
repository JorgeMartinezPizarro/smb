from flask import Flask, request, jsonify
import sqlite3, requests, smtplib, os, time, re, logging
from retriever import FAQRetriever
from email.message import EmailMessage
import subprocess
import markdown
import wikipediaapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# --- CONFIGURACIÓN NUMÉRICA ---
WIKIPEDIA_CHUNK_SIZE = int(os.getenv("WIKIPEDIA_CHUNK_SIZE", "250"))        # palabras por chunk Wikipedia
WIKIPEDIA_OVERLAP = int(os.getenv("WIKIPEDIA_OVERLAP", "50"))             # solapamiento entre chunks
WIKIPEDIA_TOP_K = int(os.getenv("WIKIPEDIA_TOP_K", "100"))                 # nº de chunks a recuperar de Wikipedia
WIKIPEDIA_MAX_CHARS = int(os.getenv("WIKIPEDIA_MAX_CHARS", "30000"))       # máximo chars para prompt Wikipedia
WIKIPEDIA_READ_CHARS = int(os.getenv("WIKIPEDIA_READ_CHARS", "50000"))  # máximo chars a descargar Wikipedia

MAIL_HISTORY_SIZE = int(os.getenv("MAIL_HISTORY_SIZE", 8))
FAQ_TOP_K = int(os.getenv("FAQ_TOP_K", 6))

# --- RUTAS Y VARIABLES ---
INDEX_PATH = "vector_db/faiss_index.bin"
DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
GPT_SERVICE = os.getenv("GPT_SERVICE", "gpt-cpu")
GPT_URL = f"http://{GPT_SERVICE}:5000/gpt"
PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "xxx") + ".txt"
SMTP_SERVER = os.getenv("SMTP_SERVER")
BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_PASS = os.getenv("BOT_PASS")


def check_and_build_index():
	if not os.path.isfile(INDEX_PATH):
		print("[INFO] Índice FAISS no encontrado, generando nuevo índice...")
		subprocess.run(["python3", "faq_ingest.py"], check=True)
		print("[INFO] Índice generado correctamente.")
	else:
		print("[INFO] Índice FAISS ya existe, cargando directamente.")

# Logging básico
logging.basicConfig(level=logging.INFO)

# --- FUNCIONES DE UTILIDAD ---

def get_user_name(email):
	with sqlite3.connect(DB_PATH) as conn:
		cur = conn.cursor()
		cur.execute("SELECT value FROM data WHERE key = ?", (f"name_{email}",))
		row = cur.fetchone()
		return row[0] if row else None

def save_user_name(email, name):
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute("""
			INSERT INTO data (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value=excluded.value
		""", (f"name_{email}", name))
		conn.commit()

def get_data(key):
	with sqlite3.connect(DB_PATH) as conn:
		cur = conn.cursor()
		cur.execute("SELECT value FROM data WHERE key = ?", (key,))
		row = cur.fetchone()
		return row[0] if row else ""

def log_history(sender, question, response):
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute(
			"INSERT INTO history (timestamp, sender, question, response) VALUES (datetime('now'), ?, ?, ?)",
			(sender, question, response)
		)

def get_history_for_sender(sender, limit=3):
	with sqlite3.connect(DB_PATH) as conn:
		cur = conn.cursor()
		cur.execute("""
			SELECT question, response FROM history
			WHERE sender = ?
			ORDER BY timestamp DESC
			LIMIT ?
		""", (sender, limit))
		return cur.fetchall()

def ask_gpt_with_retry(prompt, retries=8, delay=2):
	messages = [
		{"role": "user", "content": prompt}
	]
	for i in range(retries):
		try:
			response = requests.post(GPT_URL, json={"messages": messages}, timeout=300)
			if response.status_code == 200:
				data = response.json()
				if "response" in data:
					return data["response"].strip()
				else:
					return response.text.strip()
			else:
				logging.warning(f"GPT respuesta no OK: status {response.status_code}, contenido: {response.text}")
		except requests.exceptions.RequestException as e:
			logging.warning(f"Intento {i+1} fallido, esperando {delay} segundos... Error: {e}")
			time.sleep(delay)
	raise Exception("No se pudo conectar con GPT después de varios intentos")

def send_email(to, subject, body):
	body = body.replace('\\n', '\n')
	msg = EmailMessage()
	msg["Subject"] = subject
	msg["From"] = BOT_EMAIL
	msg["To"] = to
	msg.set_content(body)

	with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
		smtp.login(BOT_EMAIL, BOT_PASS)
		smtp.send_message(msg)

# --- EXTRACCIÓN DE NOMBRE MEJORADA ---

def extract_name_from_message(message):
	patterns = [
		r"me llamo ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)",
		r"soy ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)",
		r"mi nombre es ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)"
	]
	message_lower = message.lower()
	for pat in patterns:
		m = re.search(pat, message_lower, re.IGNORECASE)
		if m:
			name = m.group(1).title()
			logging.info(f"Nombre extraído: {name}")
			return name
	logging.info("No se pudo extraer nombre del mensaje")
	return None

def chunk_text_with_overlap(text, max_words=WIKIPEDIA_CHUNK_SIZE, overlap=WIKIPEDIA_OVERLAP):
	words = text.split()
	chunks = []
	start = 0
	while start < len(words):
		end = min(start + max_words, len(words))
		chunk = words[start:end]
		chunks.append(" ".join(chunk))
		if end == len(words):
			break
		start += max_words - overlap
	return chunks

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
	return embeddings / (norms + 1e-10)

def get_wikipedia_page(subject, lang="es"):
	wiki_wiki = wikipediaapi.Wikipedia(
		user_agent='ideniox-bot/0.7 (admin@ideniox.com)',
		language=lang
	)
	page = wiki_wiki.page(subject)
	if not page.exists():
		# fallback a búsqueda en la API
		search_url = f"https://{lang}.wikipedia.org/w/api.php"
		params = {
			"action": "query",
			"list": "search",
			"srsearch": subject,
			"format": "json"
		}
		try:
			resp = requests.get(search_url, params=params)
			data = resp.json()
			search_results = data.get("query", {}).get("search", [])
			if search_results:
				first_title = search_results[0]["title"]
				page = wiki_wiki.page(first_title)
				if page.exists():
					return page
		except Exception as e:
			logging.warning(f"No se pudo buscar en Wikipedia: {e}")
			return None
		return None
	return page

def get_relevant_wikipedia_chunks(subject, body, lang="es", WIKIPEDIA_MAX_CHARS=WIKIPEDIA_MAX_CHARS, top_k=WIKIPEDIA_TOP_K):
	query = subject.strip() if subject and subject.strip() else body.strip()
	page = get_wikipedia_page(query, lang=lang)
	if not page or not page.exists():
		logging.info(f"No se encontró página Wikipedia para '{query}'")
		return f"No se encontró información relevante sobre '{query}' en Wikipedia."

	# Tomamos resumen + texto, limitados inicialmente
	full_text = (page.summary + "\n\n" + page.text)[:WIKIPEDIA_READ_CHARS]

	# Chunkificar con solapamiento
	chunks = chunk_text_with_overlap(full_text)
	if not chunks:
		return "No hay texto suficiente en Wikipedia para este tema."

	model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
	embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
	embeddings = normalize_embeddings(embeddings)

	dimension = embeddings.shape[1]
	index = faiss.IndexFlatIP(dimension)  # usar Inner Product con embeddings normalizados
	index.add(embeddings)

	query_vec = model.encode([query], convert_to_numpy=True)
	query_vec = normalize_embeddings(query_vec)

	distances, indices = index.search(query_vec, top_k)

	selected_chunks = []
	total_chars = 0
	for idx in indices[0]:
		if idx < len(chunks):
			chunk_text = chunks[idx]
			if total_chars + len(chunk_text) > WIKIPEDIA_MAX_CHARS:
				break
			selected_chunks.append(chunk_text)
			total_chars += len(chunk_text)

	if not selected_chunks:
		return "No se encontraron fragmentos relevantes en Wikipedia."

	return "\n\n".join(selected_chunks)

# Carga global del retriever FAQ, solo una vez
retriever = FAQRetriever(INDEX_PATH)

def process_email(sender, subject, body):
	logging.info(f"Procesando email de {sender} con asunto '{subject}'")
	name = get_user_name(sender)
	if not name:
		name = extract_name_from_message(body)
		if name:
			save_user_name(sender, name)
			logging.info(f"Guardado nombre '{name}' para {sender}")
	if not name:
		name = "cliente"  # fallback

	history_entries = get_history_for_sender(sender, limit=MAIL_HISTORY_SIZE)
	if history_entries:
		history_text = "\n".join(
			f"Cliente: {q}\nBot: {r}" for q, r in reversed(history_entries)
		)
	else:
		history_text = "Este es el primer mensaje del cliente."

	
	faq_chunks = retriever.query(body, top_k=FAQ_TOP_K)
	context_text = "\n".join(faq_chunks)
	if not context_text.strip():
		context_text = "No hay información adicional disponible del FAQ."

	# Obtén fragmentos relevantes directamente de Wikipedia
	if subject != "Duda":
		wikipedia_context = get_relevant_wikipedia_chunks(subject, body, lang="es", WIKIPEDIA_MAX_CHARS=WIKIPEDIA_READ_CHARS, top_k=WIKIPEDIA_TOP_K)
	else:
		wikipedia_context = ""

	# Carga la plantilla
	with open(PROMPT_FILE, encoding="utf-8") as f:
		template = f.read()

	prompt = template.replace("{greeting}", name) \
					 .replace("{sender}", sender) \
					 .replace("{history}", history_text) \
					 .replace("{message}", body) \
					 .replace("{context}", context_text) \
					 .replace("{wikipedia}", wikipedia_context)

	logging.info(f"Prompt generado para GPT ({len(prompt)} chars)\n\n{prompt}")

	try:
		answer = ask_gpt_with_retry(prompt)
		logging.info(f"Recibida respuesta de GPT ({len(answer)} chars)")
	except Exception as e:
		logging.error(f"Error llamando a GPT: {e}")
		answer = ("Lo siento, hemos tenido un problema técnico y no puedo responder "
				  "tu consulta ahora. Por favor, inténtalo más tarde.")

	log_history(sender, body, answer)

	try:
		send_email(sender, f"Re: {subject}", answer)
		logging.info(f"Email enviado a {sender}")
	except Exception as e:
		logging.error(f"Error enviando email a {sender}: {e}")

	return answer

# --- ENDPOINT FLASK ---

@app.route("/process_email", methods=["POST"])
def api_process_email():
	data = request.json
	sender = data.get("sender")
	subject = data.get("subject")
	body = data.get("body")

	if not (sender and subject and body):
		return jsonify({"error": "Faltan campos sender, subject o body"}), 400

	try:
		answer = process_email(sender, subject, body)
		return jsonify({"status": "ok", "answer": answer})
	except Exception as e:
		logging.error(f"Error en API /process_email: {e}")
		return jsonify({"error": str(e)}), 500

# --- MAIN ---

if __name__ == "__main__":
	check_and_build_index()
	app.run(host="0.0.0.0", port=5000)
