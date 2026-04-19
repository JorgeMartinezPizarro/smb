from flask import Flask, request, jsonify
from email.header import decode_header
import sqlite3, requests, smtplib, os, time, re, logging
from email.message import EmailMessage
import subprocess
import markdown
import wikipediaapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import unicodedata
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

## TODO: separate wiki-rag, email-rag to separated files.

# Initialize models
model_embed = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

app = Flask(__name__)

CACHE_DIR = "/app/cache/wikipedia"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Configuration ---
# Wikipedia settings
WIKIPEDIA_CHUNK_SIZE = int(os.getenv("WIKIPEDIA_CHUNK_SIZE", "1000"))      # Max chars per content chunk
WIKIPEDIA_TOP_K = int(os.getenv("WIKIPEDIA_TOP_K", "5"))                   # Number of chunks to retrieve
WIKIPEDIA_MAX_CONTENT = int(os.getenv("WIKIPEDIA_MAX_CONTENT", "12000"))   # Max total chars for prompt
WIKIPEDIA_LANG = os.getenv("WIKIPEDIA_LANG", "en")                         # Default language
WIKIPEDIA_MIN_SECTION = int(os.getenv("WIKIPEDIA_MIN_SECTION", "200"))     # Min section length to consider
WIKIPEDIA_BATCH_SIZE = int(os.getenv("WIKIPEDIA_BATCH_SIZE", "32"))        # Embedding batch size
WIKIPEDIA_THRESHOLD = float(os.getenv("WIKIPEDIA_THRESHOLD", "0.5"))       # Threshold
WIKIPEDIA_CACHE_TTL = int(os.getenv("WIKIPEDIA_CACHE_TTL", "86400"))       # Cache TTL in seconds (default: 1 day)
WIKIPEDIA_FALLBACK_LANGS = os.getenv("WIKIPEDIA_FALLBACK_LANGS", "es,en")  # Comma-separated fallback langs
WIKIPEDIA_MAX_SEARCH_RESULTS = int(os.getenv("WIKIPEDIA_MAX_SEARCH_RESULTS", "5"))  # Max search candidates
TIMEOUT = int(os.getenv("TIMEOUT", "600"))     # GPT Request timeout

# Email settings
MAIL_HISTORY_SIZE = int(os.getenv("MAIL_HISTORY_SIZE", "3"))
SMTP_SERVER = os.getenv("SMTP_SERVER")
BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_PASS = os.getenv("BOT_PASS")

# Paths
INDEX_PATH = "vector_db/faiss_index.bin"
DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
GPT_SERVICE = os.getenv("GPT_SERVICE", "gpt-cpu")
GPT_URL = f"http://{GPT_SERVICE}:5000/gpt"
PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "default") + ".txt"
FOOTER_FILE = "assets/" + os.environ.get("FOOTER_FILE", "default") + ".txt"
LLM_NAME = os.getenv("LLM_NAME", "unknown")
logging.basicConfig(level=logging.INFO)

# --- Utility Functions ---

def get_user_name(email):
	"""Get stored user name from database"""
	with sqlite3.connect(DB_PATH) as conn:
		cur = conn.cursor()
		cur.execute("SELECT value FROM data WHERE key = ?", (f"name_{email}",))
		row = cur.fetchone()
		return row[0] if row else None

def save_user_name(email, name):
	"""Save user name to database"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute("""
			INSERT INTO data (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value=excluded.value
		""", (f"name_{email}", name))
		conn.commit()

def log_history(sender, question, response):
	"""Log conversation history"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute(
			"INSERT INTO history (timestamp, sender, question, response) VALUES (datetime('now'), ?, ?, ?)",
			(sender, question, response)
		)

def get_history_for_sender(sender, limit=MAIL_HISTORY_SIZE):
	"""Retrieve conversation history for sender"""
	with sqlite3.connect(DB_PATH) as conn:
		cur = conn.cursor()
		cur.execute("""
			SELECT question, response FROM history
			WHERE sender = ?
			ORDER BY timestamp DESC
			LIMIT ?
		""", (sender, limit))
		return cur.fetchall()

def ask_gpt_with_retry(prompt, retries=3, delay=3):
	"""Query GPT service with retry logic"""
	messages = [{"role": "user", "content": prompt}]
	for i in range(retries):
		try:
			response = requests.post(GPT_URL, json={"messages": messages}, timeout=TIMEOUT)
			if response.status_code == 200:
				return response.json().get("response", "").strip()
			logging.warning(f"GPT bad response: {response.status_code}")
		except requests.exceptions.RequestException as e:
			logging.warning(f"Attempt {i+1} failed: {e}")
			time.sleep(delay)
	raise Exception("Failed to connect to GPT after multiple attempts")

def send_email(to, subject, body_md, duration):
	"""
	Envía email en HTML con diseño compacto y estilo verde
	"""
	import re
	from email.message import EmailMessage
	import smtplib
	from html import escape
	import markdown

	# TODO: generalizar esto para otros modelos, es hardcodeo para GPT-OSS-20b
	# Detectar bloque analysis y bloque final
	pattern = re.compile(
		r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>(.*)",
		re.DOTALL
	)

	match = pattern.search(body_md)
	if match:
		analysis_content = match.group(1).strip()
		final_content = match.group(2).strip()
	else:
		analysis_content = ""
		final_content = body_md.strip()

	# Convertir markdown a HTML
	def convert_markdown_to_html(md_text):
		html = markdown.markdown(md_text, extensions=['fenced_code', 'nl2br'])
		return html

	final_html = convert_markdown_to_html(final_content)
	analysis_html = convert_markdown_to_html(analysis_content) if analysis_content else ""

	with open(FOOTER_FILE, encoding="utf-8") as f:
		footer = f.read()

	now = datetime.now()

	formatedTime = now.strftime("%H:%M:%S, %d/%m/%Y")

	footer = footer.replace("{time}", formatedTime)
	footer = footer.replace("{duration}", duration)
	footer = footer.replace("{model}", LLM_NAME)
	
	# TODO: MOVE html template to editable assets.
	# HTML compacto con estilo verde
	html_content = f"""
	<!DOCTYPE html>
	<html>
	<head>
	<meta charset="UTF-8">
	<style>
		body {{
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			line-height: 1.4;
			color: #333;
			max-width: 500px;
			margin: 0 auto;
			padding: 10px;
			background-color: #f5f7f9;
			font-size: 13px;
		}}
		.email-container {{
			border: 1px solid #d1d8e0;
			border-radius: 8px;
			padding: 15px;
			background-color: #ffffff;
			box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
		}}
		.card {{
			margin-bottom: 12px;
			border: 1px solid #d1d8e0;
			border-radius: 6px;
			background-color: #f8f9fa;
			overflow: hidden;
		}}
		.card-header {{
			padding: 8px 10px;
			font-weight: 600;
			background-color: #EAEAEA;
			color: #247524;
			font-size: 12px;
		}}
		.card-content {{
			padding: 12px;
			background-color: #ffffff;
		}}
		.email-title {{
			color: #1e293b;
			font-size: 15px;
			margin-bottom: 12px;
			text-align: center;
			font-weight: 600;
		}}
		.footer {{
			margin-top: 15px;
			padding-top: 10px;
			border-top: 1px solid #e2e8f0;
			font-size: 12px;
			color: #64748b;
		}}
		pre {{
			background-color: #f1f5f9;
			padding: 8px;
			border-radius: 4px;
			overflow-x: auto;
			border: 1px solid #e2e8f0;
			font-size: 11px;
			margin: 5px 0;
		}}
		code {{
			font-family: 'Consolas', 'Monaco', monospace;
			font-size: 11px;
		}}
	</style>
	</head>
	<body>
		<div class="email-container">
			<div class="card">
				<div class="card-header">
					💬 Respuesta
				</div>
				<div class="card-content">
					{final_html}
				</div>
			</div>
	"""

	html_content += f"""
			<div class="card">
				<div class="card-header">
					🧑🏼 Redactor
				</div>
				<div class="card-content">
					{footer}
				</div>
			</div>
		
	"""

	# Añadir análisis scrolleable compacto
	if analysis_html:
		html_content += f"""
				<!-- Bloque de Análisis -->
				<div class="card">
					<div class="card-header">
						🧠 Razonamiento
					</div>
					<div class="card-content" style="max-height: 112px; overflow-y: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; line-height: 1.3;">
						{analysis_html}
					</div>
				</div>
			</div>
			</body>
			</html>
		"""
	else:
		html_content += f"""
					</div>
				</body>
			</html>

		"""
	# Construir y enviar email
	msg = EmailMessage()
	msg["Subject"] = subject
	msg["From"] = BOT_EMAIL
	msg["To"] = to
	msg.add_alternative(html_content, subtype='html')

	with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
		smtp.login(BOT_EMAIL, BOT_PASS)
		smtp.send_message(msg)
				
def extract_name_from_message(message):
	"""Extract name from message text using patterns"""
	patterns = [
		r"me llamo ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)",
		r"soy ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)",
		r"mi nombre es ([a-záéíóúñ]+(?: [a-záéíóúñ]+)*)"
	]
	for pat in patterns:
		m = re.search(pat, message.lower(), re.IGNORECASE)
		if m:
			return m.group(1).title()
	return None

# --- Wikipedia Processing ---

def normalize_subject(subject):
	"""
	Normalize a subject string for more reliable Wikipedia lookups.
	Strips common filler words, normalizes unicode, and title-cases.
	"""
	# Normalize unicode (e.g., accents)
	subject = unicodedata.normalize("NFKC", subject)
	# Remove common email subject prefixes
	subject = re.sub(r"^(re|fwd|fw|asunto|pregunta|duda sobre|consulta sobre)[:\s]+", "", subject, flags=re.IGNORECASE)
	# Remove excessive punctuation and whitespace
	subject = re.sub(r"[^\w\s\-áéíóúñüÁÉÍÓÚÑÜ]", " ", subject)
	subject = re.sub(r"\s+", " ", subject).strip()
	return subject.title()


def is_cache_valid(cache_file):
	"""Check if cache file exists and is within TTL."""
	if not os.path.exists(cache_file):
		return False
	age = time.time() - os.path.getmtime(cache_file)
	return age < WIKIPEDIA_CACHE_TTL


def search_wikipedia_candidates(subject, lang, max_results=WIKIPEDIA_MAX_SEARCH_RESULTS):
	"""
	Search Wikipedia for candidate page titles.
	Returns a list of (title, snippet) tuples ranked by search relevance.
	"""
	search_url = f"https://{lang}.wikipedia.org/w/api.php"
	params = {
		"action": "query",
		"list": "search",
		"srsearch": subject,
		"srlimit": max_results,
		"srprop": "snippet|titlesnippet",
		"format": "json"
	}
	try:
		resp = requests.get(search_url, params=params, timeout=10)
		if resp.status_code == 200:
			results = resp.json().get("query", {}).get("search", [])
			return [(r["title"], r.get("snippet", "")) for r in results]
	except Exception as e:
		logging.warning(f"Wikipedia candidate search failed for '{subject}' in '{lang}': {e}")
	return []


def get_wikipedia_page(subject, lang=WIKIPEDIA_LANG):
	"""
	Retrieve the best Wikipedia page for a subject.

	Strategy:
	  1. Direct lookup by exact title.
	  2. API search for up to WIKIPEDIA_MAX_SEARCH_RESULTS candidates; pick the
	     one whose title or snippet best matches the subject (simple token overlap).
	  3. If configured, fall back to alternative languages.
	"""
	wiki_clients = {}

	def get_wiki_client(l):
		if l not in wiki_clients:
			wiki_clients[l] = wikipediaapi.Wikipedia(
				user_agent="knowledge-bot/1.0",
				language=l,
				extract_format=wikipediaapi.ExtractFormat.WIKI,
			)
		return wiki_clients[l]

	def score_candidate(title, snippet, query):
		"""Simple token-overlap score between query and title+snippet."""
		query_tokens = set(re.findall(r"\w+", query.lower()))
		target_tokens = set(re.findall(r"\w+", (title + " " + snippet).lower()))
		if not query_tokens:
			return 0.0
		return len(query_tokens & target_tokens) / len(query_tokens)

	langs_to_try = [lang] + [
		l.strip() for l in WIKIPEDIA_FALLBACK_LANGS.split(",")
		if l.strip() and l.strip() != lang
	]

	for current_lang in langs_to_try:
		wiki = get_wiki_client(current_lang)

		# 1. Direct lookup
		page = wiki.page(subject)
		if page.exists():
			logging.info(f"Wikipedia direct hit: '{subject}' in '{current_lang}'")
			return page

		# 2. Search-based lookup
		candidates = search_wikipedia_candidates(subject, current_lang)
		if candidates:
			scored = sorted(
				candidates,
				key=lambda c: score_candidate(c[0], c[1], subject),
				reverse=True,
			)
			logging.info(
				f"Wikipedia candidates for '{subject}' in '{current_lang}': "
				+ ", ".join(t for t, _ in scored[:3])
			)
			for title, _ in scored:
				page = wiki.page(title)
				if page.exists():
					logging.info(f"Wikipedia search hit: '{title}' in '{current_lang}'")
					return page

	logging.warning(f"No Wikipedia page found for '{subject}' in any language.")
	return None


def get_cached_wikipedia(subject, lang=WIKIPEDIA_LANG):
	"""
	Return (chunks, embeddings) for a subject, using disk cache when valid.
	Cache is invalidated after WIKIPEDIA_CACHE_TTL seconds.
	"""
	safe_subject = re.sub(r"[^\w\-]", "_", subject)
	cache_file = os.path.join(CACHE_DIR, f"{lang}_{safe_subject}.npz")

	if is_cache_valid(cache_file):
		try:
			data = np.load(cache_file, allow_pickle=True)
			chunks = data["chunks"].tolist()
			embeddings = data["embeddings"]
			logging.info(f"Wikipedia cache hit: '{subject}' ({len(chunks)} chunks)")
			return chunks, embeddings
		except Exception as e:
			logging.warning(f"Cache load failed for '{subject}': {e}. Re-fetching.")

	page = get_wikipedia_page(subject, lang)
	if not page or not page.exists():
		return [], np.array([])

	sections = extract_relevant_sections(page)

	chunks = []
	for title, text in sections:
		if len(text) > WIKIPEDIA_CHUNK_SIZE:
			for chunk in chunk_text(text, WIKIPEDIA_CHUNK_SIZE):
				chunks.append(f"{title}\n{chunk}")
		else:
			chunks.append(f"{title}\n{text}")

	if not chunks:
		return [], np.array([])

	embeddings = []
	for i in range(0, len(chunks), WIKIPEDIA_BATCH_SIZE):
		batch = chunks[i:i + WIKIPEDIA_BATCH_SIZE]
		emb = model_embed.encode(batch, convert_to_numpy=True)
		emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
		embeddings.append(emb)
	embeddings = np.vstack(embeddings)

	try:
		np.savez_compressed(cache_file, chunks=np.array(chunks), embeddings=embeddings)
	except Exception as e:
		logging.warning(f"Cache save failed for '{subject}': {e}")

	return chunks, embeddings


def clean_wikipedia_text(text):
	"""Normalize and clean Wikipedia text"""
	if not text:
		return ""
	text = unicodedata.normalize('NFKC', text)
	text = ''.join(ch for ch in text if (ch == '\n' or unicodedata.category(ch)[0] != 'C'))
	text = re.sub(r'[\t\r]+', ' ', text)
	text = re.sub(r' *\n+ *', '\n\n', text)
	return re.sub(r'[ ]{2,}', ' ', text).strip()


def chunk_text(text, max_chars=WIKIPEDIA_CHUNK_SIZE, overlap=None):
	"""
	Split text into semantically coherent chunks with overlap.
	Overlap defaults to 20% of max_chars.
	"""
	if overlap is None:
		overlap = max(0, max_chars // 5)

	paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
	chunks = []
	current_chunk = ""

	for p in paragraphs:
		if len(current_chunk) + len(p) + 2 <= max_chars:
			current_chunk += p + "\n\n"
		else:
			if current_chunk:
				chunks.append(current_chunk.strip())
				overlap_text = current_chunk[-overlap:] if overlap else ""
				current_chunk = overlap_text + p + "\n\n"
			else:
				# Paragraph too large on its own: hard-split by sentences
				sentences = re.split(r"(?<=[.!?])\s+", p)
				sentence_buf = ""
				for sent in sentences:
					if len(sentence_buf) + len(sent) + 1 <= max_chars:
						sentence_buf += sent + " "
					else:
						if sentence_buf:
							chunks.append(sentence_buf.strip())
						sentence_buf = sent + " "
				if sentence_buf:
					current_chunk = sentence_buf

	if current_chunk:
		chunks.append(current_chunk.strip())

	return [c for c in chunks if c]


def extract_relevant_sections(page):
	"""Extract all sections from a Wikipedia page."""
	sections = []

	# Include the page summary (top-level text) if substantial
	summary = clean_wikipedia_text(page.text)
	# page.text for wikipedia-api is the full page text; extract only the intro
	# by taking text up to the first section boundary (double newline after a sentence).
	intro_match = re.match(r"((?:.|\n){50,800}?)(?=\n\n[A-ZÁÉÍÓÚ])", summary)
	if intro_match:
		intro = intro_match.group(1).strip()
		if len(intro) >= WIKIPEDIA_MIN_SECTION:
			sections.append(("Introducción", intro))

	def process_section(section, parent_title=""):
		full_title = f"{parent_title} > {section.title}" if parent_title else section.title
		text = clean_wikipedia_text(section.text)
		if text and len(text) >= WIKIPEDIA_MIN_SECTION:
			sections.append((full_title, text))
		for subsection in section.sections:
			process_section(subsection, full_title)

	for section in page.sections:
		process_section(section)

	return sections


def filter_redundant_chunks(chunks, threshold=0.85):
	"""Remove near-duplicate chunks using cosine similarity."""
	if not chunks:
		return []
	embeddings = model_embed.encode(chunks, convert_to_numpy=True)
	filtered_chunks = []
	filtered_embeds = []

	for i, emb in enumerate(embeddings):
		if not filtered_embeds:
			filtered_chunks.append(chunks[i])
			filtered_embeds.append(emb)
			continue
		sims = cosine_similarity([emb], filtered_embeds)[0]
		if max(sims) < threshold:
			filtered_chunks.append(chunks[i])
			filtered_embeds.append(emb)

	return filtered_chunks


def expand_query(subject, body):
	"""
	Generate multiple query variants to improve retrieval recall.
	Combines the subject and key phrases from the body.
	"""
	queries = [body]

	# Add subject as an additional query signal
	if subject and subject.lower() not in body.lower():
		queries.append(subject)

	# Add a combined query
	if subject:
		queries.append(f"{subject}: {body[:200]}")

	# Extract potential named entities / key noun phrases (simple heuristic)
	key_phrases = re.findall(r"\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*\b", body)
	if key_phrases:
		queries.append(" ".join(key_phrases[:5]))

	return list(dict.fromkeys(queries))  # deduplicate preserving order


def get_wikipedia_context(subject, query, lang=WIKIPEDIA_LANG):
	"""
	Retrieve the most relevant Wikipedia chunks for a subject+query pair.

	Improvements over the original:
	  - Subject normalization before lookup.
	  - Multi-query expansion for better recall.
	  - Aggregated FAISS scores across query variants.
	  - Cross-encoder reranking on the merged candidate pool.
	  - Redundancy filtering before final selection.
	  - Graceful fallback with informative messages.
	"""
	normalized_subject = normalize_subject(subject)
	chunks, embeddings = get_cached_wikipedia(normalized_subject, lang)

	# If normalized subject fails, try raw subject as fallback
	if not chunks and normalized_subject != subject:
		logging.info(f"Retrying Wikipedia fetch with raw subject: '{subject}'")
		chunks, embeddings = get_cached_wikipedia(subject, lang)

	if not chunks:
		return f"No se encontró información en Wikipedia sobre '{subject}'."

	logging.info(f"Wikipedia context: {len(chunks)} chunks available for '{normalized_subject}'")

	# Build FAISS index
	index = faiss.IndexFlatIP(embeddings.shape[1])
	index.add(embeddings)

	# Multi-query expansion
	query_variants = expand_query(normalized_subject, query)
	logging.info(f"Query variants: {query_variants}")

	# Aggregate scores across all query variants
	k = min(WIKIPEDIA_TOP_K * 4, len(chunks))
	chunk_scores = {}  # chunk_idx -> max score across variants

	for variant in query_variants:
		q_emb = model_embed.encode([variant], convert_to_numpy=True)[0]
		q_emb /= (np.linalg.norm(q_emb) + 1e-10)
		distances, indices = index.search(q_emb.reshape(1, -1), k)
		for dist, idx in zip(distances[0], indices[0]):
			# Keep the best (max) score for each chunk across variants
			chunk_scores[idx] = max(chunk_scores.get(idx, -1.0), float(dist))

	# Sort candidates by aggregated score
	sorted_candidates = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
	top_indices = [idx for idx, _ in sorted_candidates[:k]]
	candidate_chunks = [chunks[i] for i in top_indices]

	if not candidate_chunks:
		return f"No se encontró contenido relevante en Wikipedia para '{subject}'."

	# Cross-encoder reranking against the primary query
	try:
		cross_scores = reranker.predict(
			[(query, c) for c in candidate_chunks], batch_size=16
		)
		scored_chunks = sorted(
			zip(candidate_chunks, cross_scores),
			key=lambda x: x[1],
			reverse=True,
		)
		reranked_chunks = [chunk for chunk, _ in scored_chunks]
	except Exception as e:
		logging.warning(f"Reranker failed, falling back to FAISS order: {e}")
		reranked_chunks = candidate_chunks

	# Redundancy filtering
	filtered_chunks = filter_redundant_chunks(reranked_chunks, threshold=0.85)

	# Final selection within content budget
	selected_chunks = []
	total_chars = 0
	for chunk in filtered_chunks[:WIKIPEDIA_TOP_K]:
		if total_chars + len(chunk) > WIKIPEDIA_MAX_CONTENT:
			break
		selected_chunks.append(chunk)
		total_chars += len(chunk)

	if not selected_chunks:
		return f"No se encontró contenido relevante en Wikipedia para '{subject}'."

	logging.info(f"Wikipedia context: {len(selected_chunks)} chunks selected ({total_chars} chars)")
	return "\n\n".join(selected_chunks)


# --- Email Processing ---

def decode_mime_words(text):
	"""Decode MIME encoded headers"""
	return ''.join(
		frag.decode(enc or 'utf-8') if isinstance(frag, bytes) else frag
		for frag, enc in decode_header(text)
	)

def build_prompt(sender, name, subject, body, history):
	"""Construct the complete prompt for GPT"""
	with open("assets/faq.txt", encoding="utf-8") as f:
		faq_context = f.read()
	
	with open(PROMPT_FILE, encoding="utf-8") as f:
		template = f.read()
	
	wikipedia_context = "" if subject.lower() == "duda" else get_wikipedia_context(subject, body)
	
	return template.replace("{greeting}", name) \
				  .replace("{sender}", sender) \
				  .replace("{history}", history) \
				  .replace("{message}", body) \
				  .replace("{context}", faq_context) \
				  .replace("{wikipedia}", wikipedia_context)

def process_email(sender, subject, body):
	"""Main email processing pipeline"""
	start = time.time()
	subject = decode_mime_words(subject)
	logging.info(f"Processing email from {sender} - Subject: '{subject}'")
	
	# Handle user name
	name = get_user_name(sender) or extract_name_from_message(body) or "user"
	if not get_user_name(sender) and name != "user":
		save_user_name(sender, name)
	
	# Prepare conversation history
	history = "\n".join(
		f"User: {q}\nBot: {r}" for q, r in reversed(get_history_for_sender(sender))
	)
	# Generate and send response
	prompt = build_prompt(sender, name, subject, body, history)
	logging.info(f"Prompt generated:\n\n{prompt}")
	try:
		response = ask_gpt_with_retry(prompt)
		log_history(sender, body, response)
		end = time.time()
		elapsed = end - start
		hours = int(elapsed // 3600)
		minutes = int((elapsed % 3600) // 60)
		seconds = int(elapsed % 60)
		duration = f"Elapsed time: {hours}h {minutes}m {seconds}s"
		send_email(sender, f"Re: {subject}", response, duration)
		return response
	except Exception as e:
		logging.error(f"Email processing failed: {e}")
		error_msg = "Sorry, we're experiencing technical difficulties. Please try again later."
		end = time.time()
		elapsed = end - start
		hours = int(elapsed // 3600)
		minutes = int((elapsed % 3600) // 60)
		seconds = int(elapsed % 60)
		duration = f"Elapsed time: {hours}h {minutes}m {seconds}s"
		send_email(sender, f"Re: {subject}", error_msg, duration)
		return error_msg

# --- API Endpoints ---

@app.route("/process_email", methods=["POST"])
def api_process_email():
	"""API endpoint for email processing"""
	data = request.json
	if not all(k in data for k in ["sender", "subject", "body"]):
		return jsonify({"error": "Missing required fields"}), 400
	
	try:
		response = process_email(data["sender"], data["subject"], data["body"])
		return jsonify({"status": "success", "response": response})
	except Exception as e:
		logging.error(f"API error: {e}")
		return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
