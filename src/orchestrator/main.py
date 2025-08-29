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
WIKIPEDIA_THRESHOLD = float(os.getenv("WIKIPEDIA_THRESHOLD", "0.5"))     # Threshold
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

def send_email(to, subject, body_md):
    """
    Envía email en HTML con diseño compacto y estilo verde
    """
    import re
    from email.message import EmailMessage
    import smtplib
    from html import escape
    import markdown

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
        html = markdown.markdown(md_text)
        html = re.sub(r'```([\s\S]*?)```', r'<pre style="background:#f1f5f9;padding:10px;border-radius:4px;overflow:auto;margin:8px 0;font-size:12px;"><code>\1</code></pre>', html)
        return html

    final_html = convert_markdown_to_html(final_content)
    analysis_html = convert_markdown_to_html(analysis_content) if analysis_content else ""

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
            background-color: #22c55e;  /* Verde */
            color: white;
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
		h1 {{
			font-size: 13pt;
		}}
    </style>
    </head>
    <body>
        <div class="email-container">
            <div class="card">
                <div class="card-header">
                    ✅ Respuesta final: {escape(subject)}
                </div>
                <div class="card-content">
                    {final_html}
                </div>
            </div>
    """

    html_content += """
            <div class="card">
                <div class="card-header">
					🧑🏼 Respues generada por:
				</div>
				<div class="card-content">
                    <p>Atentamente,</p>
					<p>George von Knowman</p>
					<p>Math Teacher Assistant (MTA)</p>
					<p>https://math.ideniox.com/</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

	# Añadir análisis scrolleable compacto
    if analysis_html:
        html_content += f"""
            <!-- Bloque de Análisis -->
            <div class="card">
                <div class="card-header">
                    🔍 Análisis interno
                </div>
                <div class="card-content" style="max-height: 90px; overflow-y: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; line-height: 1.3;">
                    {analysis_html}
                </div>
            </div>
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

def get_cached_wikipedia(subject, lang=WIKIPEDIA_LANG):
    safe_subject = subject.replace("/", "_").replace(" ", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_subject}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        chunks = data["chunks"].tolist()
        embeddings = data["embeddings"]
        return chunks, embeddings

    page = get_wikipedia_page(subject, lang)
    if not page or not page.exists():
        return [], np.array([])

    sections = extract_relevant_sections(page)

    chunks = []
    for title, text in sections:
        if len(text) > WIKIPEDIA_CHUNK_SIZE:
            # solo partir si es enorme
            for chunk in chunk_text(text, WIKIPEDIA_CHUNK_SIZE):
                chunks.append(f"{title}\n{chunk}")
        else:
            # sección completa como un solo chunk
            chunks.append(f"{title}\n{text}")

    embeddings = []
    for i in range(0, len(chunks), WIKIPEDIA_BATCH_SIZE):
        batch = chunks[i:i+WIKIPEDIA_BATCH_SIZE]
        emb = model_embed.encode(batch, convert_to_numpy=True)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    np.savez_compressed(cache_file, chunks=np.array(chunks), embeddings=embeddings)
    return chunks, embeddings

def get_wikipedia_page(subject, lang=WIKIPEDIA_LANG):
	"""Retrieve Wikipedia page with fallback to search"""
	wiki = wikipediaapi.Wikipedia(
		user_agent='knowledge-bot/1.0',
		language=lang,
		extract_format=wikipediaapi.ExtractFormat.WIKI
	)
	page = wiki.page(subject)
	if page.exists():
		return page
	
	# Fallback to search
	try:
		search_url = f"https://{lang}.wikipedia.org/w/api.php"
		params = {
			"action": "query",
			"list": "search",
			"srsearch": subject,
			"format": "json"
		}
		resp = requests.get(search_url, params=params)
		if resp.status_code == 200:
			first_title = resp.json().get("query", {}).get("search", [{}])[0].get("title")
			if first_title:
				return wiki.page(first_title)
	except Exception as e:
		logging.warning(f"Wikipedia search failed: {e}")
	return None

def clean_wikipedia_text(text):
	"""Normalize and clean Wikipedia text"""
	if not text:
		return ""
	text = unicodedata.normalize('NFKC', text)
	text = ''.join(ch for ch in text if (ch == '\n' or unicodedata.category(ch)[0] != 'C'))
	text = re.sub(r'[\t\r]+', ' ', text)
	text = re.sub(r' *\n+ *', '\n\n', text)
	return re.sub(r'[ ]{2,}', ' ', text).strip()

def chunk_text(text, max_chars=WIKIPEDIA_CHUNK_SIZE, overlap=100):
    """Split text into semantically coherent chunks with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= max_chars:
            current_chunk += (p + "\n\n")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Añadir solapamiento: tomar los últimos `overlap` caracteres
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + p + "\n\n"
            else:
                # Caso especial: párrafo demasiado grande, se fuerza chunk solo con ese párrafo
                chunks.append(p)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
	
def extract_relevant_sections(page):
    """Extrae secciones de la página de Wikipedia (sin prioridad hardcodeada)."""
    sections = []
    def process_section(section, parent_title=""):
        full_title = f"{parent_title} > {section.title}" if parent_title else section.title
        text = clean_wikipedia_text(section.text)
        if text and len(text) >= WIKIPEDIA_MIN_SECTION:
            sections.append((full_title, text))
        for subsection in section.sections:
            process_section(subsection, full_title)
    process_section(page)
    return sections

def filter_redundant_chunks(chunks, threshold=0.85):
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

def get_wikipedia_context(subject, query, lang=WIKIPEDIA_LANG):
    chunks, embeddings = get_cached_wikipedia(subject, lang)

    if not chunks:
        return f"No Wikipedia information found about '{subject}'"

    # Query embedding
    query_embed = model_embed.encode([query], convert_to_numpy=True)[0]
    query_embed /= (np.linalg.norm(query_embed) + 1e-10)

    # FAISS search
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    k = min(WIKIPEDIA_TOP_K * 3, len(chunks))
    distances, indices = index.search(query_embed.reshape(1, -1), k)

    # Rerank con cross-encoder
    candidates = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    cross_scores = reranker.predict([(query, c[0]) for c in candidates], batch_size=16)

    scored_chunks = sorted(
        zip((c[0] for c in candidates), cross_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Filtrar redundancias
    filtered_chunks = filter_redundant_chunks(
        [chunk for chunk, _ in scored_chunks], threshold=0.85
    )

    # Selección final
    selected_chunks = []
    total_chars = 0
    for chunk in filtered_chunks[:WIKIPEDIA_TOP_K]:
        if total_chars + len(chunk) > WIKIPEDIA_MAX_CONTENT:
            break
        selected_chunks.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(selected_chunks) if selected_chunks else "No relevant content found"

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
		send_email(sender, f"Re: {subject}", response)
		return response
	except Exception as e:
		logging.error(f"Email processing failed: {e}")
		error_msg = "Sorry, we're experiencing technical difficulties. Please try again later."
		send_email(sender, f"Re: {subject}", error_msg)
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