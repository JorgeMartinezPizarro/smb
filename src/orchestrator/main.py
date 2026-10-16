from flask import Flask, request, jsonify
from email.header import decode_header
import sqlite3, requests, smtplib, os, time, re, logging, math
from email.message import EmailMessage
import subprocess
import markdown
from datetime import datetime

logging.basicConfig(level=logging.INFO)

from rag import env, faq_hits, wikipedia_hits, normalize_subject

app = Flask(__name__)

# --- Configuration ---
TIMEOUT = int(os.getenv("TIMEOUT", "600"))     # GPT Request timeout

# Token budget: the LLM context (n_ctx) must hold the prompt plus the generated answer.
MAX_PROMPT_LENGTH = int(env("MAX_PROMPT_LENGTH", "2048"))  # LLM n_ctx in tokens
MAX_TOKENS = int(env("MAX_TOKENS", "512"))                 # Tokens reserved for the answer
PROMPT_MARGIN = int(env("PROMPT_MARGIN", "64"))            # Tokens reserved for the chat template

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
# llama-server tokenizer, served next to the OpenAI-compatible /v1 API
TOKENIZE_URL = env("TOKENIZE_URL", re.sub(r"/v1/?$", "", env("LLM_BASE_URL", f"http://{GPT_SERVICE}:8080")) + "/tokenize")
PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "default") + ".txt"
FOOTER_FILE = "assets/" + os.environ.get("FOOTER_FILE", "default") + ".txt"
LLM_NAME = os.getenv("LLM_NAME", "unknown")

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

# --- Email Processing ---

def decode_mime_words(text):
	"""Decode MIME encoded headers"""
	return ''.join(
		frag.decode(enc or 'utf-8') if isinstance(frag, bytes) else frag
		for frag, enc in decode_header(text)
	)

# --- Prompt & Token Budget ---

_tokenizer_retry_at = 0.0

def count_tokens(text):
	"""Exact count from the LLM's /tokenize endpoint, else a conservative chars/3 estimate."""
	global _tokenizer_retry_at
	if time.time() >= _tokenizer_retry_at:
		try:
			response = requests.post(TOKENIZE_URL, json={"content": text}, timeout=10)
			response.raise_for_status()
			return len(response.json()["tokens"])
		except Exception as e:
			logging.info(f"Tokenizer unavailable at {TOKENIZE_URL} ({e}); estimating chars/3 for 5 min")
			_tokenizer_retry_at = time.time() + 300
	return math.ceil(len(text) / 3)

def render(template, values):
	"""Fill {placeholders} in one pass, so placeholders inside user text are never expanded."""
	return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)

def fit_prompt(template, values, history, sources):
	"""
	Render the template so that prompt + MAX_TOKENS fits in MAX_PROMPT_LENGTH, trimming in order:
	history (oldest first), retrieved chunks of any source (lowest score first), the message tail.

	history: [(question, response)] oldest first.
	sources: {placeholder: (hits sorted best first, text used when no hit is left)}.
	"""
	budget = MAX_PROMPT_LENGTH - MAX_TOKENS - PROMPT_MARGIN
	history = list(history)
	sources = {key: (list(hits), empty) for key, (hits, empty) in sources.items()}
	message = values["message"]
	while True:
		prompt = render(template, {
			**values,
			"message": message,
			"history": "\n".join(f"User: {q}\nBot: {r}" for q, r in history),
			**{key: "\n\n".join(h.text for h in hits) or empty for key, (hits, empty) in sources.items()},
		})
		tokens = count_tokens(prompt)
		if tokens <= budget:
			logging.info(
				f"Prompt: {tokens}/{budget} tokens, {len(history)} history entries, "
				+ ", ".join(f"{len(hits)} {key} chunks" for key, (hits, _) in sources.items())
			)
			return prompt
		if history:
			history.pop(0)
		elif any(hits for hits, _ in sources.values()):
			min((hits for hits, _ in sources.values() if hits), key=lambda hits: hits[-1].score).pop()
		elif message:
			message = message[:max(len(message) - 4 * (tokens - budget), 0)]
		else:
			raise ValueError(f"Prompt template alone needs {tokens} tokens, budget is {budget}")

def build_prompt(sender, name, subject, body, history):
	"""Construct the complete prompt for GPT, retrieving only what the template uses."""
	with open(PROMPT_FILE, encoding="utf-8") as f:
		template = f.read()

	sources = {}
	if "{context}" in template:
		sources["context"] = (faq_hits(body), "")
	if "{wikipedia}" in template:
		topic = normalize_subject(subject)
		if topic.lower() in ("", "duda"):
			sources["wikipedia"] = ([], "")
		else:
			sources["wikipedia"] = (
				wikipedia_hits(subject, body),
				f"No se encontró información relevante en Wikipedia sobre '{topic}'.",
			)

	return fit_prompt(
		template,
		{"greeting": name, "sender": sender, "message": body},
		history if "{history}" in template else [],
		sources,
	)

def process_email(sender, subject, body):
	"""Main email processing pipeline"""
	start = time.time()
	subject = decode_mime_words(subject)
	logging.info(f"Processing email from {sender} - Subject: '{subject}'")
	
	# Handle user name
	name = get_user_name(sender) or extract_name_from_message(body) or "user"
	if not get_user_name(sender) and name != "user":
		save_user_name(sender, name)
	
	# Conversation history, oldest first
	history = list(reversed(get_history_for_sender(sender)))
	# Generate and send response
	try:
		prompt = build_prompt(sender, name, subject, body, history)
		logging.info(f"Prompt generated:\n\n{prompt}")
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
