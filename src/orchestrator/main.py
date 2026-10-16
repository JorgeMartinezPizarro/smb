from flask import Flask, request, jsonify
from email.header import decode_header
from email.utils import parseaddr, formatdate, make_msgid
import sqlite3, requests, smtplib, os, time, re, logging, math, hmac
from email.message import EmailMessage
import subprocess
import markdown
from datetime import datetime

logging.basicConfig(level=logging.INFO)

from rag import env, faq_hits, wikipedia_hits, normalize_subject, strip_reply_prefixes, load_models

app = Flask(__name__)

# --- Configuration ---
TIMEOUT = int(os.getenv("TIMEOUT", "600"))     # GPT Request timeout

# Token budget: the LLM context (n_ctx) must hold the prompt plus the generated answer.
MAX_PROMPT_LENGTH = int(env("MAX_PROMPT_LENGTH", "2048"))  # LLM n_ctx in tokens
MAX_TOKENS = int(env("MAX_TOKENS", "512"))                 # Tokens reserved for the answer
PROMPT_MARGIN = int(env("PROMPT_MARGIN", "64"))            # Tokens reserved for the chat template

# Sampling settings, sent with every LLM request
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.5"))
TOP_K = int(os.getenv("TOP_K", "25"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0"))

# Email settings
MAIL_HISTORY_SIZE = int(os.getenv("MAIL_HISTORY_SIZE", "3"))
SMTP_SERVER = os.getenv("SMTP_SERVER")
BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_PASS = os.getenv("BOT_PASS")

# Master settings: commands by email, only from MASTER_EMAIL and with MASTER_TOKEN as first body line
MASTER_EMAIL = os.getenv("MASTER_EMAIL", "")
MASTER_TOKEN = os.getenv("MASTER_TOKEN", "")

# Paths
INDEX_PATH = "vector_db/faiss_index.bin"
DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
GPT_SERVICE = os.getenv("GPT_SERVICE", "gpt-cpu")
# llama-server tokenizer, served next to the OpenAI-compatible /v1 API
TOKENIZE_URL = env("TOKENIZE_URL", re.sub(r"/v1/?$", "", env("LLM_BASE_URL", f"http://{GPT_SERVICE}:8080")) + "/tokenize")
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or f"http://{GPT_SERVICE}:8080/v1").rstrip("/")  # OpenAI-compatible API
PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "default") + ".txt"
FOOTER_FILE = "assets/" + os.environ.get("FOOTER_FILE", "default") + ".txt"
LLM_NAME = os.getenv("LLM_NAME", "unknown")

# --- Utility Functions ---

# Tokens de control de las plantillas de chat (Mistral, Llama, ChatML, harmony...).
# llama.cpp los interpreta como especiales aunque lleguen dentro del texto del usuario.
SPECIAL_TOKENS_RE = re.compile(r"</?s>|\[/?INST\]|\[/?SYSTEM_PROMPT\]|<</?SYS>>|<\|[a-z_]{1,32}\|>", re.IGNORECASE)

def strip_special_tokens(text):
	"""Remove chat template control tokens from untrusted text"""
	return SPECIAL_TOKENS_RE.sub("", text or "")

def format_seconds(elapsed):
	"""Format a number of seconds as 0h 0m 0s"""
	if elapsed is None:
		return "-"
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = int(elapsed % 60)
	return f"{hours}h {minutes}m {seconds}s"

def format_duration(start):
	"""Elapsed time since start, formatted for the footer"""
	return f"Elapsed time: {format_seconds(time.time() - start)}"

def init_db():
	"""Create tables if missing"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute("CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)")
		conn.execute("""
			CREATE TABLE IF NOT EXISTS history (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp TEXT, sender TEXT, question TEXT, response TEXT, duration REAL
			)
		""")

def get_data(key):
	"""Get a stored value from database"""
	with sqlite3.connect(DB_PATH) as conn:
		row = conn.execute("SELECT value FROM data WHERE key = ?", (key,)).fetchone()
		return row[0] if row else None

def set_data(key, value):
	"""Save a value to database"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute("""
			INSERT INTO data (key, value) VALUES (?, ?)
			ON CONFLICT(key) DO UPDATE SET value=excluded.value
		""", (key, value))

def delete_data(key):
	"""Delete a stored value from database"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute("DELETE FROM data WHERE key = ?", (key,))

def get_user_name(email):
	"""Get stored user name from database"""
	return get_data(f"name_{email}")

def save_user_name(email, name):
	"""Save user name to database"""
	set_data(f"name_{email}", name)

def get_prompt_template():
	"""Master prompt: the one set by email if any, otherwise PROMPT_FILE"""
	stored = get_data("prompt")
	if stored:
		return stored
	with open(PROMPT_FILE, encoding="utf-8") as f:
		return f.read()

def get_prompt_source():
	"""Describe where the current master prompt comes from"""
	if get_data("prompt"):
		return f"definido por email el {get_data('prompt_updated')}"
	return f"fichero `{PROMPT_FILE}`"

def log_history(sender, question, response, duration=None):
	"""Log conversation history"""
	with sqlite3.connect(DB_PATH) as conn:
		conn.execute(
			"INSERT INTO history (timestamp, sender, question, response, duration) VALUES (datetime('now'), ?, ?, ?, ?)",
			(sender, question, response, duration)
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
	"""Query the OpenAI-compatible chat endpoint with retry logic.
	Returns (content, reasoning); reasoning is empty for non-reasoning models."""
	payload = {
		"model": LLM_NAME,
		"messages": [{"role": "user", "content": prompt}],
		"max_tokens": MAX_TOKENS,
		"temperature": TEMPERATURE,
		"top_p": TOP_P,
		"top_k": TOP_K,
		"repeat_penalty": REPETITION_PENALTY,
		"presence_penalty": PRESENCE_PENALTY,
	}
	for i in range(retries):
		try:
			response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
			if response.status_code == 200:
				message = response.json()["choices"][0]["message"]
				content = (message.get("content") or "").strip()
				reasoning = (message.get("reasoning_content") or "").strip()
				return content, reasoning
			logging.warning(f"GPT bad response: {response.status_code} {response.text[:200]}")
		except requests.exceptions.RequestException as e:
			logging.warning(f"Attempt {i+1} failed: {e}")
		if i < retries - 1:
			time.sleep(delay)
	raise Exception("Failed to connect to GPT after multiple attempts")

def send_email(to, subject, body_md, duration, reasoning=""):
	"""
	Envía email en HTML con diseño compacto y estilo verde
	"""
	from email.message import EmailMessage
	import smtplib
	from html import escape
	import markdown

	final_content = body_md.strip()
	analysis_content = reasoning.strip()

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
	# Required by RFC 5322, Gmail rejects messages without Message-ID
	msg["Date"] = formatdate(localtime=True)
	msg["Message-ID"] = make_msgid(domain=BOT_EMAIL.split("@")[-1])
	# Plain text version first, an HTML-only email scores worse in spam filters
	msg.set_content(final_content)
	msg.add_alternative(html_content, subtype='html')

	with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
		smtp.login(BOT_EMAIL, BOT_PASS)
		smtp.send_message(msg)
	logging.info(f"Email sent to {to} - Subject: '{subject}' - {msg['Message-ID']}")
				
def extract_name_from_message(message):
	"""Extract name from message text: up to 3 capitalized words after the trigger"""
	name = r"([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?: [A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,2})"
	m = re.search(r"\b(?i:me llamo|mi nombre es|soy) " + name, message)
	return m.group(1) if m else None

# --- Master Commands ---

MASTER_HELP = """Comandos disponibles en el asunto. La primera línea del cuerpo debe ser el token.

- `#prompt`: con texto debajo del token, define el prompt maestro (debe incluir `{message}`). Sin texto, muestra el actual.
- `#reset`: vuelve al prompt de `PROMPT_FILE`.
- `#stats`: número de consultas, duración y remitentes.
- `#help`: esta ayuda."""

def is_master_command(sender, subject):
	"""Commands are enabled only with MASTER_EMAIL and MASTER_TOKEN configured"""
	return bool(MASTER_EMAIL and MASTER_TOKEN) \
		and sender.lower() == MASTER_EMAIL.lower() \
		and strip_reply_prefixes(subject).startswith("#")

def split_command_body(body):
	"""Return (token, argument) from a command body, dropping quoted replies and signature"""
	lines = []
	for line in body.replace("\r\n", "\n").split("\n"):
		if line.startswith(">") or line.rstrip() == "--" \
				or re.match(r"^(On|El) .+@.+(wrote|escribió):\s*$", line) \
				or re.match(r"^-+ ?(Original Message|Mensaje original) ?-+$", line.strip(), re.IGNORECASE):
			break
		lines.append(line)
	token, _, argument = "\n".join(lines).strip().partition("\n")
	return token.strip(), argument.strip()

def format_stats():
	"""Usage statistics from the history table, in markdown"""
	with sqlite3.connect(DB_PATH) as conn:
		total, day, week, senders, avg_duration, max_duration, last = conn.execute("""
			SELECT COUNT(*),
				SUM(timestamp >= datetime('now', '-1 day')),
				SUM(timestamp >= datetime('now', '-7 days')),
				COUNT(DISTINCT sender),
				AVG(duration), MAX(duration), MAX(timestamp)
			FROM history
		""").fetchone()
		top = conn.execute("""
			SELECT sender, COUNT(*) FROM history
			GROUP BY sender ORDER BY COUNT(*) DESC LIMIT 5
		""").fetchall()
	lines = [
		"**Estadísticas**",
		"",
		f"- Consultas: {total} en total, {day or 0} en 24h, {week or 0} en 7 días",
		f"- Remitentes distintos: {senders}",
		f"- Duración: media {format_seconds(avg_duration)}, máxima {format_seconds(max_duration)}",
		f"- Última consulta: {last or '-'} (UTC)",
		f"- Modelo: `{LLM_NAME}`",
		f"- Prompt: {get_prompt_source()}",
	]
	if top:
		lines += ["", "**Top remitentes**", ""] + [f"- `{s}`: {n}" for s, n in top]
	return "\n".join(lines)

def handle_master_command(subject, body):
	"""Run a master command and return the reply in markdown"""
	command = strip_reply_prefixes(subject).split()[0].lower()
	token, argument = split_command_body(body)
	if not hmac.compare_digest(token.encode(), MASTER_TOKEN.encode()):
		logging.warning(f"Master command '{command}' with invalid token")
		return "Token inválido."
	logging.info(f"Master command: '{command}'")
	if command == "#prompt":
		if not argument:
			return f"Prompt actual ({get_prompt_source()}):\n\n~~~~\n{get_prompt_template()}\n~~~~"
		if "{message}" not in argument:
			return "Prompt rechazado: debe incluir el placeholder `{message}`."
		set_data("prompt", argument)
		set_data("prompt_updated", datetime.now().strftime("%d/%m/%Y %H:%M"))
		return f"Prompt maestro actualizado ({len(argument)} caracteres):\n\n~~~~\n{argument}\n~~~~"
	if command == "#reset":
		delete_data("prompt")
		delete_data("prompt_updated")
		return f"Prompt maestro restaurado a `{PROMPT_FILE}`."
	if command == "#stats":
		return format_stats()
	return MASTER_HELP

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
	template = get_prompt_template()

	def hits(retrieve, *args):
		try:
			return retrieve(*args)
		except Exception as e:
			logging.warning(f"{retrieve.__name__} failed: {e}")
			return []

	sources = {}
	if "{context}" in template:
		sources["context"] = (hits(faq_hits, body), "")
	if "{wikipedia}" in template:
		# "Duda" (también "Re: Duda") o sin asunto: consulta directa al modelo, sin Wikipedia
		topic = normalize_subject(subject)
		if topic.lower() in ("", "duda"):
			sources["wikipedia"] = ([], "")
		else:
			sources["wikipedia"] = (
				hits(wikipedia_hits, subject, body),
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
	subject = strip_special_tokens(decode_mime_words(subject))
	body = strip_special_tokens(body)
	# From header: "Name" <address> -> address as key and recipient, display name as fallback name
	display_name, address = parseaddr(sender)
	sender = address or sender
	display_name = strip_special_tokens(decode_mime_words(display_name))
	reply_subject = f"Re: {strip_reply_prefixes(subject)}"
	logging.info(f"Processing email from {sender} - Subject: '{subject}'")

	if is_master_command(sender, subject):
		try:
			reply = handle_master_command(subject, body)
		except Exception as e:
			logging.error(f"Master command failed: {e}")
			reply = f"Error ejecutando el comando: {e}"
		send_email(sender, reply_subject, reply, format_duration(start))
		return reply

	# Handle user name
	name = get_user_name(sender)
	if not name:
		name = extract_name_from_message(body)
		if name:
			save_user_name(sender, name)
	name = name or display_name or "user"

	# Conversation history, oldest first, without control tokens
	history = [
		(strip_special_tokens(q), strip_special_tokens(r))
		for q, r in reversed(get_history_for_sender(sender))
	]
	try:
		# Generate and send response
		prompt = build_prompt(sender, name, subject, body, history)
		logging.info(f"Prompt generated:\n\n{prompt}")
		response, reasoning = ask_gpt_with_retry(prompt)
		send_email(sender, reply_subject, response, format_duration(start), reasoning)
	except Exception as e:
		logging.error(f"Email processing failed: {e}")
		error_msg = "Sorry, we're experiencing technical difficulties. Please try again later."
		send_email(sender, reply_subject, error_msg, format_duration(start))
		return error_msg
	try:
		log_history(sender, body, response, time.time() - start)  # final answer only, no reasoning
	except Exception as e:
		logging.warning(f"Could not log history: {e}")
	return response

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

def preload_models():
	"""Load the retrieval models at startup if the current prompt needs them, otherwise on first use"""
	try:
		if re.search(r"\{(wikipedia|context)\}", get_prompt_template()):
			load_models()
	except Exception as e:
		logging.warning(f"Could not preload retrieval models: {e}")

if __name__ == "__main__":
	init_db()
	preload_models()
	app.run(host="0.0.0.0", port=5000)
