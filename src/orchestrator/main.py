from flask import Flask, request, jsonify
import sqlite3, requests, smtplib, os, time, re, logging
from email.message import EmailMessage
import subprocess

app = Flask(__name__)

# Configuración por variables de entorno
DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
GPT_URL = os.getenv("GPT_URL", "http://gpt:5000/gpt")
PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "xxx") + ".txt"
SMTP_SERVER = os.getenv("SMTP_SERVER")
BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_PASS = os.getenv("BOT_PASS")

INDEX_PATH = "vector_db/faiss_index.bin"

def check_and_build_index():
    if not os.path.isfile(INDEX_PATH):
        print("[INFO] Índice FAISS no encontrado, generando nuevo índice...")
        # Ejecutar el script de ingest para crear índice y chunks
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

def clean_redundancy(text):
    # Elimina repeticiones 100% iguales seguidas (más de 2 veces)
    lines = text.splitlines()
    cleaned = []
    for i, line in enumerate(lines):
        if i >= 2 and line == lines[i-1] == lines[i-2]:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def ask_gpt_with_retry(prompt, retries=10, delay=10):
    messages = [
        {"role": "user", "content": prompt}
    ]
    for i in range(retries):
        try:
            response = requests.post(GPT_URL, json={"messages": messages}, timeout=240)
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
    body = body.replace('\\n', '\n')  # <-- aquí la corrección
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

# --- PROCESAMIENTO DE EMAIL ---

def clean_gpt_response(text):
    # Busca la línea con solo guiones (---)
    parts = text.split('---', 1)
    if len(parts) == 2:
        # Devuelve lo que viene después del primer '---', recortando espacios
        return parts[1].strip()
    else:
        # Si no hay '---', devuelve el texto tal cual
        return text.strip()
    
from retriever import FAQRetriever

# Carga global del retriever, solo una vez
retriever = FAQRetriever("vector_db/faiss_index.bin")

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

    history_entries = get_history_for_sender(sender, limit=5)
    if history_entries:
        history_text = "\n".join(
            f"Cliente: {q}\nBot: {r}" for q, r in reversed(history_entries)
        )
    else:
        history_text = "Este es el primer mensaje del cliente."

    # Obtén fragmentos relevantes del FAQ (top 5)
    faq_chunks = retriever.query(body, top_k=6)
    #logging.info(f"FAQs relevantes extraídas: {faq_chunks}")
    context_text = "\n".join(faq_chunks)
    if not context_text.strip():
        context_text = "No hay información adicional disponible del FAQ."

    # Carga la plantilla
    with open(PROMPT_FILE, encoding="utf-8") as f:
        template = f.read()

    prompt = template.replace("{greeting}", name) \
                     .replace("{sender}", sender) \
                     .replace("{history}", history_text) \
                     .replace("{message}", body) \
                     .replace("{context}", context_text)

    logging.info(f"Prompt generado para GPT ({len(prompt)} chars) {prompt}")

    try:
        answer = ask_gpt_with_retry(prompt)
        answer = clean_redundancy(answer)
        answer = clean_gpt_response(answer)
        
        logging.info(f"Recibida respuesta de GPT\n")
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
