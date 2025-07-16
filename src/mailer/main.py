import imaplib, email, os
import time
import requests
import ssl

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5000/process_email")

MAX_RETRIES = 5           # Número máximo de reintentos al conectar (None = infinito)
RETRY_INTERVAL = 10       # Tiempo en segundos entre reintentos

def log(msg):
	print(f"[MAILER] {msg}", flush=True)

def connect_imap():
	retries = 0
	while True:
		try:
			##log(f"Conectando a IMAP en {IMAP_SERVER}...")
			mail = imaplib.IMAP4_SSL(IMAP_SERVER)
			mail.login(EMAIL_USER, EMAIL_PASS)
			##log("✅ Conexión IMAP establecida.")
			return mail
		except ssl.SSLEOFError as e:
			log(f"⚠️ Error SSL (EOF): {e}")
		except Exception as e:
			log(f"❌ Error general al conectar a IMAP: {e}")

		retries += 1
		if MAX_RETRIES and retries >= MAX_RETRIES:
			log("🚫 Máximo número de reintentos alcanzado. Abortando.")
			raise SystemExit(1)

		log(f"🔁 Reintentando en {RETRY_INTERVAL} segundos...")
		time.sleep(RETRY_INTERVAL)

def check_inbox():
	try:
		mail = connect_imap()
		mail.select("inbox")
		status, data = mail.search(None, "UNSEEN")
		mail_ids = data[0].split()

		for num in mail_ids:
			status, data = mail.fetch(num, "(RFC822)")
			msg = email.message_from_bytes(data[0][1])
			sender = msg["From"]
			subject = msg["Subject"]

			body = ""
			if msg.is_multipart():
				for part in msg.walk():
					if part.get_content_type() == "text/plain":
						body = part.get_payload(decode=True).decode()
						break
			else:
				body = msg.get_payload(decode=True).decode()

			log(f"📩 Nuevo email de: {sender}, asunto: {subject}")

			payload = {
				"sender": sender,
				"subject": subject,
				"body": body
			}
			try:
				response = requests.post(ORCHESTRATOR_URL, json=payload)
				log(f"📤 Respuesta orquestador: {response.json()}")
			except Exception as e:
				log(f"❌ Error llamando al orquestador: {e}")
		mail.logout()
	except Exception as e:
		log(f"⚠️ Error en check_inbox(): {e}")

def main_loop():
	log("🟢 Iniciando bucle de revisión de correos.")
	while True:
		check_inbox()
		time.sleep(8)

if __name__ == "__main__":
	log("📡 Mailer iniciado correctamente.")
	main_loop()
