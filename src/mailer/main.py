import imaplib, email, os
import time
import requests

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5000/process_email")

def log(msg):
	print(f"[MAILER] {msg}", flush=True)

def check_inbox():
	mail = imaplib.IMAP4_SSL(IMAP_SERVER)
	mail.login(EMAIL_USER, EMAIL_PASS)
	mail.select("inbox")
	status, data = mail.search(None, "UNSEEN")
	mail_ids = data[0].split()

	for num in mail_ids:
		status, data = mail.fetch(num, "(RFC822)")
		msg = email.message_from_bytes(data[0][1])
		sender = msg["From"]
		subject = msg["Subject"]

		# Extraer el cuerpo del mensaje (solo texto plano simple)
		body = ""
		if msg.is_multipart():
			for part in msg.walk():
				if part.get_content_type() == "text/plain":
					body = part.get_payload(decode=True).decode()
					break
		else:
			body = msg.get_payload(decode=True).decode()

		print(f"Nuevo email de: {sender}, asunto: {subject}")

		# Llamar al orquestador
		payload = {
			"sender": sender,
			"subject": subject,
			"body": body
		}
		try:
			response = requests.post(ORCHESTRATOR_URL, json=payload)
			print("Respuesta orquestador:", response.json())
		except Exception as e:
			print("Error llamando al orquestador:", e)

def main_loop():
	log("Iniciando bucle de revisión de correos.")
	while True:
		check_inbox()
		time.sleep(1)

if __name__ == "__main__":
	log("Mailer iniciado correctamente.")
	main_loop()