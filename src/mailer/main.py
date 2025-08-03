import imaplib, email, os
import time
import requests
import ssl

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5000/process_email")

MAX_RETRIES = 5
RETRY_INTERVAL = 10

def log(msg):
    print(f"[MAILER] {msg}", flush=True)

def connect_imap():
    retries = 0
    while True:
        try:
            log(f"Conectando a IMAP en {IMAP_SERVER}...")
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL_USER, EMAIL_PASS)
            mail.select("inbox")
            log("✅ Conexión IMAP establecida y bandeja seleccionada.")
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

def fetch_one_unseen_email(mail):
    try:
        status, data = mail.search(None, "UNSEEN")
        mail_ids = data[0].split()
        if not mail_ids:
            return None

        # Solo procesamos el primero
        num = mail_ids[0]

        status, data = mail.fetch(num, "(RFC822)")
        if status != "OK":
            log(f"❌ Error al obtener email ID {num}")
            return None

        msg = email.message_from_bytes(data[0][1])

        sender = msg["From"]
        subject = msg["Subject"]

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get('Content-Disposition'):
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        return num, sender, subject, body
    except Exception as e:
        log(f"⚠️ Error fetching email: {e}")
        return None

def mark_as_seen(mail, mail_id):
    try:
        mail.store(mail_id, '+FLAGS', '\\Seen')
    except Exception as e:
        log(f"⚠️ Error marcando email {mail_id} como leído: {e}")

def main_loop():
    mail = connect_imap()
    log("🟢 Iniciando bucle de revisión de correos.")

    while True:
        try:
            email_data = fetch_one_unseen_email(mail)
            if not email_data:
                time.sleep(1)
                continue

            mail_id, sender, subject, body = email_data
            log(f"📩 Procesando email de: {sender}, asunto: {subject}")

            payload = {
                "sender": sender,
                "subject": subject,
                "body": body
            }

            try:
                response = requests.post(ORCHESTRATOR_URL, json=payload)
                log(f"📤 Respuesta orquestador: {response.json()}")
                mark_as_seen(mail, mail_id)
            except Exception as e:
                log(f"❌ Error llamando al orquestador: {e}")

        except imaplib.IMAP4.abort:
            log("❌ Conexión IMAP abortada, reconectando...")
            mail = connect_imap()
        except Exception as e:
            log(f"⚠️ Error en el loop principal: {e}")
            time.sleep(5)

if __name__ == "__main__":
    log("📡 Mailer iniciado correctamente.")
    main_loop()
