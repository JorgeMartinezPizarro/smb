import imaplib
import email
import os
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


def safe_decode(payload):
    """Convierte cualquier payload a str de forma segura."""
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode(errors="ignore")
    if isinstance(payload, str):
        return payload
    return str(payload)


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


def ensure_connection(mail):
    """Verifica si la conexión IMAP sigue viva."""
    try:
        status, _ = mail.noop()
        if status != "OK":
            raise imaplib.IMAP4.abort("No se obtuvo OK en NOOP")
    except imaplib.IMAP4.abort:
        raise
    except Exception as e:
        log(f"⚠️ Error en noop(): {e}")
        raise imaplib.IMAP4.abort("Fallo en noop()")

def safe_decode(payload, fallback=""):
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception:
            return fallback
    elif isinstance(payload, str):
        return payload
    else:
        # Por si es int u otro tipo raro
        return fallback

def fetch_one_unseen_email(mail, max_retries=3):
    """Devuelve (id, remitente, asunto, cuerpo) de un email no leído o None."""
    retries = 0
    while retries < max_retries:
        try:
            ensure_connection(mail)
            status, data = mail.search(None, "UNSEEN")
            if status != "OK" or not data or not data[0]:
                return None

            mail_ids = data[0].split()
            if not mail_ids:
                return None

            num = mail_ids[0]
            status, data = mail.fetch(num, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                log(f"❌ Error al obtener email ID {num}")
                return None

            msg = email.message_from_bytes(data[0][1])
            sender = msg.get("From", "")
            subject = msg.get("Subject", "")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain" and not part.get("Content-Disposition"):
                        body = safe_decode(part.get_payload(decode=True))
                        break
                if not body:  # fallback a HTML si no hay texto plano
                    for part in msg.walk():
                        if part.get_content_type() == "text/html":
                            body = safe_decode(part.get_payload(decode=True))
                            break
            else:
                body = safe_decode(msg.get_payload(decode=True))

            return num, sender, subject, body

        except imaplib.IMAP4.abort:
            raise
        except Exception as e:
            log(f"⚠️ Error fetching email (intento {retries+1}): {e}")
            retries += 1
            time.sleep(1)  # espera un segundo antes de reintentar

    return None

def mark_as_seen(mail, mail_id):
    try:
        ensure_connection(mail)
        mail.store(mail_id, "+FLAGS", "\\Seen")
        log(f"✅ Email {mail_id} marcado como leído.")
    except imaplib.IMAP4.abort:
        raise
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
                if response.status_code == 200:
                    log(f"📤 Respuesta orquestador: {response.json()}")
                    mark_as_seen(mail, mail_id)
                else:
                    log(f"⚠️ Orquestador respondió {response.status_code}, no se marca como leído.")
            except Exception as e:
                log(f"❌ Error llamando al orquestador: {e}")

        except imaplib.IMAP4.abort:
            log("🔄 Reconectando a IMAP...")
            mail = connect_imap()
        except Exception as e:
            log(f"⚠️ Error en el loop principal: {e}")
            time.sleep(5)


if __name__ == "__main__":
    log("📡 Mailer iniciado correctamente.")
    main_loop()
