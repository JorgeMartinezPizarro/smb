from flask import Flask, request, jsonify
from email.header import decode_header
from email.message import EmailMessage

import sqlite3
import requests
import smtplib
import os
import time
import re
import logging
import markdown
from datetime import datetime

from wiki_rag import get_wikipedia_context

# ----------------------------
# App
# ----------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Config
# ----------------------------

DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
SMTP_SERVER = os.getenv("SMTP_SERVER")
BOT_EMAIL = os.getenv("BOT_EMAIL")
BOT_PASS = os.getenv("BOT_PASS")

GPT_SERVICE = os.getenv("GPT_SERVICE", "gpt-cpu")
GPT_URL = f"http://{GPT_SERVICE}:5000/gpt"
TIMEOUT = int(os.getenv("TIMEOUT", "600"))

MAIL_HISTORY_SIZE = int(os.getenv("MAIL_HISTORY_SIZE", "3"))

PROMPT_FILE = "assets/" + os.environ.get("PROMPT_FILE", "default") + ".txt"
FOOTER_FILE = "assets/" + os.environ.get("FOOTER_FILE", "default") + ".txt"

LLM_NAME = os.getenv("LLM_NAME", "unknown")


# ----------------------------
# DB utils
# ----------------------------

def get_user_name(email):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM data WHERE key = ?", (f"name_{email}",))
        row = cur.fetchone()
        return row[0] if row else None


def save_user_name(email, name):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO data (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (f"name_{email}", name))
        conn.commit()


def get_history(sender):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT question, response FROM history
            WHERE sender = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (sender, MAIL_HISTORY_SIZE))
        return cur.fetchall()


def log_history(sender, q, r):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO history (timestamp, sender, question, response)
            VALUES (datetime('now'), ?, ?, ?)
        """, (sender, q, r))


# ----------------------------
# GPT
# ----------------------------

def ask_gpt(prompt):
    r = requests.post(
        GPT_URL,
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=TIMEOUT
    )
    return r.json().get("response", "")


# ----------------------------
# Email
# ----------------------------

def send_email(to, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = BOT_EMAIL
    msg["To"] = to
    msg.add_alternative(body, subtype="html")

    with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
        smtp.login(BOT_EMAIL, BOT_PASS)
        smtp.send_message(msg)


# ----------------------------
# Prompt builder
# ----------------------------

def build_prompt(sender, name, subject, body, history):
    with open(PROMPT_FILE, encoding="utf-8") as f:
        template = f.read()

    wiki = "" if subject.lower() == "duda" else get_wikipedia_context(subject, body)

    history_text = "\n".join(
        f"User: {q}\nBot: {r}" for q, r in history
    )

    return (
        template
        .replace("{greeting}", name)
        .replace("{sender}", sender)
        .replace("{message}", body)
        .replace("{history}", history_text)
        .replace("{wikipedia}", wiki)
    )


# ----------------------------
# Email processing
# ----------------------------

def process_email(sender, subject, body):
    subject = decode_mime(subject)

    name = get_user_name(sender) or "user"

    history = get_history(sender)

    prompt = build_prompt(sender, name, subject, body, history)

    response = ask_gpt(prompt)

    log_history(sender, body, response)

    send_email(sender, f"Re: {subject}", response)

    return response


def decode_mime(text):
    return ''.join(
        part.decode(enc or "utf-8") if isinstance(part, bytes) else part
        for part, enc in decode_header(text)
    )


# ----------------------------
# API
# ----------------------------

@app.route("/process_email", methods=["POST"])
def api():
    data = request.json

    if not all(k in data for k in ["sender", "subject", "body"]):
        return jsonify({"error": "missing fields"}), 400

    try:
        out = process_email(
            data["sender"],
            data["subject"],
            data["body"]
        )
        return jsonify({"ok": True, "response": out})
    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)