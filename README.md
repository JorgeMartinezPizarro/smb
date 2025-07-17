# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧠 What it does

SMB receives incoming emails, vectorizes them, retrieves relevant past interactions from a local SQLite database, and generates a personalized response using a lightweight containerized LLM.

## 🧱 Architecture Overview

Mailer
A minimalist Python IMAP/SMTP client that fetches incoming emails and formats them for processing.

GPT
A local containerized inference server running a quantized model (e.g., Mistral 7B Q4) using llama.cpp or compatible backends.

Orchestrator
The system core: vectorizes messages, retrieves context, assembles the prompt, and calls the model to generate responses.

Database (SQLite)
Stores historical Q&A pairs, human-verified corrections (errata), and support annotations, to improve context over time.

## ⚙️ Requirements

- Docker 28.2.2 or higher
- GNU Make 4.3 or higher

## 🚀 How to Run

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
make up
```

Configure email credentials and model parameters in the .env file before running.

## 🧪 System Flow

An email is received from a user.

The message is vectorized and semantically compared against stored interactions.

Relevant context is embedded into a dynamic prompt.

A local LLM generates a natural language reply.

The reply is sent back via SMTP, and the interaction is stored for future context.

## 📦 File Structure

```
smb/
├── assets
│   ├── faq.txt
│   └── prompt.txt
├── docker-compose.yml
├── Makefile
├── README.md
└── src
    ├── db
    │   ├── Dockerfile
    │   ├── entrypoint.sh
    │   └── init.sql
    ├── gpt
    │   ├── Dockerfile
    │   ├── entrypoint.sh
    │   ├── load_model.py
    │   └── requirements.txt
    ├── mailer
    │   ├── Dockerfile
    │   └── main.py
    └── orchestrator
        ├── data
        ├── Dockerfile
        ├── faq_ingest.py
        ├── main.py
        ├── prompt.txt
        └── retriever.py
```

## ✨ Features

Vector-based retrieval of historical support cases

Modular pipeline with shell-level orchestration

Custom prompt generation per message

Fully local LLM usage — no API keys or external services required

Flexible data ingestion (FAQs, human corrections, error reports)

## 👤 Author

Jorge Martínez Pizarro

A mathematical programmer

https://ideniox.com

## 📜 License

This product is licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
