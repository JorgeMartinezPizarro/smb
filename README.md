# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧠 What it does

SMB receives incoming emails, vectorizes them, retrieves relevant past interactions from a local SQLite database, and generates a personalized response using a lightweight containerized LLM.

## 🧱 Architecture Overview

Mailer
A mail client that scan new email and request an action to orquestrtor.

GPT
A dockerized llama model mistral 7b for text generation.

Orchestrator
The core, that create the prompt using vectorized content, historical and predefined rules. It receive an email and generate a valid response.

Database (SQLite)
Storage for metrics and other information that can be used to fix bugs and improve the system.

## ⚙️ Requirements

- Docker
- Make

## 🚀 How to Run

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
make up
```

Configure email credentials and model parameters in the .env file before running.

## 🧪 System Flow

An email is received from a user.

Relevant context is embedded into a dynamic prompt.

A local LLM generates a natural language reply.

The reply is sent back via SMTP.

The interaction is stored for future context.

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
        ├── Dockerfile
        ├── faq_ingest.py
        ├── main.py
        └── retriever.py
```

## ✨ Features

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
