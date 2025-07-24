# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

**Mailer**

A dockerized mail client that search for new emails.

**GPT**

A dockerized llama GGUF model to answer emails.

**Orchestrator**

A dockerized service that coordinate the workflow.

**Database**

Stores metrics and other information useful.

## ⚙️ Requirements

- Docker
- Make

## 🚀 How to Run

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
cp .env.sample .env
```

Write your own configuration [(See config examples)](docs/config.md).

Finally

```
make up
```

After the system is load, SMB will automatically answer to the configured mail box.

## 👤 Author

Jorge Martínez Pizarro

A mathematical programmer

https://ideniox.com

## 📜 License

This product is licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
