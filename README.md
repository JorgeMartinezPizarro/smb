# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

**Mailer**

A mail client that scan new email and talk to the orquestrtor.

**GPT**

A dockerized Mistral 7B LLaMA-based model for text generation.

**Orchestrator**

Coordinate the answer of emails using semantic context.

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
