# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

**Mailer**

A dockerized mail client looking for new emails.

**GPT**

A dockerized llama GGUF model to generate answers.

**Orchestrator**

A dockerized service that coordinates the workflow.

**Database**

Stores metrics and other useful information.

## ⚙️ Requirements

- Docker
- Make

## 🚀 How to Run

Get the sources:

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
cp .env.sample .env
```

Write your own configuration [(See config examples)](docs/config.md).

Finally, compile and run:

```
make build up
```

After the system is load, SMB will automatically answer to the configured mail box.

## 👤 Author

Jorge Martínez Pizarro

A mathematical programmer

https://ideniox.com

## 📜 License

This product is licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
