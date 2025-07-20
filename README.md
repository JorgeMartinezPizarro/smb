# Support Mail Bot (SMB) 📬🤖

SMB is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

**Mailer**

A mail client that scan new email and request an action to orquestrtor.

**GPT**

A dockerized Mistral 7B LLaMA-based model for text generation.

**Orchestrator**

The core, that create the prompt using vectorized content, historical and predefined rules. It receive an email and generate a valid response.

**Database**

Stores metrics and other information useful for debugging and improving the system.

## ⚙️ Requirements

- Docker
- Make

## 🚀 How to Run

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
make up
```

Configure email credentials and model parameters in the .env file before running. [See config examples](docs/config.md) or look at the `.env.sample` for more details.

After the setup and load of `SMB`, you can start sending emails to your `MAIL_USER` and wait for the response.

## 👤 Author

Jorge Martínez Pizarro

A mathematical programmer

https://ideniox.com

## 📜 License

This product is licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
