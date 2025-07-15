# Support Mail Bot (SMB) 📬🤖

A lightweight AI-powered email support bot.

## 🔹 Abstract

**Support Mail Bot (SMB)** is a modular system designed to automate support email responses using AI and historical context.

## 🔸 Architecture

- **DB**  
  Uses SQLite to store historical data and past email interactions.

- **GPT**  
  Runs a containerized Mistral model (tokenized) for generating natural language responses.

- **Mailer**  
  A simple Python-based email client to fetch incoming messages.

- **Orchestrator**  
  Core app that vectorizes incoming mail, fetches context from SQLite, and crafts a response using the GPT model.

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
```

2. Configure your environment by editing the .env file.

3. Run the app:

```bash
make up
```