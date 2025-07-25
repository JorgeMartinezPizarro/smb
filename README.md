# Support Mail Bot (SMB) 📬🤖

`SMB` is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

*Mailer*

A `python` mail client looking for new emails.

*GPT*

A `llama.cpp` service with a `GGUF` model.

*Orchestrator*

A `python` service that coordinates the workflow.

*Database*

`sqlite3` database to store metrics.

## ⚙️ Requirements

- Docker
- Make
- [Nvidia container toolkit]

## 🚀 How to Run

Get the sources:

```
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
cp .env.sample .env
```

Write your own configuration:

- [Config examples](docs/config.md).

- [More config examples](.env.sample)

Finally, compile and run:

```
make build up
```

After the system is up, `SMB` will answer incomming emails.

## 👤 License

This product is created by [Jorge Martinez Pizarro](https://ideniox.com) and licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
