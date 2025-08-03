# Support Mail Bot (SMB) 📬🤖

`SMB` is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

### Mailer

A `python` mail client looking for new emails.

### GPT

A `llama.cpp` service with a `GGUF` model.

### Orchestrator

A `python` service that coordinates the workflow.

### Database

`sqlite3` database to store metrics.

## 🖥️ Requirements

- Docker
- GNU Make
- [Nvidia container toolkit] (optional, for GPU usage)

## 💾 Download

```sh
git clone https://github.com/JorgeMartinezPizarro/smb
cd smb
```

## ✏️ Config

To create your own configuration, use:

```
cp .env.sample .env
```

Change `.env` according to your needs:

- [Config examples](docs/config.md).

- [More config examples](.env.sample).

To check if you system can use `CUDA`, use:

```sh
make test-gpu
```

By setting up valid values for:

```sh
REGISTRY_USER=user
REGISTRY_REPO=repo
IMAGE_TAG=first
```

you can use `make push` and `make pull` to get your own compiled images.

## 🏭 Build

To create the containers

```sh
make build
```

## 🕹️ Run

Finally, to start `SMB`:

```sh
make up
```

After the system is up, `SMB` will answer incoming emails.

Use

```sh
make logs
```

for further information.

## 🏛️ License

This product is created by [Jorge Martinez Pizarro](https://ideniox.com) and licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
