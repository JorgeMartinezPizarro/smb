# Support Mail Bot (SMB) 🍄📬🤖

`SMB` is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

### Mailer

A `python` mail client looking for new emails.

### GPT

A `llama.cpp` service with a `GGUF` model.

### Database

`sqlite3` database to store metrics.

### Orchestrator

A `python` service that uses embeddings with `GPT` to answer emails.

## 🖥️ Requirements

- 8GB VRAM + 8GB RAM for GPU or 16GB RAM for CPU.
- Docker.
- GNU Make.
- Nvidia container toolkit (optional for GPU usage).

**NOTE**: To run `SMB` on windows, you need WSL additionally.

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

- [Config examples](.env.sample)

- [Benchmarks](docs/config.md)

To check if your system can use `CUDA`, use:

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

To create custom `prompt` using `RAG`, you can use the placeholders:

- `{history}`: historical conversation data.
- `{message}`: the question of the user.
- `{context}`: the `FAQ_TOP_K` question/answer pairs of `assets/faq.txt` most relevant to the message.
- `{wikipedia}`: the `WIKIPEDIA_TOP_K` most relevant chunks of the Wikipedia article named in the subject.

Retrieval uses multilingual models (`EMBED_MODEL`, `RERANK_MODEL`), so Spanish questions match English articles. Chunks scoring below `WIKIPEDIA_THRESHOLD` / `FAQ_THRESHOLD` are dropped. If the prompt would not fit in `MAX_PROMPT_LENGTH - MAX_TOKENS`, the oldest history and then the lowest-scored chunks are removed. Placeholders missing from the template are not retrieved.

To create custom `footer`, you can use the placeholders:

- `{time}`: the current date and time.
- `{duration}`: total duration of the inference.

For advanced `Chain of Thought` you need to wait, it is under research and will be availabe soon.

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

After the system is up, `SMB` will answer incoming emails. On first start the orchestrator downloads its retrieval models into `cache/huggingface`.

Use

```sh
make logs
```

for further information.

To change the compose project name, or use another environment file, you can use:

```sh
export ENV_FILE = .env.RTX3050-math
export PROJECT_NAME = mega
make build up logs
```

## 💬 Usage

After the project is running, the service will answer questions in the configured email account.

To use the service, just send an email with subject `Duda`.
In order to enable `wikipedia` RAG, you need to provide a search query in the subject to let the model automatically read the wikipedia article.

## 🏛️ License

This product is created by [Jorge Martinez Pizarro](https://ideniox.com) and licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
