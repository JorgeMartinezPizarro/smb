# Support Mail Bot (SMB) 🍄📬🤖

`SMB` is a modular AI-powered system to automate email support using local LLMs, semantic search, and historical interaction context.

## 🧱 Architecture

### Mailer

A `python` mail client looking for new emails.

### GPT

The official `llama.cpp` server with a `GGUF` model, exposing an OpenAI-compatible API.

### Database

`sqlite3` database to store metrics.

### Orchestrator

A `python` service that uses embeddings with `GPT` to answer emails.

## 🖥️ Requirements

- 8GB VRAM + 8GB RAM for GPU or 16GB RAM for CPU.
- Docker.
- GNU Make.
- Nvidia container toolkit and a CUDA 12.8 capable driver (optional for GPU usage).

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

The model `LLM_REPO`/`LLM_NAME` is downloaded from Hugging Face into `cache/models` on first start. `LLAMA_CPP_BUILD` pins the `ghcr.io/ggml-org/llama.cpp` image build, and `LLM_BASE_URL` points the orchestrator to any OpenAI-compatible endpoint (default: `GPT_SERVICE`).

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

- `{message}`: the question of the user.
- `{history}`: historical conversation data.
- `{greeting}`: the name of the user, if known.
- `{sender}`: the email address of the user.
- `{context}`: the `FAQ_TOP_K` question/answer pairs of `assets/faq.txt` most relevant to the message.
- `{wikipedia}`: the `WIKIPEDIA_TOP_K` most relevant chunks of the Wikipedia article named in the subject.

Retrieval uses multilingual models (`EMBED_MODEL`, `RERANK_MODEL`), so Spanish questions match English articles. Chunks scoring below `WIKIPEDIA_THRESHOLD` / `FAQ_THRESHOLD` are dropped. If the prompt would not fit in `MAX_PROMPT_LENGTH - MAX_TOKENS`, the oldest history and then the lowest-scored chunks are removed. Placeholders missing from the template are not retrieved, and without `{wikipedia}` or `{context}` the retrieval models are not even loaded.

The chat template of the model is applied automatically, do not add tokens like `<s>[INST]` to the prompt.

To create custom `footer`, you can use the placeholders:

- `{time}`: the current date and time.
- `{duration}`: total duration of the inference.
- `{model}`: the name of the model.

Files in `assets/` are read on every email, changes apply without restart.

For advanced `Chain of Thought` you need to wait, it is under research and will be available soon.

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

It waits until the model is downloaded and loaded. After the system is up, `SMB` will answer incoming emails. Services restart automatically after a failure or a reboot.

Use

```sh
make logs
```

for further information.

To run several instances from the same folder, create one environment file for each, with its own `PROJECT_NAME`, and pass it to `make`:

```sh
make ENV_FILE=.env.RTX3050-math build up logs
```

Each instance keeps its own database in `cache/<PROJECT_NAME>/`.

## 💬 Usage

After the project is running, `SMB` will answer every new email in the configured account.

The subject is always the `wikipedia` article to read, and the model answers only from it. If there is no article with that name, `SMB` replies at once with the articles `wikipedia` suggests.

Emails that fail 3 times are marked as read and flagged for manual review.

## 🎛️ Commands

Set `MASTER_EMAIL` and `MASTER_TOKEN` to manage `SMB` by email. From the master address, send an email with the command as subject and the token as first line of the body:

- `#prompt`: the text below the token becomes the master prompt, it must include `{message}`. Without text, replies with the current prompt.
- `#reset`: back to the prompt of `PROMPT_FILE`.
- `#stats`: number of queries, duration and top senders.
- `#help`: list of commands.

The prompt set by email is stored in the instance database and used from the next email on, no restart needed.

## 🏛️ License

This product is created by [Jorge Martinez Pizarro](https://ideniox.com) and licensed as a [haat](https://github.com/JorgeMartinezPizarro/haat/blob/main/LICENSE.md).
