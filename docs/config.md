## LLM backend

`gpt-cpu` and `gpt-gpu` run `ghcr.io/ggml-org/llama.cpp:server[-cuda]-${LLAMA_CPP_BUILD}`, configured with `LLAMA_ARG_*` variables:

| `.env` | llama.cpp server |
|---|---|
| `LLM_REPO` / `LLM_NAME` | `--hf-repo` / `--hf-file` (cached in `cache/models`) |
| `MAX_PROMPT_LENGTH` | `--ctx-size` |
| `NUM_THREADS` | `--threads` |
| `BATCH_SIZE` | `--batch-size` |
| `GPU_LAYERS` | `--n-gpu-layers` (`gpu` profile, `0` on `cpu`) |
| `LLM_REASONING` | `--reasoning` (`on`, `off`, `auto`) |
| `LLM_REASONING_EFFORT` | `--reasoning-effort` (`low`, `medium`, `high`, `default`) |

Always set: `--jinja`, `--reasoning-format deepseek` (thinking goes to the 🧠 card of the email, not to the history), `--no-mmproj` (text only) and `--parallel 1`.

`MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `REPETITION_PENALTY` (`repeat_penalty`) and `PRESENCE_PENALTY` are sent by the orchestrator with every `POST ${LLM_BASE_URL}/chat/completions`.

## Benchmarking

The service has been tested with the following hardware and setups (measured with the former `llama-cpp-python` service):

### i5 13500 64GB RAM

**Config**

```shell
COMPOSE_PROFILES=cpu
GPT_SERVICE=gpt-cpu
BATCH_SIZE=4096 
NUM_THREADS=16
MAX_PROMPT_LENGTH=2048
MAX_TOKENS=256
GPU_LAYERS=0
LLM_REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"
LLM_NAME="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
```

**Results**

- 26ms per read token

- 130ms per generated token

### RTX 3050 8GB VRAM

**Config**

```shell
COMPOSE_PROFILES=gpu
GPT_SERVICE=gpt-gpu
BATCH_SIZE=4096 
NUM_THREADS=6
MAX_PROMPT_LENGTH=2048
MAX_TOKENS=512
GPU_LAYERS=33
LLM_REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"
LLM_NAME="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
```

**Results**

- 1.3ms per read token

- 30.7ms per generated token
