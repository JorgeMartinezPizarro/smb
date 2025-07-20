#!/bin/bash
set -e

echo "[ENTRYPOINT] Modo de ejecución: USE_GPU=$USE_GPU"

MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_FILE="$MODEL_PATH"

## TODO Move it to the config, as soon as it has been tested with other models
#REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"
#FILE="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
REPO="TheBloke/deepseek-llm-7B-chat-GGUF"
FILE="deepseek-llm-7b-chat.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/$FILE" ]; then
  echo "[ENTRYPOINT] Modelo no encontrado. Descargando $FILE..."
  python3 - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
  repo_id="$REPO",
  filename="$FILE",
  local_dir="$MODEL_DIR",
  local_dir_use_symlinks=False
)
EOF
  echo "[ENTRYPOINT] Descarga completada."
else
  echo "[ENTRYPOINT] Modelo ya existe, saltando descarga."
fi


export LLAMA_CPP_LIB=/llama-cpp-python/libllama.so

exec python /app/load_model.py
