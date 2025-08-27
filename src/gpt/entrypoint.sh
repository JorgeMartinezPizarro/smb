#!/bin/bash
set -e

echo "[ENTRYPOINT] Modo de ejecución: USE_GPU=$USE_GPU"

MODEL_DIR="/app/models"

mkdir -p "$MODEL_DIR"

export MODEL_PATH="$MODEL_DIR/$LLM_NAME"

echo "Running ${MODEL_PATH}"

export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export OPENBLAS_NUM_THREADS=${NUM_THREADS}
export NUMEXPR_NUM_THREADS=${NUM_THREADS}

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[ENTRYPOINT] Model not found. Downloading $LLM_REPO..."
  python3 - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
  repo_id="$LLM_REPO",
  filename="$LLM_NAME",
  local_dir="$MODEL_DIR",
  local_dir_use_symlinks=False
)
EOF
  echo "[ENTRYPOINT] Download complete."
else
  echo "[ENTRYPOINT] Model already exists, skipping download."
fi

export LLAMA_CPP_LIB=/llama-cpp-python/libllama.so

exec python3 /app/load_model.py
