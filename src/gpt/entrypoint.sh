#!/bin/bash
set -e

echo "[ENTRYPOINT] Modo de ejecución: USE_GPU=$USE_GPU"

MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_FILE="$MODEL_PATH"
REPO="bartowski/Mistral-7B-Instruct-v0.3-GGUF"
FILE="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
EXPECTED_FILE="mistral-7b-instruct.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/$EXPECTED_FILE" ]; then
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
  mv "$MODEL_DIR/$FILE" "$MODEL_DIR/$EXPECTED_FILE"
  echo "[ENTRYPOINT] Descarga completada y archivo renombrado."
else
  echo "[ENTRYPOINT] Modelo ya existe, saltando descarga."
fi

exec gunicorn --workers=1 --threads=10 --bind 0.0.0.0:5000 load_model:app
