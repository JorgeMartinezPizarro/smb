#!/bin/bash
set -e

echo "[ENTRYPOINT] Modo de ejecución: USE_GPU=$USE_GPU"

MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_FILE="$MODEL_PATH"
REPO="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF"
FILE="openhermes-2.5-mistral-7b.Q8_0.gguf"

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

exec gunicorn --workers=1 --threads=1 --timeout=300 --bind 0.0.0.0:5000 load_model:app
