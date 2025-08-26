import threading
import logging
from flask import Flask, request, jsonify
from llama_cpp import Llama, LoraTrainer
import os
import uuid

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

model_ready = threading.Event()
model_lock = threading.Lock()
llm = None

MODEL_PATH = os.environ.get("MODEL_PATH", "")
NUM_THREADS = int(os.environ.get("NUM_THREADS", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", 128)) 
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 128)) 
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", 2)) 
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
TOP_K = int(os.environ.get("TOP_K", 25))
TOP_P = float(os.environ.get("TOP_P", 0.5))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", 1.15))

# Carpeta donde guardamos los LoRA .bin
LORA_DIR = "./lora_bins"
os.makedirs(LORA_DIR, exist_ok=True)

def load_model():
    global llm
    logging.info(f"🔄 Cargando modelo base GGUF... GPU={USE_GPU}")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_threads=NUM_THREADS,
            n_batch=BATCH_SIZE,
            n_ctx=MAX_PROMPT_LENGTH,
            n_gpu_layers=GPU_LAYERS if USE_GPU else 0,
            use_mmap=False,
            use_mlock=False,
            verbose=True,
        )
        # priming
        with model_lock:
            llm("Priming...", max_tokens=1)
        model_ready.set()
        logging.info(f"✅ Modelo base cargado (GPU={USE_GPU})")
    except Exception as e:
        logging.error(f"❌ Error cargando modelo: {e}")

@app.route("/health", methods=["GET"])
def health():
    return ("ready", 200) if model_ready.is_set() else ("loading", 503)

@app.route("/gpt", methods=["POST"])
def chat():
    if not model_ready.is_set():
        return jsonify({"error": "Model not ready"}), 503

    data = request.json or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    logging.info(f"📝 Prompt recibido (n_messages={len(messages)})")
    try:
        with model_lock:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )
        text = response["choices"][0]["message"]["content"]
        logging.info(f"📤 Respuesta generada ({len(text)} chars)")
        return jsonify({"response": text.strip()})
    except Exception as e:
        logging.error(f"❌ Error en generación: {e}")
        return jsonify({"error": "Generation error"}), 500

@app.route("/train", methods=["POST"])
def train():
    """Recibe un texto plano y genera un LoRA .bin delta"""
    if not model_ready.is_set():
        return jsonify({"error": "Model not ready"}), 503

    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    logging.info(f"🔧 Entrenamiento LoRA iniciado para prompt de {len(prompt)} chars")
    try:
        # Entrenamiento LoRA mínimo viable
        trainer = LoraTrainer(
            base_model=llm,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )

        # Convertimos texto plano en lista de líneas
        dataset = [line for line in prompt.split("\n") if line.strip()]
        trainer.train(dataset, batch_size=1, epochs=1)  # batch/epochs mínimos

        # Guardamos .bin con nombre único
        lora_file = os.path.join(LORA_DIR, f"lora_{uuid.uuid4().hex}.bin")
        trainer.save(lora_file)

        logging.info(f"✅ LoRA guardado en {lora_file}")
        return jsonify({"status": "ok", "lora_file": lora_file})

    except Exception as e:
        logging.error(f"❌ Error en entrenamiento LoRA: {e}")
        return jsonify({"error": "Training error"}), 500

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
