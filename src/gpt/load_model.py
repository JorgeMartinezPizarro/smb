import threading
import logging
from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

model_ready = threading.Event()
model_lock = threading.Lock()
llm = None

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/mistral-7b-instruct.Q4_K_M.gguf")
NUM_THREADS = os.environ.get("NUM_THREADS", 4)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 2048)
MAX_PROMPT_LENGTH = 4096
MAX_TOKENS = 4096
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

def load_model():
    global llm
    logging.info("🔄 Cargando modelo con llama-cpp...")

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_threads=NUM_THREADS,
            n_batch=BATCH_SIZE,
            n_ctx=MAX_PROMPT_LENGTH,
            n_gpu_layers=-1 if USE_GPU else 0,
            use_mmap=True,
            use_mlock=True,
            verbose=True,
        )
        with model_lock:
            llm("Priming...", max_tokens=1)

        model_ready.set()
        logging.info(f"✅ Modelo cargado con éxito (GPU: {USE_GPU}).")

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
                temperature=0.7,
                top_p=0.9,
            )
        text = response["choices"][0]["message"]["content"]
        logging.info(f"📤 Respuesta generada ({len(text)} chars)")
        return jsonify({"response": text.strip()})
    except Exception as e:
        logging.error(f"❌ Error en generación: {e}")
        return jsonify({"error": "Generation error"}), 500

def generate_text(prompt):
    with model_lock:
        response = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            echo=False,
        )
    return response["choices"][0]["text"]

# Carga el modelo en segundo plano
threading.Thread(target=load_model, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
