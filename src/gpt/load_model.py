import threading
import logging
from flask import Flask, request, jsonify
from llama_cpp import Llama
import concurrent.futures
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

model_ready = threading.Event()
model_lock = threading.Lock()
llm = None

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/mistral-7b-instruct.Q4_K_M.gguf")
MAX_PROMPT_LENGTH = 4096   # o menos si hace falta
MAX_TOKENS = 1000           # más rápido en CPU

def load_model():
    global llm
    logging.info("🔄 Cargando modelo con llama-cpp...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_threads=10,
            n_batch=4096,
            n_ctx=4096,
            use_mmap=True,
            use_mlock=True,
            verbose=True,
            temperature=0.7,
    		top_p=0.9,
    		stop=["Un cordial saludo,"]
        )
        # Priming para asegurar que el modelo está listo
        with model_lock:
            llm("Priming...", max_tokens=1)
        model_ready.set()
        logging.info("✅ Modelo cargado con éxito.")
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

    prompt = "\n".join(m.get("content", "") for m in messages)
    prompt = prompt[-MAX_PROMPT_LENGTH:]

    logging.info(f"📝 Prompt recibido ({len(prompt)} chars)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate_text, prompt)
        try:
            response = future.result(timeout=240)
        except concurrent.futures.TimeoutError:
            return jsonify({"error": "Generation timeout"}), 504
        except Exception as e:
            logging.error(f"❌ Error en generación: {e}")
            return jsonify({"error": "Generation error"}), 500

    logging.info(f"📤 Respuesta generada ({len(response)} chars)")
    return jsonify({"response": response.strip()})

def generate_text(prompt):
    with model_lock:
        response = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            echo=False,
        )
    return response["choices"][0]["text"]

# Arranca la carga del modelo en un hilo aparte al importar el módulo
threading.Thread(target=load_model, daemon=True).start()

if __name__ == "__main__":
    # Para correr localmente con Flask dev server
    app.run(host="0.0.0.0", port=5000)
