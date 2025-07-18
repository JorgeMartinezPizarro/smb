.PHONY: build up down logs restart ps run-gpu test-gpu

include .env

build:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker build --progress=plain --build-arg USE_CUDA=cuda --build-arg GGML_CUDA=1 -t gpt-gpu ./src/gpt
else
	docker compose build --build-arg USE_CUDA=cpu
endif

up: build
	docker compose up -d
ifeq ($(COMPOSE_PROFILES),gpu)
	$(MAKE) run-gpu
endif

down:
	docker compose down --remove-orphans
	docker kill gpt-gpu 2>/dev/null || true
	docker rm gpt-gpu 2>/dev/null || true

logs:
	docker compose logs -f

restart: down build up

run-gpu:
	docker run -d --gpus all \
		--name gpt-gpu \
		--restart unless-stopped \
		--network gpt-network \
		--health-cmd="curl -f http://localhost:5000/health || exit 1" \
		--health-interval=30s \
		--health-retries=3 \
		-e MODEL_PATH=/app/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
		-e USE_GPU=true \
		-e USE_CUDA=cuda \
		-e GGML_CUDA=1 \
		-e BATCH_SIZE=$(BATCH_SIZE) \
		-e NUM_THREADS=$(NUM_THREADS) \
		-v ./cache/models:/app/models \
		-v ./cache/cache:/root/.cache/llama_cpp \
		-p 5000:5000 \
		gpt-gpu

test-gpu:
	@echo "Comprobando si Docker y NVIDIA Container Toolkit están correctamente configurados..."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || \
		(echo "ERROR: No se detecta configuración correcta para GPUs en Docker. Verifica que tengas instalado nvidia-container-toolkit y que el daemon de Docker esté configurado." && exit 1)
