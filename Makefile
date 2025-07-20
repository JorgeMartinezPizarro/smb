.PHONY: build up down logs restart ps run-gpu test-gpu

include .env

REGISTRY ?= $(REGISTRY_USER)
REPO ?= $(REGISTRY_REPO)
TAG ?= $(IMAGE_TAG)

build:
ifeq ($(COMPOSE_PROFILES),gpu)
	# Build GPU image manualmente
	docker build --progress=plain \
		--build-arg USE_CUDA=cuda \
		--build-arg GGML_CUDA=1 \
		-t gpt-gpu \
		-t $(REGISTRY)/$(REPO)-gpt-gpu:$(TAG) \
		./src/gpt

	# Build mailer, db, orchestrator desde compose
	docker compose build mailer db orchestrator
else
	# Build CPU y el resto de imágenes con Compose
	docker compose build
endif

up:
	docker compose up -d
ifeq ($(COMPOSE_PROFILES),gpu)
	$(MAKE) run-gpu
endif

down:
	docker compose down --remove-orphans
	docker kill gpt-gpu 2>/dev/null || true
	docker rm gpt-gpu 2>/dev/null || true

logs:
	@docker compose logs -f &
	@docker ps --format '{{.Names}}' | grep -q '^gpt-gpu$$' && \
		docker logs -f gpt-gpu 2>&1 | sed -u "s/^/$(shell printf '\033[34m')gpt-gpu-1 | $(shell printf '\033[0m')/" || true

restart: down build up

run-gpu:
	docker run -d --gpus all \
		--name gpt-gpu \
		--network gpt-network \
		--health-cmd="curl -f http://localhost:5000/health || exit 1" \
		--health-interval=30s \
		--health-retries=3 \
		-e MODEL_PATH=/app/models/deepseek-llm-7b-chat.Q4_K_M.gguf \
		-e USE_GPU=true \
		-e USE_CUDA=cuda \
		-e NUM_THREADS=${NUM_THREADS} \
      	-e BATCH_SIZE=${BATCH_SIZE} \
      	-e MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} \
      	-e MAX_TOKENS=${MAX_TOKENS} \
      	-e GPU_LAYERS=${GPU_LAYERS} \
		-e GGML_CUDA=1 \
		-e BATCH_SIZE=$(BATCH_SIZE) \
		-e NUM_THREADS=$(NUM_THREADS) \
		-v ./cache/models:/app/models \
		-v ./cache/cache:/root/.cache/llama_cpp \
		-p 5000:5000 \
		gpt-gpu

test-gpu:
	@echo "Comprobando si Docker y NVIDIA Container Toolkit están correctamente configurados..."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || \\
		(echo "ERROR: No se detecta configuración correcta para GPUs en Docker. Verifica que tengas instalado nvidia-container-toolkit y que el daemon de Docker esté configurado." && exit 1)

push:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker push $(REGISTRY)/$(REPO)-gpt-gpu:$(TAG)
else
	docker push $(REGISTRY)/$(REPO)-gpt-cpu:$(TAG)
endif
	docker push $(REGISTRY)/$(REPO)-mailer:$(TAG)
	docker push $(REGISTRY)/$(REPO)-db:$(TAG)
	docker push $(REGISTRY)/$(REPO)-orchestrator:$(TAG)

pull:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker pull $(REGISTRY)/$(REPO)-gpt-gpu:$(TAG)
else
	docker pull $(REGISTRY)/$(REPO)-gpt-cpu:$(TAG)
endif
	docker pull $(REGISTRY)/$(REPO)-mailer:$(TAG)
	docker pull $(REGISTRY)/$(REPO)-db:$(TAG)
	docker pull $(REGISTRY)/$(REPO)-orchestrator:$(TAG)