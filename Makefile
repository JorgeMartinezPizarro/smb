.PHONY: build up down logs restart ps run-gpu test-gpu

include .env

build:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker build --build-arg USE_CUDA=cuda -t gpt-gpu ./src/gpt
else
	docker compose build --build-arg USE_CUDA=cpu
endif

up: build
ifeq ($(COMPOSE_PROFILES),gpu)
	$(MAKE) run-gpu
endif
	docker compose up -d

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

restart: down build up

ps:
	docker compose ps

run-gpu:
	docker run -d --rm --gpus all \
		--name gpt-gpu \
		--restart unless-stopped \
		--health-cmd="curl -f http://localhost:5000/health || exit 1" \
		--health-interval=30s \
		--health-retries=3 \
		-e MODEL_PATH=/app/models/openhermes-2.5-mistral-7b.Q8_0.gguf \
		-e USE_GPU=true \
		-e BATCH_SIZE=$(BATCH_SIZE) \
		-e NUM_THREADS=$(NUM_THREADS) \
		-v ./cache/models:/app/models \
		-v ./cache/cache:/root/.cache/llama_cpp \
		-p 5000:5000 \
		gpt-gpu

test-gpu:
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
