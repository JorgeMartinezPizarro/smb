.PHONY: build up down logs restart ps

include .env

build:
	docker compose build

up: build
	docker compose up -d
	if [ "${COMPOSE_PROFILES}" = "gpu" ]; then \
		$(MAKE) run-gpu; \
	fi

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

restart: down build up

ps:
	docker compose ps

run-gpu:
	docker build -t gpt-gpu ./src/gpt
	docker run -d --rm --gpus all \
		--name gpt-gpu  \
		-e MODEL_PATH=/app/models/openhermes-2.5-mistral-7b.Q8_0.gguf \
		-e USE_GPU=true \
		-e BATCH_SIZE=${BATCH_SIZE} \
		-e NUM_THREADS=${NUM_THREADS} \
		-v ./cache/models:/app/models \
		-v ./cache/cache:/root/.cache/llama_cpp \
		-p 5000:5000 \
		gpt-gpu
