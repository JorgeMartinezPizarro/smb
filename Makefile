.PHONY: build pull push up down logs test-gpu

include .env

## NOTE:
#
#  We use docker run cause docker compose still not support 100% GPU,
#  it may be reviewed and fixed in the feature. For now it was 
#  important to ensure that it works on Windows with the WSL.
#

build:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker build --progress=plain \
		--build-arg USE_CUDA=cuda \
		--build-arg GGML_CUDA=1 \
		-t $(REGISTRY_USER)/${REGISTRY_REPO}-gpt-gpu:${IMAGE_TAG} \
		./src/gpt
else
	docker compose build
endif

push:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker push ${REGISTRY_USER}/${REGISTRY_REPO}-gpt-gpu:${IMAGE_TAG}
endif
	docker compose push

pull:
ifeq ($(COMPOSE_PROFILES),gpu)
	docker pull ${REGISTRY_USER}/${REGISTRY_REPO}-gpt-gpu:${IMAGE_TAG}
endif
	docker compose pull

up:
	docker compose up -d
ifeq ($(COMPOSE_PROFILES),gpu)
	docker run -d --gpus all \
		--name gpt-gpu \
		--network gpt-network \
		--health-cmd="curl -f http://localhost:5000/health || exit 1" \
		--health-interval=30s \
		--health-retries=3 \
		-e LLM_REPO=${LLM_REPO} \
		-e LLM_NAME=${LLM_NAME} \
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
		$(REGISTRY_USER)/${REGISTRY_REPO}-gpt-gpu:${IMAGE_TAG}
endif

down:
	docker kill gpt-gpu 2>/dev/null || true
	docker rm gpt-gpu 2>/dev/null || true
	docker compose down --remove-orphans
	
logs:
	@bash -c '\
		trap "kill 0" EXIT; \
		docker compose logs -f & \
		docker logs -f gpt-gpu 2>&1 | awk '\''{print "\033[35mgpt-gpu | \033[0m" $$0}'\'' & \
		wait \
	'

test-gpu:
	@echo "Checking if Docker and NVIDIA Container Toolkit are properly installed."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || \\
		(echo "ERROR: It seems that Docker or NVIDIA COntainer Toolkit are not properly installed." && exit 1)
