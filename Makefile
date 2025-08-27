.PHONY: build pull push up down logs test-gpu

ENV_FILE?=.env

include ${ENV_FILE}

DOCKER_CMD=docker compose

build:
	${DOCKER_CMD} build

push:
	${DOCKER_CMD} push

pull:
	${DOCKER_CMD} pull

up:
	${DOCKER_CMD} -p $(PROJECT_NAME) up -d

down:
	${DOCKER_CMD} -p $(PROJECT_NAME) down --remove-orphans
	
logs:
	${DOCKER_CMD} -p $(PROJECT_NAME) logs -f

test-env:
	@echo "SMB Configuration:\n=================="
	@grep -v -E '^\s*#|^MAIL_PASS=' $(ENV_FILE) | awk -F= '{print $$1 "=" $$2}'

test-gpu:
	@echo "Checking if Docker and NVIDIA Container Toolkit are properly installed."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || \
		(echo "ERROR: It seems that Docker or NVIDIA Container Toolkit are not properly installed. Your system is not able to use GPU." && exit 1)
