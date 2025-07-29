.PHONY: build pull push up down logs test-gpu

include .env

build:
	docker compose build

push:
	docker compose push

pull:
	docker compose pull

up:
	docker compose -p $(PROJECT_NAME) up -d

down:
	docker compose -p $(PROJECT_NAME) down --remove-orphans
	
logs:
	docker compose -p $(PROJECT_NAME) logs -f

test-gpu:
	@echo "Checking if Docker and NVIDIA Container Toolkit are properly installed."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || \
		(echo "ERROR: It seems that Docker or NVIDIA Container Toolkit are not properly installed." && exit 1)
