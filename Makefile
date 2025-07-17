.PHONY: build up down logs restart ps

build:
	docker compose build --build-arg USE_GPU=${USE_GPU}

up: build
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

restart: down build up

ps:
	docker compose ps
