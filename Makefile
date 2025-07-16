build:
	docker compose build --build-arg USE_GPU=false

up: build
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose down
	docker compose build --build-arg USE_GPU=false
	docker compose up -d --gpus all

ps:
	docker compose ps

health:
	@curl -sf http://localhost:5000/health || echo "⏳ Aún no está listo"