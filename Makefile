up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose down && docker compose up --build

ps:
	docker compose ps

health:
	@curl -sf http://localhost:5000/health || echo "⏳ Aún no está listo"
