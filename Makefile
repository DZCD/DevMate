.PHONY: up down chat run init

# One-command start + interactive chat
up:
	docker compose up -d --build
	@echo "\nServices started. Attaching to DevMate agent..."
	@sleep 3
	docker attach devmate-agent

# Stop all services
down:
	docker compose down -v

# Interactive chat (requires services running)
chat:
	docker compose run --rm devmate chat

# Single-shot command
run:
	docker compose run --rm devmate run "$(CMD)"

# Initialize docs index
init:
	docker compose run --rm devmate init
