.PHONY: help infra-up infra-down infra-logs infra-ps cloud-dev cloud-migrate cloud-test cloud-test-live cloud-lint cloud-fmt \
	builder-test builder-lint builder-fmt

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

infra-up: ## Start Postgres + RabbitMQ and wait until healthy
	docker compose up -d --wait

infra-down: ## Stop infrastructure (data volumes are kept)
	docker compose down

infra-logs: ## Follow infrastructure logs
	docker compose logs -f

infra-ps: ## Show infrastructure status
	docker compose ps

cloud-dev: ## Run the cloud API with auto-reload on :8000
	cd cloud && uv run fastapi dev

cloud-migrate: ## Apply database migrations (alembic upgrade head)
	cd cloud && uv run alembic upgrade head

cloud-test: ## Run cloud tests (needs make infra-up; uses the quipquill_test database)
	cd cloud && uv run pytest

cloud-test-live: ## Live OpenRouter tests (cost money; key from cloud/.env)
	cd cloud && OPENROUTER_API_KEY=$$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2) uv run pytest -m integration -v

cloud-lint: ## Lint and check formatting of cloud
	cd cloud && uv run ruff check && uv run ruff format --check

cloud-fmt: ## Auto-fix lint issues and format cloud
	cd cloud && uv run ruff check --fix && uv run ruff format

builder-test: ## Run builder tests (no network)
	cd builder && uv run pytest

builder-lint: ## Lint and check formatting of builder
	cd builder && uv run ruff check && uv run ruff format --check

builder-fmt: ## Auto-fix lint issues and format builder
	cd builder && uv run ruff check --fix && uv run ruff format
