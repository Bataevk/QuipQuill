.PHONY: help infra-up infra-down infra-logs infra-ps cloud-dev cloud-test cloud-lint cloud-fmt

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

cloud-test: ## Run cloud tests
	cd cloud && uv run pytest

cloud-lint: ## Lint and check formatting of cloud
	cd cloud && uv run ruff check && uv run ruff format --check

cloud-fmt: ## Auto-fix lint issues and format cloud
	cd cloud && uv run ruff check --fix && uv run ruff format
