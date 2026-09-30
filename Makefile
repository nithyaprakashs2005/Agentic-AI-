.DEFAULT_GOAL := help
.PHONY: help install dev api web test test-unit test-int cov lint fmt types check \
        services services-down services-logs mcp voice-check smoke clean

BACKEND := backend
FRONTEND := frontend
PORT ?= 8000

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------
install: ## Install backend and frontend dependencies
	cd $(BACKEND) && uv sync
	cd $(FRONTEND) && npm install

# --- Run ---------------------------------------------------------------
api: ## Run the backend API with reload
	cd $(BACKEND) && uv run uvicorn app.main:app --reload --port $(PORT)

web: ## Run the Next.js frontend
	cd $(FRONTEND) && npm run dev

dev: ## Run backend and frontend together
	@echo "API  -> http://localhost:$(PORT)/docs"
	@echo "Web  -> http://localhost:3000"
	@$(MAKE) -j2 api web

# --- Tests -------------------------------------------------------------
test: ## Run the whole test suite
	cd $(BACKEND) && uv run pytest

test-unit: ## Unit tests only
	cd $(BACKEND) && uv run pytest tests/unit -v

test-int: ## Integration tests only
	cd $(BACKEND) && uv run pytest tests/integration -v --timeout=60

cov: ## Tests with the coverage gate (80%)
	cd $(BACKEND) && uv run pytest --cov=app --cov-report=term-missing --cov-report=html

# --- Quality -----------------------------------------------------------
lint: ## Lint
	cd $(BACKEND) && uv run ruff check .
	cd $(FRONTEND) && npm run lint

fmt: ## Format
	cd $(BACKEND) && uv run ruff format .
	cd $(BACKEND) && uv run ruff check --fix .

types: ## Type-check
	cd $(BACKEND) && uv run mypy app/

check: lint types test ## Everything CI runs

# --- Services ----------------------------------------------------------
services: ## Start Qdrant, Valkey, Postgres, LiveKit
	docker compose up qdrant valkey postgres livekit -d
	@echo "Waiting for health..." && sleep 6 && docker compose ps

services-down: ## Stop all services
	docker compose down

services-logs: ## Tail service logs
	docker compose logs -f

# --- MCP / voice -------------------------------------------------------
mcp: ## List MCP servers and their tools
	cd $(BACKEND) && uv run python -c "\
import json; from app.mcp.server import describe_all; \
print(json.dumps(describe_all()['servers'], indent=2, default=str)[:4000])"

voice-check: ## Report voice-channel readiness and the latency budget
	cd $(BACKEND) && uv run python -c "\
import json; from app.voice.pipeline import preflight; \
print(json.dumps(preflight(), indent=2))"

# --- Migrations --------------------------------------------------------
# DATABASE_URL comes from the environment or .env; never from alembic.ini.
migrate: ## Apply migrations to DATABASE_URL
	cd $(BACKEND) && uv run alembic upgrade head

migrate-new: ## Autogenerate a migration (make migrate-new m="add x")
	cd $(BACKEND) && uv run alembic revision --autogenerate -m "$(m)"

migrate-down: ## Roll back one migration
	cd $(BACKEND) && uv run alembic downgrade -1

# --- Evaluation --------------------------------------------------------
eval: ## Measure FCR against the labelled case set
	cd $(BACKEND) && uv run python scripts/eval_fcr.py

eval-strict: ## Same, but fail the build below the 92% target
	cd $(BACKEND) && uv run python scripts/eval_fcr.py --fail-under 0.92

# --- Smoke -------------------------------------------------------------
smoke: ## Exercise a running API end to end
	cd $(BACKEND) && uv run python scripts/smoke.py

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/htmlcov $(BACKEND)/.coverage $(BACKEND)/.mypy_cache
	rm -rf $(FRONTEND)/.next
