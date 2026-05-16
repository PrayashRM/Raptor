# ==============================================================================
# Makefile — ResearchRAG Backend
# Shortcuts for common development and deployment commands.
#
# Usage:
#   make up          start all containers
#   make down        stop all containers
#   make build       rebuild backend image
#   make logs        tail all container logs
#   make migrate     run alembic migrations
#   make shell       open bash inside backend container
#   make psql        open psql inside postgres container
#   make fresh       wipe everything and start clean
# ==============================================================================

.PHONY: help up down build rebuild logs logs-backend logs-postgres \
        migrate migrate-new migrate-down shell psql pgadmin-url \
        test test-unit test-integration fresh clean \
        prod-up prod-down prod-migrate prod-build

# ------------------------------------------------------------------------------
# Default target — show help
# ------------------------------------------------------------------------------
help:
	@echo ""
	@echo "ResearchRAG Backend — Available Commands"
	@echo "========================================="
	@echo ""
	@echo "DEVELOPMENT"
	@echo "  make up              Start all containers (postgres + pgadmin + backend)"
	@echo "  make down            Stop all containers (data is preserved)"
	@echo "  make build           Build backend Docker image"
	@echo "  make rebuild         Force rebuild (no cache)"
	@echo "  make fresh           Wipe all data and start completely fresh"
	@echo "  make clean           Remove stopped containers and dangling images"
	@echo ""
	@echo "LOGS"
	@echo "  make logs            Tail logs from all containers"
	@echo "  make logs-backend    Tail backend logs only"
	@echo "  make logs-postgres   Tail postgres logs only"
	@echo ""
	@echo "DATABASE"
	@echo "  make migrate         Run alembic upgrade head"
	@echo "  make migrate-new m=  Create new migration (e.g. make migrate-new m=add_users)"
	@echo "  make migrate-down    Downgrade one migration step"
	@echo "  make migrate-history Show migration history"
	@echo "  make psql            Open psql shell in postgres container"
	@echo "  make db-backup       Backup database to backup.sql"
	@echo "  make db-restore      Restore database from backup.sql"
	@echo ""
	@echo "DEVELOPMENT TOOLS"
	@echo "  make shell           Open bash shell in backend container"
	@echo "  make pgadmin-url     Print pgAdmin URL and credentials"
	@echo "  make check-health    Check all container health statuses"
	@echo ""
	@echo "TESTING"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov        Run tests with coverage report"
	@echo ""
	@echo "PRODUCTION"
	@echo "  make prod-build      Build production Docker image"
	@echo "  make prod-up         Start production stack"
	@echo "  make prod-down       Stop production stack"
	@echo "  make prod-migrate    Run migrations against production DB"
	@echo ""

# ------------------------------------------------------------------------------
# Development — Container lifecycle
# ------------------------------------------------------------------------------

# Start all containers
# --build ensures the backend image is built if it doesn't exist
up:
	docker compose up --build

# Start in detached (background) mode
up-d:
	docker compose up --build -d

# Stop all containers (volumes preserved — data is safe)
down:
	docker compose down

# Build backend image only
build:
	docker compose build backend

# Force rebuild with no cache (use when deps change and cache seems stale)
rebuild:
	docker compose build --no-cache backend

# Wipe everything including volumes (DESTRUCTIVE — all data deleted)
fresh:
	@echo "WARNING: This will delete ALL database data."
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	docker compose down -v --remove-orphans
	docker compose up --build

# Remove stopped containers and dangling images
clean:
	docker compose down --remove-orphans
	docker image prune -f

# ------------------------------------------------------------------------------
# Logs
# ------------------------------------------------------------------------------

# Tail all container logs
logs:
	docker compose logs -f

# Tail backend logs only
logs-backend:
	docker compose logs -f backend

# Tail postgres logs only
logs-postgres:
	docker compose logs -f postgres

# ------------------------------------------------------------------------------
# Database — Alembic migrations
# ------------------------------------------------------------------------------

# Run all pending migrations
migrate:
	docker compose exec backend alembic upgrade head

# Create a new migration
# Usage: make migrate-new m="add_new_column_to_users"
migrate-new:
	docker compose exec backend alembic revision --autogenerate -m "$(m)"

# Downgrade one step
migrate-down:
	docker compose exec backend alembic downgrade -1

# Show migration history
migrate-history:
	docker compose exec backend alembic history --verbose

# Show current migration version
migrate-current:
	docker compose exec backend alembic current

# ------------------------------------------------------------------------------
# Database — Direct access
# ------------------------------------------------------------------------------

# Open psql shell inside postgres container
psql:
	docker compose exec postgres psql -U reslens_user -d ResLens

# Show all tables in ResLens database
db-tables:
	docker compose exec postgres psql -U reslens_user -d ResLens -c "\dt"

# Show installed extensions
db-extensions:
	docker compose exec postgres psql -U reslens_user -d ResLens -c "\dx"

# Backup database to file
db-backup:
	docker compose exec postgres pg_dump -U reslens_user ResLens > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup saved to backup_$(shell date +%Y%m%d_%H%M%S).sql"

# Restore database from backup.sql
db-restore:
	cat backup.sql | docker compose exec -T postgres psql -U reslens_user ResLens

# ------------------------------------------------------------------------------
# Development Tools
# ------------------------------------------------------------------------------

# Install a new package (run inside container)
# Usage: make add p=some-package
add:
	docker compose exec backend uv pip install $(p)
	@echo "REMINDER: Add $(p) to pyproject.toml dependencies and run 'uv lock' locally"

# Generate/update uv.lock file (run locally, not in container)
lock:
	uv lock

# Open bash shell in backend container
shell:
	docker compose exec backend bash

# Print pgAdmin connection info
pgadmin-url:
	@echo ""
	@echo "pgAdmin UI:  http://localhost:5050"
	@echo "Login Email: $$(grep PGADMIN_DEFAULT_EMAIL .env | cut -d= -f2)"
	@echo "Password:    $$(grep PGADMIN_DEFAULT_PASSWORD .env | cut -d= -f2)"
	@echo ""
	@echo "To connect to postgres inside pgAdmin:"
	@echo "  Host:     postgres"
	@echo "  Port:     5432"
	@echo "  Database: ResLens"
	@echo "  Username: reslens_user"
	@echo "  Password: $$(grep POSTGRES_PASSWORD .env | cut -d= -f2)"
	@echo ""

# Check health status of all containers
check-health:
	@echo ""
	@echo "Container Health Status:"
	@echo "========================"
	docker compose ps
	@echo ""
	@echo "Postgres health check:"
	docker compose exec postgres pg_isready -U reslens_user -d ResLens

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

# Run all tests inside backend container
test:
	docker compose exec backend pytest tests/ -v

# Run unit tests only
test-unit:
	docker compose exec backend pytest tests/unit/ -v

# Run integration tests only
test-integration:
	docker compose exec backend pytest tests/integration/ -v

# Run with coverage report
test-cov:
	docker compose exec backend pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# Run specific test file
# Usage: make test-file f=tests/unit/test_auth_service.py
test-file:
	docker compose exec backend pytest $(f) -v

# ------------------------------------------------------------------------------
# Production
# ------------------------------------------------------------------------------

# Build production image
prod-build:
	docker build \
		--target production \
		--tag reslens-backend:latest \
		--tag reslens-backend:$(shell git rev-parse --short HEAD) \
		.

# Start production stack
prod-up:
	docker compose -f docker-compose.prod.yml up -d

# Stop production stack
prod-down:
	docker compose -f docker-compose.prod.yml down

# Run migrations against production database
# Run this BEFORE deploying new version
prod-migrate:
	docker compose -f docker-compose.prod.yml exec backend alembic upgrade head

# View production logs
prod-logs:
	docker compose -f docker-compose.prod.yml logs -f