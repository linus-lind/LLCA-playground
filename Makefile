.PHONY: help venv install install-dev install-hooks lint format-check type-check config-check test check clean mlflow-ui experiment-backup

VENV := .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	PYTHON_SYS := py -3.13
	PYTHON := $(VENV_BIN)/python.exe
else
	VENV_BIN := $(VENV)/bin
	PYTHON_SYS := python3.13
	PYTHON := $(VENV_BIN)/python
endif

PIP := $(PYTHON) -m pip

help: ## Show available project tasks
	@$(PYTHON_SYS) -c "import re; from pathlib import Path; rows=[m.groups() for line in Path('Makefile').read_text().splitlines() if (m:=re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line))]; print('\n'.join(f'  {name:<18} {description}' for name, description in rows))"

$(PYTHON):
	$(PYTHON_SYS) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(PYTHON) ## Create the Python 3.13 virtual environment

install: venv ## Install runtime dependencies in editable mode
	$(PIP) install -e .

install-dev: venv ## Install runtime and development dependencies
	$(PIP) install -e ".[dev]"
	"$(MAKE)" install-hooks

install-hooks: ## Install repository-local pre-commit and pre-push hooks
	$(PYTHON) -m pre_commit install --config .git-hooks-config.yaml -t pre-commit -t pre-push

lint: ## Run Ruff without modifying files
	$(PYTHON) -m ruff check src tests scripts

format-check: ## Verify Ruff formatting
	$(PYTHON) -m ruff format --check src tests scripts

type-check: ## Run strict Mypy over the package
	$(PYTHON) -m mypy src scripts

config-check: ## Compose and validate every Hydra application root
	$(PYTHON) scripts/validate_configs.py

test: ## Run the complete unit and integration test suite
	$(PYTHON) -m unittest discover -s tests -v

check: lint format-check type-check config-check test ## Run the complete local quality gate

mlflow-ui: ## Open the local MLflow tracking UI
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

experiment-backup: ## Archive and verify the local MLflow database and artifacts
	$(PYTHON) scripts/archive_experiments.py

clean: ## Remove disposable caches and Hydra execution logs only
	$(PYTHON) scripts/clean_workspace.py

.DEFAULT_GOAL := help
