.PHONY: help venv install install-dev lint type-check test clean mlflow-ui experiment-backup install-hooks

VENV := .venv

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
	PYTHON_SYS := py -3.13
else
	VENV_BIN := $(VENV)/bin
	PYTHON_SYS := python3.13
endif

PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/python -m pip
PKG := src/llca
TORCH_INDEX := https://download.pytorch.org/whl/cu126

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' | sort

venv:
	$(PYTHON_SYS) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -e .
	$(PIP) install --force-reinstall torch --index-url $(TORCH_INDEX)
	"$(MAKE)" install-hooks

install-hooks:
	$(VENV_BIN)/pre-commit install --config .git-hooks-config.yaml -t pre-commit -t pre-push

lint:
	$(VENV_BIN)/ruff check src/ tests/

type-check:
	$(VENV_BIN)/mypy $(PKG)

mlflow-ui:
	$(VENV_BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db

experiment-backup:
	$(PYTHON) scripts/archive_experiments.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find hydra -mindepth 1 -maxdepth 1 -type d \( -name outputs -o -name multirun \) -exec rm -rf {} +
	rm -rf dist/ build/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf logs/*.log mlruns/ mlartifacts/ mlflow.db

.DEFAULT_GOAL := help
