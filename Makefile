.PHONY: help test test-all test-fast test-unit test-integration test-coverage clean lint format install install-dev

help:
	@echo "MetFish Testing Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          Install package"
	@echo "  make install-dev      Install package with dev dependencies"
	@echo "  make test             Run fast tests (no slow, torch, openfold)"
	@echo "  make test-all         Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make lint             Run linters (ruff, black, codespell)"
	@echo "  make format           Format code with black"
	@echo "  make clean            Clean up generated files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v -m "not slow and not requires_torch and not requires_openfold"

test-all:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

test-coverage:
	pytest tests/ --cov=src/metfish --cov-report=html --cov-report=term-missing
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running ruff..."
	-ruff check src/ tests/
	@echo ""
	@echo "Running black check..."
	-black --check src/ tests/
	@echo ""
	@echo "Running codespell..."
	-codespell src/ tests/ README.md

format:
	black src/ tests/

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete