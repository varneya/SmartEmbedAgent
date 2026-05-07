# SmartEmbedAgent — common developer commands.
.PHONY: help install install-dev test smoke run eval clean format lint

help:
	@echo "SmartEmbedAgent — common developer commands"
	@echo ""
	@echo "  make install      Install runtime dependencies"
	@echo "  make install-dev  Install runtime + dev dependencies (lint, format, test)"
	@echo "  make test         Run the full test suite"
	@echo "  make smoke        Quick end-to-end smoke test (no API key needed)"
	@echo "  make run          Run main.py against the general sample corpus"
	@echo "  make eval         Run the evaluation harness against the benchmark corpus"
	@echo "  make format       Run black on src/ tests/ main.py"
	@echo "  make lint         Run ruff on src/ tests/ main.py"
	@echo "  make clean        Remove caches, runs/, *.pyc"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black ruff pytest

test:
	python -m unittest discover tests -v

smoke:
	python scripts/smoke_test.py

run:
	python main.py \
		--corpus_path data/sample_general.txt \
		--config_path config/sample_config.json \
		--output_path runs/recommendation.json \
		--verbose

eval:
	python -m evals.comparison_baseline evals/benchmark_corpora/general_qa.json

format:
	black src tests main.py evals --line-length 100

lint:
	ruff check src tests main.py evals

clean:
	rm -rf .cache runs __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
