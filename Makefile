.PHONY: install run score-example test lint sensitive check

install:
	python -m pip install -e ".[dev]"

run:
	python -m next_purchase.cli run --project-root .

score-example:
	python -m next_purchase.cli score --transactions examples/supplied_transactions_v1.csv --provenance examples/supplied_provenance_v1.json --score-date 2025-06-01 --output-root artifacts/supplied-example

test:
	python -m unittest discover -s tests -v

lint:
	python -m ruff check .

sensitive:
	python scripts/check_sensitive.py

check: lint test sensitive
