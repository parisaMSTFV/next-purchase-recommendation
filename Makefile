.PHONY: install run test lint sensitive check

install:
	python -m pip install -e ".[dev]"

run:
	python -m next_purchase.cli run --project-root .

test:
	python -m unittest discover -s tests -v

lint:
	python -m ruff check .

sensitive:
	python scripts/check_sensitive.py

check: lint test sensitive
