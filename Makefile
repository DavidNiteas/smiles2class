.PHONY: install test build clean

install:
	python -m pip install -e ".[dev,vice]"

test:
	PYTHONPATH=src:. python -m pytest

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
