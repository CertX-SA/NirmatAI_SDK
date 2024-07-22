# Makefile

.PHONY: format-code check-format-code type-check test build-docs install-hooks ensure-pre-commit

format-code:
 black .

check-format-code:
 black --check .

type-check:
 mypy NirmatAI tests

test:
 pytest

build-docs:
 # Add your documentation build commands here
 echo "Building docs..."

install-hooks: ensure-pre-commit
 pre-commit install --hook-type pre-commit
 pre-commit install --hook-type pre-push

ensure-pre-commit:
 @if ! command -v pre-commit &> /dev/null; then \
  echo "pre-commit not found, installing..."; \
  pip install pre-commit; \
 fi