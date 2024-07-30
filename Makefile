# Load environment variables from .env file
ifneq ("$(wildcard .env)","")
include .env
else
$(warning .env file not found - using default values)
endif

#######
# Run #
#######

run:
	mkdir -p htmlcov
	ansible-playbook check_functions.yml \
	-e "container_state=started" \
	-e "use_gpus=true"

run-dev:
	mkdir -p htmlcov
	ansible-playbook check_functions.yml -e "container_state=started" -e "use_gpus=false" -e "start_dev_only=true"

stop:
	@export POD_NAME=$(POD_NAME) && \
	ansible-playbook check_functions.yml -e "container_state=absent"

###########
# Quality #
###########

CONTAINER_NAME := $(POD_NAME)_dev

# Documentation
## Build
build-docs:
	podman exec $(CONTAINER_NAME) sphinx-build docs/source docs/build

# Code quality
## Check format
check-format-code:
	podman exec $(CONTAINER_NAME) ruff check nirmatai_sdk

## Format
format-code:
	podman exec $(CONTAINER_NAME) ruff check nirmatai_sdk --fix

## Type check
type-check:
	podman exec $(CONTAINER_NAME) mypy nirmatai_sdk --ignore-missing-imports

# Format and type check
check: format-code type-check

# Test
test:
	podman exec $(CONTAINER_NAME) python -m pytest -k "not test_core_integration" nirmatai_sdk

test-coverage:
	podman exec $(CONTAINER_NAME) python -m pytest -k "not test_core_integration" nirmatai_sdk --cov --cov-report=html

#########
# Setup #
#########
verify-software:
	@echo "The shell being used is:"
	@echo $(shell echo $$SHELL)
	@echo "Checking if podman is installed..."
	podman --version
	@echo "Checking if Python is installed..."
	python --version

install-precommit:
	pip install pre-commit
	pre-commit install

setup: verify-software install-precommit
	@echo "You are ready to go!"
