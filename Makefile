# If environment variable POD_NAME is not set, set it to the user name
POD_NAME ?= $(shell whoami)

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
	@echo "Stopping the playbook..."
	@export POD_NAME=$(POD_NAME) && \
	ansible-playbook check_functions.yml -e "container_state=absent"

CONTAINER_NAME := $(POD_NAME)_dev

###########
# Container Check #
###########

# Check that the podman container named $(POD_NAME)_dev exists
dev-exists:
	@echo "Checking that the container $(CONTAINER_NAME) exists..."
	@podman container exists $(CONTAINER_NAME) || (echo "Container $(CONTAINER_NAME) does not exist. Please create it." && exit 1)
	@echo "Container $(CONTAINER_NAME) exists."

# Check that the podman container named $(POD_NAME)_dev is running
dev-running:
	@echo "Checking that the container $(CONTAINER_NAME) is running..."
	@podman container exists $(CONTAINER_NAME) || (echo "Container $(CONTAINER_NAME) does not exist. Please create it." && exit 1)
	@podman container inspect $(CONTAINER_NAME) | jq -r '.[0].State.Status' | grep running || (echo "Container $(CONTAINER_NAME) is not running. Please start it." && exit 1)
	@echo "Container $(CONTAINER_NAME) is running."

###########
# Quality #
###########

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
