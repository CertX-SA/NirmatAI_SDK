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

###############
# Private-GPT #
###############
FILE := privateGPT/version.txt
VERSION :=$(file < $(FILE))

# Gets rid of the .git folder to avoid VsCode from trying to index it
login:
	podman login -u $(DOCKER_USERNAME) -p $(DOCKER_PASSWORD)

clone-privateGPT:
	git clone https://github.com/imartinez/privateGPT

build-image:
	cd privateGPT && \
	podman build -t $(DOCKER_USERNAME)/privategpt:$(VERSION) -f Dockerfile.external .

push:
	podman push $(DOCKER_USERNAME)/privategpt:$(VERSION)

build-dev-and-client:
	podman build -t $(DOCKER_USERNAME)/dev:0.0.3 -f Dockerfile.dev .
	podman build -t $(DOCKER_USERNAME)/client:0.0.3 -f Dockerfile.client .

build-push: clone-privateGPT login build-image push

push-dev-and-client:
	podman push $(DOCKER_USERNAME)/dev:0.0.3
	podman push $(DOCKER_USERNAME)/client:0.0.3

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
