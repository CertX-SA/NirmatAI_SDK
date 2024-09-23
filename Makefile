# Set POD_NAME to the username if not already set
POD_NAME ?= $(shell whoami)

###########
# Commands#
###########

# Create directories and run the Ansible playbook with the specified environment variables
run:
	mkdir -p htmlcov
	ansible-playbook check_functions.yml \
	-e "container_state=started" \
	-e "use_gpus=true"

run-dev:
	mkdir -p htmlcov
	ansible-playbook check_functions.yml \
	-e "container_state=started" \
	-e "use_gpus=false" \
	-e "start_dev_only=true"

# Stop the Ansible playbook and set the container state to 'absent'
stop:
	@echo "Stopping the playbook..."
	@export POD_NAME=$(POD_NAME) && \
	ansible-playbook check_functions.yml \
	-e "container_state=absent"

# Check that the podman container named $(POD_NAME)_dev exists
dev-exists:
	@echo "Checking that the container $(POD_NAME)_dev exists..."
	@podman container exists $(POD_NAME)_dev || (echo "Container $(POD_NAME)_dev does not exist. Please create it." && exit 1)
	@echo "Container $(POD_NAME)_dev exists."

# Check that the podman container named $(POD_NAME)_dev is running
dev-running:
	@echo "Checking that the container $(POD_NAME)_dev is running..."
	@podman container exists $(POD_NAME)_dev || (echo "Container $(POD_NAME)_dev does not exist. Please create it." && exit 1)
	@podman container inspect $(POD_NAME)_dev | jq -r '.[0].State.Status' | grep running || (echo "Container $(POD_NAME)_dev is not running. Please start it." && exit 1)
	@echo "Container $(POD_NAME)_dev is running."

# Other checks run as pre-commit hooks and in github actions
ansible-lint:
	@echo "Running ansible-lint..."
	@podman exec $(POD_NAME)_dev ansible-lint check_functions.yaml

format-check:
	@echo "Running ruff-lint..."
	@podman exec $(POD_NAME)_dev ruff check

type-check:
	@echo "Running mypy..."
	@podman exec $(POD_NAME)_dev mypy .

build-docs:
	@echo "Building docs..."
	@podman exec $(POD_NAME)_dev sphinx-build docs/source docs/build

pre-commit:
	@echo "Running pre-commit..."
	pre-commit run --all-files
