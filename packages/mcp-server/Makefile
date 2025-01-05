include .env

# Variables
PACKAGE_NAME := mcp-pinecone

# Colors for better visibility
CYAN := \033[36m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# Default make command
all: help

## reinstall-deps: Reinstall dependencies with uv
reinstall-deps:
	uv sync --reinstall

## lint: Lint the code
lint:
	uv run ruff check .

## build: Build the package
build:
	uv build

## lock-upgrade: Lock dependencies to the latest version
lock-upgrade:
	uv lock --upgrade

## publish: Publish the package to PyPI
publish:
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "$(RED)Error: PYPI_TOKEN is not set$(RESET)"; \
		exit 1; \
	fi
	uv publish --username __token__ --password ${PYPI_TOKEN}

## release: Create and push a new release tag
release:
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Error: VERSION is required. Use 'make release VERSION=x.x.x'$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Creating release v$(VERSION)...$(RESET)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
	@echo "\nRelease v$(VERSION) created!"
	@echo "Users can install with:"
	@echo "  uvx install github:sirmews/$(PACKAGE_NAME)@v$(VERSION)"
	@echo "  uv pip install git+https://github.com/sirmews/$(PACKAGE_NAME).git@v$(VERSION)"

## inspect-local-server: Inspect the local MCP server
inspect-local-server:
	npx @modelcontextprotocol/inspector uv --directory . run $(PACKAGE_NAME)

## help: Show a list of commands
help : Makefile
	@echo "Usage:"
	@echo "  make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z\-_0-9%:\\]+/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = $$1; \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			gsub("\\\\", "", helpCommand); \
			gsub(":+$$", "", helpCommand); \
			printf "  $(CYAN)%-20s$(RESET) %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)


.PHONY: all help