# Default make command
all: help

## build: Build the extension
build:
	cargo build

## build-production: Build the extension for production
build-production:
	cargo build --release

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