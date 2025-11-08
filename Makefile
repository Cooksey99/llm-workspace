.PHONY: help setup install-deps pull-model run build clean test

help:
	@echo "Available commands:"
	@echo "  make setup        - Complete first-time setup"
	@echo "  make install-deps - Install Go dependencies"
	@echo "  make pull-model   - Pull the LLM model"
	@echo "  make run          - Run the application"
	@echo "  make build        - Build binary"
	@echo "  make clean        - Remove built files and data"
	@echo "  make test         - Run tests"

setup: check-ollama install-deps pull-model
	@echo "Setup complete! Run 'make run' to start."

check-ollama:
	@which ollama > /dev/null || (echo "Ollama not found. Install with: brew install ollama" && exit 1)
	@echo "Ollama found"
	@pgrep -x ollama > /dev/null || (echo "🔧 Starting Ollama..." && ollama serve > /dev/null 2>&1 &)
	@sleep 2

install-deps:
	@echo "Installing Go dependencies..."
	@test -f go.mod || go mod init llm-workspace
	@go get github.com/ollama/ollama/api
	@go get github.com/philippgille/chromem-go
	@go get gopkg.in/yaml.v3
	@go mod tidy
	@echo "Dependencies installed"

pull-model:
	@echo "Pulling Qwen 2.5 Coder 14B model..."
	@ollama pull qwen2.5-coder:14b
	@echo "Model ready"

run:
	@go run main.go

build:
	@echo "Building binary..."
	@go build -o llm-app main.go
	@echo "Built: ./llm-app"

clean:
	@echo "Cleaning..."
	@rm -f llm-app
	@rm -rf data/
	@echo "Clean complete"

test:
	@go test ./...
