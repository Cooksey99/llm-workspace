# LLM Workspace

AI-powered terminal with PTY wrapper and RAG capabilities.

## Features

- **Interactive AI** in your terminal via Unix socket
- **RAG (Retrieval Augmented Generation)** for code and knowledge
- **File Operations** with AI assistance
- **Command Interception** in PTY for seamless AI interaction

## Build

```bash
make build
```

This builds:
- `llm-server` - Go AI server (connects to Ollama)
- `llm-workspace` - Rust PTY wrapper

## Run

```bash
./llm-workspace
```

Or:

```bash
make run
```

The PTY will automatically start the AI server in the background.

## AI Commands

Once in the PTY, use these commands:

- `/ai <question>` - Chat with AI
- `/edit <request>` - Use AI with file editing capabilities
- `/add <text>` - Add knowledge to vector database
- `/index <path>` - Index a directory for RAG
- `/stats` - Show knowledge base statistics

## Standalone AI Server

Run the AI server in interactive mode:

```bash
go run ai/server.go interactive
```

## Install

Install to `/usr/local/bin`:

```bash
make install
```

Then run from anywhere:

```bash
llm-workspace
```

## Configuration

Edit `ai/config.yaml` to configure:
- Model selection
- RAG settings
- File operation preferences

## Development

```bash
make dev    # Quick dev build
make clean  # Clean all artifacts
```

## Logs

Server logs: `/tmp/llm-workspace.log`
