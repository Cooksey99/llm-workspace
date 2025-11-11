package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"llm-workspace/config"
	"llm-workspace/fileops"
	"llm-workspace/rag"

	"github.com/ollama/ollama/api"
)

const socketPath = "/tmp/llm-workspace.sock"

type Request struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type Response struct {
	Success bool   `json:"success"`
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

type Server struct {
	cfg         *config.Config
	client      *api.Client
	ragManager  *rag.Manager
	fileManager *fileops.Manager
	ctx         context.Context
}

func NewServer() (*Server, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	ragManager, err := rag.NewManager(cfg, client)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RAG: %w", err)
	}

	fileManager := fileops.NewManager(cfg, client, ragManager)

	return &Server{
		cfg:         cfg,
		client:      client,
		ragManager:  ragManager,
		fileManager: fileManager,
		ctx:         context.Background(),
	}, nil
}

func (s *Server) handleRequest(req Request) Response {
	switch req.Type {
	case "chat":
		response, err := s.fileManager.Chat(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: response}

	case "edit":
		response, err := s.fileManager.ChatWithTools(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: response}

	case "add":
		err := s.ragManager.AddKnowledge(s.ctx, req.Content, "user_input")
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: "Added to knowledge base"}

	case "index":
		err := s.ragManager.IndexDirectory(s.ctx, req.Content)
		if err != nil {
			return Response{Success: false, Error: err.Error()}
		}
		return Response{Success: true, Content: fmt.Sprintf("Indexed directory: %s", req.Content)}

	case "stats":
		count := s.ragManager.Count()
		return Response{
			Success: true,
			Content: fmt.Sprintf("Knowledge base contains %d documents", count),
		}

	default:
		return Response{Success: false, Error: "unknown request type"}
	}
}

func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)
	encoder := json.NewEncoder(conn)

	var req Request
	if err := decoder.Decode(&req); err != nil {
		log.Printf("Failed to decode request: %v", err)
		return
	}

	resp := s.handleRequest(req)

	if err := encoder.Encode(resp); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func (s *Server) Start() error {
	if err := os.RemoveAll(socketPath); err != nil {
		return fmt.Errorf("failed to remove old socket: %w", err)
	}

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		return fmt.Errorf("failed to create socket: %w", err)
	}
	defer listener.Close()

	if err := os.Chmod(socketPath, 0600); err != nil {
		return fmt.Errorf("failed to set socket permissions: %w", err)
	}

	log.Printf("AI Server listening on %s", socketPath)
	log.Printf("Model: %s", s.cfg.LLM.Model)
	log.Printf("Knowledge Base: %d documents", s.ragManager.Count())

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down...")
		listener.Close()
		os.RemoveAll(socketPath)
		os.Exit(0)
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			if strings.Contains(err.Error(), "use of closed network connection") {
				break
			}
			log.Printf("Accept error: %v", err)
			continue
		}

		go s.handleConnection(conn)
	}

	return nil
}

func runInteractive() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create Ollama client: %w", err)
	}

	ragManager, err := rag.NewManager(cfg, client)
	if err != nil {
		return fmt.Errorf("failed to initialize RAG: %w", err)
	}

	fileManager := fileops.NewManager(cfg, client, ragManager)

	fmt.Println("Local LLM with RAG Ready!")
	fmt.Printf("Model: %s\n", cfg.LLM.Model)
	fmt.Printf("Knowledge Base: %d documents\n", ragManager.Count())
	fmt.Println("\nCommands:")
	fmt.Println("  /add <text>       - Add knowledge to vector DB")
	fmt.Println("  /index <path>     - Index a directory (code files)")
	fmt.Println("  /edit <request>   - Enable file editing mode")
	fmt.Println("  /stats            - Show knowledge base stats")
	fmt.Println("  /quit             - Exit")
	fmt.Println("\nType your message:")

	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if input == "/quit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "/stats" {
			fmt.Printf("Knowledge base contains %d documents\n", ragManager.Count())
			continue
		}

		if strings.HasPrefix(input, "/add ") {
			content := strings.TrimPrefix(input, "/add ")
			err := ragManager.AddKnowledge(ctx, content, "user_input")
			if err != nil {
				fmt.Printf("Error adding knowledge: %v\n", err)
			} else {
				fmt.Println("Added to knowledge base")
			}
			continue
		}

		if strings.HasPrefix(input, "/index ") {
			dirPath := strings.TrimPrefix(input, "/index ")
			fmt.Printf("Indexing directory: %s\n", dirPath)
			err := ragManager.IndexDirectory(ctx, dirPath)
			if err != nil {
				fmt.Printf("Error indexing: %v\n", err)
			}
			continue
		}

		if strings.HasPrefix(input, "/edit ") {
			request := strings.TrimPrefix(input, "/edit ")
			response, err := fileManager.ChatWithTools(ctx, request)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else if response != "" {
				fmt.Printf("\n%s\n", response)
			}
			continue
		}

		response, err := fileManager.Chat(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\n%s\n", response)
	}

	return nil
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "interactive" {
		if err := runInteractive(); err != nil {
			log.Fatal(err)
		}
		return
	}

	pidFile := "/tmp/llm-workspace.pid"
	pid := fmt.Sprintf("%d", os.Getpid())
	if err := os.WriteFile(pidFile, []byte(pid), 0644); err != nil {
		log.Printf("Warning: could not write PID file: %v", err)
	}
	defer os.Remove(pidFile)

	server, err := NewServer()
	if err != nil {
		os.Remove(pidFile)
		log.Fatal(err)
	}

	if err := server.Start(); err != nil {
		os.Remove(pidFile)
		log.Fatal(err)
	}
}
