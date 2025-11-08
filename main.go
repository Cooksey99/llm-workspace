package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	chromem "github.com/philippgille/chromem-go"
	"gopkg.in/yaml.v3"
)

type Config struct {
	LLM struct {
		Model         string  `yaml:"model"`
		BaseURL       string  `yaml:"base_url"`
		Temperature   float64 `yaml:"temperature"`
		ContextLength int     `yaml:"context_length"`
	} `yaml:"llm"`
	SystemPrompt     string `yaml:"system_prompt"`
	RAG              RAGConfig
	Storage          StorageConfig
	Personalization  PersonalizationConfig
}

type RAGConfig struct {
	EmbeddingModel string `yaml:"embedding_model"`
	ChunkSize      int    `yaml:"chunk_size"`
	ChunkOverlap   int    `yaml:"chunk_overlap"`
	TopK           int    `yaml:"top_k"`
}

type StorageConfig struct {
	VectorDBPath     string `yaml:"vector_db_path"`
	ChatHistoryPath  string `yaml:"chat_history_path"`
}

type PersonalizationConfig struct {
	LearnFromInteractions bool   `yaml:"learn_from_interactions"`
	SaveConversations     bool   `yaml:"save_conversations"`
	UserPreferencesPath   string `yaml:"user_preferences_path"`
}

type LLMApp struct {
	config Config
	client *api.Client
	db     *chromem.DB
}

func loadConfig() (Config, error) {
	var config Config
	data, err := os.ReadFile("config.yaml")
	if err != nil {
		return config, err
	}
	err = yaml.Unmarshal(data, &config)
	return config, err
}

func NewLLMApp() (*LLMApp, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	os.MkdirAll(config.Storage.VectorDBPath, 0755)
	os.MkdirAll(config.Storage.ChatHistoryPath, 0755)
	
	db := chromem.NewDB()

	return &LLMApp{
		config: config,
		client: client,
		db:     db,
	}, nil
}

func (app *LLMApp) Chat(ctx context.Context, userMessage string) (string, error) {
	messages := []api.Message{
		{
			Role:    "system",
			Content: app.config.SystemPrompt,
		},
		{
			Role:    "user",
			Content: userMessage,
		},
	}

	req := &api.ChatRequest{
		Model:    app.config.LLM.Model,
		Messages: messages,
		Options: map[string]any{
			"temperature": app.config.LLM.Temperature,
		},
	}

	var response strings.Builder
	err := app.client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("chat failed: %w", err)
	}

	return response.String(), nil
}

func (app *LLMApp) AddKnowledge(ctx context.Context, content, metadata string) error {
	collection, err := app.db.GetOrCreateCollection("knowledge", nil, nil)
	if err != nil {
		return err
	}

	err = collection.AddDocument(ctx, chromem.Document{
		ID:       fmt.Sprintf("doc_%d", collection.Count()),
		Content:  content,
		Metadata: map[string]string{"source": metadata},
	})
	
	return err
}

func main() {
	app, err := NewLLMApp()
	if err != nil {
		log.Fatalf("Failed to initialize app: %v", err)
	}

	fmt.Println("Local LLM Ready!")
	fmt.Printf("Model: %s\n", app.config.LLM.Model)
	fmt.Println("\nCommands:")
	fmt.Println("  /add <text>  - Add knowledge to vector DB")
	fmt.Println("  /quit        - Exit")
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

		if strings.HasPrefix(input, "/add ") {
			content := strings.TrimPrefix(input, "/add ")
			err := app.AddKnowledge(ctx, content, "user_input")
			if err != nil {
				fmt.Printf("Error adding knowledge: %v\n", err)
			} else {
				fmt.Println("Added to knowledge base")
			}
			continue
		}

		response, err := app.Chat(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\n%s\n", response)
	}
}
