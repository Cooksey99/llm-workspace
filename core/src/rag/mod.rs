//! Retrieval Augmented Generation (RAG) system.
//!
//! Provides document indexing, embedding generation, and semantic search
//! for context-aware AI responses.

mod embedder;
mod indexer;
mod store;
mod types;

#[allow(unused)]
pub use types::{Document, SearchResult};

use crate::{config::Config, ollama::Client};
use embedder::Embedder;
use indexer::{chunk_text, collect_files};
use store::VectorStore;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("Embedder error: {0}")]
    Embedder(#[from] embedder::EmbedderError),
    
    #[error("Indexer error: {0}")]
    Indexer(#[from] indexer::IndexerError),
    
    #[error("Failed to retrieve context: {0}")]
    Retrieval(String),
}

pub type Result<T> = std::result::Result<T, RagError>;

/// Manages the RAG system including embeddings and vector storage.
#[derive(Clone)]
pub struct Manager {
    embedder: Embedder,
    store: VectorStore,
    chunk_size: usize,
    chunk_overlap: usize,
    top_k: usize,
}

impl Manager {
    /// Creates a new RAG manager.
    pub fn new(config: &Config, ollama_client: Client) -> Self {
        let embedder = Embedder::new(ollama_client, &config.rag.embedding_model);
        let store = VectorStore::new();
        
        Self {
            embedder,
            store,
            chunk_size: config.rag.chunk_size,
            chunk_overlap: config.rag.chunk_overlap,
            top_k: config.rag.top_k,
        }
    }
    
    /// Adds a single piece of text to the knowledge base.
    pub async fn add_knowledge(&self, content: &str, source: &str) -> Result<()> {
        let embedding = self.embedder.embed(content).await?;
        
        let id = format!("{}_{}", source, self.store.count());
        let document = Document::new(id, content, embedding)
            .with_metadata("source", source);
        
        self.store.add(document);
        Ok(())
    }
    
    /// Indexes all files in a directory.
    pub async fn index_directory(&self, dir_path: &str) -> Result<usize> {
        let files = collect_files(dir_path).await?;
        let mut indexed_count = 0;
        
        for file in files {
            let chunks = chunk_text(&file.content, self.chunk_size, self.chunk_overlap);
            
            for (i, chunk) in chunks.into_iter().enumerate() {
                let embedding = self.embedder.embed(&chunk).await?;
                
                let id = format!("{}_chunk_{}", file.path.display(), i);
                let document = Document::new(id, chunk, embedding)
                    .with_metadata("source", file.path.to_string_lossy())
                    .with_metadata("chunk", i.to_string());
                
                self.store.add(document);
            }
            
            indexed_count += 1;
            println!("✓ Indexed: {}", file.path.display());
        }
        
        Ok(indexed_count)
    }
    
    /// Retrieves relevant context for a query.
    pub async fn retrieve_context(&self, query: &str) -> Result<String> {
        if self.store.count() == 0 {
            return Ok(String::new());
        }
        
        let query_embedding = self.embedder.embed(query).await?;
        let results = self.store.search(&query_embedding, self.top_k);
        
        if results.is_empty() {
            return Ok(String::new());
        }
        
        let mut context = String::from("\n\nRelevant context from your knowledge base:\n");
        
        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!("\n[{}] {}\n", i + 1, result.document.content));
        }
        
        Ok(context)
    }
    
    /// Returns the total number of documents in the knowledge base.
    pub fn count(&self) -> usize {
        self.store.count()
    }
    
    /// Clears all documents from the knowledge base.
    pub fn clear(&self) {
        self.store.clear();
    }
}
