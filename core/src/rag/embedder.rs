use crate::ollama::{Client, EmbedRequest};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("Ollama error: {0}")]
    Ollama(#[from] crate::ollama::OllamaError),
    
    #[error("No embeddings returned")]
    NoEmbeddings,
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

/// Generates embeddings using Ollama.
#[derive(Clone)]
pub struct Embedder {
    client: Client,
    model: String,
}

impl Embedder {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
    
    /// Generates an embedding for the given text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };
        
        let response = self.client.embed(request).await?;
        
        response.embeddings
            .into_iter()
            .next()
            .ok_or(EmbedderError::NoEmbeddings)
    }
}
