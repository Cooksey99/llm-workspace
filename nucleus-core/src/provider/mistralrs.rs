//! mistral.rs provider implementation.
//!
//! This module provides an in-process LLM provider using mistral.rs.
//! Supports both local GGUF files and automatic HuggingFace downloads.

use super::types::*;
use async_trait::async_trait;
use mistralrs::{
    GgufModelBuilder, IsqType, Model, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};
use std::path::Path;
use std::sync::Arc;

/// mistral.rs in-process provider.
///
/// Automatically detects if model is:
/// 1. A local GGUF file path (loads directly)
/// 2. A HuggingFace model ID (downloads if needed)
///
/// Note: Use async `new()` - model loading requires async operations.
pub struct MistralRsProvider {
    model: Arc<Model>,
    model_name: String,
}

impl MistralRsProvider {
    /// Creates a new mistral.rs provider.
    ///
    /// Downloads and loads the model. This may take time on first use.
    ///
    /// # Model Resolution
    ///
    /// - If `model_name` ends with `.gguf`, treats it as a local file path
    /// - Otherwise, treats it as a HuggingFace model ID (auto-downloads)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use nucleus_core::provider::MistralRsProvider;
    /// # async fn example() -> anyhow::Result<()> {
    /// // HuggingFace model (auto-downloads)
    /// let provider = MistralRsProvider::new("Qwen/Qwen3-0.6B-Instruct").await?;
    ///
    /// // Local GGUF file
    /// let provider = MistralRsProvider::new("./models/qwen3-0.6b.gguf").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(model_name: impl Into<String>) -> Result<Self> {
        let model_name = model_name.into();
        let model = Self::build_model(&model_name).await?;

        Ok(Self {
            model: Arc::new(model),
            model_name,
        })
    }

    async fn build_model(model_name: &str) -> Result<Model> {
        let is_local_gguf = model_name.ends_with(".gguf") && Path::new(model_name).exists();
        
        let model = if is_local_gguf {
            // Extract path and filename to load modal
            let path = Path::new(&model_name);
            let dir = path.parent()
                .ok_or_else(|| ProviderError::Other("Invalid GGUF file path".to_string()))?
                .to_str()
                .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in path".to_string()))?;
            let filename = path.file_name()
                .ok_or_else(|| ProviderError::Other("Invalid GGUF filename".to_string()))?
                .to_str()
                .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in filename".to_string()))?;

            GgufModelBuilder::new(dir, vec![filename])
                .with_logging()
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                .map_err(|e| ProviderError::Other(format!("Failed to configure paged attention: {:?}", e)))?
                .build()
                .await
                .map_err(|e| ProviderError::Other(format!("Failed to load local GGUF '{}': {:?}", model_name, e)))?
        } else {
            // Download from HuggingFace if not cached  
            TextModelBuilder::new(&model_name)
                .with_isq(IsqType::Q4K) // 4-bit quantization
                .with_logging()
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
                .map_err(|e| ProviderError::Other(format!("Failed to configure paged attention: {:?}", e)))?
                .build()
                .await
                .map_err(|e| ProviderError::Other(
                    format!("Failed to load model '{}'. Make sure it exists on HuggingFace or is a valid local .gguf file: {:?}", 
                        model_name, e)
                ))?
        };

        Ok(model)
    }

    
}


#[async_trait]
impl Provider for MistralRsProvider {
    async fn chat<'a>(
        &'a self,
        request: ChatRequest,
        mut callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()> {
        // Convert messages to mistral.rs format
        let mut messages = TextMessages::new();
        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "system" => TextMessageRole::System,
                "user" => TextMessageRole::User,
                "assistant" => TextMessageRole::Assistant,
                _ => TextMessageRole::User,
            };
            messages = messages.add_message(role, &msg.content);
        }

        // Send request and get response
        let response = self.model
            .send_chat_request(messages)
            .await
            .map_err(|e| ProviderError::Other(format!("Chat request failed: {:?}", e)))?;

        // Extract content from first choice
        let content = response.choices
            .first()
            .and_then(|choice| choice.message.content.as_ref())
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Send complete response through callback
        callback(ChatResponse {
            model: self.model_name.clone(),
            content: content.clone(),
            done: true,
            message: Message {
                role: "assistant".to_string(),
                content,
                images: None,
                tool_calls: None,
            },
        });

        Ok(())
    }

    async fn embed(&self, _text: &str, _model: &str) -> Result<Vec<f32>> {
        Err(ProviderError::Other(
            "Embeddings not yet supported for mistral.rs provider".to_string(),
        ))
    }
}
