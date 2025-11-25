use std::collections::HashMap;

/// A document stored in the vector database.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique identifier for the document
    pub id: String,
    
    /// The text content
    pub content: String,
    
    /// Vector embedding of the content
    pub embedding: Vec<f32>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            embedding,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A search result with similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub document: Document,
    pub score: f32,
}
