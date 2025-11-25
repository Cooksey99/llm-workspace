use super::types::{Document, SearchResult};
use std::sync::{Arc, RwLock};

/// In-memory vector store using cosine similarity.
#[derive(Clone)]
pub struct VectorStore {
    documents: Arc<RwLock<Vec<Document>>>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Adds a document to the store.
    pub fn add(&self, document: Document) {
        let mut docs = self.documents.write().unwrap();
        docs.push(document);
    }
    
    /// Searches for similar documents using cosine similarity.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let docs = self.documents.read().unwrap();
        
        let mut results: Vec<SearchResult> = docs
            .iter()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        results.into_iter().take(top_k).collect()
    }
    
    /// Returns the total number of documents.
    pub fn count(&self) -> usize {
        self.documents.read().unwrap().len()
    }
    
    /// Clears all documents.
    pub fn clear(&self) {
        self.documents.write().unwrap().clear();
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (magnitude_a * magnitude_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    
    #[test]
    fn test_vector_store() {
        let store = VectorStore::new();
        
        let doc = Document::new("1", "test", vec![1.0, 0.0, 0.0]);
        store.add(doc);
        
        assert_eq!(store.count(), 1);
        
        let results = store.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 1.0);
    }
}
