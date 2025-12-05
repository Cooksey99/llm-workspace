//! Vector store abstraction and factory.
//!
//! This module provides a unified interface for different vector database implementations.

use super::types::{Document, SearchResult};
use super::qdrant_store::QdrantStore;
use super::lancedb_store::LanceDbStore;
use crate::config::StorageMode;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Unified interface for vector database operations.
///
/// Implementations handle document storage, similarity search, and metadata queries
/// across different vector database backends (LanceDB for embedded, Qdrant for gRPC).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds or updates a document in the store.
    async fn add(&self, document: Document) -> Result<()>;

    /// Searches for the most similar documents using vector similarity.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The embedding vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by descending similarity score.
    async fn search(&self, query_embedding: &[f32], top_k: u64) -> Result<Vec<SearchResult>>;

    /// Returns the total number of documents in the store.
    async fn count(&self) -> Result<usize>;

    /// Removes all documents from the store.
    async fn clear(&self) -> Result<()>;

    /// Returns all unique source file paths that have been indexed.
    async fn get_indexed_paths(&self) -> Result<Vec<String>>;

    /// Removes all documents with a matching source path.
    ///
    /// # Arguments
    ///
    /// * `source_path` - The source path to remove (file or directory)
    ///
    /// # Returns
    ///
    /// The number of documents removed.
    async fn remove_by_source(&self, source_path: &str) -> Result<usize>;
}

/// Creates a vector store instance based on the storage mode.
///
/// - `Embedded` mode uses LanceDB for zero-setup, in-process storage
/// - `Grpc` mode uses Qdrant for remote server connectivity
///
/// # Arguments
///
/// * `storage_mode` - Storage configuration (embedded with path or gRPC with URL)
/// * `collection_name` - Name of the collection/table to use
/// * `vector_size` - Dimension of the embedding vectors
///
/// # Returns
///
/// A trait object that can be used for all vector store operations.
pub async fn create_vector_store(
    storage_mode: &StorageMode,
    collection_name: &str,
    vector_size: u64,
) -> Result<Arc<dyn VectorStore>> {
    match storage_mode {
        StorageMode::Embedded { path } => {
            let store = LanceDbStore::new(path, collection_name, vector_size).await?;
            Ok(Arc::new(store))
        }
        StorageMode::Grpc { url } => {
            let grpc_mode = StorageMode::Grpc { url: url.clone() };
            let store = QdrantStore::new(&grpc_mode, collection_name, vector_size).await?;
            Ok(Arc::new(store))
        }
    }
}
