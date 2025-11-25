use std::path::{Path, PathBuf};
use tokio::fs;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, IndexerError>;

/// Splits text into overlapping chunks.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let end = (start + chunk_size).min(text.len());
        chunks.push(text[start..end].to_string());
        
        if end == text.len() {
            break;
        }
        
        start += chunk_size - overlap;
    }
    
    chunks
}

/// File to be indexed with its content and metadata.
#[derive(Debug, Clone)]
pub struct IndexedFile {
    pub path: PathBuf,
    pub content: String,
}

/// Recursively collects indexable files from a directory.
pub async fn collect_files(dir_path: impl AsRef<Path>) -> Result<Vec<IndexedFile>> {
    let mut files = Vec::new();
    collect_files_recursive(dir_path.as_ref(), &mut files).await?;
    Ok(files)
}

fn collect_files_recursive<'a>(dir: &'a Path, files: &'a mut Vec<IndexedFile>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
    let mut entries = fs::read_dir(dir).await?;
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        
        if path.is_dir() {
            collect_files_recursive(&path, files).await?;
        } else if is_indexable(&path) {
            if let Ok(content) = fs::read_to_string(&path).await {
                files.push(IndexedFile {
                    path: path.clone(),
                    content,
                });
            }
        }
    }
    
    Ok(())
    })
}

/// Checks if a file should be indexed based on extension.
fn is_indexable(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        matches!(
            ext.to_str(),
            Some("rs" | "go" | "py" | "js" | "ts" | "tsx" | "jsx" | "md" | "txt")
        )
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_text_small() {
        let text = "Hello";
        let chunks = chunk_text(text, 10, 2);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello");
    }
    
    #[test]
    fn test_chunk_text_with_overlap() {
        let text = "0123456789ABCDEF";
        let chunks = chunk_text(text, 10, 2);
        
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "0123456789");
        assert_eq!(chunks[1], "89ABCDEF");
    }
    
    #[test]
    fn test_is_indexable() {
        assert!(is_indexable(Path::new("test.rs")));
        assert!(is_indexable(Path::new("test.md")));
        assert!(!is_indexable(Path::new("test.exe")));
        assert!(!is_indexable(Path::new("test")));
    }
}
