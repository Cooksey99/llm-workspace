use std::{fs, path::PathBuf};

use crate::config::LlmConfig;

/// Takes a path to the LLM in local storage
///
/// Mutates the `LlmConfig` to include the defaults
pub fn get_llm_defaults(path: PathBuf, llm_config: &mut LlmConfig) {
    // Check for known paths
    if path.is_dir() {};
}
