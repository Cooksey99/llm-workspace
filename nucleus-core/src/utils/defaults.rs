use std::{fs, path::PathBuf};

use crate::config::LlmConfig;
use serde::Deserialize;
use tracing::{debug, warn};

#[derive(Debug, Deserialize)]
struct HuggingFaceConfig {
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

/// Takes a path to the LLM in local storage
///
/// Mutates the `LlmConfig` to include the defaults from HuggingFace config.json
pub fn get_llm_defaults(path: PathBuf, llm_config: &mut LlmConfig) {
    if !path.exists() {
        warn!("Model path does not exist: {}", path.display());
        return;
    }

    let config_path = if path.is_file() {
        if let Some(parent) = path.parent() {
            parent.join("config.json")
        } else {
            return;
        }
    } else {
        path.join("config.json")
    };

    if !config_path.exists() {
        debug!("No config.json found at {}", config_path.display());
        return;
    }

    match fs::read_to_string(&config_path) {
        Ok(contents) => {
            if let Err(e) = parse_config_json(&contents, llm_config) {
                warn!("Failed to parse config.json: {}", e);
            } else {
                debug!("Successfully loaded defaults from config.json");
            }
        }
        Err(e) => warn!("Failed to read config.json: {}", e),
    }
}

fn parse_config_json(
    contents: &str,
    llm_config: &mut LlmConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let config: HuggingFaceConfig = serde_json::from_str(contents)?;

    if let Some(max_pos) = config.max_position_embeddings {
        llm_config.context_length = max_pos;
        debug!("Set context_length to {} from config.json", max_pos);
    }

    Ok(())
}
