use super::ai_client::AiClient;
use anyhow::Result;

pub enum Command {
    AiChat(String),
    AiEdit(String),
    AiStats,
    AiAdd(String),
    AiIndex(String),
    PassThrough,
}

impl Command {
    pub fn parse(line: &str) -> Self {
        let trimmed = line.trim();
        
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        
        match parts.as_slice() {
            ["/ai", rest] => Command::AiChat(rest.to_string()),
            ["/edit", rest] => Command::AiEdit(rest.to_string()),
            ["/add", rest] => Command::AiAdd(rest.to_string()),
            ["/index", rest] => Command::AiIndex(rest.to_string()),
            ["/stats"] => Command::AiStats,
            _ => Command::PassThrough,
        }
    }
    
    pub fn execute(&self) -> Result<Option<String>> {
        match self {
            Command::AiChat(query) => {
                let response = AiClient::chat(query)?;
                Ok(Some(response))
            }
            Command::AiEdit(request) => {
                let response = AiClient::edit(request)?;
                Ok(Some(response))
            }
            Command::AiStats => {
                let response = AiClient::stats()?;
                Ok(Some(response))
            }
            Command::AiAdd(content) => {
                let response = AiClient::add_knowledge(content)?;
                Ok(Some(response))
            }
            Command::AiIndex(path) => {
                let response = AiClient::index_directory(path)?;
                Ok(Some(response))
            }
            Command::PassThrough => Ok(None),
        }
    }
}
