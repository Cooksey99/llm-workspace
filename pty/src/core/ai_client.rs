use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

const SOCKET_PATH: &str = "/tmp/llm-workspace.sock";

#[derive(Serialize)]
struct Request {
    r#type: String,
    content: String,
}

#[derive(Deserialize)]
struct Response {
    success: bool,
    content: String,
    error: Option<String>,
}

pub struct AiClient;

impl AiClient {
    pub fn send_request(request_type: &str, content: &str) -> Result<String> {
        let mut stream = UnixStream::connect(SOCKET_PATH)
            .context("Failed to connect to AI server. Is it running?")?;

        stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

        let request = Request {
            r#type: request_type.to_string(),
            content: content.to_string(),
        };

        let json = serde_json::to_string(&request)?;
        stream.write_all(json.as_bytes())?;
        stream.write_all(b"\n")?;
        stream.flush()?;

        let mut response_data = String::new();
        stream.read_to_string(&mut response_data)?;

        let response: Response = serde_json::from_str(&response_data)?;

        if response.success {
            Ok(response.content)
        } else {
            Err(anyhow::anyhow!(
                "AI request failed: {}",
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }

    pub fn chat(query: &str) -> Result<String> {
        Self::send_request("chat", query)
    }

    pub fn edit(request: &str) -> Result<String> {
        Self::send_request("edit", request)
    }

    pub fn add_knowledge(content: &str) -> Result<String> {
        Self::send_request("add", content)
    }

    pub fn index_directory(path: &str) -> Result<String> {
        Self::send_request("index", path)
    }

    pub fn stats() -> Result<String> {
        Self::send_request("stats", "")
    }
}
