mod config;
mod ollama;
mod rag;
mod server;

#[tokio::main]
async fn main() {
    let cfg = match config::Config::load_default() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    println!("Model: {}", cfg.llm.model);
    let server = server::Server::new(cfg);
    
    if let Err(e) = server.start().await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
