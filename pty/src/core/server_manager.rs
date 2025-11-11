use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

const PID_FILE: &str = "/tmp/llm-workspace.pid";
const SOCKET_PATH: &str = "/tmp/llm-workspace.sock";
const MAX_STARTUP_WAIT: u64 = 10;
const AI_BINARY_NAME: &str = "llm-server";

pub struct ServerManager;

impl ServerManager {
    fn get_project_root() -> Result<PathBuf> {
        std::env::current_dir().context("Failed to get current directory")
    }

    fn is_process_running(pid: u32) -> bool {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    fn read_pid() -> Option<u32> {
        fs::read_to_string(PID_FILE)
            .ok()
            .and_then(|content| content.trim().parse().ok())
    }

    pub fn is_server_running() -> bool {
        if let Some(pid) = Self::read_pid() {
            if Self::is_process_running(pid) {
                return std::path::Path::new(SOCKET_PATH).exists();
            }
        }
        false
    }

    pub fn start_server() -> Result<()> {
        if Self::is_server_running() {
            return Ok(());
        }

        fs::remove_file(SOCKET_PATH).ok();
        fs::remove_file(PID_FILE).ok();

        let project_root = Self::get_project_root()?;
        let server_path = project_root.join("ai");
        
        if !server_path.exists() {
            return Err(anyhow::anyhow!(
                "AI server directory not found at {:?}. Make sure you're running from the project root.",
                server_path
            ));
        }

        let binary_path = project_root.join(AI_BINARY_NAME);
        
        if !binary_path.exists() {
            return Err(anyhow::anyhow!(
                "AI server binary not found at {:?}. Run 'make build' first.",
                binary_path
            ));
        }

        let log_file = fs::File::create("/tmp/llm-workspace.log")
            .context("Failed to create log file")?;
        
        Command::new(&binary_path)
            .current_dir(&server_path)
            .stdout(Stdio::from(log_file.try_clone()?))
            .stderr(Stdio::from(log_file))
            .spawn()
            .context("Failed to start AI server")?;

        for _ in 0..(MAX_STARTUP_WAIT * 10) {
            if Self::is_server_running() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        Err(anyhow::anyhow!("AI server failed to start within timeout"))
    }

    pub fn stop_server() -> Result<()> {
        if let Some(pid) = Self::read_pid() {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            for _ in 0..50 {
                if !Self::is_process_running(pid) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            if Self::is_process_running(pid) {
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                }
            }
        }

        fs::remove_file(SOCKET_PATH).ok();
        fs::remove_file(PID_FILE).ok();

        Ok(())
    }

    pub fn ensure_server() -> Result<()> {
        if !Self::is_server_running() {
            eprintln!("Starting AI server...");
            Self::start_server()?;
            eprintln!("AI server started successfully");
        }
        Ok(())
    }
}

impl Drop for ServerManager {
    fn drop(&mut self) {
        let _ = Self::stop_server();
    }
}
