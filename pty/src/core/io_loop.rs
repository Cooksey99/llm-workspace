use std::io::{Read, Write};
use anyhow::Result;
use super::command::Command;

pub fn run_io_loop(master: &mut Box<dyn portable_pty::MasterPty + Send>) -> Result<()> {
    use std::io::{stdin, stdout};
    
    let mut stdin = stdin();
    let mut stdout = stdout();
    
    let mut reader = master.try_clone_reader()?;
    let mut writer = master.take_writer()?;
    
    let mut stdin_buf = [0u8; 4096];
    let mut pty_buf = [0u8; 4096];
    let mut line_buffer = String::new();
    
    loop {
        match stdin.read(&mut stdin_buf) {
            Ok(0) => break,
            Ok(n) => {
                let input = String::from_utf8_lossy(&stdin_buf[..n]);
                
                for ch in input.chars() {
                    if ch == '\n' || ch == '\r' {
                        let cmd = Command::parse(&line_buffer);
                        
                        match cmd {
                            Command::PassThrough => {
                                line_buffer.clear();
                            }
                            _ => {
                                stdout.write_all(b"\r\n")?;
                                stdout.flush()?;
                                
                                match cmd.execute() {
                                    Ok(Some(response)) => {
                                        stdout.write_all(response.as_bytes())?;
                                        stdout.write_all(b"\r\n")?;
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        let error_msg = format!("Error: {}\r\n", e);
                                        stdout.write_all(error_msg.as_bytes())?;
                                    }
                                }
                                stdout.flush()?;
                                
                                line_buffer.clear();
                                continue;
                            }
                        }
                    } else {
                        line_buffer.push(ch);
                    }
                }
                
                writer.write_all(&stdin_buf[..n])?;
                writer.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
            Err(e) => return Err(e.into()),
        }
        
        match reader.read(&mut pty_buf) {
            Ok(0) => break,
            Ok(n) => {
                stdout.write_all(&pty_buf[..n])?;
                stdout.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
            Err(e) => return Err(e.into()),
        }
        
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    
    Ok(())
}
