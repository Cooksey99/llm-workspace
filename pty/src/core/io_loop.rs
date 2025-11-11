use std::io::{Read, Write};
use anyhow::Result;

pub fn run_io_loop(master: &mut Box<dyn portable_pty::MasterPty + Send>) -> Result<()> {
    use std::io::{stdin, stdout};
    
    let mut stdin = stdin();
    let mut stdout = stdout();
    
    // Get reader and writer from master
    let mut reader = master.try_clone_reader()?;
    let mut writer = master.take_writer()?;
    
    let mut stdin_buf = [0u8; 4096];
    let mut pty_buf = [0u8; 4096];
    
    loop {
        // Read from user's keyboard → write to PTY (shell)
        match stdin.read(&mut stdin_buf) {
            Ok(0) => break, // EOF
            Ok(n) => {
                writer.write_all(&stdin_buf[..n])?;
                writer.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
            Err(e) => return Err(e.into()),
        }
        
        // Read from PTY (shell output) → write to user's terminal
        match reader.read(&mut pty_buf) {
            Ok(0) => break, // Shell exited
            Ok(n) => {
                stdout.write_all(&pty_buf[..n])?;
                stdout.flush()?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {},
            Err(e) => return Err(e.into()),
        }
        
        // Small sleep to avoid busy-waiting
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    
    Ok(())
}
