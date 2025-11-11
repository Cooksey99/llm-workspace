mod core;
mod utils;

use anyhow::Result;
use std::os::fd::AsRawFd;

use core::run_io_loop;
use portable_pty::{native_pty_system, CommandBuilder};
use utils::{get_terminal_size, set_raw_mode, RestoreTermios};

fn main() -> Result<()> {
    // Access user shell
    let shell = std::env::var("SHELL").unwrap();

    println!("Starting PTY wrapper with shell: {}", shell);

    let pty_system = native_pty_system();

    // Open PTY with current terminal size
    let terminal_size = get_terminal_size();
    let pair = pty_system.openpty(terminal_size)?;

    // Spawn shell
    let cmd = CommandBuilder::new(shell);
    let _child = pair.slave.spawn_command(cmd).unwrap();

    // Slave isn't needed, so it can be dropped
    drop(pair.slave);
    let mut master = pair.master;

    let stdin_fd = std::io::stdin().as_raw_fd();
    let original_termios = set_raw_mode(stdin_fd)?;

    // Restore terminal on exit
    let _guard = RestoreTermios {
        fd: stdin_fd,
        termios: original_termios,
    };

    run_io_loop(&mut master)
}
