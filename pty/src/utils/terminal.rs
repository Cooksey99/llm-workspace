use nix::{fcntl::OFlag, ioctl_read_bad};
use portable_pty::PtySize;
use std::os::fd::RawFd;

pub fn get_terminal_size() -> PtySize {
    use nix::libc::{winsize, STDOUT_FILENO};

    ioctl_read_bad!(get_winsize, nix::libc::TIOCGWINSZ, winsize);

    let mut size = winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };

    unsafe {
        let _ = get_winsize(STDOUT_FILENO, &mut size);
    }

    PtySize {
        rows: size.ws_row,
        cols: size.ws_col,
        pixel_width: size.ws_xpixel,
        pixel_height: size.ws_ypixel,
    }
}

pub fn set_raw_mode(fd: RawFd) -> Result<nix::sys::termios::Termios, nix::errno::Errno> {
    use nix::sys::termios::*;
    use std::os::fd::BorrowedFd;

    let original = tcgetattr(unsafe { BorrowedFd::borrow_raw(fd) })?;
    let mut raw = original.clone();

    // Make terminal raw
    cfmakeraw(&mut raw);

    tcsetattr(unsafe { BorrowedFd::borrow_raw(fd) }, SetArg::TCSANOW, &raw)?;

    Ok(original)
}

pub struct RestoreTermios {
    pub fd: i32,
    pub termios: nix::sys::termios::Termios,
}

#[allow(dead_code)]
pub fn set_nonblocking(fd: RawFd) -> anyhow::Result<()> {
    use nix::fcntl::{fcntl, FcntlArg};
    use std::os::fd::BorrowedFd;
    
    let borrowed_fd = unsafe { BorrowedFd::borrow_raw(fd) };
    let flags = fcntl(borrowed_fd, FcntlArg::F_GETFL)?;
    let mut flags = OFlag::from_bits_truncate(flags);
    flags.insert(OFlag::O_NONBLOCK);
    fcntl(borrowed_fd, FcntlArg::F_SETFL(flags))?;
    
    Ok(())
}

impl Drop for RestoreTermios {
    fn drop(&mut self) {
        use nix::sys::termios::*;
        use std::os::fd::BorrowedFd;
        let _ = tcsetattr(
            unsafe { BorrowedFd::borrow_raw(self.fd) },
            SetArg::TCSANOW,
            &self.termios,
        );
    }
}
