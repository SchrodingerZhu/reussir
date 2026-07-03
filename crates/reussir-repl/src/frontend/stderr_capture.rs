//! Capturing raw `stderr` while the TUI owns the screen.
//!
//! Backend failures log straight to file descriptor 2 from *native* code —
//! TPDE's spdlog, LLVM's `errs()` — which no Rust-level writer indirection
//! can intercept, and which smears escape-sequence salad across a raw-mode
//! alternate screen. The only robust interception point is the descriptor
//! itself: [`StderrCapture`] `dup2`s a pipe over fd 2 for the TUI's
//! lifetime, a reader thread accumulates lines (it must run concurrently —
//! a full pipe with no reader would deadlock the backend mid-log), and the
//! TUI drains them into the scrollback right after each evaluation.
//!
//! Restore discipline: the original fd is put back on drop *and* by a panic
//! hook installed **before** `ratatui::init` — ratatui's own hook then wraps
//! ours, so on panic the terminal is restored first, stderr second, and the
//! panic message prints to the real stderr instead of vanishing into the
//! pipe.
//!
//! Unix-only; on other platforms the capture is a no-op and backend logs
//! pass through as before. Script/pipe modes never construct one — lit
//! tests depend on untouched stderr.

#[cfg(unix)]
pub use unix::StderrCapture;

#[cfg(not(unix))]
pub use fallback::StderrCapture;

#[cfg(unix)]
mod unix {
    use std::io::Read;
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::{Arc, Mutex};

    /// The saved real-stderr fd, for the panic hook. `-1` when no capture
    /// is active.
    static SAVED_STDERR: AtomicI32 = AtomicI32::new(-1);

    pub struct StderrCapture {
        lines: Arc<Mutex<Vec<String>>>,
        /// The dup of the real stderr, restored on drop.
        saved: OwnedFd,
        reader: Option<std::thread::JoinHandle<()>>,
    }

    impl StderrCapture {
        /// Redirect fd 2 into a pipe and start the reader thread. Call
        /// before `ratatui::init` so the panic hooks chain in the right
        /// order (terminal restore, then stderr restore, then the message).
        pub fn install() -> std::io::Result<Self> {
            let mut fds = [0i32; 2];
            if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let (read_end, write_end) = (fds[0], fds[1]);
            let saved = unsafe { libc::dup(2) };
            if saved < 0 || unsafe { libc::dup2(write_end, 2) } < 0 {
                return Err(std::io::Error::last_os_error());
            }
            // fd 2 now *is* the pipe's write end; drop the original so the
            // reader sees EOF as soon as the restore puts the real stderr
            // back.
            unsafe { libc::close(write_end) };
            SAVED_STDERR.store(saved, Ordering::SeqCst);

            let lines = Arc::new(Mutex::new(Vec::new()));
            let sink = lines.clone();
            // SAFETY: `read_end` is a fresh pipe fd owned by this thread.
            let mut source = unsafe { std::fs::File::from_raw_fd(read_end) };
            let reader = std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                let mut pending = Vec::new();
                loop {
                    match source.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            pending.extend_from_slice(&buf[..n]);
                            while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                                let line: Vec<u8> = pending.drain(..=pos).collect();
                                let text = String::from_utf8_lossy(&line[..line.len() - 1]);
                                sink.lock().unwrap().push(text.into_owned());
                            }
                        }
                    }
                }
                if !pending.is_empty() {
                    let text = String::from_utf8_lossy(&pending).into_owned();
                    sink.lock().unwrap().push(text);
                }
            });

            // Restore the real stderr before the (chained) default hook
            // prints the panic message.
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                let saved = SAVED_STDERR.swap(-1, Ordering::SeqCst);
                if saved >= 0 {
                    unsafe {
                        libc::dup2(saved, 2);
                        libc::close(saved);
                    }
                }
                previous(info);
            }));

            // SAFETY: `saved` is a fresh dup owned by the capture.
            let saved = unsafe { OwnedFd::from_raw_fd(saved) };
            Ok(StderrCapture {
                lines,
                saved,
                reader: Some(reader),
            })
        }

        /// Take the lines captured so far.
        pub fn drain(&self) -> Vec<String> {
            std::mem::take(&mut self.lines.lock().unwrap())
        }
    }

    impl Drop for StderrCapture {
        fn drop(&mut self) {
            use std::os::fd::AsRawFd;
            // Put the real stderr back; this closes the pipe's only write
            // end (fd 2), so the reader sees EOF and exits.
            SAVED_STDERR.store(-1, Ordering::SeqCst);
            unsafe { libc::dup2(self.saved.as_raw_fd(), 2) };
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
    }
}

#[cfg(not(unix))]
mod fallback {
    /// No capture off Unix: backend logs pass through to the terminal as
    /// before (`_dup2` + `SetStdHandle` would be the Windows equivalent —
    /// a follow-up).
    pub struct StderrCapture;

    impl StderrCapture {
        pub fn install() -> std::io::Result<Self> {
            Ok(StderrCapture)
        }

        pub fn drain(&self) -> Vec<String> {
            Vec::new()
        }
    }
}
