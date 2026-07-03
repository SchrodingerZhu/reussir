//! Capturing raw process output while the TUI owns the screen.
//!
//! Backend failures log straight to the standard file descriptors from
//! *native* code — TPDE's spdlog writes to **stdout**, LLVM's `errs()` to
//! stderr — which no Rust-level writer indirection can intercept, and which
//! smears escape-sequence salad across a raw-mode alternate screen. The
//! robust interception point is the descriptors themselves, which requires
//! an inversion: the TUI draws to `/dev/tty` (see [`super::tui`]), so fd 1
//! and fd 2 belong exclusively to the *program's* output and can both be
//! `dup2`'d onto a pipe for the TUI's lifetime. A reader thread accumulates
//! lines (it must run concurrently — a full pipe with no reader would
//! deadlock the backend mid-log), and the TUI drains them into the
//! scrollback right after each evaluation.
//!
//! Restore discipline: the original fds are put back on drop *and* by a
//! panic hook installed **before** the terminal is set up — the TUI's own
//! hook then wraps ours, so on panic the terminal is restored first, the
//! fds second, and the panic message prints to the real stderr instead of
//! vanishing into the pipe.
//!
//! Unix-only; on other platforms the capture is a no-op and native logs
//! pass through as before. Script/pipe modes never construct one — lit
//! tests depend on untouched stdout/stderr.

#[cfg(unix)]
pub use unix::OutputCapture;

#[cfg(not(unix))]
pub use fallback::OutputCapture;

#[cfg(unix)]
mod unix {
    use std::io::Read;
    use std::os::fd::FromRawFd;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::{Arc, Mutex};

    /// The saved real stdout/stderr fds, for the panic hook. `-1` when no
    /// capture is active.
    static SAVED_STDOUT: AtomicI32 = AtomicI32::new(-1);
    static SAVED_STDERR: AtomicI32 = AtomicI32::new(-1);

    fn restore_fd(saved: &AtomicI32, fd: i32) {
        let saved = saved.swap(-1, Ordering::SeqCst);
        if saved >= 0 {
            unsafe {
                libc::dup2(saved, fd);
                libc::close(saved);
            }
        }
    }

    pub struct OutputCapture {
        lines: Arc<Mutex<Vec<String>>>,
        reader: Option<std::thread::JoinHandle<()>>,
    }

    impl OutputCapture {
        /// Redirect fd 1 and fd 2 into a pipe and start the reader thread.
        /// Call before terminal setup so the panic hooks chain in the right
        /// order (terminal restore, then fd restore, then the message).
        pub fn install() -> std::io::Result<Self> {
            let mut fds = [0i32; 2];
            if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            let (read_end, write_end) = (fds[0], fds[1]);
            let saved_out = unsafe { libc::dup(1) };
            let saved_err = unsafe { libc::dup(2) };
            if saved_out < 0
                || saved_err < 0
                || unsafe { libc::dup2(write_end, 1) } < 0
                || unsafe { libc::dup2(write_end, 2) } < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            // fds 1 and 2 now both *are* the pipe's write end; drop the
            // original so the reader sees EOF once both are restored.
            unsafe { libc::close(write_end) };
            SAVED_STDOUT.store(saved_out, Ordering::SeqCst);
            SAVED_STDERR.store(saved_err, Ordering::SeqCst);

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

            // Restore the real fds before the (chained) default hook prints
            // the panic message.
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore_fd(&SAVED_STDOUT, 1);
                restore_fd(&SAVED_STDERR, 2);
                previous(info);
            }));

            Ok(OutputCapture {
                lines,
                reader: Some(reader),
            })
        }

        /// Take the lines captured so far.
        pub fn drain(&self) -> Vec<String> {
            std::mem::take(&mut *self.lines.lock().unwrap())
        }
    }

    impl Drop for OutputCapture {
        fn drop(&mut self) {
            // Put the real fds back; this closes the pipe's only write ends,
            // so the reader sees EOF and exits.
            restore_fd(&SAVED_STDOUT, 1);
            restore_fd(&SAVED_STDERR, 2);
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
        }
    }
}

#[cfg(not(unix))]
mod fallback {
    /// No capture off Unix: native logs pass through to the terminal as
    /// before (`_dup2` + `SetStdHandle` would be the Windows equivalent —
    /// a follow-up).
    pub struct OutputCapture;

    impl OutputCapture {
        pub fn install() -> std::io::Result<Self> {
            Ok(OutputCapture)
        }

        pub fn drain(&self) -> Vec<String> {
            Vec::new()
        }
    }
}
