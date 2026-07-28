//! Stuck-build diagnostics: a watchdog armed around each child process rene
//! awaits. A build that wedges on a CI runner is killed from outside (lit's
//! timeout) with nothing but captured stderr to go on, so the watchdog's job
//! is to put the missing facts into that stderr while the process is still
//! wedged: that the child is genuinely still running (or — the smoking gun
//! for a runtime process-wait bug — that it already exited while the await
//! never resolved), and on Windows the live process subtree, naming which
//! toolchain process under the child is actually sitting on the clock.
//!
//! Reports go through `eprintln!` rather than `tracing` deliberately: the
//! watchdog must stay independent of the machinery it reports on.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{RecvTimeoutError, Sender};
use std::time::Duration;

/// An armed watchdog; dropping it disarms the reporter thread.
pub(crate) struct StuckReporter {
    _disarm: Sender<()>,
}

impl StuckReporter {
    /// Arms a watchdog over child `pid` (building `what`): every
    /// `REUSSIR_STUCK_SECS` seconds (default 240; `0` disables) it reports to
    /// stderr that the child is still being awaited, plus what the process
    /// table says about it. The report never kills anything — terminating a
    /// wedged build is the harness's job; this only makes the kill
    /// attributable.
    pub(crate) fn arm(pid: u32, what: &Path) -> Option<StuckReporter> {
        let period = std::env::var("REUSSIR_STUCK_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(240);
        if period == 0 {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let what: PathBuf = what.to_owned();
        std::thread::Builder::new()
            .name("stuck-watchdog".into())
            .spawn(move || {
                let mut waited = 0;
                loop {
                    match rx.recv_timeout(Duration::from_secs(period)) {
                        Err(RecvTimeoutError::Timeout) => {
                            waited += period;
                            report(pid, &what, waited);
                        }
                        // Disarmed: the sender half was dropped.
                        _ => return,
                    }
                }
            })
            .ok()?;
        Some(StuckReporter { _disarm: tx })
    }
}

fn report(pid: u32, what: &Path, waited: u64) {
    eprintln!(
        "rene-watchdog: still awaiting `rrc` (pid {pid}) for `{}` after {waited}s",
        what.display()
    );
    #[cfg(windows)]
    windows::report(pid);
}

#[cfg(windows)]
mod windows {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    /// `GetExitCodeProcess`'s sentinel for a live process (`STATUS_PENDING`).
    /// A process can in principle exit *with* 259; for a diagnostic that
    /// ambiguity is acceptable.
    const STILL_ACTIVE: u32 = 259;

    pub(super) fn report(pid: u32) {
        match exit_code(pid) {
            Some(code) if code != STILL_ACTIVE => eprintln!(
                "rene-watchdog: pid {pid} has ALREADY EXITED (code {code:#x}) yet the await \
                 has not resolved — the process-wait itself is stuck, not the child"
            ),
            _ => {
                eprintln!("rene-watchdog: live process tree under pid {pid}:");
                print_tree(pid);
            }
        }
    }

    fn exit_code(pid: u32) -> Option<u32> {
        // SAFETY: plain query-only process APIs; the handle is closed on every
        // path.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return None;
            }
            let mut code = 0u32;
            let ok = GetExitCodeProcess(handle, &mut code);
            CloseHandle(handle);
            (ok != 0).then_some(code)
        }
    }

    fn print_tree(root: u32) {
        // (pid, parent pid, executable name) for every process in the system
        // snapshot; the subtree under `root` is reconstructed from parent
        // links. Parent pids can be stale (recycled) on Windows — good enough
        // for naming a wedged toolchain child.
        let mut procs: Vec<(u32, u32, String)> = Vec::new();
        // SAFETY: standard ToolHelp snapshot walk over a properly sized,
        // zero-initialized PROCESSENTRY32W; the snapshot handle is closed.
        unsafe {
            let snap: HANDLE = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snap == INVALID_HANDLE_VALUE {
                return;
            }
            let mut entry: PROCESSENTRY32W = std::mem::zeroed();
            entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
            if Process32FirstW(snap, &mut entry) != 0 {
                loop {
                    let len = entry
                        .szExeFile
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(entry.szExeFile.len());
                    procs.push((
                        entry.th32ProcessID,
                        entry.th32ParentProcessID,
                        String::from_utf16_lossy(&entry.szExeFile[..len]),
                    ));
                    if Process32NextW(snap, &mut entry) == 0 {
                        break;
                    }
                }
            }
            CloseHandle(snap);
        }
        if let Some((pid, _, name)) = procs.iter().find(|(pid, _, _)| *pid == root) {
            eprintln!("rene-watchdog:   {name} (pid {pid})");
        }
        walk(&procs, root, 1);
    }

    fn walk(procs: &[(u32, u32, String)], parent: u32, depth: usize) {
        for (pid, parent_pid, name) in procs {
            if *parent_pid == parent && *pid != parent {
                eprintln!(
                    "rene-watchdog:   {}{name} (pid {pid})",
                    "  ".repeat(depth)
                );
                walk(procs, *pid, depth + 1);
            }
        }
    }
}
