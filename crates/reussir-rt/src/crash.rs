//! Last-chance crash reporting for Reussir executables.
//!
//! A fault in Reussir-generated code does not go through Rust's panic
//! machinery: on Windows it surfaces as an SEH exception that kills the
//! process *silently* — exit code `0xC0000005`, empty stdout, nothing on
//! stderr — which is indistinguishable from "printed nothing" in a captured
//! CI log. The launcher installs this reporter before entering
//! `__reussir_main`, so a fatal exception at least names itself: the
//! exception code, the faulting address as a module+offset (symbolizable
//! offline against the binary), the access-violation direction and target,
//! and a best-effort backtrace of the crashing thread.
//!
//! Off Windows this is a no-op: Unix already reports fatal signals visibly
//! (the shell prints the signal, and core/backtrace tooling is standard).

/// Installs the process-wide last-chance crash reporter. Called by the
/// generated launcher before `__reussir_main`; safe to call more than once.
#[unsafe(no_mangle)]
pub extern "C" fn __reussir_install_crash_reporter() {
    #[cfg(windows)]
    windows::install();
}

#[cfg(windows)]
mod windows {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS, SetUnhandledExceptionFilter,
    };
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameA, GetModuleHandleExW,
    };

    const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;
    const EXCEPTION_STACK_OVERFLOW: u32 = 0xC000_00FD;

    pub(super) fn install() {
        // SAFETY: installs a process-wide filter; the filter itself only
        // reads the exception record handed to it.
        unsafe { SetUnhandledExceptionFilter(Some(report)) };
    }

    /// The filter runs on the crashing thread, after which the search
    /// continues (`EXCEPTION_CONTINUE_SEARCH`) so the OS still applies its
    /// normal fatal-exit path — this only narrates, it never swallows.
    unsafe extern "system" fn report(info: *const EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the OS hands a valid EXCEPTION_POINTERS for the duration of
        // the filter call.
        let record = unsafe { (*info).ExceptionRecord };
        if !record.is_null() {
            // SAFETY: non-null record from the OS.
            let record = unsafe { &*record };
            let code = record.ExceptionCode as u32;
            let addr = record.ExceptionAddress as usize;
            eprintln!("reussir-rt: fatal exception {code:#010x} at {addr:#x}");
            if let Some((module, base)) = module_of(addr) {
                eprintln!(
                    "reussir-rt:   in {module} + {offset:#x}",
                    offset = addr - base
                );
            }
            if code == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2 {
                let op = match record.ExceptionInformation[0] {
                    0 => "reading",
                    1 => "writing",
                    8 => "executing",
                    _ => "accessing",
                };
                eprintln!(
                    "reussir-rt:   access violation {op} {target:#x}",
                    target = record.ExceptionInformation[1]
                );
            }
            // A backtrace capture needs stack the overflowed thread does not
            // have; the essentials above are already out.
            if code != EXCEPTION_STACK_OVERFLOW {
                let backtrace = backtrace::Backtrace::new();
                eprintln!("reussir-rt: crashing-thread backtrace:\n{backtrace:?}");
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    /// The module containing `addr` (its file name and load base), so the
    /// report's addresses are symbolizable offline against the PDB/binary.
    fn module_of(addr: usize) -> Option<(String, usize)> {
        let mut module: HMODULE = core::ptr::null_mut();
        // SAFETY: FROM_ADDRESS treats the "name" argument as an address;
        // UNCHANGED_REFCOUNT keeps this side-effect free. The file-name query
        // writes within the buffer it is given.
        unsafe {
            if GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                addr as *const u16,
                &mut module,
            ) == 0
            {
                return None;
            }
            let mut name = [0u8; 260];
            let len = GetModuleFileNameA(module, name.as_mut_ptr(), name.len() as u32) as usize;
            if len == 0 {
                return None;
            }
            Some((
                String::from_utf8_lossy(&name[..len]).into_owned(),
                module as usize,
            ))
        }
    }
}
