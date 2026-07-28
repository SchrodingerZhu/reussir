//! The launcher a Reussir executable is built around.
//!
//! This is a real Rust program, and that is the whole point: compiling it as
//! a `bin` makes rustc emit the C `main` that enters `std::rt::lang_start`,
//! so a Reussir executable starts under the *full* Rust runtime — main-thread
//! setup, the stack-overflow guard, argument capture, the panic hook and
//! runtime cleanup — rather than an approximation of it. None of that is
//! reachable from a library entered through C, and `lang_start` itself is
//! unstable, so the only way to get it is to have a Rust `main`.
//!
//! The driver compiles this file at link time and hands rustc the program's
//! objects, which supply `__reussir_main` — the C-ABI export of the function
//! marked `#[main]`.

unsafe extern "C" {
    /// The program's entry point: the `#[main]` function's export.
    fn __reussir_main();
    /// The runtime's last-chance crash reporter (a no-op off Windows): a
    /// fatal exception in generated code prints its code, faulting
    /// module+offset, and backtrace to stderr instead of killing the process
    /// silently.
    fn __reussir_install_crash_reporter();
}

fn main() {
    // Safety: the driver only builds this launcher for a program that has a
    // `#[main]` function, whose object it links in alongside; the symbols are
    // resolved at link time (the reporter from the runtime) or the link
    // fails.
    unsafe {
        __reussir_install_crash_reporter();
        __reussir_main()
    }
}
