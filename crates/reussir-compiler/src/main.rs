//! `rrc`: the Reussir compiler binary — a shim over
//! [`reussir_compiler::driver`], where the whole pipeline lives.

use std::process::ExitCode;

fn main() -> ExitCode {
    reussir_compiler::driver::main()
}
