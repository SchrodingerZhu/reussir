//! Demo driver for the `reussir_bytecode` library.
//!
//! Emits one of the crate's built-in example modules as an MLIR bytecode file,
//! which the lit suite renders back to textual MLIR to check the encoder. The
//! example modules live in [`demos`] and double as usage documentation; a real
//! frontend links the library directly rather than going through this tool.

mod demos;

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;
use reussir_bytecode::{context::Context, writer::write_module};
use stumpalo::Arena;

/// Emit MLIR bytecode (`.mlirbc`) for a built-in Reussir example module.
#[derive(Parser)]
#[command(name = "reussir-bytecode-demo", version)]
struct Cli {
    /// Emit the named built-in demo module. Use `--list-demos` to see options.
    #[arg(long)]
    demo: Option<String>,

    /// List the available demo module names and exit.
    #[arg(long)]
    list_demos: bool,

    /// Write the bytecode to this path instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn run(cli: &Cli) -> Result<(), String> {
    if cli.list_demos {
        for name in demos::NAMES {
            println!("{name}");
        }
        return Ok(());
    }

    let Some(name) = &cli.demo else {
        return Err("nothing to do: pass --demo NAME (or --list-demos)".to_string());
    };

    let arena = Arena::new();
    let ctx = Context::new(&arena);
    let module = demos::build(&ctx, name)
        .ok_or_else(|| format!("unknown demo '{name}'; try --list-demos"))?;
    let bytes = write_module(module);

    match &cli.output {
        Some(path) => std::fs::write(path, &bytes)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?,
        None => {
            use std::io::Write;
            std::io::stdout()
                .write_all(&bytes)
                .map_err(|e| format!("failed to write stdout: {e}"))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("reussir-bytecode-demo: {msg}");
            ExitCode::from(2)
        }
    }
}
