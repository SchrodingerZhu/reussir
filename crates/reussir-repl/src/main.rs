//! `rrepl`: the Reussir REPL binary (Rust pipeline).
//!
//! Modes: `-i FILE` runs a script; otherwise stdin is read line by line,
//! with prompts when it is a terminal. Exit 0 on a normal session (even one
//! that saw evaluation errors), 2 on usage or startup failure.

use std::io::{BufRead, BufReader, IsTerminal};
use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use reussir_jit::OptLevel;
use reussir_repl::frontend::{plain, tui};
use reussir_repl::session::{self, Config, Exit};

/// Reussir REPL (Rust pipeline).
#[derive(Parser)]
#[command(name = "rrepl", version)]
struct Cli {
    /// Optimization level: `none`, `default`, `aggressive`, `size`, or
    /// `tpde`. Defaults to `tpde` when the backend has TPDE compiled in.
    #[arg(short = 'O', long = "opt-level")]
    opt_level: Option<String>,

    /// Log level: `error`, `warning`, `info`, `debug`, or `trace`.
    #[arg(short = 'l', long = "log-level", default_value = "warning")]
    log_level: String,

    /// Input file for line-by-line execution (script mode).
    #[arg(short = 'i', long = "input")]
    input: Option<PathBuf>,
}

fn parse_log_level(level: &str) -> Result<tracing::Level, String> {
    match level {
        "error" => Ok(tracing::Level::ERROR),
        "warning" => Ok(tracing::Level::WARN),
        "info" => Ok(tracing::Level::INFO),
        "debug" => Ok(tracing::Level::DEBUG),
        "trace" => Ok(tracing::Level::TRACE),
        other => Err(format!("unknown --log-level `{other}`")),
    }
}

fn resolve_opt(cli: &Cli) -> Result<OptLevel, String> {
    match cli.opt_level.as_deref() {
        Some(level) => match reussir_repl::commands::parse_opt(level) {
            Some(OptLevel::Tpde) if !reussir_jit::orc::has_tpde() => {
                Err("TPDE support is not compiled into the backend".to_string())
            }
            Some(opt) => Ok(opt),
            None => Err(format!("unknown --opt-level `{level}`")),
        },
        // The historical default is TPDE (fastest compile); fall back when
        // the backend was built without it.
        None if reussir_jit::orc::has_tpde() => Ok(OptLevel::Tpde),
        None => Ok(OptLevel::Default),
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let log_level = parse_log_level(&cli.log_level)?;
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_writer(std::io::stderr)
        .init();

    let config = Config {
        opt: resolve_opt(cli)?,
    };

    // Mode selection: a script file or piped stdin drive the plain
    // line-based frontend; a real terminal gets the TUI.
    let mut lines: Box<dyn BufRead> = match &cli.input {
        Some(path) => {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
            Box::new(BufReader::new(file))
        }
        None => {
            if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                return tui::run(config);
            }
            Box::new(BufReader::new(std::io::stdin()))
        }
    };

    // Each `:clear` tears the whole session (arena, elaborator, JIT) down
    // and starts a fresh one; the input stream carries on where it was.
    loop {
        let exit = session::run(config, |session| plain::drive(session, &mut lines, false))?;
        match exit {
            Exit::Quit => return Ok(()),
            Exit::Clear => {
                println!("Context cleared");
                continue;
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
