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

    /// Use the plain line-based prompt even on a capable terminal
    /// (implied by `TERM=dumb`, script mode, and piped stdin/stdout).
    #[arg(long = "no-tui")]
    no_tui: bool,
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
    // line-based frontend; a real terminal gets the TUI unless opted out
    // (`--no-tui`) or incapable (`TERM=dumb` — set by e.g. Emacs shells on
    // every platform; an *unset* TERM stays on the TUI, since Windows
    // terminals conventionally don't set it at all). The plain fallback on
    // a terminal keeps its line prompts.
    let mut interactive = false;
    let mut lines: Box<dyn BufRead> = match &cli.input {
        Some(path) => {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
            Box::new(BufReader::new(file))
        }
        None => {
            let on_terminal = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
            let dumb = matches!(std::env::var("TERM").as_deref(), Ok("dumb"));
            if on_terminal && !dumb && !cli.no_tui {
                return tui::run(config);
            }
            interactive = on_terminal;
            Box::new(BufReader::new(std::io::stdin()))
        }
    };

    if interactive {
        println!(
            "Reussir REPL v{} (Rust pipeline)",
            env!("CARGO_PKG_VERSION")
        );
        println!("Type :help for available commands, :q to quit");
    }

    // Each `:clear` tears the whole session (arena, elaborator, JIT) down
    // and starts a fresh one; the input stream carries on where it was.
    loop {
        let exit = session::run(config, |session| {
            plain::drive(session, &mut lines, interactive)
        })?;
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
    // `palc` represents `--help` as a parse error. Keep the conventional CLI
    // contract used by `rrc` and `rene`: help goes to stdout with exit 0,
    // while genuine usage errors go to stderr with exit 2.
    let cli = match Cli::try_parse_from(std::env::args_os()) {
        Ok(cli) => cli,
        Err(err) => match err.try_into_help() {
            Ok(help) => {
                println!("{help}");
                return ExitCode::SUCCESS;
            }
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        },
    };
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
