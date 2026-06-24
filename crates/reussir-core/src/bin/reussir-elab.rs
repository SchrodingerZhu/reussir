//! `reussir-elab`: run the Reussir frontend up to a chosen phase and report.
//!
//! `--mode semi` parses and elaborates the surface program, reporting any
//! diagnostics; `--mode full` additionally monomorphizes and prints the ground
//! Full IR to stdout. The exit code is 0 on success, 1 on a syntax or
//! elaboration error, and 2 on a usage or I/O error.

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use reussir_core::full::mono::monomorphize;
use reussir_core::full::mir::print::Printer;
use reussir_core::semi::ctxt::Severity;
use reussir_core::semi::elaborate;
use reussir_core::surface::{self, Span};

/// Which phase to run to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Semi,
    Full,
}

/// Reussir frontend driver (semi/full elaboration).
#[derive(Parser)]
#[command(name = "reussir-elab", version)]
struct Cli {
    /// Input source file (`-` reads from stdin).
    input: PathBuf,

    /// Phase to run to: `semi` or `full`.
    #[arg(long, default_value = "full")]
    mode: String,
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    match s {
        "semi" => Ok(Mode::Semi),
        "full" => Ok(Mode::Full),
        other => Err(format!(
            "unknown --mode `{other}` (expected `semi` or `full`)"
        )),
    }
}

fn read_input(path: &PathBuf) -> Result<(String, String), String> {
    use std::io::Read;
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        Ok(("<stdin>".to_owned(), buf))
    } else {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Ok((path.display().to_string(), text))
    }
}

/// Render one diagnostic as `name:line:col: severity: message`, mapping the
/// character offset to a 1-based line and column.
fn render_report(name: &str, source: &str, span: Option<Span>, sev: &str, message: &str) {
    let loc = span.map(|s| line_col(source, s.start as usize));
    match loc {
        Some((line, col)) => eprintln!("{name}:{line}:{col}: {sev}: {message}"),
        None => eprintln!("{name}: {sev}: {message}"),
    }
}

/// Map a character offset to a 1-based `(line, column)`.
fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.chars().enumerate() {
        if i == offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn run(cli: &Cli) -> Result<bool, String> {
    let mode = parse_mode(&cli.mode)?;
    let (name, source) = read_input(&cli.input)?;

    let parse = reussir_syntax::parse(&source);
    if !parse.ok() {
        for err in &parse.errors {
            eprintln!("{name}: error: {err:?}");
        }
        eprintln!("error: {} syntax error(s) in {name}", parse.errors.len());
        return Ok(false);
    }

    let prog = surface::program(&parse.root);

    // All arena-bound work happens inside the scope; only the rendered text and
    // the success flag escape.
    let (rendered, ok) = reussir_core::in_arena(|tcx| {
        let elab = elaborate(tcx, &prog, parse.resolver());

        for report in &elab.reports {
            let sev = match report.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
            };
            render_report(&name, &source, report.span, sev, &report.message);
        }
        if elab.has_errors() {
            return (String::new(), false);
        }

        let text = match mode {
            Mode::Semi => reussir_core::semi::hir::print::Printer::new(&elab.defs, elab.resolver)
                .program(&elab.elaborated),
            Mode::Full => {
                let (full, mono_reports) = monomorphize(&elab);
                for report in &mono_reports {
                    let sev = match report.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                    };
                    render_report(&name, &source, report.span, sev, &report.message);
                }
                if mono_reports
                    .iter()
                    .any(|r| matches!(r.severity, Severity::Error))
                {
                    return (String::new(), false);
                }
                Printer::new(&elab.defs, elab.resolver).program(&full)
            }
        };
        (text, true)
    });

    if !ok {
        return Ok(false);
    }
    print!("{rendered}");
    Ok(true)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
