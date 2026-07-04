//! Command-line front door for the `reussir_syntax` library: parse a
//! Reussir source file and emit the JSON encoding of the surface AST.
//!
//! Diagnostics are rendered with full source context on stderr; the JSON
//! document goes to `--output PATH` or stdout. The exit code is 0 on
//! success, 1 when the input has syntax errors, and 2 on usage or I/O
//! errors.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;
use reussir_syntax::diagnostics::render_errors;
use reussir_syntax::source::{CharMap, FileId, SourceCache};

/// Reussir surface syntax parser (JSON AST emitter).
#[derive(Parser)]
#[command(name = "reussir-syntax", version)]
struct Cli {
    /// Input source file (`-` reads from stdin).
    input: PathBuf,

    /// Write the JSON AST to this path instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Parse and report diagnostics only; do not emit JSON.
    #[arg(long)]
    check: bool,

    /// Diagnostic colors: auto, always, or never.
    #[arg(long, default_value = "auto")]
    color: String,
}

fn run(cli: &Cli) -> Result<bool, String> {
    let mut cache = SourceCache::new();
    let file = if cli.input.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        cache.add_virtual("<stdin>", buf)
    } else {
        let text = std::fs::read_to_string(&cli.input)
            .map_err(|e| format!("failed to read {}: {e}", cli.input.display()))?;
        cache.add_file(&cli.input, text)
    };
    debug_assert_eq!(file, FileId::ROOT);
    let source = cache.source(file);

    let parse = reussir_syntax::parse(source);
    let map = CharMap::new(source);

    if !parse.errors.is_empty() {
        let color = match cli.color.as_str() {
            "always" => true,
            "never" => false,
            _ => std::io::stderr().is_terminal(),
        };
        let stderr = std::io::stderr().lock();
        render_errors(&cache, file, &parse.errors, color, stderr)
            .map_err(|e| format!("failed to render diagnostics: {e}"))?;
        eprintln!(
            "error: {} syntax error(s) in {}",
            parse.errors.len(),
            cache.name(file)
        );
        return Ok(false);
    }

    if !cli.check {
        let json = parse.to_json(&map);
        let rendered =
            serde_json::to_string(&json).map_err(|e| format!("failed to serialize JSON: {e}"))?;
        match &cli.output {
            Some(path) => {
                std::fs::write(path, rendered.as_bytes())
                    .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
            }
            None => {
                let mut stdout = std::io::stdout().lock();
                stdout
                    .write_all(rendered.as_bytes())
                    .and_then(|()| stdout.write_all(b"\n"))
                    .map_err(|e| format!("failed to write JSON: {e}"))?;
            }
        }
    }
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
