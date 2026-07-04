//! The line-based frontend: script files (`-i`), piped stdin, and the
//! prompt-per-line interactive fallback.
//!
//! Output is FileCheck-friendly and matches the established REPL contract:
//! `Definition added.` per accepted definition and `<value> : <type>` per
//! evaluated expression on stdout; diagnostics on stderr. Multiline inputs
//! are wrapped in literal `:{` / `}:` lines (GHCi-style); an unterminated
//! block consumes to end of input. Evaluation errors do not end the session.

use std::io::{BufRead, Write};

use reussir_core::semi::render_reports;
use reussir_syntax::diagnostics;
use reussir_syntax::source::SourceCache;

use crate::session::{Exit, Outcome, ReplSession};

/// Drive `session` line by line until EOF, `:q`, or `:clear`. When
/// `interactive` is set, prompts (`λ> `, `| `) go to stdout.
pub fn drive(
    session: &mut ReplSession<'_, '_>,
    lines: &mut dyn BufRead,
    interactive: bool,
) -> Exit {
    loop {
        let Some(input) = read_input(lines, interactive) else {
            return Exit::Quit;
        };
        if input.trim().is_empty() {
            continue;
        }
        let outcome = session.eval(&input);
        if let Some(exit) = render(&outcome, session.sources()) {
            return exit;
        }
    }
}

/// Read one input: a single line, or a `:{ … }:` block collected into one
/// chunk. `None` at EOF.
fn read_input(lines: &mut dyn BufRead, interactive: bool) -> Option<String> {
    prompt("λ> ", interactive);
    let line = read_line(lines)?;
    if line.trim() != ":{" {
        return Some(line);
    }
    // Collect the block; an unterminated block consumes to EOF (and warns).
    let mut block = Vec::new();
    loop {
        prompt("|  ", interactive);
        match read_line(lines) {
            Some(line) if line.trim() == "}:" => break,
            Some(line) => block.push(line),
            None => {
                eprintln!("warning: `:{{` block not terminated by `}}:`");
                break;
            }
        }
    }
    Some(block.join("\n"))
}

fn read_line(lines: &mut dyn BufRead) -> Option<String> {
    let mut line = String::new();
    match lines.read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => {
            // Strip the trailing newline (and a \r before it).
            while line.ends_with('\n') || line.ends_with('\r') {
                line.pop();
            }
            Some(line)
        }
        Err(error) => {
            eprintln!("error: failed to read input: {error}");
            None
        }
    }
}

fn prompt(text: &str, interactive: bool) {
    if interactive {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
}

/// Render one outcome (stdout for results, stderr for diagnostics).
/// `Some(exit)` when the outcome ends the drive loop.
pub fn render(outcome: &Outcome, sources: &SourceCache) -> Option<Exit> {
    match outcome {
        Outcome::Empty => {}
        Outcome::Quit => return Some(Exit::Quit),
        Outcome::ClearRequested => return Some(Exit::Clear),
        Outcome::Text(text) => println!("{text}"),
        Outcome::Definitions { count, warnings } => {
            render_reports(sources, warnings);
            for _ in 0..*count {
                println!("Definition added.");
            }
        }
        Outcome::Value {
            value,
            ty,
            warnings,
        } => {
            render_reports(sources, warnings);
            // Unit results print bare, matching the established contract.
            if ty == "()" {
                println!("()");
            } else {
                println!("{value} : {ty}");
            }
        }
        Outcome::Binding {
            name,
            value,
            ty,
            warnings,
        } => {
            render_reports(sources, warnings);
            println!("{name} = {value} : {ty}");
        }
        Outcome::ParseErrors { file, errors } => {
            let _ = diagnostics::render_errors(sources, *file, errors, false, std::io::stderr());
        }
        Outcome::Reports(reports) => {
            render_reports(sources, reports);
        }
        Outcome::Backend(message) => eprintln!("error: {message}"),
    }
    None
}
