//! The `:command` dispatcher, shared by every frontend.

use reussir_core::semi::hir::print::Printer;
use reussir_jit::OptLevel;

use crate::session::{Outcome, ReplSession};

const HELP: &str = "\
Available commands:
  :help                 Show this message
  :q, :quit             Exit the REPL
  :type <expr>          Show the type of an expression (no evaluation)
  :dump context         Pretty-print the accumulated definitions (HIR)
  :dump compiled        List JIT-compiled symbols
  :set opt <level>      Set the optimization level (none, default, aggressive, size, tpde)
  :clear                Reset the session (definitions and compiled code)

Multiline input: wrap in `:{` and `}:` lines.
Input is parsed as definitions (fn/struct/enum/...) or as an expression.";

/// Dispatch `command` (the input with the leading `:` stripped).
pub fn dispatch(session: &mut ReplSession<'_, '_>, command: &str) -> Outcome {
    let mut words = command.split_whitespace();
    match words.next() {
        None => Outcome::Text("expected a command after `:` (try :help)".to_string()),
        Some("help") => Outcome::Text(HELP.to_string()),
        Some("q" | "quit") => Outcome::Quit,
        Some("clear") => Outcome::ClearRequested,
        Some("type") => {
            let expr = command.strip_prefix("type").expect("matched above").trim();
            if expr.is_empty() {
                Outcome::Text("usage: :type <expr>".to_string())
            } else {
                session.type_of(expr)
            }
        }
        Some("dump") => match words.next() {
            Some("context") => {
                let elab = &session.elab;
                let text = Printer::new(&elab.defs, elab.resolver).program(
                    &elab.elaborated,
                    &elab.records,
                    &elab.trampolines,
                );
                Outcome::Text(text)
            }
            Some("compiled") => {
                let mut symbols: Vec<&str> = session.emitted.iter().map(String::as_str).collect();
                symbols.sort_unstable();
                let mut out = String::from("=== Compiled Functions ===\n");
                for symbol in symbols {
                    out.push_str(" - ");
                    out.push_str(symbol);
                    out.push('\n');
                }
                out.pop();
                Outcome::Text(out)
            }
            _ => Outcome::Text("usage: :dump context | :dump compiled".to_string()),
        },
        Some("set") => match (words.next(), words.next()) {
            (Some("opt"), Some(level)) => match parse_opt(level) {
                Some(opt) => {
                    session.opt = opt;
                    Outcome::Text(format!("optimization level set to {level}"))
                }
                None => Outcome::Text(format!(
                    "unknown level `{level}` (none, default, aggressive, size, tpde)"
                )),
            },
            _ => Outcome::Text("usage: :set opt <none|default|aggressive|size|tpde>".to_string()),
        },
        Some(other) => Outcome::Text(format!("Unknown command: :{other} (try :help)")),
    }
}

/// Parse an optimization-level name (also used by the CLI's `-O`).
pub fn parse_opt(level: &str) -> Option<OptLevel> {
    Some(match level {
        "none" => OptLevel::None,
        "default" => OptLevel::Default,
        "aggressive" => OptLevel::Aggressive,
        "size" => OptLevel::Size,
        "tpde" => OptLevel::Tpde,
        _ => return None,
    })
}
