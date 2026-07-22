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
  :dump bindings        List global `let` bindings and their current values
  :set opt <level>      Set the optimization level (none, default, aggressive, size, tpde)
  :clear                Reset the session (definitions and compiled code)

Multiline input: wrap in `:{` and `}:` lines.
Input is parsed as definitions (fn/struct/enum/...) or as an expression.

TUI keys:
  Enter                 Submit (inserts a newline while the input is incomplete)
  Alt+Enter             Insert a newline
  Ctrl+J                Force-submit the buffer as is
  Ctrl+U / Ctrl+R       Undo / redo in the editor
  Up / Down             Browse history (on the first/last line)
  PgUp / PgDn           Scroll the output
  Ctrl+C                Clear the buffer (twice on empty: quit)
  Ctrl+D                Quit (on an empty buffer)";

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
                // Evaluated expressions leave their `__repl_expr_N` wrappers
                // behind as (unreferenceable) dead HIR; they are driver
                // plumbing, not user context.
                let funcs: Vec<_> = elab
                    .elaborated
                    .iter()
                    .filter(|f| !elab.sym(f.name).starts_with("__"))
                    .cloned()
                    .collect();
                let records = elab
                    .records
                    .iter()
                    .filter(|(_, r)| !elab.sym(r.name).starts_with("__"))
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                let strings = elab.strings.entries();
                let text = Printer::new(&elab.defs, elab.resolver)
                    .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                    .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                    .program(&funcs, &strings, &records, &elab.trampolines);
                Outcome::Text(text)
            }
            Some("compiled") => {
                // The per-expression wrappers and dec companions are driver
                // plumbing; a session's dump would otherwise drown the user's
                // functions in them. Match the synthetic *prefix* (raw
                // exports and their `_RC<len>`-mangled forms), not a
                // substring — a user identifier may legally contain
                // `__repl_` mid-name.
                fn synthetic(sym: &str) -> bool {
                    let tail = sym
                        .strip_prefix("_RC")
                        .map(|rest| rest.trim_start_matches(|c: char| c.is_ascii_digit()))
                        .unwrap_or(sym);
                    tail.trim_start_matches('_').starts_with("repl_") && tail.starts_with("__")
                }
                let mut symbols: Vec<&str> = session
                    .emitted
                    .iter()
                    .map(String::as_str)
                    .filter(|s| !synthetic(s))
                    .collect();
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
            Some("bindings") => Outcome::Text(session.render_bindings()),
            _ => {
                Outcome::Text("usage: :dump context | :dump compiled | :dump bindings".to_string())
            }
        },
        Some("set") => match (words.next(), words.next()) {
            (Some("opt"), Some(level)) => match parse_opt(level) {
                // Same guard as the CLI's `-O tpde`: reject here rather than
                // letting the *next* input fail with a misattributed JIT error.
                Some(OptLevel::Tpde) if !reussir_jit::orc::has_tpde() => {
                    Outcome::Text("TPDE support is not compiled into the backend".to_string())
                }
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
