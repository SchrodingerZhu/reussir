//! REPL frontends: ways of feeding inputs to a
//! [`ReplSession`](crate::session::ReplSession) and rendering its
//! [`Outcome`](crate::session::Outcome)s.
//!
//! [`plain`] is the line-based frontend used for script files (`-i`), piped
//! stdin, and as the fallback interactive mode; the ratatui TUI is the
//! interactive frontend on a real terminal.

pub mod plain;
