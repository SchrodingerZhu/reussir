//! Parse error representation and user-facing rendering via [`ariadne`].
//!
//! Both the parser's own [`ParseError`]s and the middle-end's diagnostics render
//! through one path: a caller lowers each into a [`Diagnostic`] (a byte-offset
//! span, a [`Severity`], and a message) and hands the batch to [`render`], which
//! draws a source-caret report. A spanless diagnostic — one the compiler could
//! not trace back to a source location — falls back to a plain `severity:
//! message` line.

use ariadne::{Color, Config, Label, Report, ReportKind, Source};

/// A parse (or lexical) error. Spans are byte offsets into the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub span: (u32, u32),
    pub message: String,
}

/// The severity of a [`Diagnostic`]. Mirrors the middle-end's own severity so a
/// caller can map one to the other without this crate depending on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// A source-anchored diagnostic ready to render. The `span` is a byte-offset
/// half-open range into the source, or `None` when the diagnostic cannot be
/// pinned to a location (an internal/whole-program error) — in which case
/// [`render`] prints it without a caret.
#[derive(Debug, Clone)]
pub struct Diagnostic<'a> {
    pub span: Option<(u32, u32)>,
    pub severity: Severity,
    pub message: &'a str,
}

impl Severity {
    fn report_kind(self) -> ReportKind<'static> {
        match self {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }

    fn color(self) -> Color {
        match self {
            Severity::Error => Color::Red,
            Severity::Warning => Color::Yellow,
        }
    }
}

/// Maps byte offsets to character (Unicode scalar) offsets. The frontend
/// records character offsets, so all externally visible spans go through this
/// table.
pub struct SourceMap {
    /// `byte_to_char[b]` is the number of chars strictly before byte `b`.
    byte_to_char: Vec<u32>,
}

impl SourceMap {
    pub fn new(source: &str) -> Self {
        let mut byte_to_char = vec![0u32; source.len() + 1];
        let mut chars = 0u32;
        for (idx, c) in source.char_indices() {
            byte_to_char[idx..idx + c.len_utf8()].fill(chars);
            chars += 1;
        }
        byte_to_char[source.len()] = chars;
        Self { byte_to_char }
    }

    pub fn char_offset(&self, byte: u32) -> u32 {
        self.byte_to_char[byte as usize]
    }

    pub fn char_span(&self, span: (u32, u32)) -> (u32, u32) {
        (self.char_offset(span.0), self.char_offset(span.1))
    }
}

/// Render `diagnostics` with source context to `out`.
///
/// A diagnostic with a span draws an `ariadne` source-caret report; the `ariadne`
/// cache holds the single source file and spans are converted to character
/// offsets, which is what [`ariadne::Source`] indexes by. A spanless diagnostic
/// is written as a plain `severity: message` line — the compiler could not trace
/// it back to source, so there is nothing to point at.
pub fn render(
    file_name: &str,
    source: &str,
    source_map: &SourceMap,
    diagnostics: &[Diagnostic],
    color: bool,
    mut out: impl std::io::Write,
) -> std::io::Result<()> {
    let cache = (file_name, Source::from(source));
    for diag in diagnostics {
        let Some(bytes) = diag.span else {
            writeln!(
                out,
                "{file_name}: {}: {}",
                diag.severity.label(),
                diag.message
            )?;
            continue;
        };
        // Clamp to a non-empty range so a zero-width span still shows a caret.
        let (start, end) = source_map.char_span(bytes);
        let span = (file_name, start as usize..end.max(start + 1) as usize);
        // The message heads the report (a greppable summary line) and annotates
        // the underlined span, so the caret line stands on its own.
        Report::build(diag.severity.report_kind(), span.clone())
            .with_config(Config::default().with_color(color))
            .with_message(diag.message)
            .with_label(
                Label::new(span)
                    .with_message(diag.message)
                    .with_color(diag.severity.color()),
            )
            .finish()
            .write(cache.clone(), &mut out)?;
    }
    Ok(())
}

/// Render parse `errors` with source context — a thin adapter over [`render`]
/// that tags each as a [`Severity::Error`].
pub fn render_errors(
    file_name: &str,
    source: &str,
    source_map: &SourceMap,
    errors: &[ParseError],
    color: bool,
    out: impl std::io::Write,
) -> std::io::Result<()> {
    let diagnostics: Vec<Diagnostic> = errors
        .iter()
        .map(|e| Diagnostic {
            span: Some(e.span),
            severity: Severity::Error,
            message: &e.message,
        })
        .collect();
    render(file_name, source, source_map, &diagnostics, color, out)
}
