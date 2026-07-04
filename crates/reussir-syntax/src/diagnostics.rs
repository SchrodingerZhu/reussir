//! Parse error representation and user-facing rendering via [`ariadne`].
//!
//! Both the parser's own [`ParseError`]s and the middle-end's diagnostics render
//! through one path: a caller lowers each into a [`Diagnostic`] (a file, a
//! byte-offset span, a [`Severity`], and a message) and hands the batch to
//! [`render`], which draws a source-caret report against the compilation's
//! [`SourceCache`]. A spanless diagnostic — one the compiler could not trace
//! back to a source location — falls back to a plain
//! `{file_name}: severity: message` line.

use ariadne::{Color, Config, Label, Report, ReportKind};

use crate::source::{FileId, SourceCache};

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
/// half-open range into `file`, or `None` when the diagnostic cannot be pinned
/// to a location (an internal/whole-program error) — in which case [`render`]
/// prints it without a caret.
#[derive(Debug, Clone)]
pub struct Diagnostic<'a> {
    pub file: FileId,
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

/// Render `diagnostics` with source context to `out`.
///
/// A diagnostic with a span draws an `ariadne` source-caret report; `cache`
/// serves every file of the compilation, and byte spans are converted to the
/// character offsets [`ariadne::Source`] indexes by. A spanless diagnostic is
/// written as a plain `{file_name}: severity: message` line — the compiler
/// could not trace it back to source, so there is nothing to point at.
pub fn render(
    cache: &SourceCache,
    diagnostics: &[Diagnostic],
    color: bool,
    mut out: impl std::io::Write,
) -> std::io::Result<()> {
    for diag in diagnostics {
        // An unavailable or empty source has no line to anchor a caret on;
        // treat the span as absent rather than hand ariadne an out-of-bounds
        // range.
        let span = match diag.span {
            Some(span) if cache.is_available(diag.file) && cache.char_len(diag.file) > 0 => {
                Some(span)
            }
            _ => None,
        };
        let Some(bytes) = span else {
            writeln!(
                out,
                "{}: {}: {}",
                cache.name(diag.file),
                diag.severity.label(),
                diag.message
            )?;
            continue;
        };
        // Clamp to a non-empty, in-bounds range: a zero-width span still
        // shows a caret, and an end-of-input span (one past the last
        // character, e.g. an unexpected-EOF parse error) pulls back onto the
        // last character — out of bounds, ariadne would render the frame
        // with no source line at all.
        let (start, end) = cache.char_span(diag.file, bytes);
        let len = cache.char_len(diag.file);
        let start = start.min(len.saturating_sub(1));
        let end = end.clamp(start + 1, len.max(start + 1));
        let span = (diag.file, start as usize..end as usize);
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
            .write(cache, &mut out)?;
    }
    Ok(())
}

/// Render parse `errors` — which are always local to one file — against
/// `file`: a thin adapter over [`render`] that tags each as a
/// [`Severity::Error`].
pub fn render_errors(
    cache: &SourceCache,
    file: FileId,
    errors: &[ParseError],
    color: bool,
    out: impl std::io::Write,
) -> std::io::Result<()> {
    let diagnostics: Vec<Diagnostic> = errors
        .iter()
        .map(|e| Diagnostic {
            file,
            span: Some(e.span),
            severity: Severity::Error,
            message: &e.message,
        })
        .collect();
    render(cache, &diagnostics, color, out)
}
