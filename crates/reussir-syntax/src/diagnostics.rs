//! Parse error representation and user-facing rendering via [`ariadne`].

use ariadne::{Color, Config, Label, Report, ReportKind, Source};

/// A parse (or lexical) error. Spans are byte offsets into the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub span: (u32, u32),
    pub message: String,
}

/// Maps byte offsets to character (Unicode scalar) offsets. The Haskell
/// frontend records megaparsec offsets, which count characters, so all
/// externally visible spans go through this table.
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

/// Render the diagnostics with source context to the given writer.
///
/// The `ariadne` cache holds the single source file; spans are converted to
/// character offsets, which is what `ariadne::Source` indexes by.
pub fn render_errors(
    file_name: &str,
    source: &str,
    source_map: &SourceMap,
    errors: &[ParseError],
    color: bool,
    mut out: impl std::io::Write,
) -> std::io::Result<()> {
    let cache = (file_name, Source::from(source));
    for error in errors {
        let (start, end) = source_map.char_span(error.span);
        let span = (file_name, start as usize..end.max(start + 1) as usize);
        Report::build(ReportKind::Error, span.clone())
            .with_config(Config::default().with_color(color))
            .with_message("syntax error")
            .with_label(
                Label::new(span)
                    .with_message(&error.message)
                    .with_color(Color::Red),
            )
            .finish()
            .write(cache.clone(), &mut out)?;
    }
    Ok(())
}
