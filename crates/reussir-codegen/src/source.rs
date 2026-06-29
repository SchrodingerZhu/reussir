//! Source-position mapping for attaching MLIR locations to lowered ops.
//!
//! The MIR carries byte-offset [`Span`]s; a [`SourceMap`] resolves those to
//! `(line, column)` against the original source so lowering can label each op
//! with a `FileLineColLoc` (and, with debug info enabled, emit correct DWARF line
//! tables). Lowering without a source map falls back to `unknown` locations.
//!
//! TODO(multifile): this maps a *single* source file. Once `rrc` grows a
//! crate-like multi-file structure, a [`Span`] will need to identify its file
//! too, and this should become a file cache keyed by file id (ariadne's
//! [`Cache`](ariadne::Cache) / [`sources`](ariadne::sources) already model this).

use std::borrow::Cow;
use std::path::Path;

use ariadne::Source;
use reussir_core::surface::Span;

/// A line index over one source file, labelling positions with its path.
///
/// Wraps [`ariadne::Source`] (the same line cache the parser uses for
/// diagnostics) to resolve byte offsets to `(line, column)`.
pub struct SourceMap<'a> {
    path: &'a Path,
    source: Source<&'a str>,
}

impl<'a> SourceMap<'a> {
    /// Build the line index for `source`, labelling positions with `path`.
    pub fn new(path: &'a Path, source: &'a str) -> Self {
        SourceMap {
            path,
            source: Source::from(source),
        }
    }

    /// The source path positions are labelled with.
    pub fn path(&self) -> &Path {
        self.path
    }

    /// The full path as a string, for a `FileLineColLoc` filename.
    pub fn filename(&self) -> Cow<'_, str> {
        self.path.to_string_lossy()
    }

    /// The file name component, for the DWARF `DIFile` basename.
    pub fn basename(&self) -> Cow<'_, str> {
        self.path
            .file_name()
            .map_or(Cow::Borrowed(""), |n| n.to_string_lossy())
    }

    /// The parent directory, for the DWARF `DIFile` directory.
    pub fn directory(&self) -> Cow<'_, str> {
        self.path
            .parent()
            .map_or(Cow::Borrowed(""), |d| d.to_string_lossy())
    }

    /// The 1-based `(line, column)` of a byte `offset` (ariadne reports both
    /// zero-indexed).
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        match self.source.get_byte_line(offset) {
            Some((_, line, col)) => (line + 1, col + 1),
            None => (1, 1),
        }
    }

    /// The 1-based `(line, column)` at the start of `span`.
    pub fn span_start(&self, span: Span) -> (usize, usize) {
        self.line_col(span.start as usize)
    }
}
