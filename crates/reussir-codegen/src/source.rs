//! Source-position mapping for attaching MLIR locations to lowered ops.
//!
//! The MIR carries file-local byte-offset [`Span`](reussir_core::surface::Span)s
//! plus a per-function [`FileId`]; the compilation's [`SourceCache`] resolves
//! those to `(file, line, column)` so lowering can label each op with a
//! `FileLineColLoc` (and, with debug info enabled, emit correct DWARF line
//! tables). Lowering without a source cache falls back to `unknown` locations.
//!
//! Re-exported from `reussir_syntax` so the whole pipeline shares one
//! authority for offsets: diagnostics (ariadne caret reports) and debug info
//! read the same per-file line index.

pub use reussir_syntax::source::{FileId, SourceCache};
