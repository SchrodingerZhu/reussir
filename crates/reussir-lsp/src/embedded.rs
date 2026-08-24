//! Tree-sitter highlighting for syntax embedded in Reussir raw blocks.
//!
//! This module is intentionally syntactic. It does not invoke rustc, MLIR, or
//! either language's name/type resolution.

use std::sync::OnceLock;

use tree_sitter_highlight::{HighlightConfiguration, HighlightEvent, Highlighter};

use crate::semantic::{RawSemanticToken, SemanticKind};

const CAPTURE_NAMES: &[&str] = &[
    "attribute",
    "boolean",
    "comment",
    "constant",
    "constant.builtin",
    "constructor",
    "function",
    "function.builtin",
    "function.macro",
    "function.method",
    "keyword",
    "label",
    "number",
    "operator",
    "property",
    "string",
    "string.escape",
    "string.special.symbol",
    "tag",
    "type",
    "type.builtin",
    "variable",
    "variable.builtin",
    "variable.member",
    "variable.parameter",
];

static RUST_CONFIG: OnceLock<Option<HighlightConfiguration>> = OnceLock::new();
static MLIR_CONFIG: OnceLock<Option<HighlightConfiguration>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmbeddedLanguage {
    RustSource,
    RustBody,
    Mlir,
}

pub(crate) fn highlight(
    source: &str,
    source_offset: usize,
    language: EmbeddedLanguage,
) -> Vec<RawSemanticToken> {
    let (owned_source, kept_start, kept_end) = match language {
        EmbeddedLanguage::RustBody => {
            const PREFIX: &str = "fn __reussir_embedded() {\n";
            const SUFFIX: &str = "\n}";
            let masked = mask_polyffi_placeholders(source);
            let mut wrapped = String::with_capacity(PREFIX.len() + masked.len() + SUFFIX.len());
            wrapped.push_str(PREFIX);
            wrapped.push_str(&masked);
            wrapped.push_str(SUFFIX);
            let end = PREFIX.len() + masked.len();
            (wrapped, PREFIX.len(), end)
        }
        EmbeddedLanguage::RustSource => {
            let masked = mask_polyffi_placeholders(source);
            let end = masked.len();
            (masked, 0, end)
        }
        EmbeddedLanguage::Mlir => (source.to_owned(), 0, source.len()),
    };

    let Some(config) = configuration(language) else {
        tracing::warn!(
            ?language,
            "embedded Tree-sitter configuration is unavailable"
        );
        return Vec::new();
    };
    let mut highlighter = Highlighter::new();
    let Ok(events) = highlighter.highlight(config, owned_source.as_bytes(), None, |_| None) else {
        tracing::warn!(
            ?language,
            "Tree-sitter could not start highlighting embedded block"
        );
        return Vec::new();
    };

    let mut stack = Vec::new();
    let mut output = Vec::new();
    for event in events {
        let Ok(event) = event else {
            tracing::warn!(
                ?language,
                "Tree-sitter stopped while highlighting embedded block"
            );
            return Vec::new();
        };
        match event {
            HighlightEvent::HighlightStart(highlight) => stack.push(highlight.0),
            HighlightEvent::HighlightEnd => {
                stack.pop();
            }
            HighlightEvent::Source { start, end } => {
                let Some(&capture) = stack.last() else {
                    continue;
                };
                let start = start.max(kept_start);
                let end = end.min(kept_end);
                if start >= end {
                    continue;
                }
                if let Some((kind, modifiers)) = capture_style(capture) {
                    output.push(RawSemanticToken {
                        start: source_offset + start - kept_start,
                        end: source_offset + end - kept_start,
                        kind,
                        modifiers,
                    });
                }
            }
        }
    }
    output
}

fn configuration(language: EmbeddedLanguage) -> Option<&'static HighlightConfiguration> {
    match language {
        EmbeddedLanguage::RustSource | EmbeddedLanguage::RustBody => RUST_CONFIG
            .get_or_init(|| {
                let mut config = HighlightConfiguration::new(
                    tree_sitter_rust::LANGUAGE.into(),
                    "rust",
                    tree_sitter_rust::HIGHLIGHTS_QUERY,
                    tree_sitter_rust::INJECTIONS_QUERY,
                    "",
                )
                .ok()?;
                config.configure(CAPTURE_NAMES);
                Some(config)
            })
            .as_ref(),
        EmbeddedLanguage::Mlir => MLIR_CONFIG
            .get_or_init(|| {
                let mut config = HighlightConfiguration::new(
                    tree_sitter_mlir::LANGUAGE.into(),
                    "mlir",
                    tree_sitter_mlir::HIGHLIGHTS_QUERY,
                    tree_sitter_mlir::INJECTIONS_QUERY,
                    tree_sitter_mlir::LOCALS_QUERY,
                )
                .ok()?;
                config.configure(CAPTURE_NAMES);
                Some(config)
            })
            .as_ref(),
    }
}

fn capture_style(capture: usize) -> Option<(SemanticKind, u32)> {
    let name = *CAPTURE_NAMES.get(capture)?;
    let declaration = 0;
    let readonly = crate::semantic::modifier::READONLY;
    let default_library = crate::semantic::modifier::DEFAULT_LIBRARY;
    Some(match name {
        "attribute" => (SemanticKind::Decorator, declaration),
        "boolean" => (SemanticKind::Keyword, declaration),
        "comment" => (SemanticKind::Comment, declaration),
        "constant" => (SemanticKind::Variable, readonly),
        "constant.builtin" => (SemanticKind::Variable, readonly | default_library),
        "constructor" => (SemanticKind::Function, declaration),
        "function" => (SemanticKind::Function, declaration),
        "function.builtin" => (SemanticKind::Function, default_library),
        "function.macro" => (SemanticKind::Macro, declaration),
        "function.method" => (SemanticKind::Method, declaration),
        "keyword" => (SemanticKind::Keyword, declaration),
        "label" | "tag" => (SemanticKind::Variable, declaration),
        "number" => (SemanticKind::Number, declaration),
        "operator" => (SemanticKind::Operator, declaration),
        "property" | "variable.member" => (SemanticKind::Property, declaration),
        "string" | "string.escape" | "string.special.symbol" => (SemanticKind::String, declaration),
        "type" => (SemanticKind::Type, declaration),
        "type.builtin" => (SemanticKind::Type, default_library),
        "variable" => (SemanticKind::Variable, declaration),
        "variable.builtin" => (SemanticKind::Variable, default_library),
        "variable.parameter" => (SemanticKind::Parameter, declaration),
        _ => return None,
    })
}

/// Replace `[:T:]`-style poly-FFI placeholders with valid ASCII Rust
/// identifiers of exactly the same byte width. The highlighted ranges can
/// therefore be applied directly to the original Reussir document.
fn mask_polyffi_placeholders(source: &str) -> String {
    let mut bytes = source.as_bytes().to_vec();
    let mut cursor = 0;
    while cursor + 3 < bytes.len() {
        if bytes[cursor] != b'[' || bytes[cursor + 1] != b':' {
            cursor += 1;
            continue;
        }
        let Some(relative_end) = bytes[cursor + 2..]
            .windows(2)
            .position(|window| window == b":]")
        else {
            break;
        };
        let end = cursor + 2 + relative_end + 2;
        bytes[cursor..end].fill(b'_');
        bytes[cursor] = b'p';
        cursor = end;
    }
    // Every non-placeholder byte came from valid UTF-8, and placeholders are
    // replaced byte-for-byte with ASCII.
    String::from_utf8(bytes).expect("placeholder masking preserves UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_mask_preserves_byte_offsets() {
        let input = "Vec::<[:T:]>::new() + [:世界:]";
        let masked = mask_polyffi_placeholders(input);
        assert_eq!(masked.len(), input.len());
        assert_eq!(&masked[..7], "Vec::<p");
        assert!(masked.is_ascii());
    }
}
