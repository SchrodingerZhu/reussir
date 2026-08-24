//! LSP semantic-token encoding for Reussir.

use async_lsp::lsp_types::{SemanticToken, SemanticTokenModifier, SemanticTokenType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
pub(crate) enum SemanticKind {
    Namespace,
    Type,
    Struct,
    Enum,
    Interface,
    TypeParameter,
    Parameter,
    Variable,
    Property,
    EnumMember,
    Function,
    Method,
    Macro,
    Keyword,
    Modifier,
    Comment,
    String,
    Number,
    Operator,
    Decorator,
}

/// One row per `SemanticKind`, in discriminant order. The advertised legend
/// and the test decoder both derive from this table, and a test asserts each
/// row's position matches its discriminant, so the three cannot drift apart.
pub(crate) const TOKEN_LEGEND: &[(SemanticKind, SemanticTokenType)] = &[
    (SemanticKind::Namespace, SemanticTokenType::NAMESPACE),
    (SemanticKind::Type, SemanticTokenType::TYPE),
    (SemanticKind::Struct, SemanticTokenType::STRUCT),
    (SemanticKind::Enum, SemanticTokenType::ENUM),
    (SemanticKind::Interface, SemanticTokenType::INTERFACE),
    (SemanticKind::TypeParameter, SemanticTokenType::TYPE_PARAMETER),
    (SemanticKind::Parameter, SemanticTokenType::PARAMETER),
    (SemanticKind::Variable, SemanticTokenType::VARIABLE),
    (SemanticKind::Property, SemanticTokenType::PROPERTY),
    (SemanticKind::EnumMember, SemanticTokenType::ENUM_MEMBER),
    (SemanticKind::Function, SemanticTokenType::FUNCTION),
    (SemanticKind::Method, SemanticTokenType::METHOD),
    (SemanticKind::Macro, SemanticTokenType::MACRO),
    (SemanticKind::Keyword, SemanticTokenType::KEYWORD),
    (SemanticKind::Modifier, SemanticTokenType::MODIFIER),
    (SemanticKind::Comment, SemanticTokenType::COMMENT),
    (SemanticKind::String, SemanticTokenType::STRING),
    (SemanticKind::Number, SemanticTokenType::NUMBER),
    (SemanticKind::Operator, SemanticTokenType::OPERATOR),
    (SemanticKind::Decorator, SemanticTokenType::DECORATOR),
];

/// One row per modifier bit, in bit order; validated the same way as
/// [`TOKEN_LEGEND`].
pub(crate) const MODIFIER_LEGEND: &[(u32, SemanticTokenModifier)] = &[
    (modifier::DECLARATION, SemanticTokenModifier::DECLARATION),
    (modifier::READONLY, SemanticTokenModifier::READONLY),
    (modifier::DEFAULT_LIBRARY, SemanticTokenModifier::DEFAULT_LIBRARY),
];

pub(crate) fn legend_token_types() -> Vec<SemanticTokenType> {
    TOKEN_LEGEND.iter().map(|(_, ty)| ty.clone()).collect()
}

pub(crate) fn legend_token_modifiers() -> Vec<SemanticTokenModifier> {
    MODIFIER_LEGEND
        .iter()
        .map(|(_, modifier)| modifier.clone())
        .collect()
}

pub(crate) mod modifier {
    pub(crate) const DECLARATION: u32 = 1 << 0;
    pub(crate) const READONLY: u32 = 1 << 1;
    pub(crate) const DEFAULT_LIBRARY: u32 = 1 << 2;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawSemanticToken {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) kind: SemanticKind,
    pub(crate) modifiers: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AbsoluteSemanticToken {
    line: u32,
    start: u32,
    length: u32,
    kind: SemanticKind,
    modifiers: u32,
}

fn encode(source: &str, mut raw: Vec<RawSemanticToken>) -> Vec<SemanticToken> {
    let lines = LineIndex::new(source);
    // Sorting lets one forward-only UTF-16 cursor serve every token, keeping
    // column conversion linear instead of re-encoding each line prefix.
    raw.sort_unstable_by_key(|token| (token.start, token.end));
    let mut cursor = Utf16Cursor::default();
    let mut absolute = Vec::new();
    for token in raw {
        absolute.extend(lines.split(source, token, &mut cursor));
    }
    absolute.sort_by_key(|token| (token.line, token.start, token.length, token.kind));

    let mut normalized: Vec<AbsoluteSemanticToken> = Vec::with_capacity(absolute.len());
    for mut token in absolute {
        if token.length == 0 {
            continue;
        }
        if let Some(previous) = normalized
            .last_mut()
            .filter(|previous| previous.line == token.line)
        {
            let previous_end = previous.start + previous.length;
            if token.start < previous_end {
                let overlap = previous_end - token.start;
                if overlap >= token.length {
                    continue;
                }
                token.start += overlap;
                token.length -= overlap;
            }
            if previous.start + previous.length == token.start
                && previous.kind == token.kind
                && previous.modifiers == token.modifiers
            {
                previous.length += token.length;
                continue;
            }
        }
        normalized.push(token);
    }

    let mut previous_line = 0;
    let mut previous_start = 0;
    normalized
        .into_iter()
        .map(|token| {
            let delta_line = token.line - previous_line;
            let delta_start = if delta_line == 0 {
                token.start - previous_start
            } else {
                token.start
            };
            previous_line = token.line;
            previous_start = token.start;
            SemanticToken {
                delta_line,
                delta_start,
                length: token.length,
                token_type: token.kind as u32,
                token_modifiers_bitset: token.modifiers,
            }
        })
        .collect()
}

#[derive(Debug)]
struct LineInfo {
    start: usize,
    content_end: usize,
    next_start: usize,
}

#[derive(Debug)]
struct LineIndex {
    lines: Vec<LineInfo>,
}

/// Forward-only byte-offset → UTF-16-column state shared across tokens.
/// A lookup behind the previous position falls back to the line start, so it
/// stays correct for arbitrary inputs and linear for sorted ones.
#[derive(Debug, Default)]
struct Utf16Cursor {
    line: usize,
    byte: usize,
    utf16: u32,
}

impl LineIndex {
    fn new(source: &str) -> Self {
        // The LSP position contract (and VS Code's document model) treats
        // `\n`, `\r\n`, and a lone `\r` as line terminators.
        let bytes = source.as_bytes();
        let mut lines = Vec::new();
        let mut start = 0;
        let mut index = 0;
        while index < bytes.len() {
            let next_start = match bytes[index] {
                b'\n' => index + 1,
                b'\r' if bytes.get(index + 1) == Some(&b'\n') => index + 2,
                b'\r' => index + 1,
                _ => {
                    index += 1;
                    continue;
                }
            };
            lines.push(LineInfo {
                start,
                content_end: index,
                next_start,
            });
            index = next_start;
            start = next_start;
        }
        lines.push(LineInfo {
            start,
            content_end: source.len(),
            next_start: source.len(),
        });
        Self { lines }
    }

    fn split(
        &self,
        source: &str,
        token: RawSemanticToken,
        columns: &mut Utf16Cursor,
    ) -> Vec<AbsoluteSemanticToken> {
        let mut output = Vec::new();
        let mut cursor = token.start.min(source.len());
        let end = token.end.min(source.len());
        while cursor < end {
            let line_index = self
                .lines
                .partition_point(|line| line.start <= cursor)
                .saturating_sub(1);
            let line = &self.lines[line_index];
            let segment_end = end.min(line.content_end);
            if cursor < segment_end {
                let start = self.utf16_column(source, columns, line_index, cursor);
                let length = self.utf16_column(source, columns, line_index, segment_end) - start;
                output.push(AbsoluteSemanticToken {
                    line: line_index as u32,
                    start,
                    length,
                    kind: token.kind,
                    modifiers: token.modifiers,
                });
            }
            if end <= line.content_end || line.next_start <= cursor {
                break;
            }
            cursor = line.next_start;
        }
        output
    }

    fn utf16_column(
        &self,
        source: &str,
        columns: &mut Utf16Cursor,
        line_index: usize,
        byte: usize,
    ) -> u32 {
        if line_index != columns.line || byte < columns.byte {
            columns.line = line_index;
            columns.byte = self.lines[line_index].start;
            columns.utf16 = 0;
        }
        columns.utf16 += source[columns.byte..byte].encode_utf16().count() as u32;
        columns.byte = byte;
        columns.utf16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl TryFrom<u32> for SemanticKind {
        type Error = ();

        fn try_from(value: u32) -> Result<Self, Self::Error> {
            TOKEN_LEGEND
                .get(value as usize)
                .map(|(kind, _)| *kind)
                .ok_or(())
        }
    }

    #[test]
    fn legend_rows_match_their_discriminants() {
        for (index, (kind, _)) in TOKEN_LEGEND.iter().enumerate() {
            assert_eq!(*kind as usize, index, "row {index} is out of order");
        }
        for (index, (bit, _)) in MODIFIER_LEGEND.iter().enumerate() {
            assert_eq!(*bit, 1 << index, "modifier row {index} is out of order");
        }
    }

    #[test]
    fn uses_utf16_positions_and_splits_multiline_comments() {
        let source = "fn 😀() { /* α\r\nβ */ let x = 1 }";
        let raw = vec![RawSemanticToken {
            start: source.find("/*").unwrap(),
            end: source.find("*/").unwrap() + 2,
            kind: SemanticKind::Comment,
            modifiers: 0,
        }];
        let encoded = encode(source, raw);
        assert_eq!(encoded.len(), 2);
        assert_eq!((encoded[0].delta_line, encoded[0].delta_start), (0, 10));
        assert_eq!((encoded[1].delta_line, encoded[1].delta_start), (1, 0));
    }

    #[test]
    fn lone_carriage_return_terminates_a_line() {
        let source = "let a = 1\rlet b = 2";
        let start = source.rfind("let").unwrap();
        let raw = vec![RawSemanticToken {
            start,
            end: start + 3,
            kind: SemanticKind::Keyword,
            modifiers: 0,
        }];
        let encoded = encode(source, raw);
        assert_eq!(encoded.len(), 1);
        assert_eq!((encoded[0].delta_line, encoded[0].delta_start), (1, 0));
    }

    #[test]
    fn line_index_handles_mixed_line_endings() {
        let index = LineIndex::new("a\r\nb\rc\nd");
        let starts: Vec<usize> = index.lines.iter().map(|line| line.start).collect();
        let content_ends: Vec<usize> = index.lines.iter().map(|line| line.content_end).collect();
        assert_eq!(starts, [0, 3, 5, 7]);
        assert_eq!(content_ends, [1, 4, 6, 8]);
    }
}
