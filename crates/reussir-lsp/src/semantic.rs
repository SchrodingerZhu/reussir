//! Reussir cstree classification and LSP semantic-token encoding.

use async_lsp::lsp_types::{SemanticToken, SemanticTokenModifier, SemanticTokenType};
use reussir_syntax::kind::{ResolvedNode, ResolvedToken, SyntaxKind};
use reussir_syntax::parse;

use crate::embedded::{self, EmbeddedLanguage};

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

pub(crate) const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::NAMESPACE,
    SemanticTokenType::TYPE,
    SemanticTokenType::STRUCT,
    SemanticTokenType::ENUM,
    SemanticTokenType::INTERFACE,
    SemanticTokenType::TYPE_PARAMETER,
    SemanticTokenType::PARAMETER,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::PROPERTY,
    SemanticTokenType::ENUM_MEMBER,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::METHOD,
    SemanticTokenType::MACRO,
    SemanticTokenType::KEYWORD,
    SemanticTokenType::MODIFIER,
    SemanticTokenType::COMMENT,
    SemanticTokenType::STRING,
    SemanticTokenType::NUMBER,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::DECORATOR,
];

pub(crate) const TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DECLARATION,
    SemanticTokenModifier::READONLY,
    SemanticTokenModifier::DEFAULT_LIBRARY,
];

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

pub(crate) fn semantic_tokens(source: &str) -> Vec<SemanticToken> {
    let parsed = parse(source);
    let mut raw = Vec::new();
    for element in parsed.root.descendants_with_tokens() {
        let Some(token) = element.into_token() else {
            continue;
        };
        if token.kind() == SyntaxKind::RawMlirLiteral {
            raw.extend(highlight_embedded(token));
        } else if let Some((kind, modifiers)) = classify_reussir(token) {
            let range = token.text_range();
            raw.push(RawSemanticToken {
                start: u32::from(range.start()) as usize,
                end: u32::from(range.end()) as usize,
                kind,
                modifiers,
            });
        }
    }
    encode(source, raw)
}

fn highlight_embedded(token: &ResolvedToken) -> Vec<RawSemanticToken> {
    let text = token.text();
    let Some(body) = text
        .strip_prefix("[{")
        .and_then(|text| text.strip_suffix("}]"))
    else {
        return Vec::new();
    };
    let token_start = u32::from(token.text_range().start()) as usize;
    let body_start = token_start + 2;
    let language = match token.parent().kind() {
        SyntaxKind::TransformStmt => EmbeddedLanguage::Mlir,
        SyntaxKind::ExternSourceStmt => EmbeddedLanguage::RustSource,
        SyntaxKind::FnStmt => EmbeddedLanguage::RustBody,
        _ => return Vec::new(),
    };
    embedded::highlight(body, body_start, language)
}

fn classify_reussir(token: &ResolvedToken) -> Option<(SemanticKind, u32)> {
    use SemanticKind as S;
    use SyntaxKind as K;

    let parent = token.parent();
    let declaration = modifier::DECLARATION;

    // Declaration sites take priority over the lexical kind. Reussir permits
    // hard-keyword spellings wherever an identifier is expected.
    if token.kind().is_ident_like() {
        let declaration_kind = match parent.kind() {
            K::GenericParam if first_ident(parent, token) => Some(S::TypeParameter),
            K::Param | K::LambdaParam if first_ident(parent, token) => Some(S::Parameter),
            K::NamedField if first_ident(parent, token) => Some(S::Property),
            K::Variant if first_ident(parent, token) => Some(S::EnumMember),
            K::BindPat if path_basename(token) => Some(S::Variable),
            K::FnStmt if name_after(parent, token, |candidate| candidate.kind() == K::FnKw) => {
                Some(
                    if parent
                        .ancestors()
                        .any(|node| matches!(node.kind(), K::ImplStmt | K::TraitStmt))
                    {
                        S::Method
                    } else {
                        S::Function
                    },
                )
            }
            K::StructStmt
                if name_after(parent, token, |candidate| candidate.kind() == K::StructKw) =>
            {
                Some(S::Struct)
            }
            K::EnumStmt if name_after(parent, token, |candidate| candidate.kind() == K::EnumKw) => {
                Some(S::Enum)
            }
            K::ModStmt if name_after(parent, token, |candidate| candidate.kind() == K::ModKw) => {
                Some(S::Namespace)
            }
            K::TraitStmt if contextual_name_after(parent, token, "trait") => Some(S::Interface),
            K::LetExpr if name_after(parent, token, |candidate| candidate.kind() == K::LetKw) => {
                Some(S::Variable)
            }
            K::ImportStmt if name_after(parent, token, |candidate| candidate.kind() == K::AsKw) => {
                Some(S::Namespace)
            }
            K::CtorArg | K::PatArg
                if has_direct_token(parent, K::Colon) && first_ident(parent, token) =>
            {
                Some(S::Property)
            }
            _ => None,
        };
        if let Some(kind) = declaration_kind {
            return Some((kind, declaration));
        }

        if matches!(
            parent.kind(),
            K::Capability | K::FieldFlag | K::FlexFlag | K::VisFlag
        ) {
            return Some((S::Modifier, 0));
        }
        if parent.kind() == K::AttrList {
            return Some((S::Decorator, 0));
        }
        if parent.kind() == K::TransformStmt && token.text() == "transform"
            || parent.kind() == K::TraitStmt
                && token.text() == "trait"
                && first_ident(parent, token)
            || parent.kind() == K::ImplStmt && matches!(token.text(), "impl" | "for")
            || parent.kind() == K::ExternTrampolineStmt && token.text() == "trampoline"
        {
            return Some((S::Keyword, 0));
        }
        if parent.kind() == K::AccessSeg {
            let is_callee = parent
                .parent()
                .filter(|node| node.kind() == K::AccessChain)
                .and_then(ResolvedNode::parent)
                .is_some_and(|node| node.kind() == K::CallExpr);
            let kind = if is_callee { S::Method } else { S::Property };
            return Some((kind, 0));
        }
        if parent.kind() == K::PrimType {
            return Some((S::Type, modifier::DEFAULT_LIBRARY));
        }
        if parent.kind() == K::Path {
            return Some(classify_path(token));
        }
    }

    if parent.kind() == K::AccessSeg && matches!(token.kind(), K::IntLit | K::FloatLit) {
        return Some((S::Property, 0));
    }

    match token.kind() {
        K::Whitespace | K::ErrorToken => None,
        K::LineComment | K::BlockComment => Some((S::Comment, 0)),
        K::StringLit | K::CharLit => Some((S::String, 0)),
        K::IntLit | K::FloatLit => Some((S::Number, 0)),
        K::TrueKw | K::FalseKw => Some((S::Keyword, 0)),
        K::PubKw | K::RegionalKw
            if matches!(
                parent.kind(),
                K::FnStmt | K::StructStmt | K::EnumStmt | K::TraitStmt
            ) =>
        {
            Some((S::Modifier, 0))
        }
        kind if is_hard_keyword(kind) => Some((S::Keyword, 0)),
        kind if is_operator(kind, parent.kind()) => Some((S::Operator, 0)),
        K::Ident => Some((S::Variable, 0)),
        _ => None,
    }
}

fn classify_path(token: &ResolvedToken) -> (SemanticKind, u32) {
    use SemanticKind as S;
    use SyntaxKind as K;

    if !path_basename(token) {
        return (S::Namespace, 0);
    }
    let path = token.parent();
    let owner = path.parent().map(|node| node.kind());
    let ancestors = || path.ancestors().map(|node| node.kind());
    match owner {
        Some(K::PathType | K::GenericParam) => (S::Type, 0),
        Some(K::FuncCallExpr | K::ExternTrampolineStmt) => (S::Function, 0),
        Some(K::CtorCallExpr | K::CtorPat) => (S::Type, 0),
        Some(K::ImportStmt) => (S::Namespace, 0),
        Some(K::BindPat) => (S::Variable, modifier::DECLARATION),
        Some(K::VarExpr) => (S::Variable, 0),
        _ if ancestors().any(|kind| kind == K::PathType) => (S::Type, 0),
        _ => (S::Variable, 0),
    }
}

fn first_ident(node: &ResolvedNode, token: &ResolvedToken) -> bool {
    direct_tokens(node)
        .find(|candidate| candidate.kind().is_ident_like())
        .is_some_and(|candidate| same_token(candidate, token))
}

fn name_after(
    node: &ResolvedNode,
    token: &ResolvedToken,
    mut is_introducer: impl FnMut(&ResolvedToken) -> bool,
) -> bool {
    let mut seen = false;
    for candidate in direct_tokens(node) {
        if !seen && is_introducer(candidate) {
            seen = true;
        } else if seen && candidate.kind().is_ident_like() {
            return same_token(candidate, token);
        }
    }
    false
}

fn contextual_name_after(node: &ResolvedNode, token: &ResolvedToken, introducer: &str) -> bool {
    name_after(node, token, |candidate| candidate.text() == introducer)
}

fn path_basename(token: &ResolvedToken) -> bool {
    token.parent().kind() == SyntaxKind::Path
        && direct_tokens(token.parent())
            .filter(|candidate| candidate.kind().is_ident_like())
            .last()
            .is_some_and(|candidate| same_token(candidate, token))
}

fn has_direct_token(node: &ResolvedNode, kind: SyntaxKind) -> bool {
    direct_tokens(node).any(|token| token.kind() == kind)
}

fn direct_tokens(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedToken> {
    node.children_with_tokens()
        .filter_map(|element| element.as_token().copied())
        .filter(|token| !token.kind().is_trivia())
}

fn same_token(left: &ResolvedToken, right: &ResolvedToken) -> bool {
    left.text_range() == right.text_range()
}

fn is_hard_keyword(kind: SyntaxKind) -> bool {
    use SyntaxKind as K;
    matches!(
        kind,
        K::FnKw
            | K::StructKw
            | K::EnumKw
            | K::PubKw
            | K::ModKw
            | K::LetKw
            | K::IfKw
            | K::ElseKw
            | K::MatchKw
            | K::RegionalKw
            | K::ExternKw
            | K::AsKw
            | K::TrueKw
            | K::FalseKw
            | K::ImportKw
    )
}

fn is_operator(kind: SyntaxKind, parent: SyntaxKind) -> bool {
    use SyntaxKind as K;
    matches!(
        kind,
        K::Arrow
            | K::FatArrow
            | K::ColonEq
            | K::EqEq
            | K::BangEq
            | K::LtEq
            | K::GtEq
            | K::AmpAmp
            | K::PipePipe
            | K::DotDot
            | K::Eq
            | K::Plus
            | K::Minus
            | K::Star
            | K::Slash
            | K::Percent
            | K::Bang
            | K::Pipe
            | K::Amp
            | K::Caret
    ) || matches!(kind, K::LAngle | K::RAngle) && parent == K::BinExpr
}

fn encode(source: &str, raw: Vec<RawSemanticToken>) -> Vec<SemanticToken> {
    let lines = LineIndex::new(source);
    let mut absolute = Vec::new();
    for token in raw {
        absolute.extend(lines.split(source, token));
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

impl LineIndex {
    fn new(source: &str) -> Self {
        let mut lines = Vec::new();
        let mut start = 0;
        for (newline, _) in source.match_indices('\n') {
            let content_end = if newline > start && source.as_bytes()[newline - 1] == b'\r' {
                newline - 1
            } else {
                newline
            };
            lines.push(LineInfo {
                start,
                content_end,
                next_start: newline + 1,
            });
            start = newline + 1;
        }
        lines.push(LineInfo {
            start,
            content_end: source.len(),
            next_start: source.len(),
        });
        Self { lines }
    }

    fn split(&self, source: &str, token: RawSemanticToken) -> Vec<AbsoluteSemanticToken> {
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
                let start = source[line.start..cursor].encode_utf16().count() as u32;
                let length = source[cursor..segment_end].encode_utf16().count() as u32;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(source: &str) -> Vec<(String, SemanticKind, u32, u32)> {
        let encoded = semantic_tokens(source);
        let lines: Vec<&str> = source.split('\n').collect();
        let mut line = 0;
        let mut start = 0;
        encoded
            .into_iter()
            .map(|token| {
                line += token.delta_line;
                start = if token.delta_line == 0 {
                    start + token.delta_start
                } else {
                    token.delta_start
                };
                let units: Vec<u16> = lines[line as usize].encode_utf16().collect();
                let text = String::from_utf16_lossy(
                    &units[start as usize..(start + token.length) as usize],
                );
                (
                    text,
                    SemanticKind::try_from(token.token_type).unwrap(),
                    line,
                    start,
                )
            })
            .collect()
    }

    impl TryFrom<u32> for SemanticKind {
        type Error = ();

        fn try_from(value: u32) -> Result<Self, Self::Error> {
            Ok(match value {
                0 => SemanticKind::Namespace,
                1 => SemanticKind::Type,
                2 => SemanticKind::Struct,
                3 => SemanticKind::Enum,
                4 => SemanticKind::Interface,
                5 => SemanticKind::TypeParameter,
                6 => SemanticKind::Parameter,
                7 => SemanticKind::Variable,
                8 => SemanticKind::Property,
                9 => SemanticKind::EnumMember,
                10 => SemanticKind::Function,
                11 => SemanticKind::Method,
                12 => SemanticKind::Macro,
                13 => SemanticKind::Keyword,
                14 => SemanticKind::Modifier,
                15 => SemanticKind::Comment,
                16 => SemanticKind::String,
                17 => SemanticKind::Number,
                18 => SemanticKind::Operator,
                19 => SemanticKind::Decorator,
                _ => return Err(()),
            })
        }
    }

    #[test]
    fn classifies_context_before_keyword_spelling() {
        let tokens = decoded("fn fn(struct: i64) -> i64 { let if = struct; if }");
        assert!(tokens.contains(&("fn".into(), SemanticKind::Function, 0, 3)));
        assert!(tokens.contains(&("struct".into(), SemanticKind::Parameter, 0, 6)));
        assert!(tokens.contains(&("if".into(), SemanticKind::Variable, 0, 32)));
        assert!(tokens.contains(&("struct".into(), SemanticKind::Variable, 0, 37)));
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
    fn classifies_declarations_paths_and_members_from_cst_context() {
        let source = r#"#[repr(fixed)]
pub struct Point<T> { pub x: T }
enum Maybe<T> { Some(T), None }
trait Show<T> { fn show(self: Point<T>) -> T; }
impl<T> Point<T> { pub fn get(self: Point<T>) -> T { self.x } }
mod math;
import core::intrinsic::math as m;
fn use_it(p: Point<i64>) -> i64 { m::sqrt(p.get()) }"#;
        let tokens = decoded(source);
        for (text, kind) in [
            ("repr", SemanticKind::Decorator),
            ("Point", SemanticKind::Struct),
            ("T", SemanticKind::TypeParameter),
            ("x", SemanticKind::Property),
            ("Maybe", SemanticKind::Enum),
            ("Some", SemanticKind::EnumMember),
            ("Show", SemanticKind::Interface),
            ("show", SemanticKind::Method),
            ("get", SemanticKind::Method),
            ("math", SemanticKind::Namespace),
            ("m", SemanticKind::Namespace),
            ("sqrt", SemanticKind::Function),
        ] {
            assert!(
                tokens
                    .iter()
                    .any(|(actual, actual_kind, _, _)| actual == text && *actual_kind == kind),
                "missing {text:?} as {kind:?}: {tokens:#?}"
            );
        }
    }

    #[test]
    fn highlights_embedded_mlir_and_rust_without_raw_overlap() {
        let source = r#"transform [{ %x = arith.constant 1 : i64 }];
extern "rust" [{ use crate::Thing; }];
#[ffi(import)] fn make<T>() -> T [{ println!("{}", 1); Vec::<[:T:]>::new() }];"#;
        let tokens = decoded(source);
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "arith.constant"
                    && *kind == SemanticKind::Function)
        );
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "use" && *kind == SemanticKind::Keyword)
        );
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "new" && *kind == SemanticKind::Function),
            "{tokens:#?}"
        );
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "println!" && *kind == SemanticKind::Macro)
        );
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "[:T:]" && *kind == SemanticKind::Type)
        );
        assert!(!tokens.iter().any(|(text, _, _, _)| text.starts_with("[{")));
    }

    #[test]
    fn malformed_reussir_still_produces_tokens() {
        let tokens = decoded("fn broken(x: i64 { let y = 1");
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "broken" && *kind == SemanticKind::Function)
        );
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "1" && *kind == SemanticKind::Number)
        );
    }

    #[test]
    fn malformed_embedded_block_does_not_hide_outer_reussir() {
        let tokens = decoded("transform [{ %x = <not mlir }]; fn okay() -> i64 { 1 }");
        assert!(
            tokens
                .iter()
                .any(|(text, kind, _, _)| text == "okay" && *kind == SemanticKind::Function)
        );
    }
}
