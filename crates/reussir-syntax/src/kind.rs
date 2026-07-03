//! Syntax kinds for the Reussir surface language.
//!
//! A single [`SyntaxKind`] enum covers both tokens (produced by the lexer)
//! and composite nodes (produced by the parser). The enum implements
//! [`cstree::Syntax`] so it can be stored in a lossless `cstree` green tree.

use cstree::Syntax;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Syntax)]
#[repr(u32)]
#[non_exhaustive]
pub enum SyntaxKind {
    // ===== Tokens: trivia =====
    Whitespace,
    LineComment,
    BlockComment,

    // ===== Tokens: literals and identifiers =====
    Ident,
    IntLit,
    FloatLit,
    StringLit,

    // ===== Tokens: hard keywords =====
    FnKw,
    StructKw,
    EnumKw,
    PubKw,
    ModKw,
    LetKw,
    IfKw,
    ElseKw,
    MatchKw,
    RegionalKw,
    ExternKw,
    AsKw,
    TrueKw,
    FalseKw,

    // ===== Tokens: punctuation =====
    /// `::`
    PathSep,
    /// `->`
    Arrow,
    /// `=>`
    FatArrow,
    /// `:=`
    ColonEq,
    /// `==`
    EqEq,
    /// `!=`
    BangEq,
    /// `<=`
    LtEq,
    /// `>=`
    GtEq,
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,
    /// `..`
    DotDot,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `<`
    LAngle,
    /// `>`
    RAngle,
    /// `:`
    Colon,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `=`
    Eq,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `!`
    Bang,
    /// `|`
    Pipe,
    /// `_`
    Underscore,
    /// Unrecognized input or token-level error (kept in the tree for
    /// losslessness).
    ErrorToken,

    // ===== Nodes: top level =====
    SourceFile,
    /// Skipped tokens produced by error recovery.
    ErrorNode,

    // ===== Nodes: statements =====
    FnStmt,
    StructStmt,
    EnumStmt,
    ModStmt,
    ExternTrampolineStmt,
    GenericParamList,
    GenericParam,
    ParamList,
    Param,
    RetType,
    NamedFields,
    NamedField,
    UnnamedFields,
    UnnamedField,
    VariantList,
    Variant,
    /// `[shared]`, `[regional]`, ... capability annotation on records.
    Capability,
    /// `[field]` flag on record fields.
    FieldFlag,
    /// `[flex]` flag on parameters/return types/let bindings.
    FlexFlag,

    // ===== Nodes: paths and types =====
    Path,
    /// A primitive type name: `i32`, `f64`, `bool`, `str`, `unit`, ...
    PrimType,
    /// A user type expression: `Path` or `Path<T, U>`.
    PathType,
    /// `T -> U` (the left-hand side may be a parenthesized list).
    ArrowType,
    /// `(T, U)` in arrow-argument position.
    ParenTypeList,
    /// `<T, _, U>` type argument list on expressions.
    TypeArgList,
    /// `_` placeholder inside a type argument list.
    InferType,

    // ===== Nodes: expressions =====
    BlockExpr,
    ParenExpr,
    IfExpr,
    LetExpr,
    MatchExpr,
    MatchArm,
    RegionalExpr,
    LambdaExpr,
    LambdaParamList,
    LambdaParam,
    BinExpr,
    PrefixExpr,
    CastExpr,
    CallExpr,
    ArgList,
    AccessChain,
    AccessSeg,
    AssignExpr,
    LiteralExpr,
    VarExpr,
    FuncCallExpr,
    CtorCallExpr,
    CtorArgList,
    CtorArg,

    // ===== Nodes: patterns =====
    Pattern,
    PatGuard,
    WildcardPat,
    BindPat,
    CtorPat,
    ConstPat,
    PatArgList,
    PatArg,
    /// `..` inside a constructor pattern argument list.
    PatRest,

    /// End-of-file sentinel used by the parser; never stored in the tree.
    Eof,
}

use SyntaxKind::*;

impl SyntaxKind {
    pub fn is_trivia(self) -> bool {
        matches!(self, Whitespace | LineComment | BlockComment)
    }

    /// Tokens that may start a top-level statement. Used as the statement
    /// level recovery set.
    pub fn starts_stmt(self) -> bool {
        matches!(
            self,
            FnKw | StructKw | EnumKw | PubKw | ModKw | ExternKw | RegionalKw
        )
    }

    /// Can this token serve as an identifier? The surface language has no
    /// reserved words at identifier positions, so keyword tokens are
    /// accepted wherever an identifier is expected; keyword-led constructs
    /// win simply by being parsed first.
    pub fn is_ident_like(self) -> bool {
        matches!(
            self,
            Ident
                | FnKw
                | StructKw
                | EnumKw
                | PubKw
                | ModKw
                | LetKw
                | IfKw
                | ElseKw
                | MatchKw
                | RegionalKw
                | ExternKw
                | AsKw
                | TrueKw
                | FalseKw
        )
    }

    /// A human-readable description used in diagnostics.
    pub fn describe(self) -> &'static str {
        match self {
            Whitespace => "whitespace",
            LineComment | BlockComment => "a comment",
            Ident => "an identifier",
            IntLit => "an integer literal",
            FloatLit => "a floating-point literal",
            StringLit => "a string literal",
            FnKw => "`fn`",
            StructKw => "`struct`",
            EnumKw => "`enum`",
            PubKw => "`pub`",
            ModKw => "`mod`",
            LetKw => "`let`",
            IfKw => "`if`",
            ElseKw => "`else`",
            MatchKw => "`match`",
            RegionalKw => "`regional`",
            ExternKw => "`extern`",
            AsKw => "`as`",
            TrueKw => "`true`",
            FalseKw => "`false`",
            PathSep => "`::`",
            Arrow => "`->`",
            FatArrow => "`=>`",
            ColonEq => "`:=`",
            EqEq => "`==`",
            BangEq => "`!=`",
            LtEq => "`<=`",
            GtEq => "`>=`",
            AmpAmp => "`&&`",
            PipePipe => "`||`",
            DotDot => "`..`",
            LParen => "`(`",
            RParen => "`)`",
            LBrace => "`{`",
            RBrace => "`}`",
            LBracket => "`[`",
            RBracket => "`]`",
            LAngle => "`<`",
            RAngle => "`>`",
            Colon => "`:`",
            Semicolon => "`;`",
            Comma => "`,`",
            Dot => "`.`",
            Eq => "`=`",
            Plus => "`+`",
            Minus => "`-`",
            Star => "`*`",
            Slash => "`/`",
            Percent => "`%`",
            Bang => "`!`",
            Pipe => "`|`",
            Underscore => "`_`",
            Eof => "the end of the input",
            ErrorToken => "invalid input",
            _ => "a syntax node",
        }
    }
}

/// The lossless syntax tree types for the Reussir surface language.
pub type SyntaxNode = cstree::syntax::SyntaxNode<SyntaxKind>;
pub type SyntaxToken = cstree::syntax::SyntaxToken<SyntaxKind>;
pub type ResolvedNode = cstree::syntax::ResolvedNode<SyntaxKind>;
pub type ResolvedToken = cstree::syntax::ResolvedToken<SyntaxKind>;

/// The interned-string key for a token's text, and the resolver that turns a key
/// back into source text. Equal keys mean equal text within a single parse, so
/// identifiers can be stored and compared as cheap `Copy` keys. [`InternKey`]
/// exposes `into_u32`/`try_from_u32` (used to mint keys in tests).
pub use cstree::interning::{InternKey, Resolver, TokenKey};
