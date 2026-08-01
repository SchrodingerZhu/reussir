//! Surface syntax parser for the Reussir language.
//!
//! The pipeline is:
//!
//! 1. [`lexer`] — a [`logos`]-generated tokenizer (XID identifiers,
//!    string and number literals);
//! 2. `parser` — an event-driven recursive-descent parser with Pratt
//!    expression parsing, speculative parsing for the generic-argument
//!    ambiguity, structural error recovery, and [`stacker`]-guarded
//!    recursion;
//! 3. a lossless [`cstree`] green tree interned with [`lasso`];
//! 4. [`ast`] — lowering of the CST to the JSON encoding consumed (and
//!    produced) by the frontend;
//! 5. [`diagnostics`] — best-effort error reporting rendered with
//!    [`ariadne`].

pub mod ast;
pub mod diagnostics;
pub mod kind;
pub mod lexer;
pub mod literal;
pub(crate) mod parser;
pub mod source;

use diagnostics::ParseError;
use kind::{ResolvedNode, SyntaxKind};
use source::CharMap;

// The shared-interner types [`parse_with_interner`] / [`parse_repl`] take, so
// downstream crates don't need a direct cstree dependency. `Interner` is the
// trait carrying `get_or_intern` (e.g. for a REPL driver interning synthetic
// names); its `Resolver` counterpart is re-exported from [`kind`].
pub use cstree::interning::{Interner, MultiThreadedTokenInterner, new_threaded_interner};

// The single source of truth for the contextual primitive-type and
// capability names. Consumers that re-encode these sets by hand (the AST
// emitter, `surface` lowering in `reussir-core`) test against them.
pub use parser::grammar::{CAPABILITIES, PRIM_TYPES};

/// The result of parsing: a lossless syntax tree (always produced, even in
/// the presence of errors) plus collected diagnostics.
pub struct Parse {
    pub root: ResolvedNode,
    pub errors: Vec<ParseError>,
}

impl Parse {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// The interner that resolves token keys (e.g. [`kind::TokenKey`]s stored in
    /// the typed AST) back into source text.
    pub fn resolver(&self) -> &dyn kind::Resolver<kind::TokenKey> {
        &**self.root.resolver()
    }

    /// Lower the tree to the JSON document of the program (a list of
    /// statements).
    ///
    /// This is a verification oracle for cross-checking against the frontend,
    /// not part of the elaboration pipeline, so it requires a clean parse:
    /// call it only when [`Parse::ok`] holds. Lowering a tree
    /// that still contains error nodes panics (the recovery shape is not a
    /// valid AST).
    pub fn to_json(&self, map: &CharMap) -> serde_json::Value {
        assert!(
            self.ok(),
            "to_json requires an error-free parse; got {} error(s)",
            self.errors.len()
        );
        ast::prog_to_json(&self.root, map)
    }
}

/// Parse a whole source file.
pub fn parse(source: &str) -> Parse {
    parse_impl(source, lasso::Rodeo::<kind::TokenKey>::new(), |p| {
        p.source_file()
    })
}

/// Parse a whole source file, interning token text into a caller-supplied
/// shared interner.
///
/// [`TokenKey`](kind::TokenKey)s are stable across every parse that shares the
/// interner, so typed-AST keys from one parse resolve correctly against
/// another's — the property a REPL session needs to elaborate many small
/// inputs against one accumulated program. The session keeps one `Arc` clone
/// as the long-lived [`kind::Resolver`]; each parse's tree holds another.
pub fn parse_with_interner(
    source: &str,
    interner: std::sync::Arc<MultiThreadedTokenInterner>,
) -> Parse {
    parse_impl(source, interner, |p| p.source_file())
}

/// How [`parse_repl`] routed an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplInputKind {
    /// Top-level items; the tree root is a `SourceFile`.
    Items,
    /// An expression sequence (`e1; e2; ...`); the tree root is a
    /// `BlockExpr`.
    Expr,
}

/// The result of parsing one REPL input.
pub struct ReplParse {
    pub parse: Parse,
    pub kind: ReplInputKind,
}

/// Parse one REPL input as either top-level items or an expression sequence.
///
/// The route is decided by the first tokens, no backtracking: errors are
/// reported against the chosen route. Items are recognized by an item prefix
/// — `fn`/`struct`/`enum`/`mod` followed by the item's name, `extern`
/// followed by its ABI string, `pub` followed by an item form, `regional`
/// followed by `fn` — and everything else is an expression. One token of
/// lookahead past the head keyword is needed because keyword tokens are
/// contextual in identifier positions ([`SyntaxKind::is_ident_like`]): a
/// variable named `fn` may head an expression (`fn + 1`). The one ambiguous
/// prefix is an item keyword followed by `as` — `fn as i64` is a cast of a
/// variable named `fn` — which routes to expressions, so an item literally
/// named `as` cannot be entered at the REPL.
pub fn parse_repl(source: &str, interner: std::sync::Arc<MultiThreadedTokenInterner>) -> ReplParse {
    let mut kind = ReplInputKind::Expr;
    let parse = parse_impl(source, interner, |p| {
        kind = repl_route(p);
        match kind {
            ReplInputKind::Items => p.source_file(),
            ReplInputKind::Expr => p.repl_expr_seq(),
        }
    });
    ReplParse { parse, kind }
}

/// The [`parse_repl`] routing decision; see its docs for the rationale.
fn repl_route(p: &parser::Parser) -> ReplInputKind {
    use SyntaxKind::*;
    let starts_item = match p.current() {
        // `fn f`, `struct P`, ... — the keyword is an item head iff its
        // name follows; a keyword-named variable is followed by an operator
        // (`fn + 1`) or nothing. `as` is the exception: `fn as i64` is a
        // cast of a variable named `fn`.
        FnKw | ModKw | ImportKw => p.nth(1).is_ident_like() && p.nth(1) != AsKw,
        // Records may also carry a capability annotation before the name:
        // `struct [value] V { ... }`.
        StructKw | EnumKw => (p.nth(1).is_ident_like() && p.nth(1) != AsKw) || p.nth(1) == LBracket,
        // `extern "C" trampoline ...` — a string cannot follow a variable.
        ExternKw => p.nth(1) == StringLit,
        // `pub <item>` — mirrors `stmt`'s post-visibility dispatch.
        PubKw => matches!(
            p.nth(1),
            RegionalKw | FnKw | StructKw | EnumKw | ModKw | ExternKw | ImportKw
        ),
        // `regional fn` is an item; any other `regional ...` is a region
        // expression.
        RegionalKw => p.nth(1) == FnKw,
        // An outer attribute `#[...]` only ever heads an item.
        Pound => true,
        Ident if p.current_text() == "transform" && p.nth(1) == RawMlirLiteral => true,
        _ => false,
    };
    if starts_item {
        ReplInputKind::Items
    } else {
        ReplInputKind::Expr
    }
}

/// The shared parse pipeline: lex, run `entry` on the parser, then build the
/// lossless tree with `interner` (which becomes the tree's resolver).
fn parse_impl<I>(source: &str, mut interner: I, entry: impl FnOnce(&mut parser::Parser)) -> Parse
where
    I: cstree::interning::Interner<kind::TokenKey> + kind::Resolver<kind::TokenKey> + 'static,
{
    let (all_tokens, lex_errors) = lexer::tokenize(source);
    let significant: Vec<lexer::Token> = all_tokens
        .iter()
        .filter(|t| !t.kind.is_trivia() && t.kind != SyntaxKind::ErrorToken)
        .copied()
        .collect();

    let mut p = parser::Parser::new(source, significant);
    entry(&mut p);
    let (events, parse_errors) = p.finish();

    let mut errors: Vec<ParseError> = lex_errors
        .into_iter()
        .map(|e| ParseError {
            span: e.span,
            message: e.message.to_owned(),
        })
        .collect();
    errors.extend(parse_errors);
    errors.sort_by_key(|e| e.span);
    // A single unexpected token often fails several expectations in a row;
    // the first message is the most informative one.
    errors.dedup_by_key(|e| e.span.0);

    let green = parser::sink::Sink::new(source, &all_tokens, events, &mut interner).finish();
    let root = kind::SyntaxNode::new_root_with_resolver(green, interner);
    Parse { root, errors }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(source: &str) -> Parse {
        let parse = parse(source);
        assert!(
            parse.ok(),
            "unexpected errors for {source:?}: {:#?}",
            parse.errors,
        );
        // The tree is lossless: concatenating all tokens reproduces the
        // input.
        assert_eq!(parse.root.text(), source);
        parse
    }

    fn json_of(source: &str) -> serde_json::Value {
        let map = CharMap::new(source);
        parse_ok(source).to_json(&map)
    }

    /// The AST emitter re-encodes [`PRIM_TYPES`] and [`CAPABILITIES`] in
    /// `ast::prim_type_json` / the capability match, each ending in
    /// `unreachable!`. Feeding every name the parser recognizes through the
    /// emitter turns a missed mirror entry into a test failure instead of a
    /// panic on valid user input.
    #[test]
    fn every_primitive_type_reaches_the_ast_emitter() {
        for prim in PRIM_TYPES {
            json_of(&format!("fn f(x: {prim}) {{ }}"));
        }
    }

    #[test]
    fn every_capability_reaches_the_ast_emitter() {
        for cap in CAPABILITIES {
            json_of(&format!("struct [{cap}] S {{ x: i64 }}"));
        }
    }

    #[test]
    fn shared_interner_keeps_token_keys_stable_across_parses() {
        use cstree::interning::Resolver;

        let interner = std::sync::Arc::new(new_threaded_interner());
        let a = parse_with_interner("fn foo() -> i32 { 1 }", interner.clone());
        let b = parse_with_interner("fn bar() -> i32 { foo() }", interner.clone());
        assert!(a.ok() && b.ok());

        // The `foo` token in parse B resolves through parse A's resolver (and
        // the session interner) to the same text, i.e. the keys share one
        // key space.
        let key_in_a = token_key_of(&a, "foo").expect("foo in a");
        let key_in_b = token_key_of(&b, "foo").expect("foo in b");
        assert_eq!(key_in_a, key_in_b);
        assert_eq!(interner.try_resolve(key_in_b), Some("foo"));
    }

    /// The key of the first token with the given text.
    fn token_key_of(parse: &Parse, text: &str) -> Option<kind::TokenKey> {
        parse
            .root
            .descendants_with_tokens()
            .filter_map(|el| el.into_token())
            .find(|t| t.text() == text)
            .and_then(|t| t.text_key())
    }

    #[test]
    fn repl_routes_items_and_expressions() {
        let interner = std::sync::Arc::new(new_threaded_interner());
        let items = [
            "fn f() -> i32 { 1 }",
            "struct P { x: i32 }",
            "enum E { A, B }",
            "pub fn g() -> i32 { 2 }",
            "mod m;",
            "extern \"C\" trampoline \"f_ffi\" = f;",
            "regional fn h(c: [flex] L<i32>) { c->v := 1 }",
            // A capability annotation may precede the record name.
            "struct [value] V { a: i32 }",
            "enum [shared] E { A, B }",
            // Several items in one input.
            "fn a() -> i32 { 1 }\nfn b() -> i32 { 2 }",
            "import core::intrinsic::math;",
            "import core::intrinsic::math as m;",
            "import core::intrinsic::math::sqrt as rt;",
            // FFI items: a foreign source block, an opaque record, and a
            // foreign-bodied function.
            "extern \"rust\" [{ use reussir_rt::collections::vec::Vec; }];",
            "#[ffi(rust = \"::reussir_rt::collections::vec::Vec\")]\npub struct Vec<T>;",
            "#[ffi(import)]\npub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ Vec::push(v, x) }];",
        ];
        for source in items {
            let rp = parse_repl(source, interner.clone());
            assert_eq!(rp.kind, ReplInputKind::Items, "{source}");
            assert!(rp.parse.ok(), "{source}: {:#?}", rp.parse.errors);
            assert_eq!(rp.parse.root.kind(), SyntaxKind::SourceFile);
        }

        let exprs = [
            "42",
            "1 + 2",
            "f(1, 2)",
            "let a = 10; let b = 20; a + b",
            "if true { 1 } else { 0 }",
            // The `else` branch may be omitted (unit `if`).
            "if true { f() }",
            "{ let x = 1; x }",
            "|x: i32| x + 1",
            // `regional` NOT followed by `fn` is a regional expression.
            "regional { Cell { value: 1 } }",
            // Keyword tokens are contextual in identifier positions, so a
            // keyword-named variable may head an expression; the name
            // lookahead keeps these off the items route.
            "fn + 1",
            "struct(2)",
            "mod.field",
            "pub * 2",
            "extern == 3",
            // The documented `as` ambiguity resolves toward the cast.
            "fn as i64",
            // A bare keyword-named variable (nothing follows).
            "enum",
            // `import` as a keyword-named variable.
            "import + 1",
            "import as i64",
        ];
        for source in exprs {
            let rp = parse_repl(source, interner.clone());
            assert_eq!(rp.kind, ReplInputKind::Expr, "{source}");
            assert!(rp.parse.ok(), "{source}: {:#?}", rp.parse.errors);
            assert_eq!(rp.parse.root.kind(), SyntaxKind::BlockExpr);
            assert_eq!(rp.parse.root.text(), source);
        }
    }

    #[test]
    fn repl_expr_seq_reports_trailing_garbage() {
        let interner = std::sync::Arc::new(new_threaded_interner());
        let rp = parse_repl("1 + 2 3", interner);
        assert_eq!(rp.kind, ReplInputKind::Expr);
        assert!(!rp.parse.ok());
        // Lossless even under recovery.
        assert_eq!(rp.parse.root.text(), "1 + 2 3");
    }

    #[test]
    fn repl_expr_seq_recovers_at_semicolons() {
        let interner = std::sync::Arc::new(new_threaded_interner());
        // An error inside one expression resynchronizes at the `;`: the
        // expression's own diagnostic is the only one (no misleading
        // "expected `;`"), and the rest of the sequence still parses.
        let rp = parse_repl("1 + ; 2 + 3", interner);
        assert_eq!(rp.kind, ReplInputKind::Expr);
        assert_eq!(rp.parse.errors.len(), 1, "{:#?}", rp.parse.errors);
        assert_eq!(rp.parse.root.text(), "1 + ; 2 + 3");
        // The recovered tail is a real expression node, not error debris.
        let tail = rp
            .parse
            .root
            .children()
            .filter(|c| c.kind() != SyntaxKind::ErrorNode)
            .last()
            .expect("expression after the error");
        assert_eq!(tail.text(), "2 + 3");
    }

    #[test]
    fn parses_functions_with_operators() {
        json_of(
            "fn f(x: i32, y: i32) -> i32 {\n    let z = x * 2 + y % 3;\n    if (z >= 0 && !(z == 4)) { z } else { -z }\n}",
        );
    }

    /// `if` without `else` is sugar for an empty `else {}`: the emitted AST
    /// still carries three `If` parts, the last being an empty block.
    #[test]
    fn if_without_else_desugars_to_an_empty_else_block() {
        fn find_if(v: &serde_json::Value) -> Option<&serde_json::Value> {
            match v {
                serde_json::Value::Object(map) => {
                    if map.get("tag").and_then(|t| t.as_str()) == Some("If") {
                        return Some(v);
                    }
                    map.values().find_map(find_if)
                }
                serde_json::Value::Array(items) => items.iter().find_map(find_if),
                _ => None,
            }
        }

        let json = json_of("fn f(c: bool) { if c { f(c) } }");
        let if_node = find_if(&json).expect("an If node");
        let parts = if_node["contents"].as_array().expect("If parts");
        assert_eq!(parts.len(), 3, "desugared If still has three parts");
        assert_eq!(
            parts[2],
            serde_json::json!({ "tag": "ExprSeq", "contents": [] })
        );
    }

    /// A dangling `else` with no block is still a parse error.
    #[test]
    fn if_with_dangling_else_still_errors() {
        let parse = parse("fn f(c: bool) { if c { } else }");
        assert!(!parse.ok());
    }

    #[test]
    fn parses_generics_and_match() {
        json_of(
            "enum List<T> { Nil, Cons(T, List<T>) }\nfn append(a : List<i32>, b : List<i32>) -> List<i32> {\n    match a {\n        List::Nil => b,\n        List::Cons(x, xs) => List::Cons{x, append(xs, b)}\n    }\n}",
        );
    }

    #[test]
    fn parses_lambdas_and_calls() {
        json_of("fn t() -> i32 { (|x: i32| x + 1)(41) }");
        json_of("fn abstract() -> bool -> u64 { let u = 1; |x| if (x) { 0 } else { u } }");
        // `||` lexes as one token but means an empty lambda parameter list.
        json_of("fn z() -> i32 { let f = || 1; f() }");
    }

    #[test]
    fn parses_array_types() {
        // `[T; extents…]` in parameter, return, and turbofish position;
        // rank-1 and rank-3.
        json_of(
            "fn f(a : [f64; 8], m : [i32; 5, 16, 8]) -> [f64; 8] { core::intrinsic::array::tabulate<[f64; 8]>(|i| i as f64) }",
        );
        // Extents are full expressions in the tree; whether one can be
        // evaluated is decided during elaboration, not here.
        json_of("fn f(a : [f64; 2 + 2], b : [f64; n]) -> i64 { 0 }");
        // The extent list itself is mandatory.
        assert!(!super::parse("fn f(a : [f64;]) -> i64 { 0 }").ok());
    }

    #[test]
    fn parses_regional_and_assign() {
        json_of(
            "struct [regional] L<T> { v: T, next: [field] L<T> }\nregional fn push<T>(c : [flex] L<T>, e : T) { c->next := Nullable::NonNull{c} }",
        );
    }

    #[test]
    fn parses_trampoline_and_mod() {
        json_of(
            "pub mod utils;\nfn fib<T>(n: T) -> T { n }\nextern \"C\" trampoline \"fib_ffi\" = fib<u64>;",
        );
    }

    #[test]
    fn parses_import_items() {
        // Both spellings encode as one `ImportStmt` of `[name, path]`: the
        // bare form binds the last segment, `as` binds the given name — for
        // modules and individual functions alike.
        let json = json_of(
            "import core::intrinsic::math;\nimport core::intrinsic::array as arr;\nimport core::intrinsic::math::sqrt as rt;",
        );
        let cases = [
            ("math", vec!["core", "intrinsic"], "math"),
            ("arr", vec!["core", "intrinsic"], "array"),
            ("rt", vec!["core", "intrinsic", "math"], "sqrt"),
        ];
        for (item, (name, segments, basename)) in json.as_array().unwrap().iter().zip(cases) {
            let item = unwrap_span(item);
            assert_eq!(item["tag"], "ImportStmt");
            assert_eq!(item["contents"][0], name);
            assert_eq!(
                item["contents"][1]["pathSegments"],
                serde_json::json!(segments)
            );
            assert_eq!(item["contents"][1]["pathBasename"], basename);
        }
        // `import` remains a contextual word in identifier positions.
        json_of("fn import(import: i32) -> i32 { import }");
        // The trailing semicolon is mandatory.
        assert!(!super::parse("import core::intrinsic::math").ok());
    }

    #[test]
    fn parses_ffi_items() {
        // The three FFI item forms: a foreign source block, an opaque
        // (field-less) record, and a foreign-bodied function.
        let json = json_of(
            "extern \"rust\" [{ use reussir_rt::collections::vec::Vec as RVec; }];\n\
             #[ffi(rust = \"::reussir_rt::collections::vec::Vec\")]\n\
             pub struct Vec<T>;\n\
             #[ffi(import)]\n\
             pub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ RVec::push(v, x) }];",
        );
        let source_block = unwrap_span(&json[0]);
        assert_eq!(source_block["tag"], "ExternSourceStmt");
        assert_eq!(source_block["contents"][0], "rust");
        assert_eq!(
            source_block["contents"][1],
            "[{ use reussir_rt::collections::vec::Vec as RVec; }]"
        );
        let record = unwrap_span(&json[1]);
        assert_eq!(record["tag"], "RecordStmt");
        assert_eq!(record["contents"]["recordFields"]["tag"], "Opaque");
        let func = unwrap_span(&json[2]);
        assert_eq!(func["tag"], "FunctionStmt");
        assert_eq!(func["contents"]["funcBody"], serde_json::Value::Null);
        assert_eq!(
            func["contents"]["funcForeignBody"],
            "[{ RVec::push(v, x) }]"
        );
        // A foreign body requires the trailing semicolon.
        assert!(!super::parse("fn f() -> i64 [{ 42 }]").ok());
        assert!(!super::parse("extern \"rust\" [{ code }]").ok());
    }

    #[test]
    fn parses_opaque_transform_item() {
        let source = r#"transform [{
    %loops = transform.structured.match ops{["scf.for"]} in %target
        : (!transform.op<"func.func">) -> !transform.any_op
    transform.loop.unroll %loops { factor = 4 } : !transform.any_op
    transform.yield
}];"#;
        let json = json_of(source);
        let item = unwrap_span(&json[0]);
        assert_eq!(item["tag"], "TransformStmt");
        assert_eq!(
            item["contents"],
            &source["transform ".len()..source.len() - 1]
        );
    }

    #[test]
    fn transform_remains_a_contextual_word() {
        json_of("fn transform(transform: i32) -> i32 { transform }");
    }

    #[test]
    fn transform_item_requires_a_semicolon() {
        let parse = parse("transform [{ transform.yield }]");
        assert!(
            parse
                .errors
                .iter()
                .any(|error| error.message.contains("`;`"))
        );
    }

    #[test]
    fn generic_argument_ambiguity() {
        // Comparison...
        let s = json_of("fn f(a: i32, b: i32) -> bool { a < b }").to_string();
        assert!(s.contains("\"Lt\""), "expected a comparison: {s}");
        // ...vs. instantiation.
        let s = json_of("fn g() -> i32 { id<i32>(1) }").to_string();
        assert!(s.contains("funcCallTyArgs"), "expected a call: {s}");
        // ...vs. a bare instantiated constructor.
        let s = json_of("fn h() -> List<i32> { List::Nil<i32> }").to_string();
        assert!(s.contains("ctorName"), "expected a ctor call: {s}");
    }

    #[test]
    fn access_chains_group() {
        let s = json_of("fn f(p: Pair<i32, i32>) -> i32 { p.first.second(1).third }").to_string();
        assert!(s.contains("AccessChain"), "{s}");
        // Fused float in tuple-index position: two numeric accesses.
        let s = json_of("fn g(p: P) -> i32 { p.0.1 }").to_string();
        assert!(
            s.contains("\"Unnamed\"") && s.contains("\"contents\":1"),
            "{s}"
        );
    }

    #[test]
    fn casts_and_prefix() {
        // The postfix cast applies on top of the prefixed atom:
        // `-x as f64` is `Cast f64 (Negate x)`.
        let v = json_of("fn f(x: i32) -> f64 { -x as f64 }");
        let body = &unwrap_span(&v[0])["contents"]["funcBody"];
        let seq = &unwrap_span(body)["contents"][0];
        let cast = unwrap_span(seq);
        assert_eq!(cast["tag"], "Cast", "{v}");
        assert_eq!(
            unwrap_span(&cast["contents"][1])["tag"],
            "UnaryOpExpr",
            "{v}"
        );
    }

    /// Strip a `SpannedExpr` / `SpannedStmt` / `TypeSpanned` wrapper.
    fn unwrap_span(v: &serde_json::Value) -> &serde_json::Value {
        &v["contents"]["spanValue"]
    }

    #[test]
    fn patterns_with_guards_and_ellipsis() {
        json_of(
            "fn classify(p: Pair<i32, i32>) -> i32 {\n    match p {\n        Pair { first: 0, .. } => 0,\n        Pair { first: x, second: y } if x > y => 1,\n        _ => 2\n    }\n}",
        );
    }

    #[test]
    fn comments_are_preserved() {
        json_of(
            "// RUN: lit directive\n/* block\n   comment */\nfn f() -> i32 { 1 } // trailing\n",
        );
    }

    #[test]
    fn keywords_are_contextual() {
        // The surface language has no reserved identifiers.
        json_of("fn f() -> i32 { let value = 1; value }");
        json_of("fn shared(field: i32) -> i32 { field }");
    }

    #[test]
    fn error_recovery_reports_multiple_errors() {
        let source =
            "fn good() -> i32 { 1 }\nfn bad( { 2 }\nstruct Also { x: }\nfn fine() -> i32 { 3 }";
        let parse = parse(source);
        assert!(parse.errors.len() >= 2, "errors: {:#?}", parse.errors);
        // Losslessness holds under recovery too.
        assert_eq!(parse.root.text(), source);
        // All four items appear in the tree despite the errors.
        assert!(parse.root.children().count() >= 4);
    }

    #[test]
    fn nonassoc_comparison_is_diagnosed() {
        let parse = parse("fn f(a: i32, b: i32, c: i32) -> bool { a < b < c }");
        assert!(
            parse.errors.iter().any(|e| e.message.contains("chained")),
            "errors: {:#?}",
            parse.errors
        );
    }

    #[test]
    fn deep_recursion_is_survivable() {
        // Deeply nested parens would overflow the fixed test-thread stack
        // (2 MiB) without the stacker guards in the parser.
        let depth = 100_000;
        let source = format!(
            "fn f() -> i32 {{ {}1{} }}",
            "(".repeat(depth),
            ")".repeat(depth)
        );
        let parse = parse(&source);
        assert!(parse.ok(), "first error: {:#?}", parse.errors.first());
        // Tear the tree down on a spacious thread: dropping a 100k-deep
        // syntax tree recurses in cstree itself, which is outside this
        // crate's control.
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(move || drop(parse))
            .unwrap()
            .join()
            .unwrap();
    }
}
