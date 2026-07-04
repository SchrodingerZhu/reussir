//! Lowering of the lossless CST to the JSON encoding of the surface AST.
//!
//! Encoding conventions:
//!
//! * sum types: `{"tag": "Ctor", "contents": ...}` with a bare value for a
//!   single field, an array for several, and no `contents` key for nullary
//!   constructors in mixed sums;
//! * all-nullary sums are plain strings (`"Public"`, `"Add"`);
//! * single-constructor records are plain objects keyed by field name;
//! * tuples are arrays, an optional value is `null` or the value;
//! * spans are `{"spanValue": v, "spanStartOffset": n, "spanEndOffset": n}`
//!   with *character* offsets.

use serde_json::{Number, Value, json};

use crate::kind::{ResolvedNode, ResolvedToken, SyntaxKind, SyntaxKind::*};
use crate::source::CharMap;

pub fn prog_to_json(root: &ResolvedNode, map: &CharMap) -> Value {
    assert_eq!(root.kind(), SourceFile, "expected a SourceFile root");
    let e = Emitter { map };
    Value::Array(root.children().map(|stmt| e.stmt(stmt)).collect())
}

struct Emitter<'m> {
    map: &'m CharMap,
}

// ===== small helpers =====

fn tagged0(tag: &str) -> Value {
    json!({ "tag": tag })
}

fn tagged(tag: &str, contents: Value) -> Value {
    json!({ "tag": tag, "contents": contents })
}

fn tokens(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedToken> {
    node.children_with_tokens()
        .filter_map(|el| el.as_token().copied())
        .filter(|t| !t.kind().is_trivia())
}

fn nodes(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedNode> {
    node.children()
}

fn child_node(node: &ResolvedNode, kind: SyntaxKind) -> Option<&ResolvedNode> {
    node.children().find(|n| n.kind() == kind)
}

fn is_ident_like(kind: SyntaxKind) -> bool {
    kind.is_ident_like()
}

fn is_expr_kind(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        BlockExpr
            | ParenExpr
            | IfExpr
            | LetExpr
            | MatchExpr
            | RegionalExpr
            | LambdaExpr
            | BinExpr
            | PrefixExpr
            | CastExpr
            | CallExpr
            | AccessChain
            | AssignExpr
            | LiteralExpr
            | VarExpr
            | FuncCallExpr
            | CtorCallExpr
    )
}

fn is_type_kind(kind: SyntaxKind) -> bool {
    matches!(kind, PrimType | PathType | ArrowType | ParenTypeList)
}

fn expr_children(node: &ResolvedNode) -> impl Iterator<Item = &ResolvedNode> {
    node.children().filter(|n| is_expr_kind(n.kind()))
}

impl Emitter<'_> {
    fn char_span(&self, node: &ResolvedNode) -> (u64, u64) {
        let range = node.text_range();
        let (s, e) = self
            .map
            .char_span((range.start().into(), range.end().into()));
        (s as u64, e as u64)
    }

    fn with_node_span(&self, node: &ResolvedNode, value: Value) -> Value {
        let (start, end) = self.char_span(node);
        json!({
            "spanValue": value,
            "spanStartOffset": start,
            "spanEndOffset": end,
        })
    }

    fn with_token_span(&self, token: &ResolvedToken, value: Value) -> Value {
        let range = token.text_range();
        let (start, end) = self
            .map
            .char_span((range.start().into(), range.end().into()));
        json!({
            "spanValue": value,
            "spanStartOffset": start,
            "spanEndOffset": end,
        })
    }

    // ===== statements =====

    fn stmt(&self, node: &ResolvedNode) -> Value {
        let inner = match node.kind() {
            FnStmt => self.fn_stmt(node),
            StructStmt => self.record_stmt(node, "StructKind"),
            EnumStmt => self.record_stmt(node, "EnumKind"),
            ModStmt => self.mod_stmt(node),
            ExternTrampolineStmt => self.extern_stmt(node),
            k => unreachable!("unexpected statement node {k:?}"),
        };
        tagged("SpannedStmt", self.with_node_span(node, inner))
    }

    fn visibility(&self, node: &ResolvedNode) -> Value {
        let public = tokens(node).any(|t| t.kind() == PubKw);
        Value::String(if public { "Public" } else { "Private" }.into())
    }

    /// The defining name: the first identifier-like direct token after the
    /// introducing keyword.
    fn name_after<'n>(&self, node: &'n ResolvedNode, kw: SyntaxKind) -> &'n ResolvedToken {
        let mut seen_kw = false;
        for t in tokens(node) {
            if t.kind() == kw {
                seen_kw = true;
            } else if seen_kw && is_ident_like(t.kind()) {
                return t;
            }
        }
        panic!("missing name token in {:?}", node.kind())
    }

    fn generic_params(&self, node: &ResolvedNode) -> Value {
        let Some(list) = child_node(node, GenericParamList) else {
            return json!([]);
        };
        Value::Array(
            nodes(list)
                .filter(|n| n.kind() == GenericParam)
                .map(|param| {
                    let name = tokens(param)
                        .find(|t| is_ident_like(t.kind()))
                        .expect("generic parameter name");
                    let bounds: Vec<Value> = nodes(param)
                        .filter(|n| n.kind() == Path)
                        .map(|p| self.path(p))
                        .collect();
                    json!([name.text(), bounds])
                })
                .collect(),
        )
    }

    fn fn_stmt(&self, node: &ResolvedNode) -> Value {
        let body = expr_children(node).last().map(|b| self.expr(b));
        let ret = child_node(node, RetType).map(|r| {
            let flex = child_node(r, FlexFlag).is_some();
            let ty = nodes(r)
                .find(|n| is_type_kind(n.kind()))
                .expect("return type");
            json!([self.type_(ty), flex])
        });
        let params = child_node(node, ParamList).map_or_else(
            || json!([]),
            |list| {
                Value::Array(
                    nodes(list)
                        .filter(|n| n.kind() == Param)
                        .map(|param| {
                            let name = tokens(param)
                                .find(|t| is_ident_like(t.kind()))
                                .expect("parameter name");
                            let flex = child_node(param, FlexFlag).is_some();
                            let ty = nodes(param)
                                .find(|n| is_type_kind(n.kind()))
                                .expect("parameter type");
                            json!([name.text(), self.type_(ty), flex])
                        })
                        .collect(),
                )
            },
        );
        let function = json!({
            "funcVisibility": self.visibility(node),
            "funcName": self.name_after(node, FnKw).text(),
            "funcGenerics": self.generic_params(node),
            "funcParams": params,
            "funcReturnType": ret,
            "funcIsRegional": tokens(node).any(|t| t.kind() == RegionalKw),
            "funcBody": body,
        });
        tagged("FunctionStmt", function)
    }

    fn record_stmt(&self, node: &ResolvedNode, kind: &str) -> Value {
        let intro = if kind == "StructKind" {
            StructKw
        } else {
            EnumKw
        };
        let cap = child_node(node, Capability)
            .and_then(|c| tokens(c).find(|t| is_ident_like(t.kind()) || t.kind() == RegionalKw))
            .map_or("Shared", |t| match t.text() {
                "shared" => "Shared",
                "value" => "Value",
                "flex" => "Flex",
                "rigid" => "Rigid",
                "field" => "Field",
                "regional" => "Regional",
                other => unreachable!("unexpected capability {other}"),
            });
        let fields = if let Some(named) = child_node(node, NamedFields) {
            let fields: Vec<Value> = nodes(named)
                .filter(|n| n.kind() == NamedField)
                .map(|f| {
                    let name = tokens(f)
                        .find(|t| is_ident_like(t.kind()))
                        .expect("field name");
                    let flag = child_node(f, FieldFlag).is_some();
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    self.with_node_span(f, json!([name.text(), self.type_(ty), flag]))
                })
                .collect();
            tagged("Named", Value::Array(fields))
        } else if let Some(unnamed) = child_node(node, UnnamedFields) {
            let fields: Vec<Value> = nodes(unnamed)
                .filter(|n| n.kind() == UnnamedField)
                .map(|f| {
                    let flag = child_node(f, FieldFlag).is_some();
                    let ty = nodes(f)
                        .find(|n| is_type_kind(n.kind()))
                        .expect("field type");
                    self.with_node_span(f, json!([self.type_(ty), flag]))
                })
                .collect();
            tagged("Unnamed", Value::Array(fields))
        } else {
            let list = child_node(node, VariantList).expect("variant list");
            let variants: Vec<Value> = nodes(list)
                .filter(|n| n.kind() == Variant)
                .map(|v| {
                    let name = tokens(v)
                        .find(|t| is_ident_like(t.kind()))
                        .expect("variant name");
                    let tys: Vec<Value> = nodes(v)
                        .filter(|n| is_type_kind(n.kind()))
                        .map(|t| self.type_(t))
                        .collect();
                    self.with_node_span(v, json!([name.text(), tys]))
                })
                .collect();
            tagged("Variants", Value::Array(variants))
        };
        let record = json!({
            "recordName": self.name_after(node, intro).text(),
            "recordTyParams": self.generic_params(node),
            "recordFields": fields,
            "recordKind": kind,
            "recordVisibility": self.visibility(node),
            "recordDefaultCap": cap,
        });
        tagged("RecordStmt", record)
    }

    fn mod_stmt(&self, node: &ResolvedNode) -> Value {
        let name = self.name_after(node, ModKw);
        tagged("ModStmt", json!([self.visibility(node), name.text()]))
    }

    fn extern_stmt(&self, node: &ResolvedNode) -> Value {
        let mut strings = tokens(node).filter(|t| t.kind() == StringLit);
        let abi = unescape_string(strings.next().expect("ABI string").text());
        let sym = unescape_string(strings.next().expect("symbol string").text());
        let func = child_node(node, Path).expect("trampoline target path");
        let ty_args: Vec<Value> = child_node(node, TypeArgList)
            .map(|l| {
                nodes(l)
                    .filter(|n| is_type_kind(n.kind()))
                    .map(|t| self.type_(t))
                    .collect()
            })
            .unwrap_or_default();
        json!({
            "tag": "ExternTrampolineStmt",
            "etsName": sym,
            "etsABI": abi,
            "etsFunc": self.path(func),
            "etsFuncTyArgs": ty_args,
        })
    }

    // ===== paths and types =====

    fn path(&self, node: &ResolvedNode) -> Value {
        let segments: Vec<&str> = tokens(node)
            .filter(|t| is_ident_like(t.kind()))
            .map(|t| t.text())
            .collect();
        let (basename, prefix) = segments.split_last().expect("non-empty path");
        json!({ "pathBasename": basename, "pathSegments": prefix })
    }

    fn type_(&self, node: &ResolvedNode) -> Value {
        let inner = match node.kind() {
            PrimType => prim_type(tokens(node).next().expect("prim type token").text()),
            PathType => {
                let path = child_node(node, Path).expect("type path");
                let args: Vec<Value> = child_node(node, TypeArgList)
                    .map(|l| {
                        nodes(l)
                            .filter(|n| is_type_kind(n.kind()))
                            .map(|t| self.type_(t))
                            .collect()
                    })
                    .unwrap_or_default();
                tagged("TypeExpr", json!([self.path(path), args]))
            }
            ArrowType => {
                let mut children = nodes(node).filter(|n| is_type_kind(n.kind()));
                let lhs = children.next().expect("arrow lhs");
                let rhs = children.next().expect("arrow rhs");
                let args: Vec<Value> = if lhs.kind() == ParenTypeList {
                    nodes(lhs)
                        .filter(|n| is_type_kind(n.kind()))
                        .map(|t| self.type_(t))
                        .collect()
                } else {
                    vec![self.type_(lhs)]
                };
                tagged("TypeArrow", json!([args, self.type_(rhs)]))
            }
            ParenTypeList => {
                // A single-element parenthesized list is plain grouping; the
                // parser rejects multi-element lists outside arrow-lhs
                // position.
                let inner = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("parenthesized type");
                return self.type_(inner);
            }
            k => unreachable!("unexpected type node {k:?}"),
        };
        tagged("TypeSpanned", self.with_node_span(node, inner))
    }

    // ===== patterns =====

    fn pattern(&self, node: &ResolvedNode) -> Value {
        let kind = nodes(node)
            .find(|n| matches!(n.kind(), WildcardPat | BindPat | CtorPat | ConstPat))
            .expect("pattern kind");
        let guard = child_node(node, PatGuard)
            .map(|g| self.expr(expr_children(g).next().expect("guard expression")));
        json!({
            "patKind": self.pattern_kind(kind),
            "patGuard": guard,
        })
    }

    fn pattern_kind(&self, node: &ResolvedNode) -> Value {
        match node.kind() {
            WildcardPat => tagged0("WildcardPat"),
            ConstPat => {
                let token = tokens(node).next().expect("constant token");
                tagged("ConstPat", self.constant(token))
            }
            BindPat => {
                let path = child_node(node, Path).expect("binding path");
                let name = tokens(path).next().expect("binding name");
                tagged("BindPat", Value::String(name.text().into()))
            }
            CtorPat => {
                let path = child_node(node, Path).expect("constructor path");
                let (args, ellipsis, named) = match child_node(node, PatArgList) {
                    None => (Vec::new(), false, false),
                    Some(list) => {
                        let named = tokens(list).next().is_some_and(|t| t.kind() == LBrace);
                        let ellipsis = child_node(list, PatRest).is_some();
                        let args = nodes(list)
                            .filter(|n| n.kind() == PatArg)
                            .map(|arg| self.pat_arg(arg))
                            .collect();
                        (args, ellipsis, named)
                    }
                };
                json!({
                    "tag": "CtorPat",
                    "patCtorPath": self.path(path),
                    "patCtorArgs": args,
                    "patCtorHasEllipsis": ellipsis,
                    "patCtorIsNamed": named,
                })
            }
            k => unreachable!("unexpected pattern node {k:?}"),
        }
    }

    fn pat_arg(&self, node: &ResolvedNode) -> Value {
        let field = tokens(node)
            .find(|t| is_ident_like(t.kind()))
            .map(|t| t.text());
        let kind =
            nodes(node).find(|n| matches!(n.kind(), WildcardPat | BindPat | CtorPat | ConstPat));
        let kind_json = match kind {
            Some(k) => self.pattern_kind(k),
            // `{ x }` shorthand binds the field to a variable of the same
            // name.
            None => tagged(
                "BindPat",
                Value::String(field.expect("shorthand field name").into()),
            ),
        };
        json!({
            "patCtorArgField": field,
            "patCtorArgKind": kind_json,
        })
    }

    // ===== expressions =====

    fn expr(&self, node: &ResolvedNode) -> Value {
        let inner = match node.kind() {
            ParenExpr => {
                // Parentheses are transparent in the AST.
                return self.expr(expr_children(node).next().expect("inner expression"));
            }
            LiteralExpr => {
                let token = tokens(node).next().expect("literal token");
                tagged("ConstExpr", self.constant(token))
            }
            BlockExpr => {
                let exprs: Vec<Value> = expr_children(node).map(|e| self.expr(e)).collect();
                tagged("ExprSeq", Value::Array(exprs))
            }
            IfExpr => {
                let mut parts = expr_children(node).map(|e| self.expr(e));
                let cond = parts.next().expect("condition");
                let then = parts.next().expect("then branch");
                let other = parts.next().expect("else branch");
                tagged("If", json!([cond, then, other]))
            }
            LetExpr => {
                let name = self.let_name(node).expect("let binding name");
                let flex = child_node(node, FlexFlag).is_some();
                let ty = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .map(|t| json!([self.type_(t), flex]));
                let value = expr_children(node).last().expect("bound value");
                tagged(
                    "Let",
                    json!([
                        self.with_token_span(name, Value::String(name.text().into())),
                        ty,
                        self.expr(value)
                    ]),
                )
            }
            MatchExpr => {
                let scrutinee = expr_children(node).next().expect("scrutinee");
                let arms: Vec<Value> = nodes(node)
                    .filter(|n| n.kind() == MatchArm)
                    .map(|arm| {
                        let pat = child_node(arm, Pattern).expect("arm pattern");
                        let body = expr_children(arm).next().expect("arm body");
                        json!([self.pattern(pat), self.expr(body)])
                    })
                    .collect();
                tagged("Match", json!([self.expr(scrutinee), arms]))
            }
            RegionalExpr => {
                let body = expr_children(node).next().expect("regional body");
                tagged("RegionalExpr", self.expr(body))
            }
            LambdaExpr => {
                let params: Vec<Value> = child_node(node, LambdaParamList)
                    .map(|list| {
                        nodes(list)
                            .filter(|n| n.kind() == LambdaParam)
                            .map(|param| {
                                let name = tokens(param)
                                    .find(|t| is_ident_like(t.kind()))
                                    .expect("lambda parameter name");
                                let ty = nodes(param)
                                    .find(|n| is_type_kind(n.kind()))
                                    .map(|t| self.type_(t));
                                json!([name.text(), ty])
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let ret = child_node(node, RetType)
                    .map(|r| self.type_(nodes(r).find(|n| is_type_kind(n.kind())).expect("ret")));
                let body = expr_children(node).last().expect("lambda body");
                tagged(
                    "Lambda",
                    json!({ "args": params, "body": self.expr(body), "retTy": ret }),
                )
            }
            BinExpr => {
                let op = tokens(node)
                    .find_map(|t| binary_op_name(t.kind()))
                    .expect("binary operator");
                let mut operands = expr_children(node).map(|e| self.expr(e));
                let lhs = operands.next().expect("lhs");
                let rhs = operands.next().expect("rhs");
                tagged("BinOpExpr", json!([op, lhs, rhs]))
            }
            PrefixExpr => {
                let op = tokens(node)
                    .find_map(|t| match t.kind() {
                        Minus => Some("Negate"),
                        Bang => Some("Not"),
                        _ => None,
                    })
                    .expect("unary operator");
                let operand = expr_children(node).next().expect("operand");
                tagged("UnaryOpExpr", json!([op, self.expr(operand)]))
            }
            CastExpr => {
                let operand = expr_children(node).next().expect("cast operand");
                let ty = nodes(node)
                    .find(|n| is_type_kind(n.kind()))
                    .expect("cast type");
                tagged("Cast", json!([self.type_(ty), self.expr(operand)]))
            }
            CallExpr => {
                let callee = expr_children(node).next().expect("callee");
                let args: Vec<Value> = child_node(node, ArgList)
                    .map(|l| expr_children(l).map(|e| self.expr(e)).collect())
                    .unwrap_or_default();
                tagged("CallExpr", json!([self.expr(callee), args]))
            }
            AccessChain => {
                let base = expr_children(node).next().expect("access base");
                let accesses = self.access_segs(node);
                tagged("AccessChain", json!([self.expr(base), accesses]))
            }
            AssignExpr => {
                let mut operands = expr_children(node).map(|e| self.expr(e));
                let lhs = operands.next().expect("assignment target");
                let rhs = operands.next().expect("assignment value");
                let access = self.access_segs(node);
                let access = access.first().cloned().expect("assignment field");
                tagged("Assign", json!([lhs, access, rhs]))
            }
            VarExpr => {
                let path = child_node(node, Path).expect("variable path");
                tagged("Var", self.path(path))
            }
            FuncCallExpr => {
                let path = child_node(node, Path).expect("function path");
                let args: Vec<Value> = child_node(node, ArgList)
                    .map(|l| expr_children(l).map(|e| self.expr(e)).collect())
                    .unwrap_or_default();
                let call = json!({
                    "funcCallName": self.path(path),
                    "funcCallTyArgs": self.type_args(node),
                    "funcCallArgs": args,
                });
                tagged("FuncCallExpr", call)
            }
            CtorCallExpr => {
                let path = child_node(node, Path).expect("constructor path");
                let args: Vec<Value> = child_node(node, CtorArgList)
                    .map(|l| {
                        nodes(l)
                            .filter(|n| n.kind() == CtorArg)
                            .map(|arg| {
                                let named = tokens(arg).any(|t| t.kind() == Colon);
                                let field = if named {
                                    tokens(arg)
                                        .find(|t| is_ident_like(t.kind()))
                                        .map(|t| Value::String(t.text().into()))
                                        .unwrap_or(Value::Null)
                                } else {
                                    Value::Null
                                };
                                let value = expr_children(arg).next().expect("argument value");
                                json!([field, self.expr(value)])
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let call = json!({
                    "ctorName": self.path(path),
                    "ctorTyArgs": self.type_args(node),
                    "ctorArgs": args,
                });
                tagged("CtorCallExpr", call)
            }
            k => unreachable!("unexpected expression node {k:?}"),
        };
        tagged("SpannedExpr", self.with_node_span(node, inner))
    }

    /// The binding name token of a `let`: the first identifier-like direct
    /// token after the `let` keyword.
    fn let_name<'n>(&self, node: &'n ResolvedNode) -> Option<&'n ResolvedToken> {
        let mut seen_let = false;
        for t in tokens(node) {
            if t.kind() == LetKw {
                seen_let = true;
            } else if seen_let && is_ident_like(t.kind()) {
                return Some(t);
            }
        }
        None
    }

    /// `<T, _, U>` arguments on a path-based expression; `_` becomes `null`
    /// (the type is a list of optional types).
    fn type_args(&self, node: &ResolvedNode) -> Value {
        let Some(list) = child_node(node, TypeArgList) else {
            return json!([]);
        };
        Value::Array(
            nodes(list)
                .filter(|n| is_type_kind(n.kind()) || n.kind() == InferType)
                .map(|n| {
                    if n.kind() == InferType {
                        Value::Null
                    } else {
                        self.type_(n)
                    }
                })
                .collect(),
        )
    }

    /// Flatten the `AccessSeg` children into the list of accesses.
    /// A fused float token like `0.1` contributes two numeric accesses.
    fn access_segs(&self, node: &ResolvedNode) -> Vec<Value> {
        let mut out = Vec::new();
        for seg in nodes(node).filter(|n| n.kind() == AccessSeg) {
            for t in tokens(seg) {
                match t.kind() {
                    Dot | Arrow => {}
                    IntLit => out.push(tagged("Unnamed", int_value(t.text()))),
                    FloatLit => {
                        for part in t.text().split('.') {
                            out.push(tagged("Unnamed", int_value(part)));
                        }
                    }
                    k if is_ident_like(k) => {
                        out.push(tagged("Named", Value::String(t.text().into())))
                    }
                    k => unreachable!("unexpected access token {k:?}"),
                }
            }
        }
        out
    }

    fn constant(&self, token: &ResolvedToken) -> Value {
        match token.kind() {
            IntLit => tagged("ConstInt", int_value(token.text())),
            // "ConstDouble" is the wire tag the Haskell differential verifier
            // (`reussir-surface-verify`) decodes; keep it stable.
            FloatLit => tagged(
                "ConstDouble",
                // Underscore separators are not valid JSON; strip them. The
                // digits themselves are carried verbatim (arbitrary
                // precision — serde_json's `arbitrary_precision` feature).
                Value::Number(Number::from_string_unchecked(token.text().replace('_', ""))),
            ),
            StringLit => tagged("ConstString", Value::String(unescape_string(token.text()))),
            TrueKw => tagged("ConstBool", Value::Bool(true)),
            FalseKw => tagged("ConstBool", Value::Bool(false)),
            k => unreachable!("unexpected constant token {k:?}"),
        }
    }
}

/// An integer literal as a JSON value: plain decimals are numbers (arbitrary
/// precision); radix forms keep their spelling as a string, losslessly.
fn int_value(text: &str) -> Value {
    let plain = text.replace('_', "");
    if plain.bytes().all(|b| b.is_ascii_digit()) {
        Value::Number(Number::from_string_unchecked(plain))
    } else {
        Value::String(plain)
    }
}

fn binary_op_name(kind: SyntaxKind) -> Option<&'static str> {
    Some(match kind {
        Plus => "Add",
        Minus => "Sub",
        Star => "Mul",
        Slash => "Div",
        Percent => "Mod",
        LAngle => "Lt",
        RAngle => "Gt",
        LtEq => "Lte",
        GtEq => "Gte",
        EqEq => "Equ",
        BangEq => "Neq",
        AmpAmp => "And",
        PipePipe => "Or",
        _ => return None,
    })
}

fn prim_type(text: &str) -> Value {
    fn integral(tag: &str, width: u16) -> Value {
        tagged("TypeIntegral", tagged(tag, json!(width)))
    }
    fn ieee(width: u16) -> Value {
        tagged("TypeFP", tagged("IEEEFloat", json!(width)))
    }
    match text {
        "i8" => integral("Signed", 8),
        "i16" => integral("Signed", 16),
        "i32" => integral("Signed", 32),
        "i64" => integral("Signed", 64),
        "u8" => integral("Unsigned", 8),
        "u16" => integral("Unsigned", 16),
        "u32" => integral("Unsigned", 32),
        "u64" => integral("Unsigned", 64),
        "f16" => ieee(16),
        "f32" => ieee(32),
        "f64" => ieee(64),
        "bfloat16" => tagged("TypeFP", tagged0("BFloat16")),
        "float8" => tagged("TypeFP", tagged0("Float8")),
        "bool" => tagged0("TypeBool"),
        "str" => tagged0("TypeStr"),
        "unit" => tagged0("TypeUnit"),
        other => unreachable!("unexpected primitive type {other}"),
    }
}

/// Decode a string literal (including the surrounding quotes) using the
/// surface escape rules: single-character escapes, `\&` (empty), ASCII
/// mnemonics (`\NUL`, `\SOH`, ..., `\DEL`), and numeric escapes (`\65`,
/// `\x41`, `\o101`).
pub fn unescape_string(raw: &str) -> String {
    const MNEMONICS: &[(&str, char)] = &[
        // Longest first so that e.g. `SOH` wins over `SO`.
        ("NUL", '\u{00}'),
        ("SOH", '\u{01}'),
        ("STX", '\u{02}'),
        ("ETX", '\u{03}'),
        ("EOT", '\u{04}'),
        ("ENQ", '\u{05}'),
        ("ACK", '\u{06}'),
        ("BEL", '\u{07}'),
        ("DLE", '\u{10}'),
        ("DC1", '\u{11}'),
        ("DC2", '\u{12}'),
        ("DC3", '\u{13}'),
        ("DC4", '\u{14}'),
        ("NAK", '\u{15}'),
        ("SYN", '\u{16}'),
        ("ETB", '\u{17}'),
        ("CAN", '\u{18}'),
        ("SUB", '\u{1A}'),
        ("ESC", '\u{1B}'),
        ("DEL", '\u{7F}'),
        ("EM", '\u{19}'),
        ("FS", '\u{1C}'),
        ("GS", '\u{1D}'),
        ("RS", '\u{1E}'),
        ("US", '\u{1F}'),
        ("SP", ' '),
        ("BS", '\u{08}'),
        ("HT", '\u{09}'),
        ("LF", '\u{0A}'),
        ("VT", '\u{0B}'),
        ("FF", '\u{0C}'),
        ("CR", '\u{0D}'),
        ("SO", '\u{0E}'),
        ("SI", '\u{0F}'),
    ];

    let body = raw
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(raw);
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    'outer: while let Some(idx) = rest.find('\\') {
        out.push_str(&rest[..idx]);
        rest = &rest[idx + 1..];
        let Some(c) = rest.chars().next() else { break };
        match c {
            'a' => out.push('\u{07}'),
            'b' => out.push('\u{08}'),
            'f' => out.push('\u{0C}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'v' => out.push('\u{0B}'),
            '\\' => out.push('\\'),
            '"' => out.push('"'),
            '\'' => out.push('\''),
            '&' => {} // empty escape
            'x' | 'X' => {
                let digits: String = rest[1..]
                    .chars()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .collect();
                push_code(&mut out, &digits, 16);
                rest = &rest[1 + digits.len()..];
                continue 'outer;
            }
            'o' | 'O' => {
                let digits: String = rest[1..]
                    .chars()
                    .take_while(|c| ('0'..='7').contains(c))
                    .collect();
                push_code(&mut out, &digits, 8);
                rest = &rest[1 + digits.len()..];
                continue 'outer;
            }
            c if c.is_ascii_digit() => {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                push_code(&mut out, &digits, 10);
                rest = &rest[digits.len()..];
                continue 'outer;
            }
            _ => {
                for (name, value) in MNEMONICS {
                    if rest.starts_with(name) {
                        out.push(*value);
                        rest = &rest[name.len()..];
                        continue 'outer;
                    }
                }
                // Unknown escape: keep the character (an invalid escape is
                // already flagged by lex-level validation).
                out.push(c);
            }
        }
        rest = &rest[c.len_utf8()..];
    }
    out.push_str(rest);
    out
}

fn push_code(out: &mut String, digits: &str, radix: u32) {
    if let Ok(code) = u32::from_str_radix(digits, radix)
        && let Some(c) = char::from_u32(code)
    {
        out.push(c);
    }
}
