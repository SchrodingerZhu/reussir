//! A readable, **round-trippable** textual rendering of the Full (MIR) program.
//!
//! This is the `rrc --emit mir` output and the *serialization* direction of
//! the textual IR: a Rust-ish dump of each ground function keyed by its v0
//! symbol. The spelling is a contract — the lalrpop grammar (the
//! deserializer) mirrors it, and round-trip tests gate the two against each
//! other.
//!
//! Built with [`pprint`] (algebraic, Wadler-style): each node renders to a
//! [`Doc`], composed with explicit `hardline`/`indent` for deterministic block
//! layout (a stable contract the parser can round-trip). Calls, constructors and
//! record instances reference interned [`mir::Symbol`]s as `@symbol`; locals are
//! `v<id>` ([`VarId`]); types print structurally.

use std::fmt::Write;

use pprint::{Doc, Printer as PpPrinter, hardline, indent, pprint};
use reussir_syntax::kind::{Resolver, TokenKey};
use reussir_syntax::source::SourceCache;

use crate::full::mir;
use crate::semi::ctxt::DefaultCap;
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Flexivity, FpTy, IntTy, Ty, TyKind};
use crate::surface::Visibility;

/// Four-space indentation, 100-column target. Block structure is forced
/// (`hardline`), so width only affects how wide inline argument lists may run.
const IR_PRINTER: PpPrinter = PpPrinter {
    max_width: 100,
    indent: 4,
    use_tabs: false,
};

/// Renders a Full MIR program to text.
pub struct Printer<'a> {
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
    /// When present, the dump carries source locations: the file table
    /// (`0 = "path";`), each function's `in <file>`, and every node's
    /// `[start..end]` span — the lossless round-trip form (`rrc --emit mir`).
    /// Without it the dump is the bare program (human display).
    sources: Option<&'a SourceCache>,
}

impl<'a> Printer<'a> {
    pub fn new(defs: &'a DefTable, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        Printer {
            defs,
            resolver,
            sources: None,
        }
    }

    /// A printer that also serializes source locations against `sources`.
    pub fn with_sources(
        defs: &'a DefTable,
        resolver: &'a dyn Resolver<TokenKey>,
        sources: &'a SourceCache,
    ) -> Self {
        Printer {
            defs,
            resolver,
            sources: Some(sources),
        }
    }

    /// Render `program`. The program owns the symbol interner, borrowed here to
    /// resolve `@symbol` references.
    pub fn program(&self, program: &mir::Program<'_>) -> String {
        let r = Render {
            symbols: &program.symbols,
            defs: self.defs,
            resolver: self.resolver,
            sources: self.sources,
        };

        let mut items: Vec<Doc<'static>> = Vec::new();
        if let Some(cache) = self.sources {
            for id in cache.ids() {
                items.push(text(format!("{} = {:?};", id.index(), cache.name(id))));
            }
        }
        for (token, payload) in &program.string_literals {
            items.push(string_decl(*token, payload));
        }
        for rec in &program.records {
            items.push(r.record(rec));
        }
        for func in &program.functions {
            items.push(r.function(func));
        }
        for t in &program.trampolines {
            items.push(text(format!(
                "extern \"{}\" trampoline \"{}\" = @{};",
                t.abi,
                r.sym(t.export),
                r.sym(t.target),
            )));
        }

        // One blank line between top-level items.
        let mut doc = Doc::Null;
        for (i, item) in items.into_iter().enumerate() {
            if i > 0 {
                doc = doc + hardline() + hardline();
            }
            doc = doc + item;
        }
        let mut out = pprint(doc, IR_PRINTER);
        out.push('\n');
        out
    }
}

/// Borrows the symbol interner, the def table (record-type paths), and the
/// source identifier resolver for one render.
struct Render<'a> {
    symbols: &'a lasso::Rodeo,
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
    sources: Option<&'a SourceCache>,
}

impl Render<'_> {
    fn sym(&self, s: mir::Symbol) -> &str {
        self.symbols.resolve(&s.0)
    }

    /// Render a record instance: its symbol, default capability, nominal type,
    /// and ground field layout (`struct { … }` / `enum { … }`). The nominal type
    /// reprints capability-canonicalized, so the leading keyword carries the
    /// default capability instead.
    fn record(&self, rec: &mir::RecordInstance<'_>) -> Doc<'static> {
        let cap = match rec.default_cap {
            DefaultCap::Value => "value",
            DefaultCap::Shared => "shared",
            DefaultCap::Regional => "regional",
        };
        let head = text(format!("record @{} : {cap} ", self.sym(rec.symbol)));
        let body = match rec.layout {
            mir::RecordLayout::Compound(members) => {
                let parts: Vec<Doc<'static>> = members
                    .iter()
                    .map(|m| {
                        let name = match m.name {
                            Some(s) => text(format!("\"{}\": ", self.sym(s))),
                            None => Doc::Null,
                        };
                        let marker = if m.is_field {
                            text("field ")
                        } else {
                            Doc::Null
                        };
                        name + marker + self.ty(m.ty)
                    })
                    .collect();
                text("struct ") + self.ty(rec.ty) + text(" { ") + comma_sep(parts) + text(" }")
            }
            mir::RecordLayout::Variant(variants) => {
                let parts: Vec<Doc<'static>> = variants
                    .iter()
                    .map(|v| {
                        let fields: Vec<Doc<'static>> =
                            v.fields.iter().map(|&t| self.ty(t)).collect();
                        text(format!("@{} {}(", self.sym(v.symbol), self.sym(v.name)))
                            + comma_sep(fields)
                            + text(")")
                    })
                    .collect();
                text("enum ") + self.ty(rec.ty) + text(" { ") + comma_sep(parts) + text(" }")
            }
        };
        head + body + text(";")
    }

    fn function(&self, func: &mir::Function<'_>) -> Doc<'static> {
        let mut head = String::new();
        if func.visibility == Visibility::Public {
            head.push_str("pub ");
        }
        if func.is_regional {
            head.push_str("regional ");
        }
        let _ = write!(head, "fn @{}(", self.sym(func.symbol));
        let params: Vec<Doc<'static>> = func
            .params
            .iter()
            .map(|p| {
                text(format!(
                    "v{} ({}): ",
                    p.var.0,
                    self.resolver.resolve(p.name)
                )) + self.ty(p.ty)
            })
            .collect();
        let loc = if self.sources.is_some() {
            text(format!(" in {}", func.file.index()))
        } else {
            Doc::Null
        };
        let sig = text(head) + comma_sep(params) + text(") -> ") + self.ty(func.return_ty) + loc;

        match func.body {
            Some(body) => {
                sig + text(" {") + indent(hardline() + self.expr(body)) + hardline() + text("}")
            }
            None => sig + text(";"),
        }
    }

    // ----- types -----

    fn ty(&self, ty: Ty<'_>) -> Doc<'static> {
        match *ty.kind() {
            TyKind::Int(IntTy::Signed(w)) => text(format!("i{w}")),
            TyKind::Int(IntTy::Unsigned(w)) => text(format!("u{w}")),
            TyKind::Fp(FpTy::Ieee(w)) => text(format!("f{w}")),
            TyKind::Fp(FpTy::BFloat16) => text("bf16"),
            TyKind::Fp(FpTy::Float8) => text("f8"),
            TyKind::Bool => text("bool"),
            TyKind::Str => text("str"),
            TyKind::Char => text("char"),
            TyKind::Unit => text("()"),
            TyKind::Bottom => text("!"),
            TyKind::Nullable(inner) => text("Nullable<") + self.ty(inner) + text(">"),
            TyKind::Array { elem, dims } => {
                let mut d = text("array<") + self.ty(elem);
                for extent in dims {
                    d = d + text(format!(", {extent}"));
                }
                d + text(">")
            }
            TyKind::Record { def, args, flex } => {
                let mut d = match flex {
                    Flexivity::Flex | Flexivity::Rigid | Flexivity::Regional => {
                        text(format!("[{}] ", cap_name(flex)))
                    }
                    Flexivity::Irrelevant => Doc::Null,
                };
                d = d + text(self.defs.path(def).display(self.resolver));
                if !args.is_empty() {
                    // Turbofish so record type args don't clash with cmp `<`.
                    let parts: Vec<Doc<'static>> = args.iter().map(|&a| self.ty(a)).collect();
                    d = d + text("::<") + comma_sep(parts) + text(">");
                }
                d
            }
            TyKind::Closure { params, ret } => {
                let parts: Vec<Doc<'static>> = params.iter().map(|&p| self.ty(p)).collect();
                text("(") + comma_sep(parts) + text(") -> ") + self.ty(ret)
            }
            // A ground MIR carries no generics or holes; render defensively.
            TyKind::Generic(g) => text(format!("<generic {}>", g.0)),
            TyKind::Hole(h) => text(format!("<hole {}>", h.0)),
        }
    }

    // ----- expressions -----

    /// Block/statement context: a `Seq` lays its items out `;`-separated (the
    /// enclosing braces bound it); anything else renders as a single value.
    fn expr(&self, e: &mir::Expr<'_>) -> Doc<'static> {
        match e.kind {
            mir::ExprKind::Seq(items) => {
                let n = items.len();
                // The leading `[span]` is the `Seq`'s own (its items carry
                // theirs inline) — the dual of the grammar's `Block` rule.
                let mut d = match (self.sources, e.span) {
                    (Some(_), Some(sp)) => text(format!("[{}..{}]", sp.start, sp.end)) + hardline(),
                    _ => Doc::Null,
                };
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        d = d + hardline();
                    }
                    d = d + self.value(item);
                    if i + 1 < n {
                        d = d + text(";");
                    }
                }
                d
            }
            _ => self.value(e),
        }
    }

    /// A single expression, **fully typed**: every value node prints `atom : ty`,
    /// so the parser reads the type back rather than re-deriving it. `let` (always
    /// `unit`) and a sub-expression `Seq` (braced, typed by its last item) are the
    /// two forms whose type is structural and so left unprinted.
    fn value(&self, e: &mir::Expr<'_>) -> Doc<'static> {
        match e.kind {
            mir::ExprKind::Let {
                var: v,
                name,
                value,
            } => {
                text("let")
                    + self.span_doc(e.span)
                    + text(" ")
                    + var(v)
                    + text(format!(" ({}) = ", self.resolver.resolve(name)))
                    + self.value(value)
            }
            mir::ExprKind::Seq(_) => text("{ ") + self.expr(e) + text(" }"),
            mir::ExprKind::If(c, t, f) => self.typed(self.render_if(c, t, f), e),
            mir::ExprKind::Match(scrut, tree) => self.typed(self.render_match(scrut, &tree), e),
            _ => self.typed(self.atom(e), e),
        }
    }

    /// ` [start..end]` — a node's span suffix (locations mode only).
    fn span_doc(&self, span: Option<crate::surface::Span>) -> Doc<'static> {
        match (self.sources, span) {
            (Some(_), Some(sp)) => text(format!(" [{}..{}]", sp.start, sp.end)),
            _ => Doc::Null,
        }
    }

    fn typed(&self, atom: Doc<'static>, e: &mir::Expr<'_>) -> Doc<'static> {
        atom + text(" : ") + self.ty(e.ty) + self.span_doc(e.span)
    }

    /// `if C then { T } else { F }` — braces delimit the branches (no
    /// dangling-else); self-delimiting, so it round-trips at any position.
    fn render_if(&self, c: &mir::Expr<'_>, t: &mir::Expr<'_>, f: &mir::Expr<'_>) -> Doc<'static> {
        text("if ")
            + self.value(c)
            + text(" then {")
            + indent(hardline() + self.expr(t))
            + hardline()
            + text("} else {")
            + indent(hardline() + self.expr(f))
            + hardline()
            + text("}")
    }

    fn render_match(&self, scrut: &mir::Expr<'_>, tree: &mir::DecisionTree<'_>) -> Doc<'static> {
        text("match ")
            + self.value(scrut)
            + text(" {")
            + indent(hardline() + self.tree(tree))
            + hardline()
            + text("}")
    }

    /// The bare form of a value node, without its `: ty` suffix (added by
    /// [`Self::value`]). `Let`/`Seq`/`If`/`Match` are handled there.
    fn atom(&self, e: &mir::Expr<'_>) -> Doc<'static> {
        use mir::ExprKind::*;
        match e.kind {
            GlobalStr(s) => text(str_lit(s.words())),
            ConstInt(n) => text(format!("{n}")),
            ConstFloat(f) => text(f.to_string()),
            ConstBool(b) => text(format!("{b}")),
            ConstChar(c) => text(format!("char#{c}")),
            Var(v) => var(v),
            Poison => text("poison"),
            Negate(x) => text("-(") + self.value(x) + text(")"),
            Not(x) => text("!(") + self.value(x) + text(")"),
            Arith(l, op, r) => self.binop(l, arith_sym(op), r),
            Cmp(l, op, r) => self.binop(l, cmp_sym(op), r),
            Cast(x, t) => text("(") + self.value(x) + text(" as ") + self.ty(t) + text(")"),
            RegionRun(x) => text("region { ") + self.value(x) + text(" }"),
            Proj(base, path) => {
                let mut d = text("proj(") + self.value(base);
                for idx in path {
                    d = d + text(format!(", {idx}"));
                }
                d + text(")")
            }
            Assign(dst, field, src) => {
                text("assign(")
                    + self.value(dst)
                    + text(format!(", {field}, "))
                    + self.value(src)
                    + text(")")
            }
            Call {
                callee,
                args,
                regional,
            } => {
                let pre = if regional { "regional " } else { "" };
                text(format!("{pre}@{}(", self.sym(callee))) + self.arg_list(args) + text(")")
            }
            Ctor { record, args } => {
                text(format!("@{}{{", self.sym(record))) + self.arg_list(args) + text("}")
            }
            Variant {
                record,
                variant,
                args,
            } => {
                text(format!("@{}::#{variant}(", self.sym(record)))
                    + self.arg_list(args)
                    + text(")")
            }
            Intrinsic { op, args } => {
                text(format!(
                    "intrinsic#{}#{}#{}(",
                    op.family(),
                    op.name(),
                    op.imm()
                )) + self.arg_list(args)
                    + text(")")
            }
            NullableCall(inner) => match inner {
                Some(x) => text("NonNull(") + self.value(x) + text(")"),
                None => text("Null"),
            },
            ClosureCall { target, args } => {
                let mut d = text("apply(") + self.value(target);
                if !args.is_empty() {
                    d = d + text(", ") + self.arg_list(args);
                }
                d + text(")")
            }
            Closure(c) => {
                let caps: Vec<Doc<'static>> = c
                    .captures
                    .iter()
                    .map(|(v, t)| var(*v) + text(": ") + self.ty(*t))
                    .collect();
                let params: Vec<Doc<'static>> = c
                    .params
                    .iter()
                    .map(|(v, t)| var(*v) + text(": ") + self.ty(*t))
                    .collect();
                text("closure[")
                    + comma_sep(caps)
                    + text("](")
                    + comma_sep(params)
                    + text(") { ")
                    + self.value(c.body)
                    + text(" }")
            }
            ArrayOp { op, args, kernel } => {
                let mut d =
                    text(format!("array#{}(", op.as_str())) + self.arg_list(args) + text(")");
                if let Some(k) = kernel {
                    let params: Vec<Doc<'static>> = k
                        .params
                        .iter()
                        .map(|(v, t)| var(*v) + text(": ") + self.ty(*t))
                        .collect();
                    d = d
                        + text(" kernel(")
                        + comma_sep(params)
                        + text(") { ")
                        + self.value(k.body)
                        + text(" }");
                }
                d
            }
            Let { .. } | Seq(_) | If(..) | Match(..) => {
                unreachable!("structural forms are rendered by `value`")
            }
        }
    }

    fn binop(&self, l: &mir::Expr<'_>, sym: &str, r: &mir::Expr<'_>) -> Doc<'static> {
        text("(") + self.value(l) + text(format!(" {sym} ")) + self.value(r) + text(")")
    }

    fn arg_list(&self, args: &[mir::Expr<'_>]) -> Doc<'static> {
        comma_sep(args.iter().map(|a| self.value(a)).collect())
    }

    // ----- decision trees -----

    fn tree(&self, tree: &mir::DecisionTree<'_>) -> Doc<'static> {
        match *tree {
            mir::DecisionTree::Uncovered => text("uncovered"),
            mir::DecisionTree::Unreachable => text("unreachable"),
            mir::DecisionTree::Leaf { body, bindings } => {
                self.bindings(bindings) + text("=> ") + self.expr(body)
            }
            mir::DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => {
                self.bindings(bindings)
                    + text("if ")
                    + self.value(guard)
                    + text(" {")
                    + indent(hardline() + self.tree(success))
                    + hardline()
                    + text("} else {")
                    + indent(hardline() + self.tree(failure))
                    + hardline()
                    + text("}")
            }
            mir::DecisionTree::Switch { scrutinee, cases } => {
                text("switch ")
                    + pat_ref(scrutinee)
                    + text(" {")
                    + indent(hardline() + self.cases(&cases))
                    + hardline()
                    + text("}")
            }
        }
    }

    fn cases(&self, cases: &mir::SwitchCases<'_>) -> Doc<'static> {
        let arms: Vec<Doc<'static>> = match *cases {
            mir::SwitchCases::Int { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(n, t)| self.arm(text(format!("{n}")), t))
                    .collect();
                v.push(self.arm(text("_"), default));
                v
            }
            mir::SwitchCases::Bool { if_true, if_false } => vec![
                self.arm(text("true"), if_true),
                self.arm(text("false"), if_false),
            ],
            mir::SwitchCases::Char { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(c, t)| self.arm(text(format!("char#{c}")), t))
                    .collect();
                v.push(self.arm(text("_"), default));
                v
            }
            mir::SwitchCases::Ctor(arms) => arms
                .iter()
                .enumerate()
                .map(|(i, t)| self.arm(text(format!("#{i}")), t))
                .collect(),
            mir::SwitchCases::String { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(s, t)| self.arm(text(str_lit(s.words())), t))
                    .collect();
                v.push(self.arm(text("_"), default));
                v
            }
            mir::SwitchCases::Nullable { non_null, null } => vec![
                self.arm(text("NonNull"), non_null),
                self.arm(text("Null"), null),
            ],
        };
        let mut d = Doc::Null;
        for (i, a) in arms.into_iter().enumerate() {
            if i > 0 {
                d = d + hardline();
            }
            d = d + a;
        }
        d
    }

    fn arm(&self, label: Doc<'static>, tree: &mir::DecisionTree<'_>) -> Doc<'static> {
        // Braced so the sub-tree (whose leaf body is an expression) is bounded —
        // an int case label can't be mistaken for the start of the next body.
        label + text(" => {") + indent(hardline() + self.tree(tree)) + hardline() + text("}")
    }

    fn bindings(&self, bindings: &[mir::Binding<'_>]) -> Doc<'static> {
        let mut d = Doc::Null;
        for (v, path) in bindings {
            d = d + var(*v) + text("=") + pat_ref(path) + text(" ");
        }
        d
    }
}

/// Owned text node, so the [`Doc`] never borrows the render context.
fn text(s: impl Into<String>) -> Doc<'static> {
    Doc::from(s.into())
}

fn var(v: VarId) -> Doc<'static> {
    text(format!("v{}", v.0))
}

fn pat_ref(path: &[u32]) -> Doc<'static> {
    let mut s = String::from("scrut");
    for idx in path {
        let _ = write!(s, ".{idx}");
    }
    text(s)
}

fn comma_sep(docs: Vec<Doc<'static>>) -> Doc<'static> {
    let mut d = Doc::Null;
    for (i, item) in docs.into_iter().enumerate() {
        if i > 0 {
            d = d + text(", ");
        }
        d = d + item;
    }
    d
}

fn cap_name(c: Flexivity) -> &'static str {
    match c {
        Flexivity::Flex => "flex",
        Flexivity::Rigid => "rigid",
        Flexivity::Regional => "regional",
        Flexivity::Irrelevant => "",
    }
}

/// `str#w0_w1_w2_w3` — all four `StringToken` words, so the literal
/// round-trips faithfully.
fn str_lit(w: [u64; 4]) -> String {
    format!("str#{}#{}#{}#{}", w[0], w[1], w[2], w[3])
}

fn string_decl(token: crate::utils::string::StringToken, payload: &str) -> Doc<'static> {
    text(format!("{} = {:?};", str_lit(token.words()), payload))
}

fn arith_sym(op: ArithOp) -> &'static str {
    match op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Div => "/",
        ArithOp::Mod => "%",
        ArithOp::And => "&&",
        ArithOp::Or => "||",
    }
}

fn cmp_sym(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "<",
        CmpOp::Gt => ">",
        CmpOp::Le => "<=",
        CmpOp::Ge => ">=",
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::elaborate;
    use crate::{full::mono::monomorphize, surface, with_tcx};

    fn render(source: &str) -> String {
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            Printer::new(&elab.defs, elab.resolver).program(&full)
        })
    }

    #[test]
    fn prints_a_recursive_scalar_function() {
        let src = r#"
            pub fn fib(n: u64) -> u64 {
                if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
            }
        "#;
        let out = render(src);
        println!("{out}");
        assert!(out.contains("pub fn @_RC3fib("), "{out}");
        assert!(out.contains("if "), "{out}");
        assert!(out.contains("@_RC3fib("), "{out}");
        assert!(out.contains(" : u64"), "{out}");
    }
}
