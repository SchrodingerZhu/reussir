//! A readable, **round-trippable** textual rendering of the Full (MIR) program.
//!
//! This is the `reussir-elab --mode full` output and the *serialization*
//! direction of the textual IR: a Rust-ish dump of each ground function keyed by
//! its v0 symbol. The spelling is a contract — the lalrpop grammar (the
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

use crate::full::mir;
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyKind};
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
}

impl<'a> Printer<'a> {
    pub fn new(defs: &'a DefTable, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        Printer { defs, resolver }
    }

    /// Render `program`. The program owns the symbol interner, borrowed here to
    /// resolve `@symbol` references.
    pub fn program(&self, program: &mir::Program<'_>) -> String {
        let r = Render {
            symbols: &program.symbols,
            defs: self.defs,
            resolver: self.resolver,
        };

        let mut items: Vec<Doc<'static>> = Vec::new();
        for rec in &program.records {
            items.push(text(format!("record @{};", r.sym(rec.symbol))));
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
}

impl Render<'_> {
    fn sym(&self, s: mir::Symbol) -> &str {
        self.symbols.resolve(&s.0)
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
        let sig = text(head) + comma_sep(params) + text(") -> ") + self.ty(func.return_ty);

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
            TyKind::Unit => text("()"),
            TyKind::Bottom => text("!"),
            TyKind::Nullable(inner) => text("Nullable<") + self.ty(inner) + text(">"),
            TyKind::Record { def, args, flex } => {
                let mut d = match flex {
                    Capability::Flex | Capability::Rigid | Capability::Regional => {
                        text(format!("[{}] ", cap_name(flex)))
                    }
                    Capability::Irrelevant => Doc::Null,
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

    /// Block-level rendering: sequences, `if`/`else`, `match` span lines;
    /// everything else is inline.
    fn expr(&self, e: &mir::Expr<'_>) -> Doc<'static> {
        match e.kind {
            mir::ExprKind::Seq(items) => {
                // `;`-separated statements (the last is the block's value), so the
                // whitespace-insensitive grammar can find the boundaries.
                let n = items.len();
                let mut d = Doc::Null;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        d = d + hardline();
                    }
                    d = d + self.expr(item);
                    if i + 1 < n {
                        d = d + text(";");
                    }
                }
                d
            }
            mir::ExprKind::If(c, t, f) => self.render_if(c, t, f),
            mir::ExprKind::Match(scrut, tree) => self.render_match(scrut, &tree),
            _ => self.inline(e),
        }
    }

    /// `if C then { T } else { F }` — braces delimit the branches (self-contained,
    /// no dangling-else), so it round-trips both at block and inline position.
    fn render_if(&self, c: &mir::Expr<'_>, t: &mir::Expr<'_>, f: &mir::Expr<'_>) -> Doc<'static> {
        text("if ")
            + self.inline(c)
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
            + self.inline(scrut)
            + text(" {")
            + indent(hardline() + self.tree(tree))
            + hardline()
            + text("}")
    }

    /// Inline rendering (parenthesized, type-annotated subterms).
    fn inline(&self, e: &mir::Expr<'_>) -> Doc<'static> {
        use mir::ExprKind::*;
        match e.kind {
            GlobalStr(s) => text(format!("str#{}", s.words()[0])),
            ConstInt(n) => text(format!("{n} : ")) + self.ty(e.ty),
            ConstFloat(f) => text(format!("{f} : ")) + self.ty(e.ty),
            ConstBool(b) => text(format!("{b}")),
            Var(v) => var(v),
            Poison => text("poison"),
            Negate(x) => text("-(") + self.inline(x) + text(")"),
            Not(x) => text("!(") + self.inline(x) + text(")"),
            Arith(l, op, r) => self.binop(l, arith_sym(op), r, e.ty),
            Cmp(l, op, r) => self.binop(l, cmp_sym(op), r, e.ty),
            Cast(x, t) => text("(") + self.inline(x) + text(" as ") + self.ty(t) + text(")"),
            RegionRun(x) => text("region { ") + self.inline(x) + text(" }"),
            Proj(base, path) => {
                let mut d = text("proj(") + self.inline(base);
                for idx in path {
                    d = d + text(format!(", {idx}"));
                }
                d + text(")")
            }
            Assign(dst, field, src) => {
                text("assign(")
                    + self.inline(dst)
                    + text(format!(", {field}, "))
                    + self.inline(src)
                    + text(")")
            }
            Let {
                var: v,
                name,
                value,
            } => {
                text("let ")
                    + var(v)
                    + text(format!(" ({}): ", self.resolver.resolve(name)))
                    + self.ty(value.ty)
                    + text(" = ")
                    + self.inline(value)
            }
            Call {
                callee,
                args,
                regional,
            } => {
                let pre = if regional { "regional " } else { "" };
                text(format!("{pre}@{}(", self.sym(callee)))
                    + self.arg_list(args)
                    + text(") : ")
                    + self.ty(e.ty)
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
            NullableCall(inner) => match inner {
                Some(x) => text("NonNull(") + self.inline(x) + text(")"),
                None => text("Null"),
            },
            ClosureCall { target, args } => {
                let mut d = text("apply(") + self.inline(target);
                if !args.is_empty() {
                    d = d + text(", ") + self.arg_list(args);
                }
                d + text(")")
            }
            Closure(c) => {
                let caps: Vec<Doc<'static>> = c.captures.iter().map(|(v, _)| var(*v)).collect();
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
                    + self.inline(c.body)
                    + text(" }")
            }
            // Block forms nested inline. `if`/`match` are already self-delimiting;
            // a `Seq` is wrapped in braces so its `;`-separated items are bounded.
            If(c, t, f) => self.render_if(c, t, f),
            Match(scrut, tree) => self.render_match(scrut, &tree),
            Seq(_) => text("{ ") + self.expr(e) + text(" }"),
        }
    }

    fn binop(&self, l: &mir::Expr<'_>, sym: &str, r: &mir::Expr<'_>, ty: Ty<'_>) -> Doc<'static> {
        text("(")
            + self.inline(l)
            + text(format!(" {sym} "))
            + self.inline(r)
            + text(") : ")
            + self.ty(ty)
    }

    fn arg_list(&self, args: &[mir::Expr<'_>]) -> Doc<'static> {
        comma_sep(args.iter().map(|a| self.inline(a)).collect())
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
                    + self.inline(guard)
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
            mir::SwitchCases::Ctor(arms) => arms
                .iter()
                .enumerate()
                .map(|(i, t)| self.arm(text(format!("#{i}")), t))
                .collect(),
            mir::SwitchCases::String { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(s, t)| self.arm(text(format!("str#{}", s.words()[0])), t))
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
        label + text(" =>") + indent(hardline() + self.tree(tree))
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

fn cap_name(c: Capability) -> &'static str {
    match c {
        Capability::Flex => "flex",
        Capability::Rigid => "rigid",
        Capability::Regional => "regional",
        Capability::Irrelevant => "",
    }
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
            let (full, reports) = monomorphize(&elab);
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
        assert!(out.contains("pub fn @_RC3fib("), "{out}");
        assert!(out.contains("if "), "{out}");
        assert!(out.contains("@_RC3fib("), "{out}");
        assert!(out.contains(" : u64"), "{out}");
    }
}
