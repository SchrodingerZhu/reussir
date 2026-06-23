//! A readable, **round-trippable** textual rendering of the Full (MIR) program.
//!
//! This is the `reussir-elab --mode full` output and the *serialization*
//! direction of the textual IR: a Rust-ish dump of each ground function keyed by
//! its v0 symbol. Unlike the old pretty-printer, this spelling is a contract —
//! the lalrpop grammar (the deserializer) mirrors it, and round-trip tests gate
//! the two against each other.
//!
//! Calls, constructors and record instances reference interned [`mir::Symbol`]s
//! as `@symbol`; locals are `v<id>` ([`VarId`]); types print structurally.

use std::fmt::Write;

use lasso::Rodeo;
use reussir_syntax::kind::{Resolver, TokenKey};

use crate::full::mir;
use crate::semi::hir::{ArithOp, CmpOp, VarId};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyKind};
use crate::surface::Visibility;

/// Renders a Full MIR program to text.
pub struct Printer<'a> {
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
}

impl<'a> Printer<'a> {
    pub fn new(defs: &'a DefTable, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        Printer { defs, resolver }
    }

    /// Render `program`. The program owns the symbol interner, so it is borrowed
    /// here to resolve `@symbol` references.
    pub fn program(&self, program: &mir::Program<'_>) -> String {
        let r = Render {
            symbols: &program.symbols,
            defs: self.defs,
            resolver: self.resolver,
        };
        let mut out = String::new();
        for rec in &program.records {
            let _ = writeln!(out, "record @{};", r.sym(rec.symbol));
        }
        if !program.records.is_empty() {
            out.push('\n');
        }
        for (i, func) in program.functions.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            r.function(&mut out, func);
        }
        for t in &program.trampolines {
            let _ = writeln!(
                out,
                "\nextern \"{}\" trampoline \"{}\" = @{};",
                t.abi,
                r.sym(t.export),
                r.sym(t.target),
            );
        }
        out
    }
}

/// Borrows everything needed to render one program: the symbol interner, the def
/// table (for record-type paths), and the source identifier resolver.
struct Render<'a> {
    symbols: &'a Rodeo,
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
}

impl Render<'_> {
    fn sym(&self, s: mir::Symbol) -> &str {
        self.symbols.resolve(&s.0)
    }

    fn function(&self, out: &mut String, func: &mir::Function<'_>) {
        if func.visibility == Visibility::Public {
            out.push_str("pub ");
        }
        if func.is_regional {
            out.push_str("regional ");
        }
        let _ = write!(out, "fn @{}(", self.sym(func.symbol));
        for (i, p) in func.params.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            var(out, p.var);
            let _ = write!(out, " ({}): ", self.resolver.resolve(p.name));
            self.ty(out, p.ty);
        }
        out.push_str(") -> ");
        self.ty(out, func.return_ty);
        match func.body {
            Some(body) => {
                out.push_str(" {\n");
                self.expr(out, body, 1);
                out.push('\n');
                out.push_str("}\n");
            }
            None => out.push_str(";\n"),
        }
    }

    // ----- types -----

    fn ty(&self, out: &mut String, ty: Ty<'_>) {
        match *ty.kind() {
            TyKind::Int(IntTy::Signed(w)) => {
                let _ = write!(out, "i{w}");
            }
            TyKind::Int(IntTy::Unsigned(w)) => {
                let _ = write!(out, "u{w}");
            }
            TyKind::Fp(FpTy::Ieee(w)) => {
                let _ = write!(out, "f{w}");
            }
            TyKind::Fp(FpTy::BFloat16) => out.push_str("bf16"),
            TyKind::Fp(FpTy::Float8) => out.push_str("f8"),
            TyKind::Bool => out.push_str("bool"),
            TyKind::Str => out.push_str("str"),
            TyKind::Unit => out.push_str("()"),
            TyKind::Bottom => out.push('!'),
            TyKind::Nullable(inner) => {
                out.push_str("Nullable<");
                self.ty(out, inner);
                out.push('>');
            }
            TyKind::Record { def, args, flex } => {
                if let Capability::Flex | Capability::Rigid | Capability::Regional = flex {
                    let _ = write!(out, "[{}] ", cap_name(flex));
                }
                out.push_str(&self.defs.path(def).display(self.resolver));
                if !args.is_empty() {
                    out.push('<');
                    for (i, &a) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        self.ty(out, a);
                    }
                    out.push('>');
                }
            }
            TyKind::Closure { params, ret } => {
                out.push('(');
                for (i, &p) in params.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.ty(out, p);
                }
                out.push_str(") -> ");
                self.ty(out, ret);
            }
            // A ground MIR carries no generics or holes; render defensively.
            TyKind::Generic(g) => {
                let _ = write!(out, "<generic {}>", g.0);
            }
            TyKind::Hole(h) => {
                let _ = write!(out, "<hole {}>", h.0);
            }
        }
    }

    // ----- expressions -----

    /// Print an expression at the given indentation. Blocks (sequences,
    /// `if`/`else`, `match`) span multiple lines; everything else is inline.
    fn expr(&self, out: &mut String, e: &mir::Expr<'_>, depth: usize) {
        match e.kind {
            mir::ExprKind::Seq(items) => self.seq(out, items, depth),
            mir::ExprKind::If(c, t, f) => {
                out.push_str("if ");
                self.inline(out, c);
                out.push_str(" then\n");
                indent(out, depth + 1);
                self.expr(out, t, depth + 1);
                out.push('\n');
                indent(out, depth);
                out.push_str("else\n");
                indent(out, depth + 1);
                self.expr(out, f, depth + 1);
            }
            mir::ExprKind::Match(scrut, tree) => {
                out.push_str("match ");
                self.inline(out, scrut);
                out.push_str(" {\n");
                self.tree(out, &tree, depth + 1);
                indent(out, depth);
                out.push('}');
            }
            _ => self.inline(out, e),
        }
    }

    fn seq(&self, out: &mut String, items: &[mir::Expr<'_>], depth: usize) {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push('\n');
                indent(out, depth);
            }
            self.expr(out, item, depth);
        }
    }

    /// Print an expression inline (with parenthesized, type-annotated subterms).
    fn inline(&self, out: &mut String, e: &mir::Expr<'_>) {
        use mir::ExprKind::*;
        match e.kind {
            GlobalStr(s) => {
                let _ = write!(out, "str#{:x}", s.words()[0]);
            }
            ConstInt(n) => {
                let _ = write!(out, "{n} : ");
                self.ty(out, e.ty);
            }
            ConstFloat(f) => {
                let _ = write!(out, "{f} : ");
                self.ty(out, e.ty);
            }
            ConstBool(b) => {
                let _ = write!(out, "{b}");
            }
            Var(v) => var(out, v),
            Poison => out.push_str("<poison>"),
            Negate(x) => {
                out.push_str("-(");
                self.inline(out, x);
                out.push(')');
            }
            Not(x) => {
                out.push_str("!(");
                self.inline(out, x);
                out.push(')');
            }
            Arith(l, op, r) => self.binop(out, l, arith_sym(op), r, e.ty),
            Cmp(l, op, r) => self.binop(out, l, cmp_sym(op), r, e.ty),
            Cast(x, t) => {
                out.push('(');
                self.inline(out, x);
                out.push_str(" as ");
                self.ty(out, t);
                out.push(')');
            }
            RegionRun(x) => {
                out.push_str("region { ");
                self.inline(out, x);
                out.push_str(" }");
            }
            Proj(base, path) => {
                self.inline(out, base);
                for idx in path {
                    let _ = write!(out, ".{idx}");
                }
            }
            Assign(dst, field, src) => {
                self.inline(out, dst);
                let _ = write!(out, ".{field} = ");
                self.inline(out, src);
            }
            Let {
                var: v,
                name,
                value,
            } => {
                out.push_str("let ");
                var(out, v);
                let _ = write!(out, " ({}): ", self.resolver.resolve(name));
                self.ty(out, value.ty);
                out.push_str(" = ");
                self.inline(out, value);
                out.push_str(" in");
            }
            Call {
                callee,
                args,
                regional,
            } => {
                if regional {
                    out.push_str("regional ");
                }
                let _ = write!(out, "@{}", self.sym(callee));
                self.args(out, args);
                out.push_str(" : ");
                self.ty(out, e.ty);
            }
            Ctor { record, args } => {
                let _ = write!(out, "@{}{{", self.sym(record));
                self.arg_list(out, args);
                out.push('}');
            }
            Variant {
                record,
                variant,
                args,
            } => {
                let _ = write!(out, "@{}::#{variant}", self.sym(record));
                self.args(out, args);
            }
            NullableCall(inner) => match inner {
                Some(x) => {
                    out.push_str("NonNull(");
                    self.inline(out, x);
                    out.push(')');
                }
                None => out.push_str("Null"),
            },
            ClosureCall { target, args } => {
                self.inline(out, target);
                self.args(out, args);
            }
            Closure(c) => {
                out.push_str("closure[");
                for (i, (v, _)) in c.captures.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    var(out, *v);
                }
                out.push_str("](");
                for (i, (v, t)) in c.params.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    var(out, *v);
                    out.push_str(": ");
                    self.ty(out, *t);
                }
                out.push_str(") { ");
                self.inline(out, c.body);
                out.push_str(" }");
            }
            // Block forms shouldn't normally appear inline, but a nested
            // `if`/`match`/`Seq` is still renderable on one logical line.
            Seq(_) | If(..) | Match(..) => {
                out.push('(');
                self.expr(out, e, 0);
                out.push(')');
            }
        }
    }

    fn binop(&self, out: &mut String, l: &mir::Expr<'_>, sym: &str, r: &mir::Expr<'_>, ty: Ty<'_>) {
        out.push('(');
        self.inline(out, l);
        let _ = write!(out, " {sym} ");
        self.inline(out, r);
        out.push_str(") : ");
        self.ty(out, ty);
    }

    fn args(&self, out: &mut String, args: &[mir::Expr<'_>]) {
        out.push('(');
        self.arg_list(out, args);
        out.push(')');
    }

    fn arg_list(&self, out: &mut String, args: &[mir::Expr<'_>]) {
        for (i, a) in args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            self.inline(out, a);
        }
    }

    // ----- decision trees -----

    fn tree(&self, out: &mut String, tree: &mir::DecisionTree<'_>, depth: usize) {
        match *tree {
            mir::DecisionTree::Uncovered => {
                indent(out, depth);
                out.push_str("<uncovered>\n");
            }
            mir::DecisionTree::Unreachable => {
                indent(out, depth);
                out.push_str("<unreachable>\n");
            }
            mir::DecisionTree::Leaf { body, bindings } => {
                indent(out, depth);
                self.bindings(out, bindings);
                out.push_str("=> ");
                self.expr(out, body, depth);
                out.push('\n');
            }
            mir::DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => {
                indent(out, depth);
                self.bindings(out, bindings);
                out.push_str("if ");
                self.inline(out, guard);
                out.push_str(" {\n");
                self.tree(out, success, depth + 1);
                indent(out, depth);
                out.push_str("} else {\n");
                self.tree(out, failure, depth + 1);
                indent(out, depth);
                out.push_str("}\n");
            }
            mir::DecisionTree::Switch { scrutinee, cases } => {
                indent(out, depth);
                out.push_str("switch ");
                pat_ref(out, scrutinee);
                out.push_str(" {\n");
                self.cases(out, &cases, depth + 1);
                indent(out, depth);
                out.push_str("}\n");
            }
        }
    }

    fn cases(&self, out: &mut String, cases: &mir::SwitchCases<'_>, depth: usize) {
        match *cases {
            mir::SwitchCases::Int { cases, default } => {
                for (n, t) in cases {
                    indent(out, depth);
                    let _ = writeln!(out, "{n} =>");
                    self.tree(out, t, depth + 1);
                }
                indent(out, depth);
                out.push_str("_ =>\n");
                self.tree(out, default, depth + 1);
            }
            mir::SwitchCases::Bool { if_true, if_false } => {
                indent(out, depth);
                out.push_str("true =>\n");
                self.tree(out, if_true, depth + 1);
                indent(out, depth);
                out.push_str("false =>\n");
                self.tree(out, if_false, depth + 1);
            }
            mir::SwitchCases::Ctor(arms) => {
                for (i, t) in arms.iter().enumerate() {
                    indent(out, depth);
                    let _ = writeln!(out, "#{i} =>");
                    self.tree(out, t, depth + 1);
                }
            }
            mir::SwitchCases::String { cases, default } => {
                for (s, t) in cases {
                    indent(out, depth);
                    let _ = writeln!(out, "str#{:x} =>", s.words()[0]);
                    self.tree(out, t, depth + 1);
                }
                indent(out, depth);
                out.push_str("_ =>\n");
                self.tree(out, default, depth + 1);
            }
            mir::SwitchCases::Nullable { non_null, null } => {
                indent(out, depth);
                out.push_str("NonNull =>\n");
                self.tree(out, non_null, depth + 1);
                indent(out, depth);
                out.push_str("Null =>\n");
                self.tree(out, null, depth + 1);
            }
        }
    }

    fn bindings(&self, out: &mut String, bindings: &[mir::Binding<'_>]) {
        for (v, path) in bindings {
            var(out, *v);
            out.push('=');
            pat_ref(out, path);
            out.push(' ');
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

fn var(out: &mut String, v: VarId) {
    let _ = write!(out, "v{}", v.0);
}

fn pat_ref(out: &mut String, path: &[u32]) {
    out.push_str("scrut");
    for idx in path {
        let _ = write!(out, ".{idx}");
    }
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
