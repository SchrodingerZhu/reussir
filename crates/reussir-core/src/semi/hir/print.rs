//! A readable, **round-trippable** textual rendering of the Semi (HIR) program.
//!
//! The HIR is the *polymorphic* phase: functions still carry generic parameters
//! (`$n`), calls reference items by [`DefId`] path (`#path`) with explicit type
//! arguments, and types may mention [`TyKind::Generic`] (but no inference holes:
//! a fully elaborated HIR has none). This is
//! the serialize half; the lalrpop grammar (`semi/hir/grammar.lalrpop`) is the dual,
//! gated by round-trip tests.
//!
//! Built with [`pprint`], sharing the spelling conventions of the MIR printer
//! ([`crate::full::mir::print`]) — `;`-separated blocks, braced `if`, functional
//! `proj`/`assign`/`apply`, turbofish type args — but keyed by `#path`/`$n`
//! rather than `@symbol`.

use pprint::{Doc, Printer as PpPrinter, hardline, indent, pprint};
use reussir_syntax::kind::{Resolver, TokenKey};

use crate::semi::ctxt::{DefaultCap, Record, TrampolineRoot};
use crate::semi::hir::{
    ArithOp, CmpOp, DecisionTree, Expr, ExprKind, Function, PatVarRef, SwitchCases, VarId,
};
use crate::semi::resolve::DefTable;
use crate::semi::ty::{Capability, DefId, FpTy, GenericId, IntTy, Ty, TyKind};
use crate::surface::{RecordKind, Visibility};

const IR_PRINTER: PpPrinter = PpPrinter {
    max_width: 100,
    indent: 4,
    use_tabs: false,
};

/// Renders Semi (HIR) functions to text.
pub struct Printer<'a> {
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
}

impl<'a> Printer<'a> {
    pub fn new(defs: &'a DefTable, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        Printer { defs, resolver }
    }

    /// Render the resumable Semi output: record declarations, trampoline roots,
    /// then the elaborated functions. Records are emitted in `DefId` order for
    /// determinism.
    pub fn program(
        &self,
        funcs: &[Function<'_>],
        records: &rustc_hash::FxHashMap<DefId, Record<'_>>,
        trampolines: &[TrampolineRoot<'_>],
    ) -> String {
        let mut items: Vec<Doc<'static>> = Vec::new();
        let mut recs: Vec<&Record<'_>> = records.values().collect();
        recs.sort_by_key(|r| r.def.0);
        for r in recs {
            items.push(self.record(r));
        }
        for t in trampolines {
            items.push(self.trampoline(t));
        }
        for f in funcs {
            items.push(self.function(f));
        }

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

    fn path(&self, def: DefId) -> String {
        self.defs.path(def).display(self.resolver)
    }

    /// `<$0, regional $1>` — a generic binder list, shared by records and
    /// functions; `regional` marks a generic in `regional_generics`.
    fn generics_binder(
        &self,
        params: &[(TokenKey, GenericId)],
        regional: &[GenericId],
    ) -> Doc<'static> {
        if params.is_empty() {
            return Doc::Null;
        }
        let parts: Vec<Doc<'static>> = params
            .iter()
            .map(|(_, g)| {
                let r = if regional.contains(g) {
                    "regional "
                } else {
                    ""
                };
                text(format!("{r}${}", g.0))
            })
            .collect();
        text("<") + comma_sep(parts) + text(">")
    }

    fn record(&self, r: &Record<'_>) -> Doc<'static> {
        let cap = match r.default_cap {
            DefaultCap::Value => "[value] ",
            DefaultCap::Shared => "[shared] ",
            DefaultCap::Regional => "[regional] ",
        };
        let kind = match r.kind {
            RecordKind::StructKind => "struct #",
            RecordKind::EnumKind => "enum #",
        };
        text(format!("{cap}{kind}{}", self.path(r.def)))
            + self.generics_binder(&r.ty_params, &r.regional_generics)
            + text(";")
    }

    fn trampoline(&self, t: &TrampolineRoot<'_>) -> Doc<'static> {
        text(format!(
            "extern \"{}\" trampoline \"{}\" = #{}",
            t.abi,
            t.name,
            self.path(t.target)
        )) + self.ty_args(&t.ty_args)
            + text(";")
    }

    fn function(&self, f: &Function<'_>) -> Doc<'static> {
        let mut head = String::new();
        if f.visibility == Visibility::Public {
            head.push_str("pub ");
        }
        if f.is_regional {
            head.push_str("regional ");
        }
        head.push_str("fn #");
        head.push_str(&self.path(f.def));

        let mut sig = text(head) + self.generics_binder(&f.generics, &f.regional_generics);
        let params: Vec<Doc<'static>> = f
            .params
            .iter()
            .map(|(name, var, ty)| {
                text(format!("v{} ({}): ", var.0, self.resolver.resolve(*name))) + self.ty(*ty)
            })
            .collect();
        sig = sig + text("(") + comma_sep(params) + text(") -> ") + self.ty(f.return_ty);

        match &f.body {
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
            TyKind::Generic(g) => text(format!("${}", g.0)),
            TyKind::Hole(_) => unreachable!("a fully elaborated HIR carries no inference holes"),
            TyKind::Nullable(inner) => text("Nullable<") + self.ty(inner) + text(">"),
            TyKind::Record { def, args, flex } => {
                let mut d = match flex {
                    Capability::Flex | Capability::Rigid | Capability::Regional => {
                        text(format!("[{}] ", cap_name(flex)))
                    }
                    Capability::Irrelevant => Doc::Null,
                };
                d = d + text(format!("#{}", self.path(def)));
                if !args.is_empty() {
                    let parts: Vec<Doc<'static>> = args.iter().map(|&a| self.ty(a)).collect();
                    d = d + text("::<") + comma_sep(parts) + text(">");
                }
                d
            }
            TyKind::Closure { params, ret } => {
                let parts: Vec<Doc<'static>> = params.iter().map(|&p| self.ty(p)).collect();
                text("(") + comma_sep(parts) + text(") -> ") + self.ty(ret)
            }
        }
    }

    /// Optional `::<args>` type-argument list.
    fn ty_args(&self, args: &[Ty<'_>]) -> Doc<'static> {
        if args.is_empty() {
            Doc::Null
        } else {
            let parts: Vec<Doc<'static>> = args.iter().map(|&a| self.ty(a)).collect();
            text("::<") + comma_sep(parts) + text(">")
        }
    }

    // ----- expressions -----

    /// Block/statement context: a `Seq` lays out `;`-separated; else a value.
    fn expr(&self, e: &Expr<'_>) -> Doc<'static> {
        match &e.kind {
            ExprKind::Seq(items) => {
                let n = items.len();
                let mut d = Doc::Null;
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

    /// A single expression, **fully typed**: every value node prints `atom : ty`.
    /// `let` (always `unit`) and a sub-expression `Seq` (braced) are unprinted.
    fn value(&self, e: &Expr<'_>) -> Doc<'static> {
        match &e.kind {
            ExprKind::Let {
                var: v,
                name,
                value,
                ..
            } => {
                text("let ")
                    + var(*v)
                    + text(format!(" ({}) = ", self.resolver.resolve(*name)))
                    + self.value(value)
            }
            ExprKind::Seq(_) => text("{ ") + self.expr(e) + text(" }"),
            ExprKind::If(c, t, f) => self.typed(self.render_if(c, t, f), e.ty),
            ExprKind::Match(scrut, tree) => self.typed(self.render_match(scrut, tree), e.ty),
            _ => self.typed(self.atom(e), e.ty),
        }
    }

    fn typed(&self, atom: Doc<'static>, ty: Ty<'_>) -> Doc<'static> {
        atom + text(" : ") + self.ty(ty)
    }

    fn render_if(&self, c: &Expr<'_>, t: &Expr<'_>, f: &Expr<'_>) -> Doc<'static> {
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

    fn render_match(&self, scrut: &Expr<'_>, tree: &DecisionTree<'_>) -> Doc<'static> {
        text("match ")
            + self.value(scrut)
            + text(" {")
            + indent(hardline() + self.tree(tree))
            + hardline()
            + text("}")
    }

    /// The bare form of a value node, without its `: ty` suffix.
    fn atom(&self, e: &Expr<'_>) -> Doc<'static> {
        match &e.kind {
            ExprKind::GlobalStr(s) => text(str_lit(s.words())),
            ExprKind::ConstInt(n) => text(format!("{n}")),
            ExprKind::ConstFloat(f) => text(float_lit(*f)),
            ExprKind::ConstBool(b) => text(format!("{b}")),
            ExprKind::Var(v) => var(*v),
            ExprKind::Poison => text("poison"),
            ExprKind::Negate(x) => text("-(") + self.value(x) + text(")"),
            ExprKind::Not(x) => text("!(") + self.value(x) + text(")"),
            ExprKind::Arith(l, op, r) => self.binop(l, arith_sym(*op), r),
            ExprKind::Cmp(l, op, r) => self.binop(l, cmp_sym(*op), r),
            ExprKind::Cast(x, t) => {
                text("(") + self.value(x) + text(" as ") + self.ty(*t) + text(")")
            }
            ExprKind::RegionRun(x) => text("region { ") + self.value(x) + text(" }"),
            ExprKind::Proj(base, path) => {
                let mut d = text("proj(") + self.value(base);
                for idx in path {
                    d = d + text(format!(", {idx}"));
                }
                d + text(")")
            }
            ExprKind::Assign(dst, field, src) => {
                text("assign(")
                    + self.value(dst)
                    + text(format!(", {field}, "))
                    + self.value(src)
                    + text(")")
            }
            ExprKind::FuncCall {
                target,
                ty_args,
                args,
                regional,
            } => {
                let pre = if *regional { "regional " } else { "" };
                text(format!("{pre}#{}", self.path(*target)))
                    + self.ty_args(ty_args)
                    + text("(")
                    + self.arg_list(args)
                    + text(")")
            }
            ExprKind::CompoundCall {
                target,
                ty_args,
                args,
            } => {
                text(format!("#{}", self.path(*target)))
                    + self.ty_args(ty_args)
                    + text("{")
                    + self.arg_list(args)
                    + text("}")
            }
            ExprKind::VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => {
                text(format!("#{}", self.path(*target)))
                    + self.ty_args(ty_args)
                    + text(format!("#{variant}("))
                    + self.arg_list(args)
                    + text(")")
            }
            ExprKind::NullableCall(inner) => match inner {
                Some(x) => text("NonNull(") + self.value(x) + text(")"),
                None => text("Null"),
            },
            ExprKind::ClosureCall { target, args } => {
                let mut d = text("apply(") + self.value(target);
                if !args.is_empty() {
                    d = d + text(", ") + self.arg_list(args);
                }
                d + text(")")
            }
            ExprKind::Closure(c) => {
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
                    + self.value(&c.body)
                    + text(" }")
            }
            ExprKind::Let { .. } | ExprKind::Seq(_) | ExprKind::If(..) | ExprKind::Match(..) => {
                unreachable!("structural forms are rendered by `value`")
            }
        }
    }

    fn binop(&self, l: &Expr<'_>, sym: &str, r: &Expr<'_>) -> Doc<'static> {
        text("(") + self.value(l) + text(format!(" {sym} ")) + self.value(r) + text(")")
    }

    fn arg_list(&self, args: &[Expr<'_>]) -> Doc<'static> {
        comma_sep(args.iter().map(|a| self.value(a)).collect())
    }

    // ----- decision trees -----

    fn tree(&self, tree: &DecisionTree<'_>) -> Doc<'static> {
        match tree {
            DecisionTree::Uncovered => text("uncovered"),
            DecisionTree::Unreachable => text("unreachable"),
            DecisionTree::Leaf { body, bindings } => {
                self.bindings(bindings) + text("=> ") + self.expr(body)
            }
            DecisionTree::Guard {
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
            DecisionTree::Switch { scrutinee, cases } => {
                text("switch ")
                    + pat_ref(scrutinee)
                    + text(" {")
                    + indent(hardline() + self.cases(cases))
                    + hardline()
                    + text("}")
            }
        }
    }

    fn cases(&self, cases: &SwitchCases<'_>) -> Doc<'static> {
        let arms: Vec<Doc<'static>> = match cases {
            SwitchCases::Int { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(n, t)| self.arm(text(format!("{n}")), t))
                    .collect();
                v.push(self.arm(text("_"), default));
                v
            }
            SwitchCases::Bool { if_true, if_false } => vec![
                self.arm(text("true"), if_true),
                self.arm(text("false"), if_false),
            ],
            SwitchCases::Ctor(arms) => arms
                .iter()
                .enumerate()
                .map(|(i, t)| self.arm(text(format!("#{i}")), t))
                .collect(),
            SwitchCases::String { cases, default } => {
                let mut v: Vec<Doc<'static>> = cases
                    .iter()
                    .map(|(s, t)| self.arm(text(str_lit(s.words())), t))
                    .collect();
                v.push(self.arm(text("_"), default));
                v
            }
            SwitchCases::Nullable { non_null, null } => vec![
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

    fn arm(&self, label: Doc<'static>, tree: &DecisionTree<'_>) -> Doc<'static> {
        label + text(" => {") + indent(hardline() + self.tree(tree)) + hardline() + text("}")
    }

    fn bindings(&self, bindings: &[(VarId, PatVarRef)]) -> Doc<'static> {
        let mut d = Doc::Null;
        for (v, path) in bindings {
            d = d + var(*v) + text("=") + pat_ref(path) + text(" ");
        }
        d
    }
}

fn text(s: impl Into<String>) -> Doc<'static> {
    Doc::from(s.into())
}

fn var(v: VarId) -> Doc<'static> {
    text(format!("v{}", v.0))
}

fn pat_ref(path: &PatVarRef) -> Doc<'static> {
    let mut s = String::from("scrut");
    for idx in &path.0 {
        s.push_str(&format!(".{idx}"));
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

/// `str#w0_w1_w2_w3` — all four `StringToken` words, so the literal
/// round-trips faithfully.
fn str_lit(w: [u64; 4]) -> String {
    format!("str#{}#{}#{}#{}", w[0], w[1], w[2], w[3])
}

/// Print a float so it always lexes back as a `Token::Float` (`digits.digits`):
/// `f64`'s `Display` drops the point for integral values (`1.0` -> `1`), which
/// would otherwise re-lex as an integer.
fn float_lit(f: f64) -> String {
    let s = format!("{f}");
    if s.contains('.') { s } else { format!("{s}.0") }
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
