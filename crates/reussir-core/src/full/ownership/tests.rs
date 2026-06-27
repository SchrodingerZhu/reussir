//! Test harness for the ownership analysis.
//!
//! The analysis is a pure `(&Function, &RecordTable) → OwnershipTable`, so it is
//! exercised without any MLIR or runtime. Three pieces, mirroring
//! `docs/ownership-analysis.md` §9:
//!
//! * [`MirBuilder`] — constructs `mir::Function`s directly in the arena, sharing
//!   the production [`mir::ExprIdGen`] anchor stamping and using **real interned
//!   types** so [`Rr::is_rr`] is tested for real, not stubbed.
//! * [`render`] — an annotated pretty-printer that weaves each emitted
//!   `dup`/`drop` into the tree. Every corpus test prints its rendering (captured
//!   by `cargo test`; pass `--nocapture` to watch them), so the placement is
//!   eyeball-checkable as well as asserted.
//! * [`check_balanced`] — a generic safety net that abstractly interprets the
//!   annotated tree and asserts every owned RR var settles exactly once.

use lasso::Rodeo;
use rustc_hash::FxHashMap;

use super::{Managed, OwnershipTable, RcOp, RecordShape, RecordTable, Rr, analyze_function};
use crate::full::mir::{
    self, ClosureExpr, DecisionTree, Expr, ExprKind, Function, Param, SwitchCases,
};
use crate::semi::hir::{ExprId, VarId};
use crate::semi::ty::{Capability, DefId, IntTy, Ty, TyCtxt};
use crate::surface::Visibility;
use crate::with_tcx;

use reussir_syntax::kind::{InternKey, TokenKey};

/// A name token whose text is irrelevant to the analysis (it keys on `VarId` /
/// `Symbol`, never source names).
fn dummy_tok() -> TokenKey {
    TokenKey::try_from_u32(0).expect("token key")
}

/// Builds ground `mir::Function`s and their backing [`RecordTable`] in an arena.
struct MirBuilder<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    ids: mir::ExprIdGen,
    next_var: u32,
    next_def: u32,
    symbols: Rodeo,
    table: RecordTable<'tcx>,
}

impl<'a, 'tcx> MirBuilder<'a, 'tcx> {
    fn new(tcx: &'a TyCtxt<'tcx>) -> Self {
        MirBuilder {
            tcx,
            ids: mir::ExprIdGen::default(),
            next_var: 0,
            next_def: 0,
            symbols: Rodeo::default(),
            table: RecordTable::new(),
        }
    }

    // ----- ids, vars, symbols -----

    fn fresh_var(&mut self) -> VarId {
        let v = VarId(self.next_var);
        self.next_var += 1;
        v
    }

    fn sym(&mut self, name: &str) -> mir::Symbol {
        mir::Symbol(self.symbols.get_or_intern(name))
    }

    fn fresh_def(&mut self) -> DefId {
        let d = DefId(self.next_def);
        self.next_def += 1;
        d
    }

    // ----- types (real, interned) -----

    fn i64(&self) -> Ty<'tcx> {
        self.tcx.mk_int(IntTy::Signed(64))
    }

    /// A heap, reference-counted `Shared` record (the default `struct`/`enum`).
    fn shared(&mut self) -> Ty<'tcx> {
        let def = self.fresh_def();
        let ty = self.tcx.mk_record(def, &[], Capability::Irrelevant);
        self.table.insert(
            ty,
            RecordShape {
                managed: Managed::Shared,
                fields: vec![],
            },
        );
        ty
    }

    /// An inline `Value` record with the given ground field types.
    fn value(&mut self, fields: Vec<Ty<'tcx>>) -> Ty<'tcx> {
        let def = self.fresh_def();
        let ty = self.tcx.mk_record(def, &[], Capability::Irrelevant);
        self.table.insert(
            ty,
            RecordShape {
                managed: Managed::Value,
                fields,
            },
        );
        ty
    }

    /// A `[regional]` record. It is rc-managed like a shared record; the table
    /// (keyed by the canonical `Irrelevant` type) records that, while the
    /// returned type keeps its `Regional` value coloring.
    fn regional(&mut self) -> Ty<'tcx> {
        let def = self.fresh_def();
        let canonical = self.tcx.mk_record(def, &[], Capability::Irrelevant);
        self.table.insert(
            canonical,
            RecordShape {
                managed: Managed::Regional,
                fields: vec![],
            },
        );
        self.tcx.mk_record(def, &[], Capability::Regional)
    }

    // ----- expressions (each stamped with a fresh anchor) -----

    fn mk(&mut self, kind: ExprKind<'tcx>, ty: Ty<'tcx>) -> Expr<'tcx> {
        Expr {
            id: self.ids.fresh(),
            kind,
            ty,
            span: None,
        }
    }

    fn var(&mut self, v: VarId, ty: Ty<'tcx>) -> Expr<'tcx> {
        self.mk(ExprKind::Var(v), ty)
    }

    fn const_int(&mut self, n: i128) -> Expr<'tcx> {
        let ty = self.i64();
        self.mk(ExprKind::ConstInt(n), ty)
    }

    fn const_bool(&mut self, b: bool) -> Expr<'tcx> {
        let ty = self.tcx.mk_bool();
        self.mk(ExprKind::ConstBool(b), ty)
    }

    fn if_(&mut self, c: Expr<'tcx>, t: Expr<'tcx>, e: Expr<'tcx>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let c = self.tcx.alloc(c);
        let t = self.tcx.alloc(t);
        let e = self.tcx.alloc(e);
        self.mk(ExprKind::If(c, t, e), ty)
    }

    /// `let v = value;` (a Seq statement; its own result type is unit).
    fn let_(&mut self, v: VarId, value: Expr<'tcx>) -> Expr<'tcx> {
        let value = self.tcx.alloc(value);
        let unit = self.tcx.mk_unit();
        self.mk(
            ExprKind::Let {
                var: v,
                name: dummy_tok(),
                value,
            },
            unit,
        )
    }

    fn seq(&mut self, items: Vec<Expr<'tcx>>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let items = self.tcx.alloc_slice(&items);
        self.mk(ExprKind::Seq(items), ty)
    }

    fn call(&mut self, callee: &str, args: Vec<Expr<'tcx>>, ret: Ty<'tcx>) -> Expr<'tcx> {
        let callee = self.sym(callee);
        let args = self.tcx.alloc_slice(&args);
        self.mk(
            ExprKind::Call {
                callee,
                args,
                regional: false,
            },
            ret,
        )
    }

    fn ctor(&mut self, record: &str, args: Vec<Expr<'tcx>>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let record = self.sym(record);
        let args = self.tcx.alloc_slice(&args);
        self.mk(ExprKind::Ctor { record, args }, ty)
    }

    /// An enum variant constructor (`record#variant(args)`).
    fn variant(
        &mut self,
        record: &str,
        variant: usize,
        args: Vec<Expr<'tcx>>,
        ty: Ty<'tcx>,
    ) -> Expr<'tcx> {
        let record = self.sym(record);
        let args = self.tcx.alloc_slice(&args);
        self.mk(
            ExprKind::Variant {
                record,
                variant,
                args,
            },
            ty,
        )
    }

    // ----- decision trees -----

    fn leaf(&self, body: Expr<'tcx>, bindings: Vec<(VarId, Vec<u32>)>) -> DecisionTree<'tcx> {
        let body = self.tcx.alloc(body);
        let bs: Vec<(VarId, &'tcx [u32])> = bindings
            .iter()
            .map(|(y, p)| (*y, self.tcx.alloc_slice(p)))
            .collect();
        let bindings = self.tcx.alloc_slice(&bs);
        DecisionTree::Leaf { body, bindings }
    }

    /// A variant switch on the value at `path` (root = `[]`), one arm per variant.
    fn switch_ctor(&self, path: Vec<u32>, arms: Vec<DecisionTree<'tcx>>) -> DecisionTree<'tcx> {
        let scrutinee = self.tcx.alloc_slice(&path);
        let arms = self.tcx.alloc_slice(&arms);
        DecisionTree::Switch {
            scrutinee,
            cases: SwitchCases::Ctor(arms),
        }
    }

    fn match_(&mut self, scrut: Expr<'tcx>, tree: DecisionTree<'tcx>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let scrut = self.tcx.alloc(scrut);
        self.mk(ExprKind::Match(scrut, tree), ty)
    }

    // ----- containers / regions / closures -----

    /// A nested field projection `base.<path>`, yielding a value of type `ty`.
    fn proj(&mut self, base: Expr<'tcx>, path: Vec<u32>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let base = self.tcx.alloc(base);
        let path = self.tcx.alloc_slice(&path);
        self.mk(ExprKind::Proj(base, path), ty)
    }

    /// `dst->field := src` (result type unit).
    fn assign(&mut self, dst: Expr<'tcx>, field: u32, src: Expr<'tcx>) -> Expr<'tcx> {
        let dst = self.tcx.alloc(dst);
        let src = self.tcx.alloc(src);
        let unit = self.tcx.mk_unit();
        self.mk(ExprKind::Assign(dst, field, src), unit)
    }

    /// A `region-run { body }` scope; its result is moved out as `ty`.
    fn region(&mut self, body: Expr<'tcx>, ty: Ty<'tcx>) -> Expr<'tcx> {
        let body = self.tcx.alloc(body);
        self.mk(ExprKind::RegionRun(body), ty)
    }

    /// A closure value capturing `captures` and taking `params`, with the given
    /// closure `ty` (build it with [`closure_ty`](Self::closure_ty)).
    fn closure(
        &mut self,
        captures: Vec<(VarId, Ty<'tcx>)>,
        params: Vec<(VarId, Ty<'tcx>)>,
        body: Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Expr<'tcx> {
        let captures = self.tcx.alloc_slice(&captures);
        let params = self.tcx.alloc_slice(&params);
        let body = self.tcx.alloc(body);
        self.mk(
            ExprKind::Closure(ClosureExpr {
                captures,
                params,
                body,
            }),
            ty,
        )
    }

    fn closure_ty(&self, params: &[Ty<'tcx>], ret: Ty<'tcx>) -> Ty<'tcx> {
        self.tcx.mk_closure(params, ret)
    }

    /// `target(args)` — an indirect call through a closure value.
    fn closure_call(
        &mut self,
        target: Expr<'tcx>,
        args: Vec<Expr<'tcx>>,
        ty: Ty<'tcx>,
    ) -> Expr<'tcx> {
        let target = self.tcx.alloc(target);
        let args = self.tcx.alloc_slice(&args);
        self.mk(ExprKind::ClosureCall { target, args }, ty)
    }

    /// A guarded decision node: bind `bindings`, test `guard`; take `success` if it
    /// holds, else `failure`.
    fn guard(
        &self,
        bindings: Vec<(VarId, Vec<u32>)>,
        guard: Expr<'tcx>,
        success: DecisionTree<'tcx>,
        failure: DecisionTree<'tcx>,
    ) -> DecisionTree<'tcx> {
        let bs: Vec<(VarId, &'tcx [u32])> = bindings
            .iter()
            .map(|(y, p)| (*y, self.tcx.alloc_slice(p)))
            .collect();
        let bindings = self.tcx.alloc_slice(&bs);
        let guard = self.tcx.alloc(guard);
        let success = self.tcx.alloc(success);
        let failure = self.tcx.alloc(failure);
        DecisionTree::Guard {
            bindings,
            guard,
            success,
            failure,
        }
    }

    fn param(&mut self, v: VarId, ty: Ty<'tcx>) -> Param<'tcx> {
        Param {
            name: dummy_tok(),
            var: v,
            ty,
        }
    }

    fn function(
        &mut self,
        name: &str,
        params: Vec<Param<'tcx>>,
        ret: Ty<'tcx>,
        body: Expr<'tcx>,
    ) -> Function<'tcx> {
        let symbol = self.sym(name);
        let body = self.tcx.alloc(body);
        Function {
            symbol,
            visibility: Visibility::Private,
            is_regional: false,
            params,
            return_ty: ret,
            body: Some(body),
        }
    }
}

// ---------------------------------------------------------------------------
// Annotated rendering
// ---------------------------------------------------------------------------

fn ops_str(ops: &[RcOp]) -> String {
    ops.iter()
        .map(|op| match op {
            RcOp::Dup(v) => format!("dup v{}", v.0),
            RcOp::Drop(v) => format!("drop v{}", v.0),
            RcOp::DropValue(id) => format!("drop-value #{}", id.0),
            RcOp::DupValue(id) => format!("dup-value #{}", id.0),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A short name for an [`ExprKind`], for the renderer's fallback line (the kinds
/// without a dedicated arm: `GlobalStr` / `ConstFloat` / `Poison`).
fn kind_name(kind: &ExprKind<'_>) -> &'static str {
    match kind {
        ExprKind::GlobalStr(_) => "GlobalStr",
        ExprKind::ConstInt(_) => "ConstInt",
        ExprKind::ConstFloat(_) => "ConstFloat",
        ExprKind::ConstBool(_) => "ConstBool",
        ExprKind::Var(_) => "Var",
        ExprKind::Negate(_) => "Negate",
        ExprKind::Not(_) => "Not",
        ExprKind::Arith(..) => "Arith",
        ExprKind::Cmp(..) => "Cmp",
        ExprKind::Cast(..) => "Cast",
        ExprKind::If(..) => "If",
        ExprKind::RegionRun(_) => "RegionRun",
        ExprKind::Proj(..) => "Proj",
        ExprKind::Assign(..) => "Assign",
        ExprKind::Let { .. } => "Let",
        ExprKind::Seq(_) => "Seq",
        ExprKind::Call { .. } => "Call",
        ExprKind::Ctor { .. } => "Ctor",
        ExprKind::Variant { .. } => "Variant",
        ExprKind::NullableCall(_) => "NullableCall",
        ExprKind::Closure(_) => "Closure",
        ExprKind::ClosureCall { .. } => "ClosureCall",
        ExprKind::Match(..) => "Match",
        ExprKind::Poison => "Poison",
    }
}

struct Renderer<'a> {
    out: String,
    ot: &'a OwnershipTable,
    syms: &'a Rodeo,
}

impl Renderer<'_> {
    fn line(&mut self, indent: usize, s: &str) {
        for _ in 0..indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    /// Emit a node's header line with its own rc ops woven inline: `before` ops as
    /// a `»` prefix, `after` ops as a `«` suffix. Keeping the ops on the node's own
    /// line (rather than on separate lines that visually float between siblings)
    /// makes it unambiguous *which* node each op belongs to — e.g. a `dup` on the
    /// first argument of a call reads as the argument's op, not the call's.
    fn head(&mut self, indent: usize, id: ExprId, text: &str) {
        let before = self.ot.before(id);
        let after = self.ot.after(id);
        let prefix = if before.is_empty() {
            String::new()
        } else {
            format!("» {}  ", ops_str(before))
        };
        let suffix = if after.is_empty() {
            String::new()
        } else {
            format!("  « {}", ops_str(after))
        };
        self.line(indent, &format!("{prefix}{text}{suffix}"));
    }

    fn expr(&mut self, e: &Expr<'_>, indent: usize) {
        match e.kind {
            ExprKind::Var(x) => self.head(indent, e.id, &format!("var v{}", x.0)),
            ExprKind::ConstInt(n) => self.head(indent, e.id, &format!("const {n}")),
            ExprKind::ConstBool(b) => self.head(indent, e.id, &format!("const {b}")),
            ExprKind::Let { var, value, .. } => {
                self.head(indent, e.id, &format!("let v{} =", var.0));
                self.expr(value, indent + 1);
            }
            ExprKind::Seq(es) => {
                self.head(indent, e.id, "seq");
                for s in es {
                    self.expr(s, indent + 1);
                }
            }
            ExprKind::Call { callee, args, .. } => {
                self.head(
                    indent,
                    e.id,
                    &format!("call @{}", self.syms.resolve(&callee.0)),
                );
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::Ctor { record, args } => {
                self.head(
                    indent,
                    e.id,
                    &format!("ctor @{}", self.syms.resolve(&record.0)),
                );
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::Variant {
                record,
                variant,
                args,
            } => {
                self.head(
                    indent,
                    e.id,
                    &format!("variant @{}#{variant}", self.syms.resolve(&record.0)),
                );
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::NullableCall(opt) => match opt {
                Some(x) => {
                    self.head(indent, e.id, "nullable");
                    self.expr(x, indent + 1);
                }
                None => self.head(indent, e.id, "null"),
            },
            ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
                self.head(indent, e.id, "binop");
                self.expr(l, indent + 1);
                self.expr(r, indent + 1);
            }
            ExprKind::Negate(x) | ExprKind::Not(x) | ExprKind::Cast(x, _) => {
                self.head(indent, e.id, "unop");
                self.expr(x, indent + 1);
            }
            ExprKind::If(c, t, e2) => {
                self.head(indent, e.id, "if");
                self.expr(c, indent + 1);
                self.line(indent, "then");
                self.expr(t, indent + 1);
                self.line(indent, "else");
                self.expr(e2, indent + 1);
            }
            ExprKind::Match(scrut, tree) => {
                self.head(indent, e.id, "match");
                self.expr(scrut, indent + 1);
                self.tree(&tree, indent + 1);
            }
            ExprKind::Proj(base, path) => {
                self.head(indent, e.id, &format!("proj {path:?}"));
                self.expr(base, indent + 1);
            }
            ExprKind::Assign(dst, field, src) => {
                self.head(indent, e.id, &format!("assign .{field} :="));
                self.expr(dst, indent + 1);
                self.expr(src, indent + 1);
            }
            ExprKind::RegionRun(x) => {
                self.head(indent, e.id, "region-run");
                self.expr(x, indent + 1);
            }
            ExprKind::Closure(c) => {
                let caps = c
                    .captures
                    .iter()
                    .map(|(v, _)| format!("v{}", v.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.head(indent, e.id, &format!("closure [{caps}]"));
                self.expr(c.body, indent + 1);
            }
            ExprKind::ClosureCall { target, args } => {
                self.head(indent, e.id, "closure-call");
                self.expr(target, indent + 1);
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            _ => self.head(indent, e.id, &format!("<{}>", kind_name(&e.kind))),
        }
    }

    fn tree(&mut self, t: &DecisionTree<'_>, indent: usize) {
        match *t {
            DecisionTree::Uncovered => self.line(indent, "uncovered"),
            DecisionTree::Unreachable => self.line(indent, "unreachable"),
            DecisionTree::Leaf { body, bindings } => {
                let bs = bindings
                    .iter()
                    .map(|(y, p)| format!("v{}@{p:?}", y.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.line(indent, &format!("leaf [{bs}]"));
                self.expr(body, indent + 1);
            }
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => {
                let bs = bindings
                    .iter()
                    .map(|(y, p)| format!("v{}@{p:?}", y.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                // `if` / `success` / `failure` are the guard's three parts, so
                // indent them under it (and their contents one deeper still),
                // rather than leaving them flush with the `guard` line.
                self.line(indent, &format!("guard [{bs}]"));
                self.line(indent + 1, "if");
                self.expr(guard, indent + 2);
                self.line(indent + 1, "success");
                self.tree(success, indent + 2);
                self.line(indent + 1, "failure");
                self.tree(failure, indent + 2);
            }
            DecisionTree::Switch { cases, .. } => {
                self.line(indent, "switch");
                for sub in super::subtrees(&cases) {
                    self.tree(sub, indent + 1);
                }
            }
        }
    }
}

/// Render a function body with its inc/dec ops woven in (`»` before, `«` after).
fn render(func: &Function<'_>, ot: &OwnershipTable, syms: &Rodeo) -> String {
    let mut r = Renderer {
        out: String::new(),
        ot,
        syms,
    };
    let params: Vec<String> = func
        .params
        .iter()
        .map(|p| format!("v{}", p.var.0))
        .collect();
    let header = format!(
        "fn @{}({}):",
        syms.resolve(&func.symbol.0),
        params.join(", ")
    );
    r.line(0, &header);
    if let Some(body) = func.body {
        r.expr(body, 1);
    }
    r.out
}

// ---------------------------------------------------------------------------
// Balanced-rc safety net
// ---------------------------------------------------------------------------

fn apply(op: RcOp, rc: &mut FxHashMap<VarId, i32>) {
    match op {
        // A `dup` on an owned var increments it; a `dup` on an as-yet-untracked
        // var *establishes* it (this is how a borrow-extracted pattern binding
        // takes ownership — `bind y = project(scrut, path); rc.inc(y)`).
        RcOp::Dup(x) => {
            *rc.entry(x).or_insert(0) += 1;
        }
        RcOp::Drop(x) => {
            let c = rc
                .get_mut(&x)
                .unwrap_or_else(|| panic!("drop of unowned v{}", x.0));
            assert!(*c >= 1, "drop of settled v{} (rc {c})", x.0);
            *c -= 1;
        }
        // Touch an anonymous result (discarded statement / borrow-extracted
        // field), which this named-var checker does not track — nothing to charge.
        RcOp::DropValue(_) | RcOp::DupValue(_) => {}
    }
}

/// Abstractly interpret `e` in evaluation order, applying the surrounding ops and
/// charging each consuming RR use one owned reference.
fn interp<'tcx>(
    e: &Expr<'tcx>,
    ot: &OwnershipTable,
    rr: &Rr<'_, 'tcx>,
    rc: &mut FxHashMap<VarId, i32>,
) {
    for &op in ot.before(e.id) {
        apply(op, rc);
    }
    match e.kind {
        ExprKind::Var(x) => {
            if rr.is_rr(e.ty) {
                let c = rc
                    .get_mut(&x)
                    .unwrap_or_else(|| panic!("use of unowned v{}", x.0));
                assert!(*c >= 1, "consuming settled v{} (rc {c})", x.0);
                *c -= 1;
            }
        }
        ExprKind::Let { var, value, .. } => {
            interp(value, ot, rr, rc);
            if rr.is_rr(value.ty) {
                rc.insert(var, 1);
            }
        }
        ExprKind::Seq(es) => {
            for s in es {
                interp(s, ot, rr, rc);
            }
        }
        ExprKind::Call { args, .. }
        | ExprKind::Ctor { args, .. }
        | ExprKind::Variant { args, .. } => {
            for a in args {
                interp(a, ot, rr, rc);
            }
        }
        ExprKind::NullableCall(opt) => {
            if let Some(x) = opt {
                interp(x, ot, rr, rc);
            }
        }
        ExprKind::Negate(x) | ExprKind::Not(x) | ExprKind::Cast(x, _) => interp(x, ot, rr, rc),
        ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
            interp(l, ot, rr, rc);
            interp(r, ot, rr, rc);
        }
        // Each arm is interpreted from the same entry state; reconciliation must
        // leave both with identical ownership (the join is well-defined).
        ExprKind::If(c, t, e) => {
            interp(c, ot, rr, rc);
            let mut rc_t = rc.clone();
            let mut rc_e = rc.clone();
            interp(t, ot, rr, &mut rc_t);
            interp(e, ot, rr, &mut rc_e);
            assert_eq!(
                rc_t, rc_e,
                "if-arms leave different ownership: then={rc_t:?} else={rc_e:?}"
            );
            *rc = rc_t;
        }
        // The scrutinee is borrowed; the decision tree is walked path-by-path,
        // each terminal leaf reconciling to identical ownership (pattern bindings
        // are dropped from the join).
        ExprKind::Match(scrut, tree) => {
            if !matches!(scrut.kind, ExprKind::Var(_)) {
                interp(scrut, ot, rr, rc);
            }
            let mut states = Vec::new();
            interp_tree(scrut, &tree, ot, rr, rc.clone(), &[], &mut states);
            for s in &states[1..] {
                assert_eq!(
                    states[0], *s,
                    "match paths differ: {:?} vs {s:?}",
                    states[0]
                );
            }
            if let Some(joined) = states.into_iter().next() {
                *rc = joined;
            }
        }
        // Projection borrows its base (the root is not consumed here; its drop, if
        // any, is an after-op); a temporary base is consumed.
        ExprKind::Proj(base, _) => interp_borrow(base, ot, rr, rc),
        // Assignment: borrow the dst (after-op settles its root), consume the src.
        ExprKind::Assign(dst, _, src) => {
            interp_borrow(dst, ot, rr, rc);
            interp(src, ot, rr, rc);
        }
        ExprKind::RegionRun(x) => interp(x, ot, rr, rc),
        // Creating a closure consumes its RR captures; the body is a separate
        // scope, validated on its own, not here.
        ExprKind::Closure(c) => {
            for &(v, ty) in c.captures {
                if rr.is_rr(ty) {
                    let cc = rc
                        .get_mut(&v)
                        .unwrap_or_else(|| panic!("capture of unowned v{}", v.0));
                    assert!(*cc >= 1, "capturing settled v{} (rc {cc})", v.0);
                    *cc -= 1;
                }
            }
        }
        ExprKind::ClosureCall { target, args } => {
            interp(target, ot, rr, rc);
            for a in args {
                interp(a, ot, rr, rc);
            }
        }
        ExprKind::GlobalStr(_)
        | ExprKind::ConstInt(_)
        | ExprKind::ConstFloat(_)
        | ExprKind::ConstBool(_)
        | ExprKind::Poison => {}
    }
    for &op in ot.after(e.id) {
        apply(op, rc);
    }
}

/// Navigate a borrowed base / assign-dst: a `Var` root is left untouched (its
/// drop, if its last use, is an explicit after-op); nested projections recurse; a
/// temporary base is consumed (interpreted).
fn interp_borrow<'tcx>(
    e: &Expr<'tcx>,
    ot: &OwnershipTable,
    rr: &Rr<'_, 'tcx>,
    rc: &mut FxHashMap<VarId, i32>,
) {
    match e.kind {
        ExprKind::Var(_) => {}
        ExprKind::Proj(inner, _) => interp_borrow(inner, ot, rr, rc),
        _ => interp(e, ot, rr, rc),
    }
}

/// Walk the decision tree, collecting one terminal ownership state per leaf path.
/// `bound_above` are the pattern vars bound by enclosing guards (removed from the
/// state at each leaf, like the leaf's own bindings, since they are match-local).
fn interp_tree<'tcx>(
    scrut: &Expr<'tcx>,
    tree: &DecisionTree<'tcx>,
    ot: &OwnershipTable,
    rr: &Rr<'_, 'tcx>,
    rc: FxHashMap<VarId, i32>,
    bound_above: &[VarId],
    states: &mut Vec<FxHashMap<VarId, i32>>,
) {
    match *tree {
        DecisionTree::Uncovered | DecisionTree::Unreachable => {}
        DecisionTree::Leaf { body, bindings } => {
            let mut rcl = rc;
            // A moved whole-binding has no op: transfer the scrutinee's ownership
            // to the bound var so it can be consumed in the body.
            for &(y, path) in bindings {
                if path.is_empty() && uses_var(body, y) {
                    let dup_established = ot.before(body.id).contains(&RcOp::Dup(y));
                    if !dup_established {
                        let taken = match scrut.kind {
                            ExprKind::Var(s) => rcl.remove(&s).unwrap_or(1),
                            _ => 1,
                        };
                        rcl.insert(y, taken);
                    }
                }
            }
            interp(body, ot, rr, &mut rcl);
            for &(y, _) in bindings {
                rcl.remove(&y);
            }
            for &y in bound_above {
                rcl.remove(&y);
            }
            states.push(rcl);
        }
        DecisionTree::Switch { cases, .. } => {
            for sub in super::subtrees(&cases) {
                interp_tree(scrut, sub, ot, rr, rc.clone(), bound_above, states);
            }
        }
        DecisionTree::Guard {
            bindings,
            guard,
            success,
            failure,
        } => {
            let mut rcg = rc;
            interp(guard, ot, rr, &mut rcg);
            let mut nested = bound_above.to_vec();
            for &(y, _) in bindings {
                nested.push(y);
            }
            interp_tree(scrut, success, ot, rr, rcg.clone(), &nested, states);
            interp_tree(scrut, failure, ot, rr, rcg, &nested, states);
        }
    }
}

/// Whether `e`'s tree contains a use of variable `y`.
fn uses_var(e: &Expr<'_>, y: VarId) -> bool {
    let mut found = false;
    fn go(e: &Expr<'_>, y: VarId, found: &mut bool) {
        if *found {
            return;
        }
        if let ExprKind::Var(x) = e.kind
            && x == y
        {
            *found = true;
            return;
        }
        for child in children(e) {
            go(child, y, found);
        }
    }
    go(e, y, &mut found);
    found
}

/// The immediate sub-expressions of `e` (decision-tree arms reached separately).
fn children<'tcx>(e: &Expr<'tcx>) -> Vec<&'tcx Expr<'tcx>> {
    use ExprKind::*;
    match e.kind {
        GlobalStr(_) | ConstInt(_) | ConstFloat(_) | ConstBool(_) | Var(_) | Poison => vec![],
        Negate(x) | Not(x) | Cast(x, _) | RegionRun(x) | Proj(x, _) => vec![x],
        Arith(l, _, r) | Cmp(l, _, r) | Assign(l, _, r) => vec![l, r],
        If(c, t, e) => vec![c, t, e],
        Let { value, .. } => vec![value],
        Seq(es) => es.iter().collect(),
        Call { args, .. } | Ctor { args, .. } | Variant { args, .. } => args.iter().collect(),
        NullableCall(opt) => opt.into_iter().collect(),
        ClosureCall { target, args } => std::iter::once(target).chain(args.iter()).collect(),
        Closure(c) => vec![c.body],
        Match(scrut, _) => vec![scrut],
    }
}

/// Every owned RR var must reach a settled state (rc 0) exactly once: nothing is
/// left live at return except the moved-out result (which the checker does not
/// track, as it is the caller's to own).
fn check_balanced<'tcx>(func: &Function<'tcx>, ot: &OwnershipTable, rr: &Rr<'_, 'tcx>) {
    let mut rc: FxHashMap<VarId, i32> = FxHashMap::default();
    for p in &func.params {
        if rr.is_rr(p.ty) {
            rc.insert(p.var, 1);
        }
    }
    if let Some(body) = func.body {
        interp(body, ot, rr, &mut rc);
    }
    for (v, c) in &rc {
        assert_eq!(*c, 0, "v{} left unsettled (rc {c})", v.0);
    }
}

/// Analyze, print the annotated rendering (for examination), and run the
/// balanced-rc safety net. Returns the table for pinpoint probes.
fn run<'tcx>(
    tcx: &TyCtxt<'tcx>,
    func: &Function<'tcx>,
    builder: &MirBuilder<'_, 'tcx>,
) -> OwnershipTable {
    let ot = analyze_function(tcx, func, &builder.table);
    println!("{}", render(func, &ot, &builder.symbols));
    let rr = Rr::new(tcx, &builder.table);
    check_balanced(func, &ot, &rr);
    ot
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

#[test]
fn is_rr_classification() {
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let val_with_rc = b.value(vec![rc]);
        let i64t = b.i64();
        let val_plain = b.value(vec![i64t]);
        let reg = b.regional();
        let rr = Rr::new(tcx, &b.table);

        assert!(rr.is_rr(rc), "shared record is rc");
        assert!(rr.is_rr(val_with_rc), "value record holding an rc is RR");
        assert!(!rr.is_rr(val_plain), "scalar-only value record is not RR");
        assert!(
            rr.is_rr(reg),
            "regional record is RR (rc-managed, per table)"
        );
        assert!(rr.is_rr(tcx.mk_nullable(rc)), "nullable rc is RR");
        assert!(
            !rr.is_rr(tcx.mk_nullable(i64t)),
            "nullable scalar is not RR"
        );
        assert!(!rr.is_rr(i64t), "scalar is not RR");
    });
}

#[test]
fn case1_dead_let_is_dropped_after_binding() {
    // fn f(x: Rc) -> i64 { let y = wrap(x); 0 }
    //   x: last use, moved into `wrap`; y: bound but unused ⇒ drop after the let.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let x = b.fresh_var();
        let y = b.fresh_var();

        let use_x = b.var(x, rc);
        let value = b.call("wrap", vec![use_x], rc);
        let let_stmt = b.let_(y, value);
        let let_id = let_stmt.id;
        let zero = b.const_int(0);
        let i64t = b.i64();
        let body = b.seq(vec![let_stmt, zero], i64t);

        let p = b.param(x, rc);
        let func = b.function("f", vec![p], i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.after(let_id),
            &[RcOp::Drop(y)],
            "y dropped right after binding"
        );
        assert!(
            ot.before(use_x.id).is_empty(),
            "x is a last use ⇒ moved, no dup"
        );
    });
}

#[test]
fn case2_two_uses_dup_all_but_last() {
    // fn f(x: Rc) -> Pair { mk(x, x) }
    //   first x is live across the second ⇒ dup; second x is the last use ⇒ move.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let pair = b.value(vec![rc, rc]);
        let x = b.fresh_var();

        let a0 = b.var(x, rc);
        let a1 = b.var(x, rc);
        let (a0_id, a1_id) = (a0.id, a1.id);
        let body = b.call("mk", vec![a0, a1], pair);

        let p = b.param(x, rc);
        let func = b.function("f", vec![p], pair, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(ot.before(a0_id), &[RcOp::Dup(x)], "first occurrence dup'd");
        assert!(ot.before(a1_id).is_empty(), "last occurrence moved");
    });
}

#[test]
fn case5_returned_var_is_moved() {
    // fn f(x: Rc) -> Rc { x }  — moved to the caller, no dup and no drop.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let x = b.fresh_var();
        let body = b.var(x, rc);
        let body_id = body.id;

        let p = b.param(x, rc);
        let func = b.function("f", vec![p], rc, body);
        let ot = run(tcx, &func, &b);

        assert!(ot.get(body_id).is_none(), "returned var: no ops at all");
    });
}

#[test]
fn case6_transitive_value_record_is_dropped() {
    // A value record transitively holding an rc is RR.
    // fn f(v: Box) -> i64 { 0 }  — v unused ⇒ dropped on entry.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let boxed = b.value(vec![rc]); // Box { inner: Rc }
        let v = b.fresh_var();

        let body = b.const_int(0);
        let body_id = body.id;
        let i64t = b.i64();
        let p = b.param(v, boxed);
        let func = b.function("f", vec![p], i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(body_id),
            &[RcOp::Drop(v)],
            "unused transitively-rc value record dropped on entry"
        );
    });
}

#[test]
fn case7_unused_param_dropped_on_entry() {
    // fn f(x: Rc, n: i64) -> i64 { n }  — x never used ⇒ dropped on entry; n scalar.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let i64t = b.i64();
        let x = b.fresh_var();
        let n = b.fresh_var();

        let body = b.var(n, i64t);
        let body_id = body.id;
        let params = vec![b.param(x, rc), b.param(n, i64t)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(body_id),
            &[RcOp::Drop(x)],
            "unused rc param dropped on entry"
        );
    });
}

#[test]
fn ctor_consumes_its_rr_field() {
    // fn f(x: Rc) -> Box { Box { inner: x } }  — x's last use is moved into the
    // constructor; the resulting (transitively-rc) value record is moved out.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let boxed = b.value(vec![rc]);
        let x = b.fresh_var();

        let arg = b.var(x, rc);
        let body = b.ctor("Box", vec![arg], boxed);

        let params = vec![b.param(x, rc)];
        let func = b.function("f", params, boxed, body);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.before(arg.id).is_empty(),
            "x moved into the ctor ⇒ no dup"
        );
        assert!(ot.after(arg.id).is_empty(), "no stray drop");
    });
}

#[test]
fn chained_moves_need_no_rc_ops() {
    // fn f(x: Rc) -> Rc { let y = x; y }  — x moved into y, y moved out. No ops.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let x = b.fresh_var();
        let y = b.fresh_var();

        let xval = b.var(x, rc);
        let let_stmt = b.let_(y, xval);
        let let_id = let_stmt.id;
        let yval = b.var(y, rc);
        let body = b.seq(vec![let_stmt, yval], rc);

        let params = vec![b.param(x, rc)];
        let func = b.function("f", params, rc, body);
        let ot = run(tcx, &func, &b);

        assert!(ot.get(let_id).is_none(), "y is moved out ⇒ no drop");
        assert!(ot.before(xval.id).is_empty(), "x moved into y ⇒ no dup");
    });
}

// ----- control flow (If reconciliation, discarded results) -----

#[test]
fn case3_if_var_used_in_one_arm_dropped_in_other() {
    // fn f(x: Rc) -> i64 { if c { consume(x) } else { 0 } }
    //   x dies (returns i64): moved in the `then` arm, dropped in the `else` arm.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let i64t = b.i64();
        let x = b.fresh_var();

        let xuse = b.var(x, rc);
        let then = b.call("consume", vec![xuse], i64t);
        let then_id = then.id;
        let els = b.const_int(0);
        let els_id = els.id;
        let cond = b.const_bool(true);
        let body = b.if_(cond, then, els, i64t);

        let params = vec![b.param(x, rc)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert!(ot.before(then_id).is_empty(), "then moves x ⇒ no drop");
        assert_eq!(
            ot.before(els_id),
            &[RcOp::Drop(x)],
            "else never uses x ⇒ drop on entry to that arm"
        );
    });
}

#[test]
fn if_returns_one_of_two_params() {
    // fn f(x: Rc, y: Rc) -> Rc { if c { x } else { y } }
    //   each arm moves one and drops the other, so both exit owning nothing.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let x = b.fresh_var();
        let y = b.fresh_var();

        let then = b.var(x, rc);
        let then_id = then.id;
        let els = b.var(y, rc);
        let els_id = els.id;
        let cond = b.const_bool(true);
        let body = b.if_(cond, then, els, rc);

        let params = vec![b.param(x, rc), b.param(y, rc)];
        let func = b.function("f", params, rc, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(then_id),
            &[RcOp::Drop(y)],
            "then drops the unused y"
        );
        assert_eq!(
            ot.before(els_id),
            &[RcOp::Drop(x)],
            "else drops the unused x"
        );
    });
}

#[test]
fn if_var_live_after_dups_in_each_arm() {
    // fn f(x: Rc) -> Pair { let a = if c { g(x) } else { h(x) }; mk(x, a) }
    //   x is used in each arm *and* again in `mk`, so each arm dup's it; the
    //   trailing `mk` is x's last use and moves it.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let pair = b.value(vec![rc, rc]);
        let x = b.fresh_var();
        let a = b.fresh_var();

        let xg = b.var(x, rc);
        let xg_id = xg.id;
        let gx = b.call("g", vec![xg], rc);
        let xh = b.var(x, rc);
        let xh_id = xh.id;
        let hx = b.call("h", vec![xh], rc);
        let cond = b.const_bool(true);
        let iff = b.if_(cond, gx, hx, rc);
        let let_a = b.let_(a, iff);

        let xm = b.var(x, rc);
        let xm_id = xm.id;
        let am = b.var(a, rc);
        let mk = b.call("mk", vec![xm, am], pair);
        let body = b.seq(vec![let_a, mk], pair);

        let params = vec![b.param(x, rc)];
        let func = b.function("f", params, pair, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(xg_id),
            &[RcOp::Dup(x)],
            "then dup's the still-live x"
        );
        assert_eq!(
            ot.before(xh_id),
            &[RcOp::Dup(x)],
            "else dup's the still-live x"
        );
        assert!(
            ot.before(xm_id).is_empty(),
            "trailing use of x is the last ⇒ move"
        );
    });
}

#[test]
fn discarded_rr_statement_gets_drop_value() {
    // fn f(x: Rc) -> i64 { effect(x); 0 }
    //   `effect(x)` returns an Rc that is discarded ⇒ drop-value its result.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let i64t = b.i64();
        let x = b.fresh_var();

        let xuse = b.var(x, rc);
        let effect = b.call("effect", vec![xuse], rc);
        let effect_id = effect.id;
        let zero = b.const_int(0);
        let body = b.seq(vec![effect, zero], i64t);

        let params = vec![b.param(x, rc)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.after(effect_id),
            &[RcOp::DropValue(effect_id)],
            "discarded Rc result is dropped by value"
        );
    });
}

// ----- pattern matching (borrow-dup, Perceus-style) -----

#[test]
fn match_extracts_field_and_drops_scrutinee() {
    // fn f(e: E) -> i64 { match e { #0 => 0, #1(x) => use(x) } }
    //   e dies (returns i64) ⇒ consumed: each arm dups the field it extracts, then
    //   drops the scrutinee shell (which recursively frees what it didn't extract).
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let e_ty = b.shared(); // the enum, rc-managed
        let rc = b.shared(); // the payload field type
        let i64t = b.i64();
        let e = b.fresh_var();
        let x = b.fresh_var();

        let body0 = b.const_int(0);
        let body0_id = body0.id;
        let leaf0 = b.leaf(body0, vec![]);

        let vx = b.var(x, rc);
        let use_x = b.call("use", vec![vx], i64t);
        let body1_id = use_x.id;
        let leaf1 = b.leaf(use_x, vec![(x, vec![0])]);

        let tree = b.switch_ctor(vec![], vec![leaf0, leaf1]);
        let scrut = b.var(e, e_ty);
        let body = b.match_(scrut, tree, i64t);

        let params = vec![b.param(e, e_ty)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(body0_id),
            &[RcOp::Drop(e)],
            "empty arm drops the consumed scrutinee"
        );
        assert_eq!(
            ot.before(body1_id),
            &[RcOp::Dup(x), RcOp::Drop(e)],
            "extracting arm dups the field, then drops the scrutinee"
        );
    });
}

#[test]
fn match_whole_scrutinee_moves() {
    // fn f(e: E) -> i64 { match e { y => use(y) } }
    //   binding the whole scrutinee is an identity ⇒ a plain move: no inc, no dec.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let e_ty = b.shared();
        let i64t = b.i64();
        let e = b.fresh_var();
        let y = b.fresh_var();

        let vy = b.var(y, e_ty);
        let use_y = b.call("use", vec![vy], i64t);
        let use_y_id = use_y.id;
        let tree = b.leaf(use_y, vec![(y, vec![])]);
        let scrut = b.var(e, e_ty);
        let body = b.match_(scrut, tree, i64t);

        let params = vec![b.param(e, e_ty)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.get(use_y_id).is_none(),
            "whole-scrutinee bind is a move: no rc.inc on the root, no rc.dec"
        );
    });
}

#[test]
fn match_unused_binding_freed_by_scrutinee_drop() {
    // fn f(e: E) -> i64 { match e { #0 => 0, #1(x) => 0 } }  — x bound but unused.
    //   No dup for x; the scrutinee drop recursively frees the unextracted field.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let e_ty = b.shared();
        let i64t = b.i64();
        let e = b.fresh_var();
        let x = b.fresh_var();

        let body0 = b.const_int(0);
        let leaf0 = b.leaf(body0, vec![]);
        let body1 = b.const_int(0);
        let body1_id = body1.id;
        let leaf1 = b.leaf(body1, vec![(x, vec![0])]);

        let tree = b.switch_ctor(vec![], vec![leaf0, leaf1]);
        let scrut = b.var(e, e_ty);
        let body = b.match_(scrut, tree, i64t);

        let params = vec![b.param(e, e_ty)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(body1_id),
            &[RcOp::Drop(e)],
            "unused binding gets no dup; only the scrutinee drop"
        );
    });
}

#[test]
fn match_reconciles_an_outer_var() {
    // fn f(e: E, w: Rc) -> i64 { match e { #0 => use(w), #1(x) => use2(x) } }
    //   w (an outer var) is used only in arm #0; arm #1 must drop it.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let e_ty = b.shared();
        let rc = b.shared();
        let i64t = b.i64();
        let e = b.fresh_var();
        let w = b.fresh_var();
        let x = b.fresh_var();

        let vw = b.var(w, rc);
        let use_w = b.call("use", vec![vw], i64t);
        let body0_id = use_w.id;
        let leaf0 = b.leaf(use_w, vec![]);

        let vx = b.var(x, rc);
        let use_x = b.call("use2", vec![vx], i64t);
        let body1_id = use_x.id;
        let leaf1 = b.leaf(use_x, vec![(x, vec![0])]);

        let tree = b.switch_ctor(vec![], vec![leaf0, leaf1]);
        let scrut = b.var(e, e_ty);
        let body = b.match_(scrut, tree, i64t);

        let params = vec![b.param(e, e_ty), b.param(w, rc)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(body0_id),
            &[RcOp::Drop(e)],
            "arm #0 uses w ⇒ only the scrutinee drop"
        );
        assert_eq!(
            ot.before(body1_id),
            &[RcOp::Dup(x), RcOp::Drop(e), RcOp::Drop(w)],
            "arm #1 dups x, drops the scrutinee, and reconciles by dropping w"
        );
    });
}

// ----- containers (Proj / Assign) -----

#[test]
fn proj_extracts_field_and_drops_base() {
    // fn f(p: Pair) -> Rc { p.0 }
    //   the projected field is borrow-extracted (inc'd, the parent still owns the
    //   original); p is dead after ⇒ its root drops early at the projection.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let pair = b.value(vec![rc, rc]);
        let p = b.fresh_var();

        let vp = b.var(p, pair);
        let body = b.proj(vp, vec![0], rc);
        let body_id = body.id;

        let param = b.param(p, pair);
        let func = b.function("f", vec![param], rc, body);
        let ot = run(tcx, &func, &b);

        assert!(ot.before(body_id).is_empty(), "no op before the projection");
        assert_eq!(
            ot.after(body_id),
            &[RcOp::DupValue(body_id), RcOp::Drop(p)],
            "extracted field inc'd, dead base dropped early"
        );
    });
}

#[test]
fn proj_keeps_base_when_still_live() {
    // fn f(p: Pair) -> Box { mk(p.0, p) }
    //   p.0 is borrow-extracted, but p itself is used again in the same call ⇒ its
    //   root is *not* dropped at the projection (it is moved by the later whole use).
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let pair = b.value(vec![rc, rc]);
        let boxed = b.shared();
        let p = b.fresh_var();

        let vp_field = b.var(p, pair);
        let proj = b.proj(vp_field, vec![0], rc);
        let proj_id = proj.id;
        let vp_whole = b.var(p, pair);
        let vp_whole_id = vp_whole.id;
        let body = b.call("mk", vec![proj, vp_whole], boxed);

        let param = b.param(p, pair);
        let func = b.function("f", vec![param], boxed, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.after(proj_id),
            &[RcOp::DupValue(proj_id)],
            "extracted field inc'd, but the still-live base is not dropped"
        );
        assert!(
            ot.before(vp_whole_id).is_empty(),
            "the trailing whole use of p is the last ⇒ moved"
        );
    });
}

#[test]
fn nested_proj_navigates_to_root() {
    // fn f(p: Outer) -> Rc { p.0.0 }
    //   only the outermost projection inc's the extracted leaf; the inner one is
    //   pure navigation (no op); the dead root drops at the outer projection.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let inner_ty = b.value(vec![rc]);
        let outer_ty = b.value(vec![inner_ty]);
        let p = b.fresh_var();

        let vp = b.var(p, outer_ty);
        let inner_proj = b.proj(vp, vec![0], inner_ty);
        let inner_proj_id = inner_proj.id;
        let outer_proj = b.proj(inner_proj, vec![0], rc);
        let outer_proj_id = outer_proj.id;

        let param = b.param(p, outer_ty);
        let func = b.function("f", vec![param], rc, outer_proj);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.get(inner_proj_id).is_none(),
            "inner projection is pure navigation: no ops"
        );
        assert_eq!(
            ot.after(outer_proj_id),
            &[RcOp::DupValue(outer_proj_id), RcOp::Drop(p)],
            "outer projection inc's the leaf and drops the dead root"
        );
    });
}

#[test]
fn assign_moves_src_and_borrows_dst() {
    // fn f(c: Flex, x: Rc) -> i64 { c.0 := x; 0 }
    //   x is moved into the field; c is borrowed (mutated in place) and, being dead
    //   after, dropped early at the store; the old field link is freed with c.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let flex = b.regional();
        let i64t = b.i64();
        let c = b.fresh_var();
        let x = b.fresh_var();

        let vc = b.var(c, flex);
        let vx = b.var(x, rc);
        let vx_id = vx.id;
        let assign = b.assign(vc, 0, vx);
        let assign_id = assign.id;
        let zero = b.const_int(0);
        let body = b.seq(vec![assign, zero], i64t);

        let params = vec![b.param(c, flex), b.param(x, rc)];
        let func = b.function("f", params, i64t, body);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.before(vx_id).is_empty(),
            "x moved into the field ⇒ no dup"
        );
        assert_eq!(
            ot.after(assign_id),
            &[RcOp::Drop(c)],
            "dead borrowed dst dropped after the store; no old-field dec"
        );
    });
}

// ----- regions -----

#[test]
fn region_run_is_transparent() {
    // fn f(x: Rc) -> Rc { region { x } }  — a region scope is rc-transparent.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let x = b.fresh_var();

        let vx = b.var(x, rc);
        let vx_id = vx.id;
        let body = b.region(vx, rc);
        let body_id = body.id;

        let param = b.param(x, rc);
        let func = b.function("f", vec![param], rc, body);
        let ot = run(tcx, &func, &b);

        assert!(ot.get(body_id).is_none(), "region scope adds no ops");
        assert!(ot.get(vx_id).is_none(), "x moved straight through ⇒ no ops");
    });
}

// ----- closures -----

#[test]
fn closure_moves_its_capture() {
    // fn f(x: Rc) -> Clos { || x }  — x is captured at its last use ⇒ moved in.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let clos_ty = b.closure_ty(&[], rc);
        let x = b.fresh_var();

        let vx = b.var(x, rc);
        let body = b.closure(vec![(x, rc)], vec![], vx, clos_ty);
        let body_id = body.id;

        let param = b.param(x, rc);
        let func = b.function("f", vec![param], clos_ty, body);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.get(body_id).is_none(),
            "capture is the last use of x ⇒ moved into the closure, no dup"
        );
    });
}

#[test]
fn closure_dups_a_still_live_capture() {
    // fn f(x: Rc) -> Box { mk(|| x, x) }
    //   x is captured *and* used again in the call ⇒ the closure dup's it.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let boxed = b.shared();
        let clos_ty = b.closure_ty(&[], rc);
        let x = b.fresh_var();

        let vx_body = b.var(x, rc);
        let clos = b.closure(vec![(x, rc)], vec![], vx_body, clos_ty);
        let clos_id = clos.id;
        let vx_arg = b.var(x, rc);
        let vx_arg_id = vx_arg.id;
        let body = b.call("mk", vec![clos, vx_arg], boxed);

        let param = b.param(x, rc);
        let func = b.function("f", vec![param], boxed, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(clos_id),
            &[RcOp::Dup(x)],
            "x is live after the closure ⇒ dup'd into the capture"
        );
        assert!(
            ot.before(vx_arg_id).is_empty(),
            "the trailing argument use is the last ⇒ moved"
        );
    });
}

#[test]
fn closure_call_consumes_target_and_args() {
    // fn f(g: Clos, x: Rc) -> Rc { g(x) }  — both g and x are at last use ⇒ moved.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let clos_ty = b.closure_ty(&[rc], rc);
        let g = b.fresh_var();
        let x = b.fresh_var();

        let vg = b.var(g, clos_ty);
        let vg_id = vg.id;
        let vx = b.var(x, rc);
        let vx_id = vx.id;
        let body = b.closure_call(vg, vec![vx], rc);

        let params = vec![b.param(g, clos_ty), b.param(x, rc)];
        let func = b.function("f", params, rc, body);
        let ot = run(tcx, &func, &b);

        assert!(
            ot.before(vg_id).is_empty(),
            "target closure moved into the call"
        );
        assert!(ot.before(vx_id).is_empty(), "argument moved into the call");
    });
}

// ----- match guards -----

#[test]
fn guard_keeps_scrutinee_until_terminal_leaves() {
    // fn f(e: E) -> i64 { match e { #_(x) if pred(x) => use(x), _ => 0 } }
    //   The guard borrow-extracts x (dup at the guard, before either branch). The
    //   scrutinee stays live across the guard (the failure path re-tests it), so its
    //   drop lands in *each* terminal leaf. x, surviving into the branches, is moved
    //   by `use` on success but must be dropped on failure.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let e_ty = b.shared();
        let rc = b.shared();
        let i64t = b.i64();
        let bool_t = b.tcx.mk_bool();
        let e = b.fresh_var();
        let x = b.fresh_var();

        let vx_guard = b.var(x, rc);
        let guard_expr = b.call("pred", vec![vx_guard], bool_t);
        let guard_id = guard_expr.id;

        let vx_succ = b.var(x, rc);
        let use_x = b.call("use", vec![vx_succ], i64t);
        let succ_body_id = use_x.id;
        let success = b.leaf(use_x, vec![]);

        let zero = b.const_int(0);
        let fail_body_id = zero.id;
        let failure = b.leaf(zero, vec![]);

        let tree = b.guard(vec![(x, vec![0])], guard_expr, success, failure);
        let scrut = b.var(e, e_ty);
        let body = b.match_(scrut, tree, i64t);

        let param = b.param(e, e_ty);
        let func = b.function("f", vec![param], i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(guard_id),
            &[RcOp::Dup(x)],
            "the guard binding is borrow-extracted before the test"
        );
        assert_eq!(
            ot.before(succ_body_id),
            &[RcOp::Drop(e)],
            "success consumes x and drops the now-dead scrutinee"
        );
        assert_eq!(
            ot.before(fail_body_id),
            &[RcOp::Drop(e), RcOp::Drop(x)],
            "failure drops the scrutinee and the extracted-but-unused x"
        );
    });
}

// ----- closures are RR: multiple uses follow dup-all-but-last -----

#[test]
fn closure_value_used_twice_dups_all_but_last() {
    // fn f(g: Clos) -> Box { mk(g, g) }
    //   A closure is itself an RR value, so passing it twice follows exactly the
    //   same rule as any rc (cf. `case2`): the first occurrence is dup'd (live
    //   across the second), the last is moved.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let i64t = b.i64();
        let boxed = b.shared();
        let clos_ty = b.closure_ty(&[], i64t);
        let g = b.fresh_var();

        let a0 = b.var(g, clos_ty);
        let a1 = b.var(g, clos_ty);
        let (a0_id, a1_id) = (a0.id, a1.id);
        let body = b.call("mk", vec![a0, a1], boxed);

        let param = b.param(g, clos_ty);
        let func = b.function("f", vec![param], boxed, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(a0_id),
            &[RcOp::Dup(g)],
            "first use of the RR closure is dup'd"
        );
        assert!(
            ot.before(a1_id).is_empty(),
            "last use of the closure is moved"
        );
    });
}

#[test]
fn closure_called_twice_dups_for_the_earlier_call() {
    // fn f(x: Rc) -> i64 { let g = || use(x); g(); g() }
    //   Invoking a closure consumes a reference to it (owned convention), so the
    //   closure — being RR — is dup'd for the first (non-last) call and moved into
    //   the second. The capture `x` is consumed once, when the closure is built.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let rc = b.shared();
        let i64t = b.i64();
        let clos_ty = b.closure_ty(&[], i64t);
        let x = b.fresh_var();
        let g = b.fresh_var();

        let vx = b.var(x, rc);
        let use_x = b.call("use", vec![vx], i64t);
        let clos = b.closure(vec![(x, rc)], vec![], use_x, clos_ty);
        let let_g = b.let_(g, clos);

        let vg1 = b.var(g, clos_ty);
        let vg1_id = vg1.id;
        let call1 = b.closure_call(vg1, vec![], i64t);
        let vg2 = b.var(g, clos_ty);
        let vg2_id = vg2.id;
        let call2 = b.closure_call(vg2, vec![], i64t);
        let body = b.seq(vec![let_g, call1, call2], i64t);

        let param = b.param(x, rc);
        let func = b.function("f", vec![param], i64t, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(vg1_id),
            &[RcOp::Dup(g)],
            "the closure is live across the second call ⇒ dup'd for the first"
        );
        assert!(
            ot.before(vg2_id).is_empty(),
            "the second call is the closure's last use ⇒ moved"
        );
    });
}

// ----- a realistic function: tail-recursive list reverse -----

#[test]
fn tail_recursive_list_reverse() {
    // fn reverse(xs: List, acc: List) -> List {
    //     match xs {
    //         Nil        => acc,
    //         Cons(h, t) => reverse(t, Cons(h, acc)),
    //     }
    // }
    //
    // `List` is an rc-managed enum (Nil | Cons(head, tail)); the element `head` is
    // itself an rc. Nil arm: `acc` is moved out as the result and the matched (now
    // dead) Nil cell is dropped. Cons arm: `h` and `t` are borrow-extracted from
    // the scrutinee (each dup'd), the consumed Cons cell is dropped, and h / t /
    // acc are all moved on into the rebuilt `Cons(h, acc)` and the tail call.
    // Recursion is just a `Call` — a dataflow leaf — so tail position is ordinary.
    with_tcx(|tcx| {
        let mut b = MirBuilder::new(tcx);
        let list = b.shared(); // List: rc enum
        let elem = b.shared(); // the element type, also rc
        let xs = b.fresh_var();
        let acc = b.fresh_var();
        let h = b.fresh_var();
        let t = b.fresh_var();

        // Nil => acc
        let nil_body = b.var(acc, list);
        let nil_body_id = nil_body.id;
        let nil_arm = b.leaf(nil_body, vec![]);

        // Cons(h, t) => reverse(t, Cons(h, acc))
        let vh = b.var(h, elem);
        let vacc = b.var(acc, list);
        let rebuilt = b.variant("List", 1, vec![vh, vacc], list);
        let vt = b.var(t, list);
        let rec = b.call("reverse", vec![vt, rebuilt], list);
        let cons_body_id = rec.id;
        let cons_arm = b.leaf(rec, vec![(h, vec![0]), (t, vec![1])]);

        let tree = b.switch_ctor(vec![], vec![nil_arm, cons_arm]);
        let scrut = b.var(xs, list);
        let body = b.match_(scrut, tree, list);

        let params = vec![b.param(xs, list), b.param(acc, list)];
        let func = b.function("reverse", params, list, body);
        let ot = run(tcx, &func, &b);

        assert_eq!(
            ot.before(nil_body_id),
            &[RcOp::Drop(xs)],
            "Nil arm: acc is moved out; the matched Nil cell is dropped"
        );
        assert_eq!(
            ot.before(cons_body_id),
            &[RcOp::Dup(h), RcOp::Dup(t), RcOp::Drop(xs)],
            "Cons arm: head & tail borrow-extracted, then the Cons cell dropped"
        );
    });
}
