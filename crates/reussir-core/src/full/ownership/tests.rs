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
use crate::full::mir::{self, Expr, ExprKind, Function, Param};
use crate::semi::hir::VarId;
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
        })
        .collect::<Vec<_>>()
        .join(", ")
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

    fn expr(&mut self, e: &Expr<'_>, indent: usize) {
        let before = self.ot.before(e.id);
        if !before.is_empty() {
            self.line(indent, &format!("» {}", ops_str(before)));
        }
        match e.kind {
            ExprKind::Var(x) => self.line(indent, &format!("var v{}", x.0)),
            ExprKind::ConstInt(n) => self.line(indent, &format!("const {n}")),
            ExprKind::ConstBool(b) => self.line(indent, &format!("const {b}")),
            ExprKind::Let { var, value, .. } => {
                self.line(indent, &format!("let v{} =", var.0));
                self.expr(value, indent + 1);
            }
            ExprKind::Seq(es) => {
                self.line(indent, "seq");
                for s in es {
                    self.expr(s, indent + 1);
                }
            }
            ExprKind::Call { callee, args, .. } => {
                self.line(indent, &format!("call @{}", self.syms.resolve(&callee.0)));
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::Ctor { record, args } => {
                self.line(indent, &format!("ctor @{}", self.syms.resolve(&record.0)));
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::Variant {
                record,
                variant,
                args,
            } => {
                self.line(
                    indent,
                    &format!("variant @{}#{variant}", self.syms.resolve(&record.0)),
                );
                for a in args {
                    self.expr(a, indent + 1);
                }
            }
            ExprKind::NullableCall(opt) => match opt {
                Some(x) => {
                    self.line(indent, "nullable");
                    self.expr(x, indent + 1);
                }
                None => self.line(indent, "null"),
            },
            ExprKind::Arith(l, _, r) | ExprKind::Cmp(l, _, r) => {
                self.line(indent, "binop");
                self.expr(l, indent + 1);
                self.expr(r, indent + 1);
            }
            ExprKind::Negate(x) | ExprKind::Not(x) | ExprKind::Cast(x, _) => {
                self.line(indent, "unop");
                self.expr(x, indent + 1);
            }
            _ => self.line(indent, &format!("<{}>", super::kind_name(&e.kind))),
        }
        let after = self.ot.after(e.id);
        if !after.is_empty() {
            self.line(indent, &format!("« {}", ops_str(after)));
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
        RcOp::Dup(x) => {
            let c = rc
                .get_mut(&x)
                .unwrap_or_else(|| panic!("dup of unowned v{}", x.0));
            assert!(*c >= 1, "dup of settled v{} (rc {c})", x.0);
            *c += 1;
        }
        RcOp::Drop(x) => {
            let c = rc
                .get_mut(&x)
                .unwrap_or_else(|| panic!("drop of unowned v{}", x.0));
            assert!(*c >= 1, "drop of settled v{} (rc {c})", x.0);
            *c -= 1;
        }
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
        ExprKind::GlobalStr(_)
        | ExprKind::ConstInt(_)
        | ExprKind::ConstFloat(_)
        | ExprKind::ConstBool(_)
        | ExprKind::Poison => {}
        _ => panic!(
            "checker: unexpected deferred form {}",
            super::kind_name(&e.kind)
        ),
    }
    for &op in ot.after(e.id) {
        apply(op, rc);
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
