//! Function and expression lowering: the per-function recursive tree-walk that
//! emits MLIR ops for each Full MIR expression.
//!
//! # SSA environment
//!
//! The variable environment ([`Env`]) maps each [`VarId`] to the melior [`Value`]
//! that holds it, and carries the enclosing region handle. melior values borrow
//! the block they were built in, so an `scf.if` branch or a `region-run` body —
//! which builds into its own block — is a nested scope. Because variable ids are
//! globally unique, such a scope only ever appends bindings, so rather than
//! copying the environment it is entered through [`Env::subscope`], which lends
//! it reborrowed at the inner block's shorter lifetime and rewinds the additions
//! (and any region swap) on return.

use std::cell::RefCell;

use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};

use reussir_backend::builders;
use reussir_backend::dialect;
use reussir_backend::dialect::ty::ReussirCapability;
use reussir_backend::melior::Context;
use reussir_backend::melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use reussir_backend::melior::dialect::{func, scf};
use reussir_backend::melior::ir::attribute::{
    FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use reussir_backend::melior::ir::r#type::{FunctionType, IntegerType};
use reussir_backend::melior::ir::{
    Block, BlockLike, Identifier, Location, Operation, Region, RegionLike, Type, Value,
};

use reussir_core::full::mir::{self, Expr, ExprKind};
use reussir_core::full::ownership::{OwnershipTable, RcOp, RecordTable, analyze_function};
use reussir_core::semi::hir::{ArithOp, CmpOp, ExprId, VarId};
use reussir_core::semi::ty::{IntTy, Ty, TyCtxt, TyKind};
use reussir_core::surface::{Span, Visibility};
use reussir_syntax::kind::{Resolver, TokenKey};
use smallvec::SmallVec;

use crate::source::SourceMap;

use super::ty::{TypeCtx, field_link_element, is_unit, num_class, regional_capability};
use super::{LoweringError, Result, err};

/// The lexical environment threaded through one block scope: the SSA value bound
/// to each in-scope variable, together with the enclosing region handle regional
/// allocations flow into.
///
/// The region is genuinely environmental — it is inherited unchanged across an
/// `if` branch and swapped for the body's own arena handle inside a `region-run`
/// — so it rides here rather than as a separate threaded argument. It is `None`
/// outside any region: a `regional` function seeds it with its implicit region
/// parameter, and a plain function leaves it unset.
///
/// A nested block (an `if` branch, a `region-run` body) is a nested scope that
/// only ever *appends* bindings — variable ids are globally unique, so a scope
/// never rebinds one already in `vars`. That lets a subscope be recovered by
/// truncating `vars` back to its length on entry (see [`Env::subscope`]) instead
/// of copying the whole map, which is why `vars` is an insertion-ordered map.
struct Env<'c, 'b> {
    vars: IndexMap<VarId, Value<'c, 'b>, rustc_hash::FxBuildHasher>,
    region: Option<Value<'c, 'b>>,
}

impl<'c, 'b> Env<'c, 'b> {
    /// An empty environment carrying `region` as its enclosing region handle.
    fn new(region: Option<Value<'c, 'b>>) -> Self {
        Self {
            vars: IndexMap::default(),
            region,
        }
    }

    /// Run `f` in a nested block scope: it borrows the environment reborrowed at
    /// the nested block's (shorter) lifetime `'s`, and everything it adds is
    /// discarded afterwards — `vars` is truncated back to its length on entry and
    /// `region` is restored (a `region-run` body swaps in its own handle for the
    /// duration).
    ///
    /// The nested block's SSA values are shorter-lived than this environment
    /// (they borrow a block the caller created for the scope), and `&mut` is
    /// invariant, so the borrow checker will not let `f` insert them through a
    /// `&mut Env<'c, 'b>`. This performs the one reborrow that shortens the value
    /// lifetime; it is sound because every binding `f` makes at `'s` is removed
    /// before returning, so no `'s`-lived value is ever observed at `'b`.
    fn subscope<'s, R>(&mut self, f: impl FnOnce(&mut Env<'c, 's>) -> R) -> R
    where
        'b: 's,
    {
        let mark = self.vars.len();
        let saved_region = self.region;
        // SAFETY: `'b: 's`, so viewing the stored `Value<'c, 'b>`s as
        // `Value<'c, 's>` is a lifetime *shortening* — a sound covariant coercion
        // the compiler declines to perform through the invariant `&mut`. `short`
        // aliases `*self`, but `self` is untouched until `f` returns, and the
        // truncate/restore below removes every `'s`-lived value `f` added before
        // `self` is next read at `'b`.
        let short: &mut Env<'c, 's> = unsafe { &mut *(self as *mut Env<'c, 'b>).cast() };
        let r = f(short);
        self.vars.truncate(mark);
        self.region = saved_region;
        r
    }
}

/// The state threaded through a field-projection walk: either a loaded value or a
/// borrowed reference into a record, each tagged with its ground MIR type so the
/// next step can consult the record layout and pick the right op.
enum Cursor<'c, 'b, 'tcx> {
    /// A loaded SSA value of record or scalar type `ty`. For a `[value]` record
    /// this is the inline aggregate; for a `[shared]`/`[regional]` record it is
    /// the `rc` pointer.
    Value { val: Value<'c, 'b>, ty: Ty<'tcx> },
    /// A borrowed `!reussir.ref<…, cap>` into a record whose MIR type is `ty`.
    /// `cap` is the reference's capability, carried so a further projection out
    /// of it picks the matching projected-reference capability.
    Ref {
        val: Value<'c, 'b>,
        ty: Ty<'tcx>,
        cap: ReussirCapability,
    },
}

/// How a single field projects out of a record reference: the field's MIR type,
/// the projected reference's element type and capability, and whether the
/// projection is a pointer link to load (an `rc` member) rather than an inline
/// sub-field to keep navigating by reference. See
/// [`projected_member`](Lowerer::projected_member).
struct ProjectedMember<'c, 'tcx> {
    field_ty: Ty<'tcx>,
    elem_ty: Type<'c>,
    proj_cap: ReussirCapability,
    is_link: bool,
}

/// Per-program lowering state: the context to build in, the type arena, the
/// program whose symbol table resolves callee/trampoline names, the [`TypeCtx`]
/// that lowers types and resolves record layouts, and the record table the
/// ownership analysis consults. Reused across functions.
///
/// Two side-tables hold state for the function currently being lowered (reset on
/// entry to [`function`](Self::function)): the [`OwnershipTable`] that says where
/// to emit reference-count ops, and the type of each named local (so a `dup`/
/// `drop` keyed by a variable knows whether to emit an `rc` op or to recurse
/// through an inline record).
pub(super) struct Lowerer<'c, 'p, 'tcx> {
    pub(super) context: &'c Context,
    pub(super) tcx: &'p TyCtxt<'tcx>,
    pub(super) program: &'p mir::Program<'tcx>,
    pub(super) tys: TypeCtx<'c, 'p, 'tcx>,
    records: RecordTable<'tcx>,
    /// Resolves MIR byte spans to file/line/column; `None` lowers to `unknown`
    /// locations.
    pub(super) source: Option<&'p SourceMap<'p>>,
    /// Resolves interned source names for debug info; `Some` (with `source`)
    /// enables DWARF variable/type emission. See [`debug`](super::debug).
    pub(super) names: Option<&'p dyn Resolver<TokenKey>>,
    ownership: RefCell<OwnershipTable>,
    var_tys: RefCell<FxHashMap<VarId, Ty<'tcx>>>,
    /// The location attached to ops as they are built. Set to the current
    /// expression's span by [`expr`](Self::expr) (saved and restored around each
    /// node), so every op a node emits shares that node's source position.
    cur_loc: RefCell<Location<'c>>,
    /// Record symbols whose debug composite is currently being built — the
    /// recursion guard for [`debug`](super::debug). A type that reaches itself
    /// (directly or through a boxed field) is emitted as a memberless
    /// forward-declared composite instead of expanding forever.
    pub(super) dbg_building: RefCell<FxHashSet<mir::Symbol>>,
}

impl<'c, 'p, 'tcx> Lowerer<'c, 'p, 'tcx> {
    pub(super) fn new(
        context: &'c Context,
        tcx: &'p TyCtxt<'tcx>,
        program: &'p mir::Program<'tcx>,
        source: Option<&'p SourceMap<'p>>,
        names: Option<&'p dyn Resolver<TokenKey>>,
    ) -> Self {
        Lowerer {
            context,
            tcx,
            program,
            tys: TypeCtx::new(context, program),
            records: RecordTable::from_records(&program.records),
            source,
            names,
            ownership: RefCell::new(OwnershipTable::default()),
            var_tys: RefCell::new(FxHashMap::default()),
            cur_loc: RefCell::new(Location::unknown(context)),
            dbg_building: RefCell::new(FxHashSet::default()),
        }
    }

    /// Whether debug info should be emitted (both a source map and a name
    /// resolver were supplied).
    pub(super) fn debug_enabled(&self) -> bool {
        self.source.is_some() && self.names.is_some()
    }

    /// The location currently attached to emitted ops (see [`cur_loc`](Self::cur_loc)).
    pub(super) fn loc(&self) -> Location<'c> {
        *self.cur_loc.borrow()
    }

    /// Resolve a MIR span to a `FileLineColLoc`, or `unknown` without a source
    /// map or span.
    pub(super) fn location(&self, span: Option<Span>) -> Location<'c> {
        match (self.source, span) {
            (Some(map), Some(span)) => {
                let (line, col) = map.span_start(span);
                Location::new(self.context, &map.filename(), line, col)
            }
            _ => Location::unknown(self.context),
        }
    }

    /// Record the ground type of a local, so reference-count ops keyed by it can
    /// pick the right instruction.
    fn bind_var_ty(&self, var: VarId, ty: Ty<'tcx>) {
        self.var_tys.borrow_mut().insert(var, ty);
    }

    fn var_ty(&self, var: VarId) -> Option<Ty<'tcx>> {
        self.var_tys.borrow().get(&var).copied()
    }

    /// Lower one MIR function to a `func.func` operation.
    pub(super) fn function(&self, func: &mir::Function<'tcx>) -> Result<Operation<'c>> {
        // The function and its entry-level ops (block args, return) take the
        // body's source position; nested nodes refine it as they are walked.
        let loc = self.location(func.body.and_then(|b| b.span));
        self.cur_loc.replace(loc);
        // A `regional` function takes the region it allocates into as an implicit
        // leading `!reussir.region` parameter, ahead of its source parameters.
        let user_param_tys = func
            .params
            .iter()
            .map(|p| self.tys.mlir_ty(p.ty))
            .collect::<Result<Vec<_>>>()?;
        let mut param_tys = Vec::with_capacity(user_param_tys.len() + 1);
        if func.is_regional {
            param_tys.push(self.tys.region_type());
        }
        param_tys.extend_from_slice(&user_param_tys);
        let ret = func.return_ty;
        let result_tys = if is_unit(ret) {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(ret)?]
        };
        let fn_ty = FunctionType::new(self.context, &param_tys, &result_tys);

        // Reset the per-function side-tables, then run the ownership analysis for
        // this body so the tree-walk can emit each node's reference-count ops.
        *self.ownership.borrow_mut() = analyze_function(self.tcx, func, &self.records);
        self.var_tys.borrow_mut().clear();

        let region = Region::new();
        if let Some(body) = func.body {
            let block_args: SmallVec<[(Type<'c>, Location<'c>); 8]> =
                param_tys.iter().map(|t| (*t, loc)).collect();
            let block = Block::new(&block_args);
            // The implicit region parameter (regional functions only) leads the
            // block arguments and is not a source variable; it seeds the
            // environment's region rather than being bound as a local. Source
            // parameters follow it.
            let (region_handle, param_base) = if func.is_regional {
                let arg = block
                    .argument(0)
                    .map_err(|e| LoweringError(format!("missing region argument: {e}").into()))?;
                (Some(arg.into()), 1)
            } else {
                (None, 0)
            };
            let mut env = Env::new(region_handle);
            for (i, p) in func.params.iter().enumerate() {
                let arg = block
                    .argument(param_base + i)
                    .map_err(|e| LoweringError(format!("missing block argument: {e}").into()))?;
                env.vars.insert(p.var, arg.into());
                self.bind_var_ty(p.var, p.ty);
            }
            let value = self.expr(&block, &mut env, body)?;
            if is_unit(ret) {
                block.append_operation(func::r#return(&[], loc));
            } else {
                let v =
                    value.ok_or_else(|| LoweringError("non-unit body produced no value".into()))?;
                block.append_operation(func::r#return(&[v], loc));
            }
            region.append_block(block);
        }

        // A `private` function carries an explicit `sym_visibility` attribute;
        // the printer renders it as the `private` keyword.
        let mut attributes = Vec::new();
        if func.visibility == Visibility::Private {
            attributes.push((
                Identifier::new(self.context, "sym_visibility"),
                StringAttribute::new(self.context, "private").into(),
            ));
        }
        // Debug info (when enabled): the parameters' names/types as an attribute
        // the conversion pass reads, and the function's own location fused with a
        // subprogram attribute.
        if let Some(args) = self.dbg_func_args_attr(func) {
            attributes.push(args);
        }
        let func_loc = self.subprogram_location(func);

        Ok(func::func(
            self.context,
            StringAttribute::new(self.context, self.program.symbol(func.symbol)),
            TypeAttribute::new(fn_ty.into()),
            region,
            &attributes,
            func_loc,
        ))
    }

    /// Lower an expression into `block`, returning the SSA value holding its
    /// result (`None` for a unit-typed expression).
    ///
    /// Around the node's own ops, this emits the reference-count ops the
    /// ownership analysis placed at its anchor: the `before` ops first, then the
    /// node, then the `after` ops (which may reference the node's freshly-produced
    /// value, e.g. to increment a borrowed field or drop a discarded result).
    ///
    /// The enclosing region a regional construction or `regional` call allocates
    /// into travels in `env` (see [`Env`]), so it need not be threaded here.
    fn expr<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        // Attach this node's source position to every op it emits, restoring the
        // enclosing node's position afterwards.
        let prev = self.cur_loc.replace(self.location(e.span));
        let result = (|| {
            self.emit_rc_before(block, env, e.id)?;
            let value = self.expr_inner(block, env, e)?;
            self.emit_rc_after(block, env, e.id, e.ty, value)?;
            Ok(value)
        })();
        self.cur_loc.replace(prev);
        result
    }

    /// Lower the node itself, without its surrounding reference-count ops.
    fn expr_inner<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        use ExprKind::*;
        let loc = self.loc();
        match &e.kind {
            ConstInt(n) => {
                let attr = IntegerAttribute::new(self.tys.mlir_ty(e.ty)?, *n as i64).into();
                Ok(Some(
                    self.append(block, arith::constant(self.context, attr, loc)),
                ))
            }
            ConstBool(b) => {
                let i1 = IntegerType::new(self.context, 1).into();
                let attr = IntegerAttribute::new(i1, i64::from(*b)).into();
                Ok(Some(
                    self.append(block, arith::constant(self.context, attr, loc)),
                ))
            }
            ConstFloat(f) => {
                let attr = FloatAttribute::new(self.context, self.tys.mlir_ty(e.ty)?, *f).into();
                Ok(Some(
                    self.append(block, arith::constant(self.context, attr, loc)),
                ))
            }
            Var(v) => {
                // A unit-typed variable has no SSA value (e.g. `let x = (); x`),
                // so it is never stored in `env`; lower it to `None` rather than
                // reporting it as unbound.
                if is_unit(e.ty) {
                    Ok(None)
                } else {
                    env.vars
                        .get(v)
                        .copied()
                        .map(Some)
                        .ok_or_else(|| LoweringError(format!("unbound variable {v:?}").into()))
                }
            }
            Negate(x) => self.negate(block, env, x).map(Some),
            Not(x) => self.not(block, env, x).map(Some),
            Arith(l, op, r) => self.arith(block, env, l, *op, r, e.ty).map(Some),
            Cmp(l, op, r) => self.cmp(block, env, l, *op, r).map(Some),
            Cast(x, t) => self.cast(block, env, x, *t),
            If(c, t, f) => self.lower_if(block, env, c, t, f, e.ty),
            Seq(items) => {
                let mut last = None;
                for item in items.iter() {
                    last = self.expr(block, env, item)?;
                }
                Ok(last)
            }
            Let { var, name, value } => {
                self.bind_var_ty(*var, value.ty);
                if let Some(v) = self.expr(block, env, value)? {
                    self.tag_local(v, *name, value.ty);
                    env.vars.insert(*var, v);
                }
                Ok(None)
            }
            Call {
                callee,
                args,
                regional,
            } => {
                // A `regional` callee takes the enclosing region as an implicit
                // leading argument; thread it ahead of the source arguments.
                let mut operands = Vec::with_capacity(args.len() + usize::from(*regional));
                if *regional {
                    operands.push(env.region.ok_or_else(|| {
                        LoweringError("regional call outside a region scope".into())
                    })?);
                }
                for a in args.iter() {
                    operands.push(
                        self.expr(block, env, a)?
                            .ok_or_else(|| LoweringError("call argument is unit".into()))?,
                    );
                }
                let result_tys = if is_unit(e.ty) {
                    Vec::new()
                } else {
                    vec![self.tys.mlir_ty(e.ty)?]
                };
                let symbol =
                    FlatSymbolRefAttribute::new(self.context, self.program.symbol(*callee));
                let op = block.append_operation(func::call(
                    self.context,
                    symbol,
                    &operands,
                    &result_tys,
                    loc,
                ));
                if is_unit(e.ty) {
                    Ok(None)
                } else {
                    Ok(Some(op.result(0).unwrap().into()))
                }
            }
            RegionRun(body) => self.region_run(block, env, body, e.ty),
            Proj(base, path) => self.proj(block, env, base, path).map(Some),
            Assign(dst, field, src) => {
                self.assign(block, env, dst, *field, src)?;
                Ok(None)
            }
            Match(..) => err("match lowering not yet implemented"),
            Ctor { args, .. } => self.compound(block, env, e, args).map(Some),
            Variant { variant, args, .. } => self.variant(block, env, e, *variant, args).map(Some),
            NullableCall(inner) => self.nullable_call(block, env, e, *inner).map(Some),
            Closure(_) | ClosureCall { .. } => err("closure lowering not yet implemented"),
            GlobalStr(_) => err("string literal lowering not yet implemented"),
            Poison => err("poison expression reached lowering"),
        }
    }

    /// Construct a record from its (declaration-ordered) field args.
    ///
    /// The fields are packed into the inline record payload with
    /// `reussir.record.compound`, which [`box_record`](Self::box_record) then
    /// boxes according to the record's management — a `[value]` record stays
    /// inline, a `[shared]` record is heap-boxed, and a `[regional]` record is
    /// region-allocated.
    fn compound<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
        args: &[Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let payload_ty = self.tys.record_inner_of(e.ty)?;
        let mut operands = Vec::with_capacity(args.len());
        for a in args.iter() {
            operands.push(
                self.expr(block, env, a)?
                    .ok_or_else(|| LoweringError("record field is unit".into()))?,
            );
        }
        let payload = self.append(block, builders::record_compound(&operands, payload_ty, loc));
        self.box_record(block, env, e, payload, payload_ty)
    }

    /// Construct an enum value: select a case by tag and supply its payload.
    ///
    /// The case's fields are packed into the `{enum}::{case}` payload compound
    /// with `reussir.record.compound` (an empty compound for a fieldless case),
    /// which `reussir.record.variant [tag]` then tags into the enum's variant
    /// record. [`box_record`](Self::box_record) then boxes the tagged value per
    /// the enum's management (the rc-create fusion pass later folds the
    /// `record.compound` + `record.variant` + `rc.create` chain into a single
    /// `reussir.rc.create_variant` allocation).
    fn variant<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
        variant: usize,
        args: &[Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let variant_ty = self.tys.record_inner_of(e.ty)?;
        let payload_ty = self.tys.variant_payload_of(e.ty, variant)?;
        let mut operands = Vec::with_capacity(args.len());
        for a in args.iter() {
            operands.push(
                self.expr(block, env, a)?
                    .ok_or_else(|| LoweringError("variant field is unit".into()))?,
            );
        }
        let payload = self.append(block, builders::record_compound(&operands, payload_ty, loc));
        let tagged = self.append(
            block,
            builders::record_variant(self.context, variant, payload, variant_ty, loc),
        );
        self.box_record(block, env, e, tagged, variant_ty)
    }

    /// Box a freshly-built record payload according to the record's management.
    ///
    /// A `[value]` record is held inline (the payload is the value). A `[shared]`
    /// record is heap-boxed into a fresh `!reussir.rc<…, shared>` with
    /// `reussir.rc.create`. A `[regional]` record is region-allocated: it is
    /// boxed into a `flex` `rc` pointer with `reussir.rc.create … region(%r)`,
    /// which both makes its fields mutable and ties its lifetime to the enclosing
    /// region (the region patterns pass freezes it to `rigid` where it escapes).
    fn box_record<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b>,
        e: &Expr<'tcx>,
        payload: Value<'c, 'b>,
        inner_ty: Type<'c>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        if self.tys.is_shared_record(e.ty) {
            let rc_ty = self.tys.rc_type(inner_ty);
            Ok(self.append(
                block,
                builders::rc_create(self.context, payload, rc_ty, loc),
            ))
        } else if self.tys.is_regional_record(e.ty) {
            // A regional record is constructed region-local, so its box is always
            // `flex`; the frozen `rigid` form only arises later, at region exit.
            let cap = regional_capability(e.ty)?;
            if cap != ReussirCapability::Flex {
                return err("a regional record is constructed flex, but its type is not flex");
            }
            let region = env.region.ok_or_else(|| {
                LoweringError("regional record constructed outside a region scope".into())
            })?;
            let rc_ty = self.tys.rc_type_with_cap(inner_ty, cap);
            Ok(self.append(
                block,
                builders::rc_create_in_region(self.context, payload, region, rc_ty, loc),
            ))
        } else {
            Ok(payload)
        }
    }

    /// Lower a `region-run` scope to `reussir.region.run`.
    ///
    /// The body lowers into a fresh region whose entry block takes the arena
    /// handle (`!reussir.region`) as its single argument; that handle is threaded
    /// as the region for the body, so regional constructions inside allocate into
    /// it. The body terminates with `reussir.region.yield`, carrying its result
    /// out — a `flex` (region-local) result is frozen to `rigid` as it escapes
    /// (the region patterns pass inserts the freeze).
    fn region_run<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        body: &Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let unit = is_unit(ty);
        let result_ty = if unit {
            None
        } else {
            Some(self.tys.mlir_ty(ty)?)
        };
        let region_ty = self.tys.region_type();
        let body_block = Block::new(&[(region_ty, loc)]);
        let region_arg: Value<'c, '_> = body_block
            .argument(0)
            .map_err(|e| LoweringError(format!("missing region argument: {e}").into()))?
            .into();
        // The body sees the enclosing locals but allocates into this region's own
        // arena handle; its bindings and the swapped-in region are rewound when
        // the scope returns, before the block is moved into its region below.
        env.subscope(|scope| {
            scope.region = Some(region_arg);
            let value = self.expr(&body_block, scope, body)?;
            if unit {
                body_block.append_operation(builders::region_yield(None, loc));
            } else {
                let v =
                    value.ok_or_else(|| LoweringError("region body produced no value".into()))?;
                body_block.append_operation(builders::region_yield(Some(v), loc));
            }
            Ok::<_, LoweringError>(())
        })?;
        let body_region = Region::new();
        body_region.append_block(body_block);
        let op = block.append_operation(builders::region_run(result_ty, body_region, loc));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Lower `dst->field := src` — store into a mutable `[field]` link.
    ///
    /// Mutation is only legal on a region-local `flex` regional record, so the
    /// destination is borrowed at `flex` capability, the `[field]` slot is
    /// projected out to a writable `field`-capability reference, and the source
    /// (a `nullable<rc<…>>` value) is stored into it.
    fn assign<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        dst: &Expr<'tcx>,
        field: u32,
        src: &Expr<'tcx>,
    ) -> Result<()> {
        let loc = self.loc();
        let dst_val = self
            .expr(block, env, dst)?
            .ok_or_else(|| LoweringError("assignment target is unit".into()))?;
        if regional_capability(dst.ty)? != ReussirCapability::Flex {
            return err("assignment target is not a mutable (flex) regional record");
        }
        let inner = self.tys.record_inner_of(dst.ty)?;
        let ref_ty = self.tys.ref_type_with_cap(inner, ReussirCapability::Flex);
        let borrowed = self.append(
            block,
            dialect::rc_borrow(self.context, ref_ty, dst_val, loc).into(),
        );
        let proj = self.projected_member(dst.ty, ReussirCapability::Flex, field)?;
        let field_ref_ty = self.tys.ref_type_with_cap(proj.elem_ty, proj.proj_cap);
        let index = IntegerAttribute::new(Type::index(self.context), i64::from(field));
        let field_ref = self.append(
            block,
            dialect::ref_project(self.context, field_ref_ty, borrowed, index, loc).into(),
        );
        let src_val = self
            .expr(block, env, src)?
            .ok_or_else(|| LoweringError("assignment source is unit".into()))?;
        block.append_operation(dialect::ref_store(self.context, field_ref, src_val, loc).into());
        Ok(())
    }

    /// Lower a nullable constructor (`Nullable::NonNull{e}` or `Nullable::Null`)
    /// to `reussir.nullable.create`, wrapping the pointer value (or building the
    /// null pointer when there is no inner expression).
    fn nullable_call<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
        inner: Option<&'tcx Expr<'tcx>>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let result_ty = self.tys.mlir_ty(e.ty)?;
        let value = match inner {
            Some(x) => Some(
                self.expr(block, env, x)?
                    .ok_or_else(|| LoweringError("nullable payload is unit".into()))?,
            ),
            None => None,
        };
        Ok(self.append(block, builders::nullable_create(value, result_ty, loc)))
    }

    /// Project a chain of fields out of a record value.
    ///
    /// The walk is type-directed (see [`Cursor`]): an inline `[value]` record is
    /// read field-by-field with `reussir.record.extract`; a `[shared]` record is
    /// borrowed (`reussir.rc.borrow`) and then navigated with
    /// `reussir.ref.project`, loading (`reussir.ref.load`) wherever a value is
    /// needed — either to cross into a nested `rc` link or to materialize the
    /// final result. The result is always a loaded value; if it is itself an rc
    /// resource, the ownership analysis records the matching increment in the
    /// projection's `after` ops.
    fn proj<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        base: &Expr<'tcx>,
        path: &[u32],
    ) -> Result<Value<'c, 'b>> {
        let base_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("projection base is unit".into()))?;
        let mut cursor = Cursor::Value {
            val: base_val,
            ty: base.ty,
        };
        for &idx in path.iter() {
            cursor = self.project_one(block, cursor, idx)?;
        }
        self.load_cursor(block, cursor)
    }

    /// Advance the projection [`Cursor`] by one field index.
    fn project_one<'b>(
        &self,
        block: &'b Block<'c>,
        cursor: Cursor<'c, 'b, 'tcx>,
        idx: u32,
    ) -> Result<Cursor<'c, 'b, 'tcx>> {
        let loc = self.loc();
        match cursor {
            Cursor::Value { val, ty } => {
                if let Some(cap) = self.record_ref_cap(ty)? {
                    // A boxed record value (shared or regional) is an rc pointer:
                    // borrow it at its capability to obtain a reference, then
                    // project through that reference.
                    let inner = self.tys.record_inner_of(ty)?;
                    let ref_ty = self.tys.ref_type_with_cap(inner, cap);
                    let borrowed = self.append(
                        block,
                        dialect::rc_borrow(self.context, ref_ty, val, loc).into(),
                    );
                    self.project_ref(block, borrowed, ty, cap, idx)
                } else {
                    // An inline `[value]` record value: read the field out by value
                    // (an inline record has no `[field]` link members).
                    let (field_ty, _) = self.field_at(ty, idx)?;
                    let field_mlir = self.tys.mlir_ty(field_ty)?;
                    let extracted = self.append(
                        block,
                        builders::record_extract(self.context, val, idx as usize, field_mlir, loc),
                    );
                    Ok(Cursor::Value {
                        val: extracted,
                        ty: field_ty,
                    })
                }
            }
            Cursor::Ref { val, ty, cap } => self.project_ref(block, val, ty, cap, idx),
        }
    }

    /// Project field `idx` out of a reference (`ref<record_ty, ref_cap>`) into a
    /// record. A field stored as a pointer link — a `[shared]`/`[regional]`
    /// record member — is loaded to its `rc` value, ready to be re-borrowed or
    /// returned; an inline field stays a reference until the walk finishes.
    fn project_ref<'b>(
        &self,
        block: &'b Block<'c>,
        reference: Value<'c, 'b>,
        record_ty: Ty<'tcx>,
        ref_cap: ReussirCapability,
        idx: u32,
    ) -> Result<Cursor<'c, 'b, 'tcx>> {
        let loc = self.loc();
        let proj = self.projected_member(record_ty, ref_cap, idx)?;
        let proj_ref_ty = self.tys.ref_type_with_cap(proj.elem_ty, proj.proj_cap);
        let index = IntegerAttribute::new(Type::index(self.context), i64::from(idx));
        let projected = self.append(
            block,
            dialect::ref_project(self.context, proj_ref_ty, reference, index, loc).into(),
        );
        if proj.is_link {
            let loaded = self.append(
                block,
                dialect::ref_load(self.context, proj.elem_ty, projected, loc).into(),
            );
            Ok(Cursor::Value {
                val: loaded,
                ty: proj.field_ty,
            })
        } else {
            Ok(Cursor::Ref {
                val: projected,
                ty: proj.field_ty,
                cap: proj.proj_cap,
            })
        }
    }

    /// The capability to borrow a record-typed value at, or `None` for an inline
    /// `[value]` record (or a non-record type) that is held by value rather than
    /// behind an `rc` pointer.
    fn record_ref_cap(&self, ty: Ty<'tcx>) -> Result<Option<ReussirCapability>> {
        if self.tys.is_shared_record(ty) {
            Ok(Some(ReussirCapability::Shared))
        } else if self.tys.is_regional_record(ty) {
            Ok(Some(regional_capability(ty)?))
        } else {
            Ok(None)
        }
    }

    /// How field `idx` of `record_ty` projects out of a `ref<…, ref_cap>`: its
    /// MIR type, the projected reference's element type and capability, and
    /// whether that projection is a pointer link to load (an `rc` member) or an
    /// inline sub-field to keep navigating by reference.
    ///
    /// This mirrors the dialect's member-projection rule: a `[shared]` record
    /// member projects to an `rc` link; anything else projects to its own type,
    /// inline. The projected reference inherits the parent reference's capability.
    fn projected_member(
        &self,
        record_ty: Ty<'tcx>,
        ref_cap: ReussirCapability,
        idx: u32,
    ) -> Result<ProjectedMember<'c, 'tcx>> {
        let (field_ty, is_field) = self.field_at(record_ty, idx)?;
        if is_field {
            // A mutable link: `nullable<rc<inner, flex|rigid>>`. Taken out of a
            // `flex` parent the slot is itself writable (`field` capability); out
            // of a `rigid` (frozen) parent it is read-only. The MIR types the
            // member `Nullable<Record>`, so name the link's element record.
            let inner = self.tys.record_inner_of(field_link_element(field_ty))?;
            let link_cap = if ref_cap == ReussirCapability::Flex {
                ReussirCapability::Flex
            } else {
                ReussirCapability::Rigid
            };
            let elem_ty = self
                .tys
                .nullable_type(self.tys.rc_type_with_cap(inner, link_cap));
            let proj_cap = if ref_cap == ReussirCapability::Flex {
                ReussirCapability::Field
            } else {
                ref_cap
            };
            Ok(ProjectedMember {
                field_ty,
                elem_ty,
                proj_cap,
                is_link: true,
            })
        } else if self.tys.is_shared_record(field_ty) {
            let inner = self.tys.record_inner_of(field_ty)?;
            Ok(ProjectedMember {
                field_ty,
                elem_ty: self.tys.rc_type(inner),
                proj_cap: ref_cap,
                is_link: true,
            })
        } else if self.tys.is_regional_record(field_ty) {
            // A non-`[field]` regional member would be a frozen (`rigid`) link, but
            // it is not projectable yet: elaboration leaves such a member's
            // flexivity unrefined (`Regional`), so the loaded value has no concrete
            // `flex`/`rigid` coloring to project further through. Error explicitly
            // here rather than surface a confusing "unrefined flexivity" downstream.
            err("projection of a non-`[field]` regional record member not yet implemented")
        } else {
            Ok(ProjectedMember {
                field_ty,
                elem_ty: self.tys.mlir_ty(field_ty)?,
                proj_cap: ref_cap,
                is_link: false,
            })
        }
    }

    /// The MIR type of field `idx` of a compound record, and whether it is a
    /// mutable `[field]` link.
    fn field_at(&self, record_ty: Ty<'tcx>, idx: u32) -> Result<(Ty<'tcx>, bool)> {
        let rec = self
            .tys
            .record_of(record_ty)
            .ok_or_else(|| LoweringError("projection base is not a record".into()))?;
        let members = match rec.layout {
            mir::RecordLayout::Compound(ms) => ms,
            mir::RecordLayout::Variant(_) => return err("projection of an enum value"),
        };
        let field = members
            .get(idx as usize)
            .ok_or_else(|| LoweringError(format!("field index {idx} out of range").into()))?;
        Ok((field.ty, field.is_field))
    }

    /// Materialize a [`Cursor`] as a loaded SSA value, loading through a trailing
    /// reference if the walk ended on one.
    fn load_cursor<'b>(
        &self,
        block: &'b Block<'c>,
        cursor: Cursor<'c, 'b, 'tcx>,
    ) -> Result<Value<'c, 'b>> {
        match cursor {
            Cursor::Value { val, .. } => Ok(val),
            Cursor::Ref { val, ty, .. } => {
                let inner = self.tys.mlir_ty(ty)?;
                Ok(self.append(
                    block,
                    dialect::ref_load(self.context, inner, val, self.loc()).into(),
                ))
            }
        }
    }

    /// Append `op` to `block` and return its single result as a value.
    fn append<'b>(&self, block: &'b Block<'c>, op: Operation<'c>) -> Value<'c, 'b> {
        block.append_operation(op).result(0).unwrap().into()
    }

    // --- Reference-count op emission ----------------------------------------

    /// Emit the reference-count ops the ownership analysis placed *before* node
    /// `id`. These only ever target named locals here: a `dup`/`drop` of an
    /// anonymous result before a node would refer to an already-evaluated
    /// temporary, which the constructs lowered here do not produce.
    fn emit_rc_before<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b>,
        id: ExprId,
    ) -> Result<()> {
        let ops = self.ownership.borrow().before(id).to_vec();
        for op in ops {
            match op {
                RcOp::Dup(v) => self.emit_inc_var(block, env, v)?,
                RcOp::Drop(v) => self.emit_dec_var(block, env, v)?,
                RcOp::DupValue(_) | RcOp::DropValue(_) => {
                    return err("reference-count op on an unbound temporary is not supported");
                }
            }
        }
        Ok(())
    }

    /// Emit the reference-count ops the ownership analysis placed *after* node
    /// `id`, whose `ty`/`value` are the node's just-produced result. A `DupValue`/
    /// `DropValue` keyed by `id` acts on that result (a borrowed field to retain,
    /// or a discarded result to release).
    fn emit_rc_after<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b>,
        id: ExprId,
        ty: Ty<'tcx>,
        value: Option<Value<'c, 'b>>,
    ) -> Result<()> {
        let ops = self.ownership.borrow().after(id).to_vec();
        for op in ops {
            match op {
                RcOp::Dup(v) => self.emit_inc_var(block, env, v)?,
                RcOp::Drop(v) => self.emit_dec_var(block, env, v)?,
                RcOp::DupValue(target) if target == id => {
                    let val = value
                        .ok_or_else(|| LoweringError("retain of a node with no value".into()))?;
                    self.emit_inc(block, val, ty)?;
                }
                RcOp::DropValue(target) if target == id => {
                    let val = value
                        .ok_or_else(|| LoweringError("release of a node with no value".into()))?;
                    self.emit_dec(block, val, ty)?;
                }
                RcOp::DupValue(_) | RcOp::DropValue(_) => {
                    return err("reference-count op on an unbound temporary is not supported");
                }
            }
        }
        Ok(())
    }

    /// Increment the refcount owned by local `v`. A value-less local (e.g. unit)
    /// holds no resource, so there is nothing to do.
    fn emit_inc_var<'b>(&self, block: &'b Block<'c>, env: &Env<'c, 'b>, v: VarId) -> Result<()> {
        match (env.vars.get(&v).copied(), self.var_ty(v)) {
            (Some(val), Some(ty)) => self.emit_inc(block, val, ty),
            _ => Ok(()),
        }
    }

    /// Decrement the refcount owned by local `v` (see [`emit_inc_var`](Self::emit_inc_var)).
    fn emit_dec_var<'b>(&self, block: &'b Block<'c>, env: &Env<'c, 'b>, v: VarId) -> Result<()> {
        match (env.vars.get(&v).copied(), self.var_ty(v)) {
            (Some(val), Some(ty)) => self.emit_dec(block, val, ty),
            _ => Ok(()),
        }
    }

    /// Increment a value's refcount: an `rc` pointer is incremented directly,
    /// while an inline record that transitively owns rc fields is acquired through
    /// a reference (which increments every rc pointer it reaches).
    fn emit_inc<'b>(&self, block: &'b Block<'c>, val: Value<'c, 'b>, ty: Ty<'tcx>) -> Result<()> {
        let loc = self.loc();
        if self.tys.is_shared_record(ty) {
            block.append_operation(dialect::rc_inc(self.context, val, loc).into());
            Ok(())
        } else if matches!(ty.kind(), TyKind::Record { .. }) {
            let reference = self.spill(block, val, ty)?;
            block.append_operation(dialect::ref_acquire(self.context, reference, loc).into());
            Ok(())
        } else {
            err("reference-count increment on an unsupported type")
        }
    }

    /// Decrement a value's refcount, the dual of [`emit_inc`](Self::emit_inc): an
    /// `rc` pointer is decremented directly, an inline record is dropped through a
    /// reference (releasing every rc pointer it reaches).
    fn emit_dec<'b>(&self, block: &'b Block<'c>, val: Value<'c, 'b>, ty: Ty<'tcx>) -> Result<()> {
        let loc = self.loc();
        if self.tys.is_shared_record(ty) {
            block.append_operation(dialect::rc_dec(self.context, val, loc).into());
            Ok(())
        } else if matches!(ty.kind(), TyKind::Record { .. }) {
            let reference = self.spill(block, val, ty)?;
            block.append_operation(dialect::ref_drop(self.context, reference, loc).into());
            Ok(())
        } else {
            err("reference-count decrement on an unsupported type")
        }
    }

    /// Spill a value to a fresh stack slot and return an (unspecified-capability)
    /// reference to it, so the recursive acquire/drop ops can walk its fields.
    fn spill<'b>(
        &self,
        block: &'b Block<'c>,
        val: Value<'c, 'b>,
        ty: Ty<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let inner = self.tys.mlir_ty(ty)?;
        let ref_ty = self.tys.unspecified_ref_type(inner);
        Ok(self.append(
            block,
            dialect::ref_spilled(self.context, ref_ty, val, self.loc()).into(),
        ))
    }

    fn negate<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        x: &Expr<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let xv = self
            .expr(block, env, x)?
            .ok_or_else(|| LoweringError("negate on unit".into()))?;
        match x.ty.kind() {
            TyKind::Fp(_) => Ok(self.append(block, arith::negf(xv, loc))),
            TyKind::Int(_) => {
                let ty = self.tys.mlir_ty(x.ty)?;
                let zero = self.append(
                    block,
                    arith::constant(self.context, IntegerAttribute::new(ty, 0).into(), loc),
                );
                Ok(self.append(block, arith::subi(zero, xv, loc)))
            }
            _ => err("negate on non-numeric type"),
        }
    }

    fn not<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        x: &Expr<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let xv = self
            .expr(block, env, x)?
            .ok_or_else(|| LoweringError("not on unit".into()))?;
        let i1 = IntegerType::new(self.context, 1).into();
        let one = self.append(
            block,
            arith::constant(self.context, IntegerAttribute::new(i1, 1).into(), loc),
        );
        Ok(self.append(block, arith::xori(xv, one, loc)))
    }

    fn arith<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        l: &Expr<'tcx>,
        op: ArithOp,
        r: &Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let lv = self
            .expr(block, env, l)?
            .ok_or_else(|| LoweringError("arith operand is unit".into()))?;
        let rv = self
            .expr(block, env, r)?
            .ok_or_else(|| LoweringError("arith operand is unit".into()))?;
        let signed = matches!(ty.kind(), TyKind::Int(IntTy::Signed(_)));
        let float = matches!(ty.kind(), TyKind::Fp(_));
        let op = match (op, float, signed) {
            (ArithOp::Add, false, _) => arith::addi(lv, rv, loc),
            (ArithOp::Add, true, _) => arith::addf(lv, rv, loc),
            (ArithOp::Sub, false, _) => arith::subi(lv, rv, loc),
            (ArithOp::Sub, true, _) => arith::subf(lv, rv, loc),
            (ArithOp::Mul, false, _) => arith::muli(lv, rv, loc),
            (ArithOp::Mul, true, _) => arith::mulf(lv, rv, loc),
            (ArithOp::Div, true, _) => arith::divf(lv, rv, loc),
            (ArithOp::Div, false, true) => arith::divsi(lv, rv, loc),
            (ArithOp::Div, false, false) => arith::divui(lv, rv, loc),
            (ArithOp::Mod, true, _) => arith::remf(lv, rv, loc),
            (ArithOp::Mod, false, true) => arith::remsi(lv, rv, loc),
            (ArithOp::Mod, false, false) => arith::remui(lv, rv, loc),
            (ArithOp::And, ..) => arith::andi(lv, rv, loc),
            (ArithOp::Or, ..) => arith::ori(lv, rv, loc),
        };
        Ok(self.append(block, op))
    }

    fn cmp<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        l: &Expr<'tcx>,
        op: CmpOp,
        r: &Expr<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let operand_ty = l.ty;
        let lv = self
            .expr(block, env, l)?
            .ok_or_else(|| LoweringError("cmp operand is unit".into()))?;
        let rv = self
            .expr(block, env, r)?
            .ok_or_else(|| LoweringError("cmp operand is unit".into()))?;
        let signed = matches!(operand_ty.kind(), TyKind::Int(IntTy::Signed(_)));
        let float = matches!(operand_ty.kind(), TyKind::Fp(_));
        let op = if float {
            // Ordered relational/equality predicates (NaN operand ⇒ false),
            // except `!=`, which is unordered so that `NaN != x` is true — the
            // usual IEEE/C semantics, matching how `==`/`!=` lower elsewhere.
            let pred = match op {
                CmpOp::Lt => CmpfPredicate::Olt,
                CmpOp::Gt => CmpfPredicate::Ogt,
                CmpOp::Le => CmpfPredicate::Ole,
                CmpOp::Ge => CmpfPredicate::Oge,
                CmpOp::Eq => CmpfPredicate::Oeq,
                CmpOp::Ne => CmpfPredicate::Une,
            };
            arith::cmpf(self.context, pred, lv, rv, loc)
        } else {
            let pred = match (op, signed) {
                (CmpOp::Eq, _) => CmpiPredicate::Eq,
                (CmpOp::Ne, _) => CmpiPredicate::Ne,
                (CmpOp::Lt, true) => CmpiPredicate::Slt,
                (CmpOp::Le, true) => CmpiPredicate::Sle,
                (CmpOp::Gt, true) => CmpiPredicate::Sgt,
                (CmpOp::Ge, true) => CmpiPredicate::Sge,
                (CmpOp::Lt, false) => CmpiPredicate::Ult,
                (CmpOp::Le, false) => CmpiPredicate::Ule,
                (CmpOp::Gt, false) => CmpiPredicate::Ugt,
                (CmpOp::Ge, false) => CmpiPredicate::Uge,
            };
            arith::cmpi(self.context, pred, lv, rv, loc)
        };
        Ok(self.append(block, op))
    }

    fn cast<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        x: &Expr<'tcx>,
        to: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let xv = self
            .expr(block, env, x)?
            .ok_or_else(|| LoweringError("cast operand is unit".into()))?;
        let src = num_class(x.ty)?;
        let dst = num_class(to)?;
        let to_ty = self.tys.mlir_ty(to)?;
        let op = match (src.float, dst.float) {
            (false, false) => {
                if dst.width == src.width {
                    return Ok(Some(xv)); // same integer width, no-op
                } else if dst.width < src.width {
                    arith::trunci(xv, to_ty, loc)
                } else if src.signed {
                    arith::extsi(xv, to_ty, loc)
                } else {
                    arith::extui(xv, to_ty, loc)
                }
            }
            (false, true) => {
                if src.signed {
                    arith::sitofp(xv, to_ty, loc)
                } else {
                    arith::uitofp(xv, to_ty, loc)
                }
            }
            (true, false) => {
                if dst.signed {
                    arith::fptosi(xv, to_ty, loc)
                } else {
                    arith::fptoui(xv, to_ty, loc)
                }
            }
            (true, true) => {
                if dst.width == src.width {
                    return Ok(Some(xv));
                } else if dst.width < src.width {
                    builders::truncf(xv, to_ty, loc)
                } else {
                    arith::extf(xv, to_ty, loc)
                }
            }
        };
        Ok(Some(self.append(block, op)))
    }

    fn lower_if<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        c: &Expr<'tcx>,
        t: &Expr<'tcx>,
        f: &Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let cv = self
            .expr(block, env, c)?
            .ok_or_else(|| LoweringError("if condition is unit".into()))?;
        let result_tys = if is_unit(ty) {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(ty)?]
        };
        let then_region = self.branch_region(env, t, ty)?;
        let else_region = self.branch_region(env, f, ty)?;
        let op = block.append_operation(scf::r#if(cv, &result_tys, then_region, else_region, loc));
        if is_unit(ty) {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Build one `scf.if` branch region: an entry block whose body lowers `e` in
    /// a nested scope over the enclosing environment and terminates with
    /// `scf.yield` (yielding its value unless unit-typed).
    fn branch_region<'b>(
        &self,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Region<'c>> {
        let loc = self.loc();
        let block = Block::new(&[]);
        // Lower the branch in a nested scope over the same environment: it sees
        // the outer locals and region, and any bindings it adds are rewound when
        // the scope returns, before the block is moved into its region below.
        env.subscope(|scope| {
            let value = self.expr(&block, scope, e)?;
            if is_unit(ty) {
                block.append_operation(scf::r#yield(&[], loc));
            } else {
                let v = value.ok_or_else(|| LoweringError("if branch produced no value".into()))?;
                block.append_operation(scf::r#yield(&[v], loc));
            }
            Ok::<_, LoweringError>(())
        })?;
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }
}
