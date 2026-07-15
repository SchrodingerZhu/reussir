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
use rapidhash::RapidHashMap;
use rustc_hash::{FxHashMap, FxHashSet};

use reussir_backend::builders;
use reussir_backend::dialect;
use reussir_backend::dialect::ty::{ReussirCapability, ReussirLifeScope};
use reussir_backend::melior::Context;
use reussir_backend::melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use reussir_backend::melior::dialect::{func, memref, scf};
use reussir_backend::melior::ir::attribute::{
    ArrayAttribute, Attribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute,
    StringAttribute, TypeAttribute,
};
use reussir_backend::melior::ir::r#type::{FunctionType, IntegerType, MemRefType};
use reussir_backend::melior::ir::{
    Block, BlockLike, Identifier, Location, Operation, Region, RegionLike, Type, Value,
};

use reussir_core::full::mir::{self, DecisionTree, Expr, ExprKind, SwitchCases};
use reussir_core::full::ownership::{OwnershipTable, RcOp, RecordTable, analyze_function};
use reussir_core::intrinsic::{ArrayFn, CellFn, IntrinsicOp};
use reussir_core::literal::{self, Integer};
use reussir_core::semi::hir::{ArithOp, CmpOp, ExprId, VarId};
use reussir_core::semi::ty::{Flexivity, IntTy, Ty, TyCtxt, TyKind};
use reussir_core::surface::{Span, Visibility};
use reussir_core::utils::string::StringToken;
use reussir_syntax::kind::{Resolver, TokenKey};
use smallvec::SmallVec;

use crate::source::{FileId, SourceCache};

use super::ty::{TypeCtx, field_link_element, is_unit, num_class, regional_capability};
use super::{LinkagePolicy, LoweringError, Result, err};

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
/// A nested block (an `if` branch, a `region-run` body, a `match` arm) is a
/// nested scope that only ever *appends* to its append-only tables — variable
/// ids are globally unique, so a scope never rebinds one already in `vars`, and
/// `temps`/`anchors` are searched most-recent-first so an inner entry shadows an
/// outer one at the same key. That lets a subscope be recovered by truncating
/// each table back to its length on entry (see [`Env::subscope`]) instead of
/// copying, which is why `vars` is an insertion-ordered map.
///
/// The last two tables serve `match` lowering, which walks a compiled
/// [`DecisionTree`](reussir_core::full::mir::DecisionTree) rather than plain
/// arms:
///
/// * `temps` maps a temporary expression's [`ExprId`] to the SSA value it
///   produced. `vars` names *variables* for the reference-count ops, but a
///   decision tree may match on an anonymous temporary (`match tail(xs) { … }`),
///   whose per-leaf drop the ownership pass keys to the scrutinee's own id — this
///   is the id→value lookup that lets that drop name the value. `temp_tys` on the
///   [`Lowerer`] holds the matching type (it outlives the block, so it cannot
///   ride here).
/// * `anchors` records the value or reference reached at each scrutinee
///   [`Path`](reussir_core::full::mir::Path) as the decision tree descends, so a
///   leaf binding or a nested `Switch` projects the rest of its path from the
///   nearest already-materialized point. A dispatch arm pushes the narrowed case
///   reference at the switch's path, shadowing the enclosing entry there.
struct Env<'c, 'b, 'tcx> {
    vars: IndexMap<VarId, Value<'c, 'b>, rustc_hash::FxBuildHasher>,
    region: Option<Value<'c, 'b>>,
    temps: IndexMap<ExprId, Value<'c, 'b>, rustc_hash::FxBuildHasher>,
    anchors: Vec<(SmallVec<[u32; 4]>, Anchor<'c, 'b, 'tcx>)>,
}

/// A point already materialized while walking a `match`
/// [`DecisionTree`](reussir_core::full::mir::DecisionTree), keyed in
/// [`Env::anchors`] by the [`Path`](reussir_core::full::mir::Path) it sits at.
#[derive(Clone, Copy)]
enum Anchor<'c, 'b, 'tcx> {
    /// The match root: the loaded scrutinee value (an `rc` pointer for a boxed
    /// enum, the inline aggregate for a `[value]` one).
    Root { val: Value<'c, 'b>, ty: Ty<'tcx> },
    /// A dispatch arm's narrowed reference to a variant's case payload. The case
    /// payload compound has no MIR [`Ty`], so its fields are projected through
    /// the variant's own field-type list rather than the record layout.
    ///
    /// `whole` is the *un-narrowed* enum value that was dispatched at this path —
    /// the [`Cursor`] `resolve` produced before the borrow. A binding (or nested
    /// switch) that names the whole switched value resolves to this path with an
    /// empty suffix, and gets `whole` back rather than a projection into the case
    /// payload (which has no field for "the value itself").
    Case {
        reference: Value<'c, 'b>,
        variant: &'tcx mir::VariantDef<'tcx>,
        cap: ReussirCapability,
        whole: Cursor<'c, 'b, 'tcx>,
    },
}

impl<'c, 'b, 'tcx> Env<'c, 'b, 'tcx> {
    /// An empty environment carrying `region` as its enclosing region handle.
    fn new(region: Option<Value<'c, 'b>>) -> Self {
        Self {
            vars: IndexMap::default(),
            region,
            temps: IndexMap::default(),
            anchors: Vec::new(),
        }
    }

    /// Run `f` in a nested block scope: it borrows the environment reborrowed at
    /// the nested block's (shorter) lifetime `'s`, and everything it adds is
    /// discarded afterwards — `vars`/`temps`/`anchors` are truncated back to their
    /// lengths on entry and `region` is restored (a `region-run` body swaps in its
    /// own handle for the duration).
    ///
    /// The nested block's SSA values are shorter-lived than this environment
    /// (they borrow a block the caller created for the scope), and `&mut` is
    /// invariant, so the borrow checker will not let `f` insert them through a
    /// `&mut Env<'c, 'b, 'tcx>`. This performs the one reborrow that shortens the
    /// value lifetime; it is sound because every binding `f` makes at `'s` is
    /// removed before returning, so no `'s`-lived value is ever observed at `'b`.
    fn subscope<'s, R>(&mut self, f: impl FnOnce(&mut Env<'c, 's, 'tcx>) -> R) -> R
    where
        'b: 's,
    {
        let vars_mark = self.vars.len();
        let temps_mark = self.temps.len();
        let anchors_mark = self.anchors.len();
        let saved_region = self.region;
        // SAFETY: `'b: 's`, so viewing the stored `Value<'c, 'b>`s as
        // `Value<'c, 's>` is a lifetime *shortening* — a sound covariant coercion
        // the compiler declines to perform through the invariant `&mut`. `short`
        // aliases `*self`, but `self` is untouched until `f` returns, and the
        // truncate/restore below removes every `'s`-lived value `f` added before
        // `self` is next read at `'b`.
        let short: &mut Env<'c, 's, 'tcx> =
            unsafe { &mut *(self as *mut Env<'c, 'b, 'tcx>).cast() };
        let r = f(short);
        self.vars.truncate(vars_mark);
        self.temps.truncate(temps_mark);
        self.anchors.truncate(anchors_mark);
        self.region = saved_region;
        r
    }
}

/// The state threaded through a field-projection walk: either a loaded value or a
/// borrowed reference into a record, each tagged with its ground MIR type so the
/// next step can consult the record layout and pick the right op.
#[derive(Clone, Copy)]
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
    pub(super) source: Option<&'p SourceCache>,
    /// Resolves interned source names for debug info; `Some` (with `source`)
    /// enables DWARF variable/type emission. See [`debug`](super::debug).
    pub(super) names: Option<&'p dyn Resolver<TokenKey>>,
    ownership: RefCell<OwnershipTable>,
    var_tys: RefCell<FxHashMap<VarId, Ty<'tcx>>>,
    /// Ground types of the temporary expressions currently held in
    /// [`Env::temps`], so a reference-count op the ownership pass keyed to such a
    /// temporary (a `match` scrutinee's per-arm drop) can pick the right op.
    temp_tys: RefCell<FxHashMap<ExprId, Ty<'tcx>>>,
    /// The location attached to ops as they are built. Set to the current
    /// expression's span by [`expr`](Self::expr) (saved and restored around each
    /// node), so every op a node emits shares that node's source position.
    cur_loc: RefCell<Location<'c>>,
    /// The file the current function's spans index (functions from different
    /// files share one module); set by [`function`](Self::function).
    cur_file: std::cell::Cell<FileId>,
    /// The codegen unit being emitted; functions outside it lower to
    /// declarations (see [`super::CodegenUnit`]).
    unit: super::CodegenUnit,
    /// Record symbols whose debug composite is currently being built — the
    /// recursion guard for [`debug`](super::debug). A type that reaches itself
    /// (directly or through a boxed field) is emitted as a memberless
    /// forward-declared composite instead of expanding forever.
    pub(super) dbg_building: RefCell<FxHashSet<mir::Symbol>>,
    /// String-literal payloads by token, for `str.select` pattern emission.
    string_payloads: RapidHashMap<StringToken, &'p str>,
    /// How definitions map to LLVM linkage (see [`LinkagePolicy`]).
    linkage: LinkagePolicy,
    /// Sanitizer function-attribute strings (e.g. `sanitize_address`) each
    /// emitted function definition carries via the `passthrough` attribute —
    /// LLVM's sanitizer passes only instrument plain memory accesses in
    /// functions attributed for them (see [`super::Sanitizer`]).
    sanitize_attrs: Vec<&'static str>,
}

impl<'c, 'p, 'tcx> Lowerer<'c, 'p, 'tcx> {
    pub(super) fn new(
        context: &'c Context,
        tcx: &'p TyCtxt<'tcx>,
        program: &'p mir::Program<'tcx>,
        source: Option<&'p SourceCache>,
        names: Option<&'p dyn Resolver<TokenKey>>,
        linkage: LinkagePolicy,
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
            temp_tys: RefCell::new(FxHashMap::default()),
            cur_loc: RefCell::new(Location::unknown(context)),
            cur_file: std::cell::Cell::new(FileId::ROOT),
            unit: super::CodegenUnit { index: 0, count: 1 },
            dbg_building: RefCell::new(FxHashSet::default()),
            string_payloads: program
                .string_literals
                .iter()
                .map(|(token, payload)| (*token, payload.as_str()))
                .collect(),
            linkage,
            sanitize_attrs: Vec::new(),
        }
    }

    /// Restrict this lowerer to one codegen unit of a partitioned build.
    pub(super) fn with_unit(mut self, unit: super::CodegenUnit) -> Self {
        self.unit = unit;
        self
    }

    /// Set the sanitizer attribute strings each emitted definition carries.
    pub(super) fn with_sanitize_attrs(mut self, attrs: Vec<&'static str>) -> Self {
        self.sanitize_attrs = attrs;
        self
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

    /// Resolve a MIR span (an offset into the current function's file) to a
    /// `FileLineColLoc`, or `unknown` without a source cache or span — or when
    /// the file's content could not be obtained (a re-ingested dump whose
    /// source is virtual or missing), since offsets then resolve to nothing.
    pub(super) fn location(&self, span: Option<Span>) -> Location<'c> {
        match (self.source, span) {
            (Some(cache), Some(span)) if cache.is_available(self.cur_file.get()) => {
                let file = self.cur_file.get();
                let (line, col) = cache.line_col(file, span.start as usize);
                Location::new(self.context, cache.name(file), line, col)
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

    fn temp_ty(&self, id: ExprId) -> Option<Ty<'tcx>> {
        self.temp_tys.borrow().get(&id).copied()
    }

    /// Lower one MIR function to a `func.func` operation.
    pub(super) fn function(&self, func: &mir::Function<'tcx>) -> Result<Operation<'c>> {
        // The function's spans index its own declaration file; every
        // `location` below resolves against it.
        self.cur_file.set(func.file);
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

        // A partitioned build emits the body only in the function's home
        // unit; everywhere else the symbol is a declaration the linker
        // resolves across objects.
        let symbol = self.program.symbol(func.symbol);
        let emit_body = self.unit.is_home(symbol);

        // Reset the per-function side-tables, then run the ownership analysis for
        // this body so the tree-walk can emit each node's reference-count ops.
        *self.ownership.borrow_mut() = analyze_function(self.tcx, func, &self.records);
        self.var_tys.borrow_mut().clear();
        self.temp_tys.borrow_mut().clear();

        let region = Region::new();
        if let Some(body) = func.body.filter(|_| emit_body) {
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
        // the printer renders it as the `private` keyword. A partitioned build
        // promotes every *defined* function to external visibility — a private
        // symbol could not be called from another unit — while a body-less
        // declaration must be `private` (an MLIR `func.func` rule; its LLVM
        // linkage is external either way, which is what cross-unit resolution
        // needs).
        let mut attributes = Vec::new();
        if func.transform_anchor && emit_body {
            attributes.push((
                Identifier::new(self.context, "reussir.transform_anchor"),
                Attribute::unit(self.context),
            ));
            attributes.push((
                Identifier::new(self.context, "no_inline"),
                Attribute::unit(self.context),
            ));
        }
        if !emit_body || (func.visibility == Visibility::Private && !self.unit.is_split()) {
            attributes.push((
                Identifier::new(self.context, "sym_visibility"),
                StringAttribute::new(self.context, "private").into(),
            ));
        }
        // Sanitizer attributes ride the `passthrough` attribute FuncToLLVM
        // copies onto the `llvm.func` verbatim; LLVM's sanitizer passes only
        // instrument plain memory accesses in functions carrying them. A
        // declaration needs none — its home unit annotates the body.
        if emit_body && !self.sanitize_attrs.is_empty() {
            let strings: Vec<Attribute> = self
                .sanitize_attrs
                .iter()
                .map(|a| StringAttribute::new(self.context, a).into())
                .collect();
            attributes.push((
                Identifier::new(self.context, "passthrough"),
                ArrayAttribute::new(self.context, &strings).into(),
            ));
        }
        // The function's LLVM linkage (an `llvm.linkage` attribute FuncToLLVM
        // carries onto the `llvm.func`). Declarations stay plain external.
        if let Some(linkage) = self.llvm_linkage(func) {
            let attr = format!("#llvm.linkage<{linkage}>");
            attributes.push((
                Identifier::new(self.context, "llvm.linkage"),
                Attribute::parse(self.context, &attr)
                    .unwrap_or_else(|| panic!("malformed linkage attribute {attr}")),
            ));
        }
        // Debug info (when enabled): the parameters' names/types as an attribute
        // the conversion pass reads, and the function's own location fused with a
        // subprogram attribute. A body-less declaration gets neither — its home
        // unit describes it (a declaration must not carry a DISubprogram
        // definition).
        let func_loc = if emit_body {
            if let Some(args) = self.dbg_func_args_attr(func) {
                attributes.push(args);
            }
            self.subprogram_location(func)
        } else {
            self.location(func.body.and_then(|b| b.span))
        };

        Ok(func::func(
            self.context,
            StringAttribute::new(self.context, symbol),
            TypeAttribute::new(fn_ty.into()),
            region,
            &attributes,
            func_loc,
        ))
    }

    /// The LLVM linkage for a lowered function under the session's
    /// [`LinkagePolicy`], or `None` to leave the (external) default.
    ///
    /// A monomorphized generic instance is recognized by its v0 symbol: mono
    /// mangles an instantiated item as `path-with-args` (`"I" path {type}+
    /// "E"`), so exactly the instantiated symbols start with `_RI`. That makes
    /// the classification stable across the textual-MIR round trip without
    /// widening the MIR itself.
    fn llvm_linkage(&self, func: &mir::Function<'tcx>) -> Option<&'static str> {
        let symbol = self.program.symbol(func.symbol);
        // Declarations — body-less functions, and every function outside its
        // home codegen unit — stay plain external for cross-object resolution.
        if func.body.is_none() || !self.unit.is_home(symbol) {
            return None;
        }
        match self.linkage {
            LinkagePolicy::Jit => None,
            LinkagePolicy::Aot { dedup_instances } => {
                if symbol.starts_with("_RI") {
                    // Another compilation unit may instantiate the same
                    // generic — emit the definition as mergeable. This holds
                    // for private generics too: a pub generic caller can force
                    // their instantiation from downstream units. Partitioned
                    // builds keep this: the home unit's `linkonce_odr`
                    // definition satisfies the other units' external
                    // declarations.
                    dedup_instances.then_some("linkonce_odr")
                } else if func.visibility == Visibility::Private && !self.unit.is_split() {
                    // Mirrors the `sym_visibility` promotion above: in a
                    // partitioned build any other unit may call this symbol,
                    // so the definition must stay externally linkable.
                    Some("internal")
                } else {
                    None
                }
            }
        }
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        use ExprKind::*;
        let loc = self.loc();
        match &e.kind {
            // Emitted at full width via the textual APInt path — the value
            // was range-checked against `e.ty` at monomorphization.
            ConstInt(n) => Ok(Some(self.wide_int_constant(block, n, e.ty)?)),
            ConstBool(b) => {
                let i1 = IntegerType::new(self.context, 1).into();
                let attr = IntegerAttribute::new(i1, i64::from(*b)).into();
                Ok(Some(
                    self.append(block, arith::constant(self.context, attr, loc)),
                ))
            }
            ConstChar(c) => {
                let i32_ty = IntegerType::new(self.context, 32).into();
                let attr = IntegerAttribute::new(i32_ty, i64::from(*c)).into();
                Ok(Some(
                    self.append(block, arith::constant(self.context, attr, loc)),
                ))
            }
            // The single decimal→binary rounding in the pipeline: the exact
            // literal is rounded into the target format host-side (no Rust
            // float involved, so bf16/f16 work like f32/f64) and the bits are
            // emitted through MLIR's hexadecimal form, which is exact.
            ConstFloat(f) => {
                let fp = match *e.ty.kind() {
                    TyKind::Fp(fp) => fp,
                    _ => return err("float literal with a non-float type"),
                };
                let (exp_bits, mant_bits) = literal::ieee_params(fp).ok_or_else(|| {
                    LoweringError(format!("`{fp:?}` has no settled encoding").into())
                })?;
                let bits = f.to_ieee_bits(exp_bits, mant_bits).map_err(|_| {
                    // Unreachable after the mono range check; keep it an error.
                    LoweringError(format!("float literal `{f}` overflows its type").into())
                })?;
                let mlir_ty = self.tys.mlir_ty(e.ty)?;
                let digits = ((1 + exp_bits + mant_bits) as usize).div_ceil(4);
                let text = format!("0x{bits:0digits$X} : {mlir_ty}");
                let attr = Attribute::parse(self.context, &text).ok_or_else(|| {
                    LoweringError(format!("could not build float constant `{text}`").into())
                })?;
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
            // A `core::intrinsic::<family>` call: dispatch to the family's op
            // builder over the (scalar) operands. Each family owns its own
            // dialect ops and attributes; the backend pipeline's dialect
            // conversions take the emitted op from there.
            Intrinsic { op, args } => match op {
                IntrinsicOp::Math { func, flag } => {
                    let mut operands = Vec::with_capacity(args.len());
                    for a in args.iter() {
                        let v = self.expr(block, env, a)?.ok_or_else(|| {
                            LoweringError("intrinsic operand produced no value".into())
                        })?;
                        operands.push(v);
                    }
                    let result = self.tys.mlir_ty(e.ty)?;
                    let fastmath =
                        reussir_core::intrinsic::FastMath(*flag)
                            .mlir_attr()
                            .map(|text| {
                                Attribute::parse(self.context, &text)
                                    .expect("valid fastmath attribute")
                            });
                    let built = super::math::math_operation(
                        self.context,
                        *func,
                        &operands,
                        result,
                        fastmath,
                        loc,
                    );
                    Ok(Some(self.append(block, built)))
                }
                IntrinsicOp::Cell { func } => self.lower_cell_intrinsic(block, env, e, *func, args),
            },
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
            Match(scrut, tree) => self.lower_match(block, env, e, scrut, tree),
            Ctor { args, .. } => self.compound(block, env, e, args).map(Some),
            Variant { variant, args, .. } => self.variant(block, env, e, *variant, args).map(Some),
            NullableCall(inner) => self.nullable_call(block, env, e, *inner).map(Some),
            Closure(c) => self.closure(block, env, c).map(Some),
            ClosureCall { target, args } => self.closure_call(block, env, target, args),
            ArrayOp { op, args } => self.lower_array_op(block, env, e, *op, args),
            GlobalStr(token) => self.global_str(block, e, *token).map(Some),
            Poison => err("poison expression reached lowering"),
        }
    }

    fn global_str<'b>(
        &self,
        block: &'b Block<'c>,
        e: &Expr<'tcx>,
        token: StringToken,
    ) -> Result<Value<'c, 'b>> {
        let ty = self.tys.mlir_ty(e.ty)?;
        Ok(self.append(
            block,
            builders::str_literal(self.context, &token.mangle(), ty, self.loc()),
        ))
    }

    fn string_payload(&self, token: StringToken) -> Result<&'p str> {
        self.string_payloads.get(&token).copied().ok_or_else(|| {
            LoweringError(
                format!(
                    "string literal payload missing for token {:?}",
                    token.words()
                )
                .into(),
            )
        })
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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

    /// Lower a closure creation.
    ///
    /// The runtime closure has no separate capture environment: captures and
    /// parameters share one input list with the captures leading, so a closure
    /// lowers to a shared `rc<closure<(captures ++ params) -> ret>>` built inline
    /// (`reussir.closure.create` with a body region), and each captured value is
    /// then supplied with one `reussir.closure.apply`, leaving the user-visible
    /// `rc<closure<params -> ret>>`.
    ///
    /// The body region is `IsolatedFromAbove`: its block arguments *are* the
    /// closure inputs (captures then parameters), so it is lowered in a fresh
    /// environment that binds them to those arguments and inherits no enclosing
    /// region — a closure cannot capture a `flex`/region-local value, so the body
    /// never allocates into the creator's region. The reference-count ops the
    /// ownership analysis placed for the captures (a `dup` before this node for
    /// each still-live captured `rc`) and inside the body ride on the surrounding
    /// [`expr`](Self::expr) walk as usual.
    fn closure<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        c: &mir::ClosureExpr<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let ret = c.body.ty;
        // The closure's inputs at creation: every capture, then every parameter.
        let inputs: Vec<(VarId, Ty<'tcx>)> =
            c.captures.iter().chain(c.params.iter()).copied().collect();
        let input_tys: Vec<Ty<'tcx>> = inputs.iter().map(|&(_, t)| t).collect();
        let created_ty = self.tys.closure_type(&input_tys, ret)?;

        // The body block takes one argument per closure input, in order.
        let arg_mlir = input_tys
            .iter()
            .map(|&t| self.tys.mlir_ty(t))
            .collect::<Result<Vec<_>>>()?;
        let block_args: SmallVec<[(Type<'c>, Location<'c>); 8]> =
            arg_mlir.iter().map(|t| (*t, loc)).collect();
        let body_block = Block::new(&block_args);
        let mut body_env = Env::new(None);
        for (i, &(var, ty)) in inputs.iter().enumerate() {
            let arg = body_block
                .argument(i)
                .map_err(|e| LoweringError(format!("missing closure body argument: {e}").into()))?;
            body_env.vars.insert(var, arg.into());
            self.bind_var_ty(var, ty);
        }
        let value = self.expr(&body_block, &mut body_env, c.body)?;
        if is_unit(ret) {
            body_block.append_operation(builders::closure_yield(None, loc));
        } else {
            let v = value.ok_or_else(|| LoweringError("closure body produced no value".into()))?;
            body_block.append_operation(builders::closure_yield(Some(v), loc));
        }
        let body_region = Region::new();
        body_region.append_block(body_block);
        let mut closure = self.append(
            block,
            builders::closure_create(created_ty, body_region, loc),
        );

        // Supply the captured values, consuming the leading inputs in order; each
        // apply drops one input, so after capture `i` the residual closure keeps
        // `input_tys[i + 1..]` — a slice of the original inputs, no per-step shift.
        for (i, &(var, _)) in c.captures.iter().enumerate() {
            let cap_val = env.vars.get(&var).copied().ok_or_else(|| {
                LoweringError(format!("unbound captured variable {var:?}").into())
            })?;
            let applied_ty = self.tys.closure_type(&input_tys[i + 1..], ret)?;
            closure = self.append(
                block,
                builders::closure_apply(cap_val, closure, applied_ty, loc),
            );
        }
        Ok(closure)
    }

    /// Lower a closure call `target(args)`.
    ///
    /// The target lowers to `rc<closure<params -> ret>>`; each argument is
    /// supplied with a `reussir.closure.apply` that consumes the leading input.
    /// A partial application (fewer arguments than parameters) stops at the
    /// residual closure; a full application evaluates it with
    /// `reussir.closure.eval` and produces the result (`None` for a unit return).
    ///
    /// Both `apply` (which writes into the closure box in place) and `eval` (which
    /// hands the box's payload to the body as owned, to be consumed) are only
    /// sound on a uniquely-owned closure. The target here is an arbitrary closure
    /// value that may be shared (the ownership analysis hands us one owned
    /// reference, dup'ing it when the closure is reused), so it is always first
    /// passed through `reussir.closure.uniqify`, which deep-clones it when it is
    /// shared and returns it as-is when unique. (The capture-application in
    /// [`closure`](Self::closure) needs no uniqify because it feeds directly from
    /// `closure.create`, which is already unique.)
    fn closure_call<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        target: &Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (params, ret) = match target.ty.kind() {
            TyKind::Closure { params, ret } => (*params, *ret),
            _ => return err("closure call target is not a closure"),
        };
        let closure_val = self
            .expr(block, env, target)?
            .ok_or_else(|| LoweringError("closure call target is unit".into()))?;
        // Make the target unique before touching it. `apply` writes its argument
        // into the closure box in place, and `eval` hands the box's payload (the
        // values applied by earlier, possibly-curried applications) to the body as
        // owned — the body consumes them. A shared closure is referenced by other
        // holders that still own that same payload, so applying to *or* evaluating
        // it in place would corrupt or double-free their values. This is needed
        // even for a zero-argument (eval-only) call, since the target may already
        // carry payload. `uniqify` deep-clones a shared closure and is a no-op on
        // a unique one.
        let target_ty = self.tys.mlir_ty(target.ty)?;
        let mut closure = self.append(
            block,
            builders::closure_uniqify(closure_val, target_ty, loc),
        );
        for (i, a) in args.iter().enumerate() {
            let arg_val = self
                .expr(block, env, a)?
                .ok_or_else(|| LoweringError("closure call argument is unit".into()))?;
            // Apply consumes the leading input, so after argument `i` the residual
            // closure keeps `params[i + 1..]` — a slice of the original parameters,
            // no per-step shift or copy.
            let applied_ty = self.tys.closure_type(&params[i + 1..], ret)?;
            closure = self.append(
                block,
                builders::closure_apply(arg_val, closure, applied_ty, loc),
            );
        }
        // A partial application leaves a closure value; only a fully-applied
        // closure (every parameter supplied) is evaluated. Comparing arities
        // rather than inspecting the result type stays correct when `ret` is
        // itself a closure type.
        if args.len() < params.len() {
            return Ok(Some(closure));
        }
        let result_ty = if is_unit(ret) {
            None
        } else {
            Some(self.tys.mlir_ty(ret)?)
        };
        let op = block.append_operation(builders::closure_eval(closure, result_ty, loc));
        if result_ty.is_some() {
            Ok(Some(op.result(0).unwrap().into()))
        } else {
            Ok(None)
        }
    }

    // --- Pattern matching ---------------------------------------------------

    /// Lower a `match`: evaluate the scrutinee once, then walk the compiled
    /// decision tree, dispatching enums to `reussir.record.dispatch`, booleans and
    /// integers to `scf.if`, and materializing each leaf's variable bindings by
    /// projecting the scrutinee along their paths.
    ///
    /// The scrutinee is **borrowed** (its reference-count settlement — a per-arm
    /// drop when it dies in the match — is placed by the ownership pass at each
    /// leaf, keyed to the scrutinee's own id for a temporary or to its variable).
    /// It is recorded as the root [`Anchor`] at the empty path so every binding
    /// and nested switch resolves its path from the nearest already-materialized
    /// point; a dispatch arm narrows that anchor to the case payload it receives.
    fn lower_match<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        scrut: &Expr<'tcx>,
        tree: &DecisionTree<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let scrut_val = self
            .expr(block, env, scrut)?
            .ok_or_else(|| LoweringError("match scrutinee is unit".into()))?;
        // Record the scrutinee's value and type so the ownership pass's per-arm
        // drop (keyed to the scrutinee id for a non-variable scrutinee) resolves
        // it.
        let temps_mark = env.temps.len();
        let anchors_mark = env.anchors.len();
        self.remember_temp(env, scrut, scrut_val);
        let root_idx = env.anchors.len();
        env.anchors.push((
            SmallVec::new(),
            Anchor::Root {
                val: scrut_val,
                ty: scrut.ty,
            },
        ));
        let result = self.lower_tree(block, env, root_idx, tree, e.ty);
        env.anchors.truncate(anchors_mark);
        env.temps.truncate(temps_mark);
        result
    }

    /// Lower one decision-tree node into `block`, yielding the match's result
    /// value (`None` if the match is unit-typed). `root_idx` indexes the root
    /// [`Anchor`] of the enclosing match, so a whole-scrutinee binding (path `[]`)
    /// resolves to the original scrutinee value.
    fn lower_tree<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        tree: &DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        match *tree {
            DecisionTree::Leaf { body, bindings } => {
                self.lower_leaf(block, env, root_idx, body, bindings, result_ty)
            }
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => self.lower_guard(
                block, env, root_idx, bindings, guard, success, failure, result_ty,
            ),
            DecisionTree::Switch { scrutinee, cases } => {
                self.lower_switch(block, env, root_idx, scrutinee, cases, result_ty)
            }
            // A non-exhaustive gap or a redundant arm: both were already diagnosed
            // by the frontend, so this position is dead — a poison of the result
            // type terminates it without observing an undefined value.
            DecisionTree::Uncovered | DecisionTree::Unreachable => self.dead_arm(block, result_ty),
        }
    }

    /// Lower a `Switch` on the sub-value at `path`, choosing the dispatch shape by
    /// the case family.
    fn lower_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        cases: SwitchCases<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        match cases {
            SwitchCases::Ctor(subtrees) => {
                self.ctor_switch(block, env, root_idx, path, subtrees, result_ty)
            }
            SwitchCases::Bool { if_true, if_false } => {
                self.bool_switch(block, env, root_idx, path, if_true, if_false, result_ty)
            }
            SwitchCases::Int { cases, default } => {
                self.int_switch(block, env, root_idx, path, cases, default, result_ty)
            }
            SwitchCases::Char { cases, default } => {
                self.char_switch(block, env, root_idx, path, cases, default, result_ty)
            }
            SwitchCases::Nullable { non_null, null } => {
                self.nullable_switch(block, env, root_idx, path, non_null, null, result_ty)
            }
            SwitchCases::String { cases, default } => {
                self.string_switch(block, env, root_idx, path, cases, default, result_ty)
            }
        }
    }

    /// Lower a nullable `Switch` to `reussir.nullable.dispatch`: the non-null
    /// region's block receives the unwrapped pointer as its argument, anchored
    /// at the switch's *own* path (the payload occupies the scrutinee's
    /// occurrence — see the `Family::Nullable` compilation in
    /// `semi::pattern`); the null region receives nothing.
    #[allow(clippy::too_many_arguments)]
    fn nullable_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        non_null: &'tcx DecisionTree<'tcx>,
        null: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (scrut_val, scrut_ty) = self.resolve_loaded(block, env, root_idx, path)?;
        let TyKind::Nullable(inner) = *scrut_ty.kind() else {
            return err("nullable match on a non-nullable scrutinee");
        };
        let unit = is_unit(result_ty);
        let result_type = if unit {
            None
        } else {
            Some(self.tys.mlir_ty(result_ty)?)
        };

        // Non-null region: the block argument is the unwrapped pointer,
        // pushed as a fresh anchor at `path` so bindings (and any deeper
        // projections) in the sub-tree resolve to it.
        let inner_mlir = self.tys.mlir_ty(inner)?;
        let non_null_block = Block::new(&[(inner_mlir, loc)]);
        let arg: Value<'c, '_> = non_null_block
            .argument(0)
            .map_err(|e| LoweringError(format!("missing dispatch block argument: {e}").into()))?
            .into();
        let anchor_path = SmallVec::<[u32; 4]>::from_slice(path);
        env.subscope(|scope| {
            scope.anchors.push((
                anchor_path,
                Anchor::Root {
                    val: arg,
                    ty: inner,
                },
            ));
            let value = self.lower_tree(&non_null_block, scope, root_idx, non_null, result_ty)?;
            non_null_block.append_operation(builders::scf_yield(value, loc));
            Ok::<_, LoweringError>(())
        })?;
        let non_null_region = Region::new();
        non_null_region.append_block(non_null_block);

        // Null region: no argument, no new anchor.
        let null_block = Block::new(&[]);
        env.subscope(|scope| {
            let value = self.lower_tree(&null_block, scope, root_idx, null, result_ty)?;
            null_block.append_operation(builders::scf_yield(value, loc));
            Ok::<_, LoweringError>(())
        })?;
        let null_region = Region::new();
        null_region.append_block(null_block);

        let op = block.append_operation(builders::nullable_dispatch(
            scrut_val,
            result_type,
            non_null_region,
            null_region,
            loc,
        ));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Lower an enum `Switch` to `reussir.record.dispatch`: one arm per variant
    /// (in tag order), each receiving a `!reussir.ref` to that variant's case
    /// payload, which becomes the narrowed anchor for the arm's sub-tree.
    fn ctor_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        subtrees: &'tcx [DecisionTree<'tcx>],
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        // Resolve the enum once: `whole` is the un-narrowed value at this path
        // (kept in each arm's `Case` anchor for a whole-value binding), and the
        // variant reference feeds the dispatch.
        let whole = self.resolve(block, env, root_idx, path)?;
        let (variant_ref, enum_ty, cap) = self.resolve_variant_ref(block, whole)?;
        let variants = match self.tys.record_of(enum_ty).map(|r| r.layout) {
            Some(mir::RecordLayout::Variant(variants)) => variants,
            _ => return err("match scrutinee is not an enum"),
        };
        if subtrees.len() != variants.len() {
            return err("match arm count does not match the enum's variant count");
        }
        let unit = is_unit(result_ty);
        let result_type = if unit {
            None
        } else {
            Some(self.tys.mlir_ty(result_ty)?)
        };
        let mut regions = Vec::with_capacity(subtrees.len());
        let mut tags = Vec::with_capacity(subtrees.len());
        for (i, subtree) in subtrees.iter().enumerate() {
            let payload_ty = self.tys.variant_payload_of(enum_ty, i)?;
            let arg_ref_ty = self.tys.ref_type_with_cap(payload_ty, cap);
            let arm_block = Block::new(&[(arg_ref_ty, loc)]);
            let arg: Value<'c, '_> = arm_block
                .argument(0)
                .map_err(|e| LoweringError(format!("missing dispatch arm argument: {e}").into()))?
                .into();
            let variant = &variants[i];
            let anchor_path = SmallVec::<[u32; 4]>::from_slice(path);
            env.subscope(|scope| {
                scope.anchors.push((
                    anchor_path,
                    Anchor::Case {
                        reference: arg,
                        variant,
                        cap,
                        whole,
                    },
                ));
                let value = self.lower_tree(&arm_block, scope, root_idx, subtree, result_ty)?;
                arm_block.append_operation(builders::scf_yield(value, loc));
                Ok::<_, LoweringError>(())
            })?;
            let region = Region::new();
            region.append_block(arm_block);
            regions.push(region);
            tags.push(i as i64);
        }
        let op = block.append_operation(builders::record_dispatch(
            self.context,
            variant_ref,
            &tags,
            result_type,
            regions,
            loc,
        ));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Lower a boolean `Switch` to `scf.if`, its condition the (loaded) `i1` at
    /// `path` and its two branches the `if_true`/`if_false` sub-trees.
    #[allow(clippy::too_many_arguments)]
    fn bool_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        if_true: &'tcx DecisionTree<'tcx>,
        if_false: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (cond, _) = self.resolve_loaded(block, env, root_idx, path)?;
        let unit = is_unit(result_ty);
        let result_tys = if unit {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(result_ty)?]
        };
        let then_region = self.tree_branch_region(env, root_idx, if_true, result_ty)?;
        let else_region = self.tree_branch_region(env, root_idx, if_false, result_ty)?;
        let op =
            block.append_operation(scf::r#if(cond, &result_tys, then_region, else_region, loc));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Lower a character `Switch` like a ≤64-bit unsigned integer switch: the
    /// scrutinee (a 32-bit code point) casts to `index` and each code point is
    /// its own case value.
    #[allow(clippy::too_many_arguments)]
    fn char_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        cases: &'tcx [(u32, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (scrut_val, scrut_ty) = self.resolve_loaded(block, env, root_idx, path)?;
        if !matches!(scrut_ty.kind(), TyKind::Char) {
            return err("character match on a non-character scrutinee");
        }
        let index_ty = Type::index(self.context);
        let selector = self.append(block, arith::index_castui(scrut_val, index_ty, loc));
        let case_values: Vec<i64> = cases.iter().map(|(c, _)| i64::from(*c)).collect();
        self.index_switch_dispatch(
            block,
            env,
            root_idx,
            selector,
            &case_values,
            cases.iter().map(|(_, t)| t),
            default,
            result_ty,
        )
    }

    /// Lower a string `Switch`: `reussir.str.select` reduces the scrutinee to
    /// the index of the first matching pattern plus a found bit, an
    /// `arith.select` maps a miss to the slot one past the last case, and a
    /// single `scf.index_switch` dispatches — the default tree is lowered
    /// exactly once, as the switch's own (reachable) default region.
    #[allow(clippy::too_many_arguments)]
    fn string_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        cases: &'tcx [(StringToken, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (scrut_val, scrut_ty) = self.resolve_loaded(block, env, root_idx, path)?;
        if !matches!(scrut_ty.kind(), TyKind::Str) {
            return err("string match on a non-string scrutinee");
        }
        let local_ty = dialect::ty::str(self.context, ReussirLifeScope::Local);
        let local = self.append(block, builders::str_cast(scrut_val, local_ty, loc));
        let patterns: Vec<&str> = cases
            .iter()
            .map(|(token, _)| self.string_payload(*token))
            .collect::<Result<_>>()?;
        let select =
            block.append_operation(builders::str_select(self.context, local, &patterns, loc));
        let selected: Value<'c, '_> = select.result(0).unwrap().into();
        let found: Value<'c, '_> = select.result(1).unwrap().into();
        let default_slot = self.index_const(block, cases.len() as i64);
        let selector = self.append(block, arith::select(found, selected, default_slot, loc));
        let case_values: Vec<i64> = (0..cases.len() as i64).collect();
        self.index_switch_dispatch(
            block,
            env,
            root_idx,
            selector,
            &case_values,
            cases.iter().map(|(_, t)| t),
            default,
            result_ty,
        )
    }

    /// Dispatch a decision-tree switch through one `scf.index_switch`: a
    /// region per case (in `subtrees` order, keyed by `case_values`) and the
    /// `default` tree as the switch's own default region, every region
    /// terminated by `scf.yield`. Shared tail of the int/char/string switches.
    #[allow(clippy::too_many_arguments)]
    fn index_switch_dispatch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        selector: Value<'c, 'b>,
        case_values: &[i64],
        subtrees: impl Iterator<Item = &'tcx DecisionTree<'tcx>>,
        default: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let unit = is_unit(result_ty);
        let result_tys = if unit {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(result_ty)?]
        };
        let mut regions = Vec::with_capacity(case_values.len() + 1);
        regions.push(self.tree_branch_region(env, root_idx, default, result_ty)?);
        for subtree in subtrees {
            regions.push(self.tree_branch_region(env, root_idx, subtree, result_ty)?);
        }
        let cases_attr = DenseI64ArrayAttribute::new(self.context, case_values);
        let op = block.append_operation(scf::index_switch(
            self.context,
            selector,
            &result_tys,
            cases_attr,
            regions,
            self.loc(),
        ));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Lower an integer `Switch` to `scf.index_switch`. The case bodies always run
    /// through the switch (one region per key, `default` as its default region —
    /// the open family's fall-through), each terminated by `scf.yield`; only the
    /// switched-on value differs by scrutinee width:
    ///
    /// * width ≤ 64 — the (loaded) scrutinee round-trips through `index`
    ///   (`arith.index_cast` when signed, `arith.index_castui` when unsigned) and
    ///   each key's low 64 bits is an exact `i64` case value.
    /// * width > 64 — an `index` cannot hold the scrutinee and the keys may exceed
    ///   `i64`, so a full-width `scf.if` compare-chain first reduces the keys to a
    ///   compact selector in `[0, cases.len()]` (case `i` → `i`, no match → the
    ///   default), and the switch dispatches on that. (Keys are arbitrary
    ///   precision end to end; this arm becomes reachable once the surface
    ///   grows `i128`/`u128` type names.)
    #[allow(clippy::too_many_arguments)]
    fn int_switch<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
        cases: &'tcx [(&'tcx Integer, DecisionTree<'tcx>)],
        default: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (scrut_val, scrut_ty) = self.resolve_loaded(block, env, root_idx, path)?;
        let width = match scrut_ty.kind() {
            TyKind::Int(IntTy::Signed(w) | IntTy::Unsigned(w)) => *w,
            _ => return err("integer match on a non-integer scrutinee"),
        };
        let (selector, case_values) = if width > 64 {
            // Reduce the wide keys to a dense selector, then switch on that.
            let selector = self.wide_int_selector(block, scrut_val, scrut_ty, cases, 0)?;
            (selector, (0..cases.len() as i64).collect::<Vec<_>>())
        } else {
            let index_ty = Type::index(self.context);
            let signed = matches!(scrut_ty.kind(), TyKind::Int(IntTy::Signed(_)));
            let cast = if signed {
                arith::index_cast(scrut_val, index_ty, loc)
            } else {
                arith::index_castui(scrut_val, index_ty, loc)
            };
            let arg = self.append(block, cast);
            // The key fits its (≤64-bit) scrutinee type, so its two's
            // complement low word is the exact case value.
            (
                arg,
                cases
                    .iter()
                    .map(|(key, _)| literal::low_u64(key) as i64)
                    .collect::<Vec<_>>(),
            )
        };
        self.index_switch_dispatch(
            block,
            env,
            root_idx,
            selector,
            &case_values,
            cases.iter().map(|(_, t)| t),
            default,
            result_ty,
        )
    }

    /// Reduce a wide-integer switch to a dense `index` selector: a right-nested
    /// `scf.if` chain that tests the scrutinee against each key (full-width
    /// `arith.cmpi`) and yields that case's position `idx`, falling through to the
    /// next key on a miss and to `cases.len()` (the default's slot) when every key
    /// has been ruled out.
    fn wide_int_selector<'b>(
        &self,
        block: &'b Block<'c>,
        scrut_val: Value<'c, 'b>,
        scrut_ty: Ty<'tcx>,
        cases: &'tcx [(&'tcx Integer, DecisionTree<'tcx>)],
        idx: i64,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let index_ty = Type::index(self.context);
        let Some(((key, _), rest)) = cases.split_first() else {
            // Every key was ruled out: select the default's slot.
            return Ok(self.index_const(block, idx));
        };
        let key_val = self.wide_int_constant(block, key, scrut_ty)?;
        let eq = self.append(
            block,
            arith::cmpi(self.context, CmpiPredicate::Eq, scrut_val, key_val, loc),
        );
        let then_block = Block::new(&[]);
        let hit = self.index_const(&then_block, idx);
        then_block.append_operation(scf::r#yield(&[hit], loc));
        let then_region = Region::new();
        then_region.append_block(then_block);
        let else_block = Block::new(&[]);
        let miss = self.wide_int_selector(&else_block, scrut_val, scrut_ty, rest, idx + 1)?;
        else_block.append_operation(scf::r#yield(&[miss], loc));
        let else_region = Region::new();
        else_region.append_block(else_block);
        let op = block.append_operation(scf::r#if(eq, &[index_ty], then_region, else_region, loc));
        Ok(op.result(0).unwrap().into())
    }

    /// An `index`-typed constant.
    fn index_const<'b>(&self, block: &'b Block<'c>, n: i64) -> Value<'c, 'b> {
        let index_ty = Type::index(self.context);
        self.append(
            block,
            arith::constant(
                self.context,
                IntegerAttribute::new(index_ty, n).into(),
                self.loc(),
            ),
        )
    }

    /// Materialize an integer constant of `ty` holding `key`, at any width.
    /// melior's `IntegerAttribute::new` takes only an `i64`, so the constant is
    /// built by parsing its textual attribute form (`<value> : iN`), which
    /// goes through MLIR's arbitrary-precision APInt.
    fn wide_int_constant<'b>(
        &self,
        block: &'b Block<'c>,
        key: &Integer,
        ty: Ty<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let mlir_ty = self.tys.mlir_ty(ty)?;
        let text = format!("{key} : {mlir_ty}");
        let attr = Attribute::parse(self.context, &text).ok_or_else(|| {
            LoweringError(format!("could not build integer constant `{text}`").into())
        })?;
        Ok(self.append(block, arith::constant(self.context, attr, loc)))
    }

    /// Build an `scf.if` branch region whose body lowers a decision sub-tree and
    /// terminates with `scf.yield` (yielding its value unless the match is unit).
    fn tree_branch_region<'b>(
        &self,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        tree: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Region<'c>> {
        let loc = self.loc();
        let block = Block::new(&[]);
        env.subscope(|scope| {
            let value = self.lower_tree(&block, scope, root_idx, tree, result_ty)?;
            if is_unit(result_ty) {
                block.append_operation(scf::r#yield(&[], loc));
            } else {
                let value =
                    value.ok_or_else(|| LoweringError("match branch produced no value".into()))?;
                block.append_operation(scf::r#yield(&[value], loc));
            }
            Ok::<_, LoweringError>(())
        })?;
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }

    /// Lower a leaf: bind each pattern variable to the scrutinee value at its path
    /// (a whole-scrutinee binding reuses the root value; a field binding projects
    /// and loads it — the ownership pass's `dup` on the leaf retains an aliased rc
    /// field), then lower the arm body. Bindings ride the enclosing branch scope,
    /// so they are visible to the body and rewound with it.
    fn lower_leaf<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        body: &'tcx Expr<'tcx>,
        bindings: &'tcx [(VarId, &'tcx [u32])],
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let _ = result_ty;
        self.bind_pattern(block, env, root_idx, bindings)?;
        self.expr(block, env, body)
    }

    /// Lower a guarded arm: bind the guard's pattern variables, evaluate the guard,
    /// and branch to the `success`/`failure` sub-trees with `scf.if`. The bindings
    /// stay live across both branches (the ownership pass dups them at the guard).
    #[allow(clippy::too_many_arguments)]
    fn lower_guard<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        bindings: &'tcx [(VarId, &'tcx [u32])],
        guard: &'tcx Expr<'tcx>,
        success: &'tcx DecisionTree<'tcx>,
        failure: &'tcx DecisionTree<'tcx>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        self.bind_pattern(block, env, root_idx, bindings)?;
        let cond = self
            .expr(block, env, guard)?
            .ok_or_else(|| LoweringError("match guard is unit".into()))?;
        let unit = is_unit(result_ty);
        let result_tys = if unit {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(result_ty)?]
        };
        let then_region = self.tree_branch_region(env, root_idx, success, result_ty)?;
        let else_region = self.tree_branch_region(env, root_idx, failure, result_ty)?;
        let op =
            block.append_operation(scf::r#if(cond, &result_tys, then_region, else_region, loc));
        if unit {
            Ok(None)
        } else {
            Ok(Some(op.result(0).unwrap().into()))
        }
    }

    /// Bind each `(var, path)` pattern binding into `env` by resolving the
    /// scrutinee value at `path` (its type recorded so reference-count ops keyed
    /// to the binding pick the right instruction).
    fn bind_pattern<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        root_idx: usize,
        bindings: &'tcx [(VarId, &'tcx [u32])],
    ) -> Result<()> {
        for &(var, path) in bindings {
            // An empty path goes through `resolve` like everything else: a
            // dispatch may have re-anchored the root's own path (a nullable
            // switch on the scrutinee binds its unwrapped payload there), and
            // the latest anchor must win over the raw root.
            let (val, ty) = self.resolve_loaded(block, env, root_idx, path)?;
            self.bind_var_ty(var, ty);
            env.vars.insert(var, val);
        }
        Ok(())
    }

    /// Terminate a provably-dead decision position with a poison value of the
    /// result type (or nothing for a unit match).
    fn dead_arm<'b>(
        &self,
        block: &'b Block<'c>,
        result_ty: Ty<'tcx>,
    ) -> Result<Option<Value<'c, 'b>>> {
        if is_unit(result_ty) {
            Ok(None)
        } else {
            let ty = self.tys.mlir_ty(result_ty)?;
            Ok(Some(self.append(
                block,
                builders::poison(self.context, ty, self.loc()),
            )))
        }
    }

    /// Resolve the scrutinee sub-value at `path` to a loaded SSA value and its MIR
    /// type, projecting from the nearest anchor (see [`resolve`](Self::resolve)).
    fn resolve_loaded<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
    ) -> Result<(Value<'c, 'b>, Ty<'tcx>)> {
        let cursor = self.resolve(block, env, root_idx, path)?;
        let ty = match &cursor {
            Cursor::Value { ty, .. } | Cursor::Ref { ty, .. } => *ty,
        };
        Ok((self.load_cursor(block, cursor)?, ty))
    }

    /// Turn an already-resolved enum [`Cursor`] into a `!reussir.ref` to its
    /// variant record, ready to feed `reussir.record.dispatch`: a boxed enum is
    /// borrowed, an inline `[value]` enum is spilled, and a value already reached
    /// by reference is used as-is. Returns the reference, the enum's MIR type, and
    /// the reference capability (which the arm payload references inherit).
    fn resolve_variant_ref<'b>(
        &self,
        block: &'b Block<'c>,
        cursor: Cursor<'c, 'b, 'tcx>,
    ) -> Result<(Value<'c, 'b>, Ty<'tcx>, ReussirCapability)> {
        let loc = self.loc();
        match cursor {
            Cursor::Ref { val, ty, cap } => Ok((val, ty, cap)),
            Cursor::Value { val, ty } => {
                let inner = self.tys.record_inner_of(ty)?;
                if let Some(cap) = self.record_ref_cap(ty)? {
                    let ref_ty = self.tys.ref_type_with_cap(inner, cap);
                    let reference = self.append(
                        block,
                        dialect::rc_borrow(self.context, ref_ty, val, loc).into(),
                    );
                    Ok((reference, ty, cap))
                } else {
                    // A spilled reference is always at `unspecified` capability
                    // (`reussir.ref.spilled` rejects anything else); the arm
                    // payload references inherit it.
                    let cap = ReussirCapability::Unspecified;
                    let ref_ty = self.tys.unspecified_ref_type(inner);
                    let reference = self.append(
                        block,
                        dialect::ref_spilled(self.context, ref_ty, val, loc).into(),
                    );
                    Ok((reference, ty, cap))
                }
            }
        }
    }

    /// Resolve a scrutinee `path` to a [`Cursor`] by projecting the tail of the
    /// path from the deepest anchor that is a prefix of it (most recent on a tie,
    /// so a dispatch arm's narrowed case reference shadows the enclosing enum).
    fn resolve<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        root_idx: usize,
        path: &[u32],
    ) -> Result<Cursor<'c, 'b, 'tcx>> {
        let mut best: Option<(usize, usize)> = None;
        // Only this match's anchors are candidates: `path` is relative to the
        // root at `root_idx`, while earlier anchors belong to *enclosing*
        // matches whose paths live in a different coordinate space (a nested
        // match's `[1]` must not hit an outer switch's `[1]` anchor).
        for (i, (anchor_path, _)) in env.anchors.iter().enumerate().skip(root_idx) {
            if anchor_path.len() <= path.len() && path[..anchor_path.len()] == anchor_path[..] {
                let take = best.is_none_or(|(_, len)| anchor_path.len() >= len);
                if take {
                    best = Some((i, anchor_path.len()));
                }
            }
        }
        let (index, prefix_len) =
            best.ok_or_else(|| LoweringError("no anchor covers the scrutinee path".into()))?;
        let suffix = &path[prefix_len..];
        match env.anchors[index].1 {
            Anchor::Root { val, ty } => {
                let mut cursor = Cursor::Value { val, ty };
                for &idx in suffix {
                    cursor = self.project_one(block, cursor, idx)?;
                }
                Ok(cursor)
            }
            Anchor::Case {
                reference,
                variant,
                cap,
                whole,
            } => {
                // Naming the whole switched value (empty suffix): the case payload
                // has no field for "the value itself", so hand back the enum value
                // the dispatch narrowed, exactly as the un-switched anchor would.
                let Some((first, rest)) = suffix.split_first() else {
                    return Ok(whole);
                };
                let mut cursor =
                    self.project_variant_field(block, reference, variant, cap, *first)?;
                for &idx in rest {
                    cursor = self.project_one(block, cursor, idx)?;
                }
                Ok(cursor)
            }
        }
    }

    /// Project field `idx` out of a reference to a variant case payload. Mirrors
    /// [`project_ref`](Self::project_ref) but reads the field type from the
    /// variant's field list — a case payload compound has no MIR record [`Ty`] to
    /// look a layout up in, and its fields are never `[field]` links.
    fn project_variant_field<'b>(
        &self,
        block: &'b Block<'c>,
        case_ref: Value<'c, 'b>,
        variant: &'tcx mir::VariantDef<'tcx>,
        cap: ReussirCapability,
        idx: u32,
    ) -> Result<Cursor<'c, 'b, 'tcx>> {
        let loc = self.loc();
        let field_ty = *variant
            .fields
            .get(idx as usize)
            .ok_or_else(|| LoweringError("variant field index out of range".into()))?;
        let proj = self.projected_member_of(field_ty, false, cap)?;
        let proj_ref_ty = self.tys.ref_type_with_cap(proj.elem_ty, proj.proj_cap);
        let index = IntegerAttribute::new(Type::index(self.context), i64::from(idx));
        let projected = self.append(
            block,
            dialect::ref_project(self.context, proj_ref_ty, case_ref, index, loc).into(),
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
        env: &mut Env<'c, 'b, 'tcx>,
        base: &Expr<'tcx>,
        path: &[u32],
    ) -> Result<Value<'c, 'b>> {
        let base_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("projection base is unit".into()))?;
        // A borrow-projection through a *temporary* rr base (`foo(x).field`, not
        // `xs.field`) leaves the base owned by no variable; the ownership pass
        // frees it with a `DropValue` keyed to this `Proj` node, which resolves
        // the base's value through `env.temps`. Record it so that drop can land.
        // A `Var` root is dropped via `env.vars`, and a nested `Proj` root is
        // registered by its own recursive lowering, so neither needs this.
        if !matches!(base.kind, ExprKind::Var(_) | ExprKind::Proj(..)) {
            self.remember_temp(env, base, base_val);
        }
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
        self.projected_member_of(field_ty, is_field, ref_cap)
    }

    /// The projection of a member given its MIR type and link flag directly, the
    /// core of [`projected_member`](Self::projected_member). Split out so a
    /// variant case payload — which has no MIR record [`Ty`] to look a layout up
    /// in, only a [`VariantDef`](mir::VariantDef) field list — can project its
    /// fields the same way (a case field is never a `[field]` link).
    fn projected_member_of(
        &self,
        field_ty: Ty<'tcx>,
        is_field: bool,
        ref_cap: ReussirCapability,
    ) -> Result<ProjectedMember<'c, 'tcx>> {
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
            // A non-`[field]` regional member is a frozen link: the dialect
            // projects it to `rc<inner, rigid>` unconditionally
            // (`getProjectedType`, lib/IR/ReussirTypes.cpp) — exactly the
            // shared-member shape, at `rigid` capability. The record *layout*
            // keeps the member's unrefined `Regional` coloring, so refine the
            // loaded cursor's type to `Rigid` here, matching the coloring
            // elaboration gives the projection expression itself.
            let TyKind::Record { def, args, .. } = *field_ty.kind() else {
                return err("regional member is not a record type");
            };
            let rigid_ty = self.tcx.mk_record(def, args, Flexivity::Rigid);
            let inner = self.tys.record_inner_of(field_ty)?;
            Ok(ProjectedMember {
                field_ty: rigid_ty,
                elem_ty: self.tys.rc_type_with_cap(inner, ReussirCapability::Rigid),
                proj_cap: ref_cap,
                is_link: true,
            })
        } else if matches!(field_ty.kind(), TyKind::Cell { .. }) {
            // A cell member is stored as a bare payload type in the record
            // declaration, but dialect projection materializes a shared rc link.
            Ok(ProjectedMember {
                field_ty,
                elem_ty: self.tys.mlir_ty(field_ty)?,
                proj_cap: ref_cap,
                is_link: true,
            })
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
    /// `id`. A `dup`/`drop` targets a named local; a `dup-value`/`drop-value`
    /// targets a live temporary held in [`Env::temps`] — the `match` scrutinee
    /// whose per-arm settlement (its drop, placed before each leaf body) is keyed
    /// to the temporary's own id rather than to a variable.
    fn emit_rc_before<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        id: ExprId,
    ) -> Result<()> {
        let ops = self.ownership.borrow().before(id).to_vec();
        for op in ops {
            match op {
                RcOp::Dup(v) => self.emit_inc_var(block, env, v)?,
                RcOp::Drop(v) => self.emit_dec_var(block, env, v)?,
                RcOp::DupValue(target) => self.emit_inc_temp(block, env, target)?,
                RcOp::DropValue(target) => self.emit_dec_temp(block, env, target)?,
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
        env: &Env<'c, 'b, 'tcx>,
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
                RcOp::DupValue(target) => self.emit_inc_temp(block, env, target)?,
                RcOp::DropValue(target) => self.emit_dec_temp(block, env, target)?,
            }
        }
        Ok(())
    }

    /// Record a temporary's value and type under its [`ExprId`] so a value-keyed
    /// reference-count op the ownership pass anchored to a *later* node can find
    /// it. A deferred `DropValue`/`DupValue` (the `match` scrutinee's per-arm
    /// drop, or a projection base freed after the field is borrowed out) fires
    /// in [`emit_rc_after`](Self::emit_rc_after) of the node it is keyed to, not
    /// of the temporary itself, so the value must outlive its own lowering; this
    /// stashes it. Entries live for the rest of the (sub)scope — the ids are
    /// unique, so a stale entry is never mislooked-up, and subscope truncation
    /// reclaims them.
    fn remember_temp<'b>(&self, env: &mut Env<'c, 'b, 'tcx>, e: &Expr<'tcx>, val: Value<'c, 'b>) {
        self.temp_tys.borrow_mut().insert(e.id, e.ty);
        env.temps.insert(e.id, val);
    }

    /// Increment the refcount of the temporary held in [`Env::temps`] under `id`
    /// (see [`emit_rc_before`](Self::emit_rc_before)); an unknown id is a genuine
    /// gap — a value op keyed to a temporary the walk did not record.
    fn emit_inc_temp<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        id: ExprId,
    ) -> Result<()> {
        match (env.temps.get(&id).copied(), self.temp_ty(id)) {
            (Some(val), Some(ty)) => self.emit_inc(block, val, ty),
            _ => err("reference-count op on an unbound temporary is not supported"),
        }
    }

    /// Decrement the refcount of the temporary held in [`Env::temps`] under `id`
    /// (the dual of [`emit_inc_temp`](Self::emit_inc_temp)).
    fn emit_dec_temp<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        id: ExprId,
    ) -> Result<()> {
        match (env.temps.get(&id).copied(), self.temp_ty(id)) {
            (Some(val), Some(ty)) => self.emit_dec(block, val, ty),
            _ => err("reference-count op on an unbound temporary is not supported"),
        }
    }

    /// Increment the refcount owned by local `v`. A value-less local (e.g. unit)
    /// holds no resource, so there is nothing to do.
    fn emit_inc_var<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        v: VarId,
    ) -> Result<()> {
        match (env.vars.get(&v).copied(), self.var_ty(v)) {
            (Some(val), Some(ty)) => self.emit_inc(block, val, ty),
            _ => Ok(()),
        }
    }

    /// Decrement the refcount owned by local `v` (see [`emit_inc_var`](Self::emit_inc_var)).
    fn emit_dec_var<'b>(
        &self,
        block: &'b Block<'c>,
        env: &Env<'c, 'b, 'tcx>,
        v: VarId,
    ) -> Result<()> {
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
        // A managed `rc` pointer — a shared record, a rigid regional record, or a
        // closure — is bumped directly; an inline `[value]` record is acquired
        // through a reference (which reaches the rc pointers inside it).
        if self.tys.is_managed_rc(ty) {
            block.append_operation(dialect::rc_inc(self.context, val, loc).into());
            Ok(())
        } else if matches!(ty.kind(), TyKind::Record { .. }) {
            let reference = self.spill(block, val, ty)?;
            block.append_operation(dialect::ref_acquire(self.context, reference, loc).into());
            Ok(())
        } else if let TyKind::Nullable(inner) = *ty.kind() {
            self.emit_rc_nullable(block, val, inner, true)
        } else {
            err("reference-count increment on an unsupported type")
        }
    }

    /// Decrement a value's refcount, the dual of [`emit_inc`](Self::emit_inc): an
    /// `rc` pointer is decremented directly, an inline record is dropped through a
    /// reference (releasing every rc pointer it reaches).
    fn emit_dec<'b>(&self, block: &'b Block<'c>, val: Value<'c, 'b>, ty: Ty<'tcx>) -> Result<()> {
        let loc = self.loc();
        // The dual of [`emit_inc`](Self::emit_inc): a managed `rc` pointer (shared
        // record, rigid regional record, or closure) is decremented directly, an
        // inline `[value]` record dropped through a reference.
        if self.tys.is_managed_rc(ty) {
            block.append_operation(dialect::rc_dec(self.context, val, loc).into());
            Ok(())
        } else if matches!(ty.kind(), TyKind::Record { .. }) {
            let reference = self.spill(block, val, ty)?;
            block.append_operation(dialect::ref_drop(self.context, reference, loc).into());
            Ok(())
        } else if let TyKind::Nullable(inner) = *ty.kind() {
            self.emit_rc_nullable(block, val, inner, false)
        } else {
            err("reference-count decrement on an unsupported type")
        }
    }

    /// Null-guarded refcount op on a `Nullable<inner>` value: `rc.inc`/`rc.dec`
    /// reject a nullable operand and do not null-check, so the pointer is
    /// checked, unwrapped (`nullable.coerce`), and counted only when non-null —
    /// a null link is a no-op, never a deallocation (issue 213).
    fn emit_rc_nullable<'b>(
        &self,
        block: &'b Block<'c>,
        val: Value<'c, 'b>,
        inner: Ty<'tcx>,
        inc: bool,
    ) -> Result<()> {
        let loc = self.loc();
        let i1 = IntegerType::new(self.context, 1).into();
        let flag = self.append(
            block,
            dialect::nullable_check(self.context, i1, val, loc).into(),
        );
        let then_block = Block::new(&[]);
        let inner_mlir = self.tys.mlir_ty(inner)?;
        let unwrapped = self.append(
            &then_block,
            dialect::nullable_coerce(self.context, inner_mlir, val, loc).into(),
        );
        if inc {
            self.emit_inc(&then_block, unwrapped, inner)?;
        } else {
            self.emit_dec(&then_block, unwrapped, inner)?;
        }
        then_block.append_operation(scf::r#yield(&[], loc));
        let then_region = Region::new();
        then_region.append_block(then_block);
        let else_block = Block::new(&[]);
        else_block.append_operation(scf::r#yield(&[], loc));
        let else_region = Region::new();
        else_region.append_block(else_block);
        block.append_operation(scf::r#if(flag, &[], then_region, else_region, loc));
        Ok(())
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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
        env: &mut Env<'c, 'b, 'tcx>,
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

    // ----- built-in cell operations -----

    fn lower_cell_intrinsic<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        func: CellFn,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        match func {
            CellFn::Alloc => {
                let value = self
                    .expr(block, env, &args[0])?
                    .ok_or_else(|| LoweringError("cell element produced no value".into()))?;
                let result_ty = self.tys.mlir_ty(e.ty)?;
                Ok(Some(self.append(
                    block,
                    builders::cell_create(value, result_ty, loc),
                )))
            }
            CellFn::Get => {
                let base = &args[0];
                let cell = self
                    .expr(block, env, base)?
                    .ok_or_else(|| LoweringError("cell operand produced no value".into()))?;
                self.remember_temp(env, base, cell);
                let result_ty = self.tys.mlir_ty(e.ty)?;
                Ok(Some(
                    self.append(block, builders::cell_get(cell, result_ty, loc)),
                ))
            }
            CellFn::Set => {
                let base = &args[0];
                let cell = self
                    .expr(block, env, base)?
                    .ok_or_else(|| LoweringError("cell operand produced no value".into()))?;
                self.remember_temp(env, base, cell);
                let value = self
                    .expr(block, env, &args[1])?
                    .ok_or_else(|| LoweringError("cell element produced no value".into()))?;
                block.append_operation(builders::cell_set(value, cell, loc));
                Ok(None)
            }
            CellFn::InUse => {
                let base = &args[0];
                if !matches!(
                    *base.ty.kind(),
                    TyKind::Cell {
                        exclusive: true,
                        ..
                    }
                ) {
                    return err("cell in_use operand is not an exclusive cell");
                }
                let cell = self
                    .expr(block, env, base)?
                    .ok_or_else(|| LoweringError("cell operand produced no value".into()))?;
                self.remember_temp(env, base, cell);
                let result_ty = self.tys.mlir_ty(e.ty)?;
                Ok(Some(self.append(
                    block,
                    builders::cell_in_use(cell, result_ty, loc),
                )))
            }
            CellFn::Rmw => {
                let base = &args[0];
                let cell = self
                    .expr(block, env, base)?
                    .ok_or_else(|| LoweringError("cell operand produced no value".into()))?;
                self.remember_temp(env, base, cell);
                let kernel = &args[1];
                let closure = self
                    .expr(block, env, kernel)?
                    .ok_or_else(|| LoweringError("cell update closure produced no value".into()))?;
                let TyKind::Cell {
                    elem,
                    exclusive: true,
                } = *base.ty.kind()
                else {
                    return err("cell RMW operand is not an exclusive cell");
                };
                let elem_ty = self.tys.mlir_ty(elem)?;
                let body = Block::new(&[(elem_ty, loc)]);
                let current = body
                    .argument(0)
                    .expect("cell RMW body takes the current element")
                    .into();
                // Preserve the updater's outer owner while eval consumes one
                // reference. Structural beta reduction removes this pair for a
                // literal closure nested in cell.rmw.
                body.append_operation(dialect::rc_inc(self.context, closure, loc).into());
                let replacement = self
                    .call_kernel(&body, closure, kernel.ty, &[current])?
                    .ok_or_else(|| LoweringError("cell update closure returned unit".into()))?;
                body.append_operation(builders::cell_yield(replacement, None, loc));
                let region = Region::new();
                region.append_block(body);
                block.append_operation(builders::cell_rmw(cell, None, region, loc));
                // The intrinsic consumes the updater closure; the in-region inc
                // paid only for the eval above.
                block.append_operation(dialect::rc_dec(self.context, closure, loc).into());
                Ok(None)
            }
        }
    }

    // ----- built-in array operations (issue #344) -----

    /// Lower an [`ArrayOp`](ExprKind::ArrayOp). The shapes:
    ///
    /// * `get` — borrow + view + a bounds-guarded `memref.load`;
    /// * `set` — a bounds-guarded `array.with_unique_view` storing the
    ///   element (the empty-yield implicit result is the updated array);
    /// * `splat` — a fresh `rc.create` over a poison payload (uniquely owned
    ///   by construction, so a plain borrow + view is a mutable view), filled
    ///   by a check-free `scf.for` nest (the loops iterate exactly the
    ///   extents);
    /// * `tabulate` — the same fresh box, each element computed by calling
    ///   the kernel closure;
    /// * `fold` — borrow + view and an `scf.for` nest threading the
    ///   accumulator through `iter_args`, the kernel closure called per
    ///   element.
    ///
    /// The kernel closure is called through the frontend's standard
    /// per-call chain — a per-iteration `rc.inc` pre-paying the consume,
    /// then `uniqify → apply… → eval` — which is exactly the loop-carried
    /// redex the closure beta reduction inlines at `-Oaggressive`: a literal
    /// lambda kernel fuses into the nest, leaving check-free loops with no
    /// closure calls and no per-iteration rc traffic.
    fn lower_array_op<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        op: ArrayFn,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Option<Value<'c, 'b>>> {
        match op {
            ArrayFn::Get => self.array_get(block, env, e, args).map(Some),
            ArrayFn::Set => self.array_set(block, env, args).map(Some),
            ArrayFn::Splat => self.array_splat(block, env, e, args).map(Some),
            ArrayFn::Tabulate => self.array_tabulate(block, env, e, args).map(Some),
            ArrayFn::Fold => self.array_fold(block, env, e, args).map(Some),
        }
    }

    /// The extents of an array-typed MIR type.
    fn array_dims(ty: Ty<'tcx>) -> Result<&'tcx [u64]> {
        match *ty.kind() {
            TyKind::Array { dims, .. } => Ok(dims),
            _ => err("array operation on a non-array type"),
        }
    }

    /// The `memref<dims x elem>` view type of an array-typed `ty`.
    fn memref_of(&self, ty: Ty<'tcx>) -> Result<Type<'c>> {
        let TyKind::Array { elem, dims } = *ty.kind() else {
            return err("array view requested for a non-array type");
        };
        let elem = self.tys.mlir_ty(elem)?;
        let shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        Ok(MemRefType::new(elem, &shape, None, None).into())
    }

    /// Borrow an rc array and materialize the memref view of its payload.
    fn array_view_of<'b>(
        &self,
        block: &'b Block<'c>,
        rc_val: Value<'c, 'b>,
        arr_ty: Ty<'tcx>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let inner = self.tys.array_inner_of(arr_ty)?;
        let ref_ty = self.tys.unspecified_ref_type(inner);
        let borrowed = self.append(
            block,
            dialect::rc_borrow(self.context, ref_ty, rc_val, loc).into(),
        );
        let view_ty = self.memref_of(arr_ty)?;
        Ok(self.append(
            block,
            dialect::array_view(self.context, view_ty, borrowed, loc).into(),
        ))
    }

    /// Lower `i64` index expressions and bounds-check them against `dims`:
    /// each index is cast to `index` (a negative value wraps to a huge
    /// unsigned one) and compared `ult` against its extent, the verdicts
    /// and-reduced into one guard. The caller sinks the access into the
    /// guarded branch ([`Self::guarded`]); statically provable checks fold
    /// away downstream (the comparisons are against constant extents).
    fn checked_indices<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        idx_exprs: &'tcx [Expr<'tcx>],
        dims: &[u64],
    ) -> Result<(Vec<Value<'c, 'b>>, Value<'c, 'b>)> {
        let loc = self.loc();
        let index_ty = Type::index(self.context);
        let mut idxs = Vec::with_capacity(idx_exprs.len());
        let mut ok: Option<Value<'c, 'b>> = None;
        for (ie, &d) in idx_exprs.iter().zip(dims) {
            let raw = self
                .expr(block, env, ie)?
                .ok_or_else(|| LoweringError("array index is unit".into()))?;
            let idx = self.append(block, arith::index_cast(raw, index_ty, loc));
            let ext = self.index_const(block, d as i64);
            let in_range = self.append(
                block,
                arith::cmpi(self.context, CmpiPredicate::Ult, idx, ext, loc),
            );
            ok = Some(match ok {
                None => in_range,
                Some(acc) => self.append(block, arith::andi(acc, in_range, loc)),
            });
            idxs.push(idx);
        }
        let ok = ok.ok_or_else(|| LoweringError("array access without indices".into()))?;
        Ok((idxs, ok))
    }

    /// An access guarded by a bounds verdict: an `scf.if` whose then-branch
    /// is the access itself (built by `body`, yielding the result) and whose
    /// else-branch panics — yielding a poison of the result type, since the
    /// branch must still produce one past the (aborting) panic.
    fn guarded<'b>(
        &self,
        block: &'b Block<'c>,
        ok: Value<'c, 'b>,
        result_ty: Type<'c>,
        body: impl FnOnce(&Block<'c>) -> Result<()>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let then_block = Block::new(&[]);
        body(&then_block)?;
        let then_region = Region::new();
        then_region.append_block(then_block);
        let else_block = Block::new(&[]);
        else_block.append_operation(
            dialect::panic(
                self.context,
                StringAttribute::new(self.context, "array index out of bounds"),
                loc,
            )
            .into(),
        );
        let poison = else_block
            .append_operation(builders::poison(self.context, result_ty, loc))
            .result(0)
            .expect("poison result")
            .into();
        else_block.append_operation(scf::r#yield(&[poison], loc));
        let else_region = Region::new();
        else_region.append_block(else_block);
        let guard =
            block.append_operation(scf::r#if(ok, &[result_ty], then_region, else_region, loc));
        Ok(guard.result(0).expect("guarded access result").into())
    }

    fn array_get<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let base = &args[0];
        let dims = Self::array_dims(base.ty)?;
        let rc_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("array base is unit".into()))?;
        // A temporary base (e.g. `tabulate(…).get(i)`) is released *after*
        // this node by a value-keyed drop; stash it where that op looks.
        self.remember_temp(env, base, rc_val);
        let view = self.array_view_of(block, rc_val, base.ty)?;
        let (idxs, ok) = self.checked_indices(block, env, &args[1..], dims)?;
        let elem_ty = self.tys.mlir_ty(e.ty)?;
        self.guarded(block, ok, elem_ty, |then| {
            let loaded = then
                .append_operation(memref::load(view, &idxs, loc))
                .result(0)
                .expect("load result")
                .into();
            then.append_operation(scf::r#yield(&[loaded], loc));
            Ok(())
        })
    }

    fn array_set<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let base = &args[0];
        let dims = Self::array_dims(base.ty)?;
        let rank = dims.len();
        let rc_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("array base is unit".into()))?;
        let (idxs, ok) = self.checked_indices(block, env, &args[1..=rank], dims)?;
        let value = self
            .expr(block, env, &args[rank + 1])?
            .ok_or_else(|| LoweringError("array element is unit".into()))?;
        let memref_ty = self.memref_of(base.ty)?;
        let rc_ty = self.tys.mlir_ty(base.ty)?;
        self.guarded(block, ok, rc_ty, |then| {
            let body = Block::new(&[(memref_ty, loc)]);
            let view = body
                .argument(0)
                .expect("with_unique_view body takes the view")
                .into();
            body.append_operation(memref::store(value, view, &idxs, loc));
            body.append_operation(builders::scf_yield(None, loc));
            let region = Region::new();
            region.append_block(body);
            let updated = then
                .append_operation(builders::array_with_unique_view(rc_val, rc_ty, region, loc))
                .result(0)
                .expect("with_unique_view result")
                .into();
            then.append_operation(scf::r#yield(&[updated], loc));
            Ok(())
        })
    }
    /// A fresh, uniquely-owned rc array over an uninitialized (poison)
    /// payload; the caller fills every element before the value escapes.
    fn fresh_array<'b>(&self, block: &'b Block<'c>, arr_ty: Ty<'tcx>) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let inner = self.tys.array_inner_of(arr_ty)?;
        let rc_ty = self.tys.mlir_ty(arr_ty)?;
        let poison = self.append(block, builders::poison(self.context, inner, loc));
        Ok(self.append(block, builders::rc_create(self.context, poison, rc_ty, loc)))
    }

    /// Call an already-lowered kernel closure value with the given argument
    /// values: the frontend's standard chain — `uniqify → apply… → eval` —
    /// minus the pre-paying `rc.inc`, which the caller emits (per iteration
    /// for a loop). Returns the result value (`None` for a unit result).
    fn call_kernel<'b>(
        &self,
        block: &'b Block<'c>,
        closure: Value<'c, 'b>,
        kernel_ty: Ty<'tcx>,
        args: &[Value<'c, 'b>],
    ) -> Result<Option<Value<'c, 'b>>> {
        let loc = self.loc();
        let (params, ret) = match kernel_ty.kind() {
            TyKind::Closure { params, ret } => (*params, *ret),
            _ => return err("array kernel is not a closure"),
        };
        let target_ty = self.tys.mlir_ty(kernel_ty)?;
        let mut chain = self.append(block, builders::closure_uniqify(closure, target_ty, loc));
        for (i, &arg) in args.iter().enumerate() {
            let applied_ty = self.tys.closure_type(&params[i + 1..], ret)?;
            chain = self.append(block, builders::closure_apply(arg, chain, applied_ty, loc));
        }
        let result_ty = if is_unit(ret) {
            None
        } else {
            Some(self.tys.mlir_ty(ret)?)
        };
        let op = block.append_operation(builders::closure_eval(chain, result_ty, loc));
        Ok(result_ty.map(|_| op.result(0).expect("eval result").into()))
    }

    fn array_splat<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let value = self
            .expr(block, env, &args[0])?
            .ok_or_else(|| LoweringError("array element is unit".into()))?;
        let rc_val = self.fresh_array(block, e.ty)?;
        let view = self.array_view_of(block, rc_val, e.ty)?;
        let dims = Self::array_dims(e.ty)?;
        self.splat_nest(block, view, value, dims, Vec::new())?;
        Ok(rc_val)
    }

    /// The `scf.for` nest of `splat`: store `value` at every index tuple.
    fn splat_nest<'b>(
        &self,
        block: &'b Block<'c>,
        view: Value<'c, 'b>,
        value: Value<'c, 'b>,
        dims: &[u64],
        ivs: Vec<Value<'c, 'b>>,
    ) -> Result<()> {
        let loc = self.loc();
        if dims.is_empty() {
            block.append_operation(memref::store(value, view, &ivs, loc));
            return Ok(());
        }
        let lb = self.index_const(block, 0);
        let ub = self.index_const(block, dims[0] as i64);
        let step = self.index_const(block, 1);
        let inner = Block::new(&[(Type::index(self.context), loc)]);
        {
            let iv = inner.argument(0).expect("loop induction variable").into();
            let mut ivs2: Vec<Value<'c, '_>> = ivs.iter().copied().collect();
            ivs2.push(iv);
            self.splat_nest(&inner, view, value, &dims[1..], ivs2)?;
            inner.append_operation(scf::r#yield(&[], loc));
        }
        let region = Region::new();
        region.append_block(inner);
        block.append_operation(scf::r#for(lb, ub, step, region, loc));
        Ok(())
    }

    fn array_tabulate<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let kernel = &args[0];
        let closure = self
            .expr(block, env, kernel)?
            .ok_or_else(|| LoweringError("array kernel is unit".into()))?;
        let rc_val = self.fresh_array(block, e.ty)?;
        let view = self.array_view_of(block, rc_val, e.ty)?;
        let dims = Self::array_dims(e.ty)?;
        self.tabulate_nest(block, view, closure, kernel.ty, dims, Vec::new())?;
        // The op consumed the kernel operand; the per-iteration incs paid
        // for the evals, so one release remains.
        block.append_operation(dialect::rc_dec(self.context, closure, loc).into());
        Ok(rc_val)
    }

    /// The `scf.for` nest of `tabulate`: at each index tuple, call the
    /// kernel on the (i64-cast) indices and store the element.
    fn tabulate_nest<'b>(
        &self,
        block: &'b Block<'c>,
        view: Value<'c, 'b>,
        closure: Value<'c, 'b>,
        kernel_ty: Ty<'tcx>,
        dims: &[u64],
        ivs: Vec<Value<'c, 'b>>,
    ) -> Result<()> {
        let loc = self.loc();
        if dims.is_empty() {
            block.append_operation(dialect::rc_inc(self.context, closure, loc).into());
            let i64_ty = IntegerType::new(self.context, 64).into();
            let idxs: Vec<Value<'c, 'b>> = ivs
                .iter()
                .map(|&iv| self.append(block, arith::index_cast(iv, i64_ty, loc)))
                .collect();
            let elem = self
                .call_kernel(block, closure, kernel_ty, &idxs)?
                .ok_or_else(|| LoweringError("array kernel yields unit".into()))?;
            block.append_operation(memref::store(elem, view, &ivs, loc));
            return Ok(());
        }
        let lb = self.index_const(block, 0);
        let ub = self.index_const(block, dims[0] as i64);
        let step = self.index_const(block, 1);
        let inner = Block::new(&[(Type::index(self.context), loc)]);
        {
            let iv = inner.argument(0).expect("loop induction variable").into();
            let mut ivs2: Vec<Value<'c, '_>> = ivs.iter().copied().collect();
            ivs2.push(iv);
            self.tabulate_nest(&inner, view, closure, kernel_ty, &dims[1..], ivs2)?;
            inner.append_operation(scf::r#yield(&[], loc));
        }
        let region = Region::new();
        region.append_block(inner);
        block.append_operation(scf::r#for(lb, ub, step, region, loc));
        Ok(())
    }

    fn array_fold<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b, 'tcx>,
        e: &Expr<'tcx>,
        args: &'tcx [Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let base = &args[0];
        let dims = Self::array_dims(base.ty)?;
        let rc_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("array base is unit".into()))?;
        // A temporary base is released after the op by a value-keyed drop.
        self.remember_temp(env, base, rc_val);
        let view = self.array_view_of(block, rc_val, base.ty)?;
        let init = self
            .expr(block, env, &args[1])?
            .ok_or_else(|| LoweringError("fold accumulator is unit".into()))?;
        let kernel = &args[2];
        let closure = self
            .expr(block, env, kernel)?
            .ok_or_else(|| LoweringError("array kernel is unit".into()))?;
        let acc_ty = self.tys.mlir_ty(e.ty)?;
        let result = self.fold_nest(
            block,
            view,
            closure,
            kernel.ty,
            acc_ty,
            dims,
            Vec::new(),
            init,
        )?;
        block.append_operation(dialect::rc_dec(self.context, closure, loc).into());
        Ok(result)
    }

    /// The `scf.for` nest of `fold`: the accumulator threads through
    /// `iter_args` at every level; innermost, the kernel is called on
    /// `(acc, element)`.
    #[allow(clippy::too_many_arguments)]
    fn fold_nest<'b>(
        &self,
        block: &'b Block<'c>,
        view: Value<'c, 'b>,
        closure: Value<'c, 'b>,
        kernel_ty: Ty<'tcx>,
        acc_ty: Type<'c>,
        dims: &[u64],
        ivs: Vec<Value<'c, 'b>>,
        acc: Value<'c, 'b>,
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        if dims.is_empty() {
            block.append_operation(dialect::rc_inc(self.context, closure, loc).into());
            let x = self.append(block, memref::load(view, &ivs, loc));
            return self
                .call_kernel(block, closure, kernel_ty, &[acc, x])?
                .ok_or_else(|| LoweringError("fold kernel yields unit".into()));
        }
        let lb = self.index_const(block, 0);
        let ub = self.index_const(block, dims[0] as i64);
        let step = self.index_const(block, 1);
        let inner = Block::new(&[(Type::index(self.context), loc), (acc_ty, loc)]);
        {
            let iv = inner.argument(0).expect("loop induction variable").into();
            let carried = inner.argument(1).expect("loop-carried accumulator").into();
            let mut ivs2: Vec<Value<'c, '_>> = ivs.iter().copied().collect();
            ivs2.push(iv);
            let next = self.fold_nest(
                &inner,
                view,
                closure,
                kernel_ty,
                acc_ty,
                &dims[1..],
                ivs2,
                carried,
            )?;
            inner.append_operation(scf::r#yield(&[next], loc));
        }
        let region = Region::new();
        region.append_block(inner);
        let op = block.append_operation(
            reussir_backend::melior::dialect::ods::scf::r#for(
                self.context,
                &[acc_ty],
                lb,
                ub,
                step,
                &[acc],
                region,
                loc,
            )
            .into(),
        );
        Ok(op.result(0).expect("loop-carried result").into())
    }
}
