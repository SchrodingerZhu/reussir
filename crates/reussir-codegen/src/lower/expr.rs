//! Function and expression lowering: the per-function recursive tree-walk that
//! emits MLIR ops for each Full MIR expression.
//!
//! # SSA environment
//!
//! The variable environment maps each [`VarId`] to the melior [`Value`] that
//! holds it. melior values borrow the block they were built in, so an `scf.if`
//! branch — which builds into its own block — gets a fresh environment copied
//! from its parent (outer values outlive the inner block, so the copy is a plain
//! lifetime-shortening reborrow). Bindings introduced inside a branch stay
//! scoped to it.

use rustc_hash::FxHashMap;

use reussir_backend::builders;
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
use reussir_core::semi::ctxt::DefaultCap;
use reussir_core::semi::hir::{ArithOp, CmpOp, VarId};
use reussir_core::semi::ty::{IntTy, Ty, TyKind};
use reussir_core::surface::Visibility;

use super::ty::{TypeCtx, is_unit, num_class};
use super::{LoweringError, Result, err};

/// The variable → SSA-value environment for one block scope.
type Env<'c, 'b> = FxHashMap<VarId, Value<'c, 'b>>;

/// Per-program lowering state: the context to build in, the program whose symbol
/// table resolves callee/trampoline names, and the [`TypeCtx`] that lowers types
/// and resolves record layouts. Reused across functions.
pub(super) struct Lowerer<'c, 'p, 'tcx> {
    context: &'c Context,
    program: &'p mir::Program<'tcx>,
    tys: TypeCtx<'c, 'p, 'tcx>,
}

impl<'c, 'p, 'tcx> Lowerer<'c, 'p, 'tcx> {
    pub(super) fn new(context: &'c Context, program: &'p mir::Program<'tcx>) -> Self {
        Lowerer {
            context,
            program,
            tys: TypeCtx::new(context, program),
        }
    }

    fn loc(&self) -> Location<'c> {
        Location::unknown(self.context)
    }

    /// Lower one MIR function to a `func.func` operation.
    pub(super) fn function(&self, func: &mir::Function<'tcx>) -> Result<Operation<'c>> {
        let loc = self.loc();
        let param_tys = func
            .params
            .iter()
            .map(|p| self.tys.mlir_ty(p.ty))
            .collect::<Result<Vec<_>>>()?;
        let ret = func.return_ty;
        let result_tys = if is_unit(ret) {
            Vec::new()
        } else {
            vec![self.tys.mlir_ty(ret)?]
        };
        let fn_ty = FunctionType::new(self.context, &param_tys, &result_tys);

        let region = Region::new();
        if let Some(body) = func.body {
            let block_args: Vec<(Type<'c>, Location<'c>)> =
                param_tys.iter().map(|t| (*t, loc)).collect();
            let block = Block::new(&block_args);
            let mut env: Env<'c, '_> = FxHashMap::default();
            for (i, p) in func.params.iter().enumerate() {
                let arg = block
                    .argument(i)
                    .map_err(|e| LoweringError(format!("missing block argument: {e}").into()))?;
                env.insert(p.var, arg.into());
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
        let attributes = if func.visibility == Visibility::Private {
            vec![(
                Identifier::new(self.context, "sym_visibility"),
                StringAttribute::new(self.context, "private").into(),
            )]
        } else {
            Vec::new()
        };

        Ok(func::func(
            self.context,
            StringAttribute::new(self.context, self.program.symbol(func.symbol)),
            TypeAttribute::new(fn_ty.into()),
            region,
            &attributes,
            loc,
        ))
    }

    /// Lower an expression into `block`, returning the SSA value holding its
    /// result (`None` for a unit-typed expression).
    fn expr<'b>(
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
                    env.get(v)
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
            Let { var, value, .. } => {
                if let Some(v) = self.expr(block, env, value)? {
                    env.insert(*var, v);
                }
                Ok(None)
            }
            Call { callee, args, .. } => {
                let mut operands = Vec::with_capacity(args.len());
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
            RegionRun(_) => err("region-run lowering not yet implemented"),
            Proj(base, path) => self.proj(block, env, base, path).map(Some),
            Assign(..) => err("assignment lowering not yet implemented"),
            Match(..) => err("match lowering not yet implemented"),
            Ctor { args, .. } => self.compound(block, env, e, args).map(Some),
            Variant { .. } | NullableCall(_) => err("enum/nullable lowering not yet implemented"),
            Closure(_) | ClosureCall { .. } => err("closure lowering not yet implemented"),
            GlobalStr(_) => err("string literal lowering not yet implemented"),
            Poison => err("poison expression reached lowering"),
        }
    }

    /// Construct a `[value]` record from its (declaration-ordered) field args via
    /// `reussir.record.compound`.
    fn compound<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        e: &Expr<'tcx>,
        args: &[Expr<'tcx>],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        // `record_type` also enforces the value-only restriction.
        let record_ty = self.tys.record_type(e.ty)?;
        let mut operands = Vec::with_capacity(args.len());
        for a in args.iter() {
            operands.push(
                self.expr(block, env, a)?
                    .ok_or_else(|| LoweringError("record field is unit".into()))?,
            );
        }
        Ok(self.append(block, builders::record_compound(&operands, record_ty, loc)))
    }

    /// Project a chain of fields out of a `[value]` record value with a sequence
    /// of `reussir.record.extract` ops.
    fn proj<'b>(
        &self,
        block: &'b Block<'c>,
        env: &mut Env<'c, 'b>,
        base: &Expr<'tcx>,
        path: &[u32],
    ) -> Result<Value<'c, 'b>> {
        let loc = self.loc();
        let mut cur_val = self
            .expr(block, env, base)?
            .ok_or_else(|| LoweringError("projection base is unit".into()))?;
        let mut cur_ty = base.ty;
        for &idx in path.iter() {
            let rec = self
                .tys
                .record_of(cur_ty)
                .ok_or_else(|| LoweringError("projection base is not a record".into()))?;
            if rec.default_cap != DefaultCap::Value {
                return err("projection of shared/regional records not yet implemented");
            }
            let members = match rec.layout {
                mir::RecordLayout::Compound(ms) => ms,
                mir::RecordLayout::Variant(_) => return err("projection of an enum value"),
            };
            let field = members
                .get(idx as usize)
                .ok_or_else(|| LoweringError(format!("field index {idx} out of range").into()))?;
            let field_ty = field.ty;
            let field_mlir = self.tys.mlir_ty(field_ty)?;
            let op = builders::record_extract(self.context, cur_val, idx as usize, field_mlir, loc);
            cur_val = self.append(block, op);
            cur_ty = field_ty;
        }
        Ok(cur_val)
    }

    /// Append `op` to `block` and return its single result as a value.
    fn append<'b>(&self, block: &'b Block<'c>, op: Operation<'c>) -> Value<'c, 'b> {
        block.append_operation(op).result(0).unwrap().into()
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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

    /// Build one `scf.if` branch region: an entry block whose body lowers `e`
    /// (over a copy of the enclosing environment) and terminates with
    /// `scf.yield` (yielding its value unless unit-typed).
    fn branch_region<'b>(
        &self,
        env: &Env<'c, 'b>,
        e: &Expr<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<Region<'c>> {
        let loc = self.loc();
        let block = Block::new(&[]);
        // Reborrow the outer values at the inner block's (shorter) lifetime;
        // bindings made in the branch stay local to this copy.
        let mut child: Env<'c, '_> = FxHashMap::default();
        for (k, v) in env {
            child.insert(*k, *v);
        }
        let value = self.expr(&block, &mut child, e)?;
        if is_unit(ty) {
            block.append_operation(scf::r#yield(&[], loc));
        } else {
            let v = value.ok_or_else(|| LoweringError("if branch produced no value".into()))?;
            block.append_operation(scf::r#yield(&[v], loc));
        }
        let region = Region::new();
        region.append_block(block);
        Ok(region)
    }
}
