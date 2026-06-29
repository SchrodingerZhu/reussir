//! Debug-info emission: building the Reussir `DBG*` attributes from the MIR and
//! attaching them so the backend's debug-info conversion pass can emit DWARF.
//!
//! The attributes ride on locations: a function's `func.func` is given a
//! `FusedLoc` whose metadata is a [`dbg::subprogram`] attribute, and its
//! parameters are described by a `reussir.dbg_func_args` array attribute the pass
//! reads. A variable's *type* is built precisely from its ground MIR type
//! ([`dbg_type`](Lowerer::dbg_type)).
//!
//! Only what the conversion pass already handles is emitted: scalars and
//! by-value records. Reference-counted (`[shared]`) records are boxed and need a
//! pointer type plus a `DIExpression`; they are skipped here until that lands.
//! Local (`let`) variables are likewise not emitted yet — they require attaching
//! the attribute to a specific defining op.

use reussir_backend::builders;
use reussir_backend::dialect::dbg;
use reussir_backend::melior::ir::attribute::{ArrayAttribute, StringAttribute};
use reussir_backend::melior::ir::operation::{OperationMutLike, OperationResult};
use reussir_backend::melior::ir::{Attribute, Identifier, Location, Module, Type, Value};

use reussir_core::full::mir;
use reussir_core::semi::ty::{FpTy, IntTy, Ty, TyKind};
use reussir_syntax::kind::TokenKey;

use super::expr::Lowerer;

impl<'c, 'p, 'tcx> Lowerer<'c, 'p, 'tcx> {
    /// Resolve an interned source name, if a resolver is available.
    fn source_name(&self, token: TokenKey) -> Option<&'p str> {
        self.names.map(|r| r.resolve(token))
    }

    /// Attach a local-variable debug attribute to the op that defines a `let`
    /// binding (a fused location carrying a `DBGLocalVar`). The conversion pass
    /// then describes the variable from that op's lowered storage.
    ///
    /// A binding whose value merely aliases a parameter or another local is a
    /// block argument with no defining op, and is skipped — as is a value with no
    /// representable debug type.
    pub(super) fn tag_local<'b>(&self, value: Value<'c, 'b>, name: TokenKey, ty: Ty<'tcx>) {
        if !self.debug_enabled() {
            return;
        }
        let (Some(dbg_type), Some(var_name)) = (self.dbg_type(ty), self.source_name(name)) else {
            return;
        };
        let Ok(result) = OperationResult::try_from(value) else {
            return;
        };
        let attr = dbg::local_var(
            self.context,
            dbg_type,
            StringAttribute::new(self.context, var_name),
        );
        let location = Location::fused(self.context, &[self.loc()], attr);
        builders::set_location(result.owner(), location);
    }

    /// Set the module-level debug attributes the conversion pass reads (the
    /// source file's basename and directory). No-op unless debug info is enabled.
    pub(super) fn set_module_debug_attrs(&self, module: &mut Module<'c>) {
        if !self.debug_enabled() {
            return;
        }
        let Some(source) = self.source else { return };
        let basename = StringAttribute::new(self.context, &source.basename());
        let directory = StringAttribute::new(self.context, &source.directory());
        let mut op = module.as_operation_mut();
        op.set_attribute("reussir.dbg.file_basename", basename.into());
        op.set_attribute("reussir.dbg.file_directory", directory.into());
    }

    /// The location for a function's `func.func`: its source position fused with a
    /// debug subprogram attribute, or just the position when debug info is off.
    pub(super) fn subprogram_location(&self, func: &mir::Function<'tcx>) -> Location<'c> {
        let base = self.location(func.body.and_then(|b| b.span));
        if !self.debug_enabled() {
            return base;
        }
        let raw_name = StringAttribute::new(self.context, self.program.symbol(func.symbol));
        let param_types: Vec<Attribute<'c>> = func
            .params
            .iter()
            .filter_map(|p| self.dbg_type(p.ty))
            .collect();
        let subprogram = dbg::subprogram(self.context, raw_name, &param_types);
        Location::fused(self.context, &[base], subprogram)
    }

    /// The `reussir.dbg_func_args` attribute describing a function's parameters
    /// (name, debug type, 1-based index), or `None` when debug info is off or no
    /// parameter has a representable debug type.
    pub(super) fn dbg_func_args_attr(
        &self,
        func: &mir::Function<'tcx>,
    ) -> Option<(Identifier<'c>, Attribute<'c>)> {
        if !self.debug_enabled() {
            return None;
        }
        let args: Vec<Attribute<'c>> = func
            .params
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let dbg_type = self.dbg_type(p.ty)?;
                let name = StringAttribute::new(self.context, self.source_name(p.name)?);
                Some(dbg::func_arg(self.context, dbg_type, name, i as u32 + 1))
            })
            .collect();
        if args.is_empty() {
            return None;
        }
        let array = ArrayAttribute::new(self.context, &args);
        Some((
            Identifier::new(self.context, "reussir.dbg_func_args"),
            array.into(),
        ))
    }

    /// Build the debug type attribute for a ground MIR type, or `None` if it has
    /// no representation the conversion pass handles yet (boxed records,
    /// closures, strings, ...).
    pub(super) fn dbg_type(&self, ty: Ty<'tcx>) -> Option<Attribute<'c>> {
        let mlir = self.tys.mlir_ty(ty).ok()?;
        match *ty.kind() {
            TyKind::Int(IntTy::Signed(w)) => Some(dbg::int_type(
                self.context,
                mlir,
                true,
                StringAttribute::new(self.context, &format!("i{w}")),
            )),
            TyKind::Int(IntTy::Unsigned(w)) => Some(dbg::int_type(
                self.context,
                mlir,
                false,
                StringAttribute::new(self.context, &format!("u{w}")),
            )),
            TyKind::Bool => Some(dbg::int_type(
                self.context,
                mlir,
                false,
                StringAttribute::new(self.context, "bool"),
            )),
            TyKind::Fp(FpTy::Ieee(w)) => Some(dbg::fp_type(
                self.context,
                mlir,
                StringAttribute::new(self.context, &format!("f{w}")),
            )),
            TyKind::Fp(FpTy::BFloat16) => Some(dbg::fp_type(
                self.context,
                mlir,
                StringAttribute::new(self.context, "bf16"),
            )),
            // A `[shared]` record is an `rc` box: describe the payload composite
            // (built over the inline record type) wrapped as a boxed type, so the
            // pass can reach it by dereferencing past the box header.
            TyKind::Record { .. } if self.tys.is_shared_record(ty) => {
                let payload_mlir = self.tys.record_inner_of(ty).ok()?;
                let payload = self.dbg_record(ty, payload_mlir)?;
                Some(dbg::boxed_type(self.context, payload))
            }
            // A by-value record: a composite of its (precisely-typed) fields. If
            // a field has no debug type the whole record is skipped.
            TyKind::Record { .. } => self.dbg_record(ty, mlir),
            _ => None,
        }
    }

    /// Build a `dbg_recordtype` for a by-value record, or `None` if any field
    /// lacks a debug type.
    fn dbg_record(&self, ty: Ty<'tcx>, underlying: Type<'c>) -> Option<Attribute<'c>> {
        let rec = self.tys.record_of(ty)?;
        let members = match rec.layout {
            mir::RecordLayout::Compound(members) => members,
            mir::RecordLayout::Variant(_) => return None,
        };
        let member_attrs: Option<Vec<Attribute<'c>>> = members
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let field_type = self.dbg_type(m.ty)?;
                // The source field name, falling back to the positional index for
                // a tuple field (or layout rebuilt from textual MIR).
                let field_name = m
                    .name
                    .map(|s| self.program.symbol(s).to_string())
                    .unwrap_or_else(|| i.to_string());
                let name = StringAttribute::new(self.context, &field_name);
                Some(dbg::record_member(self.context, name, field_type))
            })
            .collect();
        let name = StringAttribute::new(self.context, self.program.symbol(rec.symbol));
        Some(dbg::record_type(
            self.context,
            &member_attrs?,
            false,
            underlying,
            name,
        ))
    }
}
