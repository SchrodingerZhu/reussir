//! Debug-info emission: building the Reussir `DBG*` attributes from the MIR and
//! attaching them so the backend's debug-info conversion pass can emit DWARF.
//!
//! The attributes ride on locations: a function's `func.func` is given a
//! `FusedLoc` whose metadata is a [`dbg::subprogram`] attribute, and its
//! parameters are described by a `reussir.dbg_func_args` array attribute the pass
//! reads. A variable's *type* is built precisely from its ground MIR type
//! ([`dbg_type`](Lowerer::dbg_type)).
//!
//! Emitted debug types cover scalars, by-value and managed (`[shared]`) records
//! — the latter boxed, reached through a `DIExpression` past the rc-box header —
//! and enums (`variant`), described as a tag plus a union of the per-case
//! payloads. Both `let` locals and parameters are emitted. A recursive type is
//! emitted as a memberless placeholder at the recursion point (see
//! [`Self::dbg_record`]); the backend ties it back to the enclosing composite as
//! a recursive self-reference, so the recursive field stays walkable.

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
            // A managed record is an `rc` box: describe the payload composite
            // (built over the inline record type) wrapped as a boxed type, so the
            // pass can reach it by dereferencing past the box header. `regional`
            // selects which header size to skip — shared boxes have a one-word
            // ref-count, regional boxes a three-word header.
            TyKind::Record { .. }
                if self.tys.is_shared_record(ty) || self.tys.is_regional_record(ty) =>
            {
                let payload_mlir = self.tys.record_inner_of(ty).ok()?;
                let payload = self.dbg_record(ty, payload_mlir)?;
                let regional = self.tys.is_regional_record(ty);
                Some(dbg::boxed_type(self.context, payload, regional))
            }
            // A by-value record: a composite of its (precisely-typed) fields. If
            // a field has no debug type the whole record is skipped.
            TyKind::Record { .. } => self.dbg_record(ty, mlir),
            _ => None,
        }
    }

    /// Build a `dbg_recordtype` for a record, or `None` if any field lacks a debug
    /// type. A struct becomes a composite of its precisely-typed fields; an enum
    /// becomes a variant composite whose members are the per-case payload structs
    /// (the conversion pass lays these out as a tag plus an overlapping payload).
    ///
    /// A record reached recursively (through a boxed field or variant payload) is
    /// emitted as a memberless composite carrying only the record's name. The
    /// recursion always crosses a pointer (a managed field); the backend matches
    /// that name to the enclosing composite being built and resolves the
    /// placeholder into a recursive self-reference, so the field's pointee is the
    /// full type and the debugger can descend it.
    fn dbg_record(&self, ty: Ty<'tcx>, underlying: Type<'c>) -> Option<Attribute<'c>> {
        let rec = self.tys.record_of(ty)?;
        let name = StringAttribute::new(self.context, self.program.symbol(rec.symbol));
        let is_variant = matches!(rec.layout, mir::RecordLayout::Variant(_));
        // Recursion guard: a record already being expanded reappears as an empty
        // (forward-declared) composite rather than recursing forever.
        if !self.dbg_building.borrow_mut().insert(rec.symbol) {
            return Some(dbg::record_type(
                self.context,
                &[],
                is_variant,
                underlying,
                name,
            ));
        }
        let members = match rec.layout {
            mir::RecordLayout::Compound(members) => self.dbg_struct_members(members),
            mir::RecordLayout::Variant(variants) => variants
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let case = self.dbg_variant_case(ty, i, v)?;
                    let case_name = StringAttribute::new(self.context, self.program.symbol(v.name));
                    Some(dbg::record_member(self.context, case_name, case))
                })
                .collect(),
        };
        self.dbg_building.borrow_mut().remove(&rec.symbol);
        Some(dbg::record_type(
            self.context,
            &members?,
            is_variant,
            underlying,
            name,
        ))
    }

    /// Build the `dbg_record_member`s for a struct's fields (named, or positional
    /// for tuple fields / a layout rebuilt from textual MIR), or `None` if any
    /// field lacks a debug type.
    fn dbg_struct_members(&self, members: &[mir::Member<'tcx>]) -> Option<Vec<Attribute<'c>>> {
        members
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let field_type = self.dbg_type(m.ty)?;
                // Named field, else a positional `fN` (a bare number confuses some
                // debuggers' member rendering).
                let field_name = m
                    .name
                    .map(|s| self.program.symbol(s).to_string())
                    .unwrap_or_else(|| format!("f{i}"));
                let name = StringAttribute::new(self.context, &field_name);
                Some(dbg::record_member(self.context, name, field_type))
            })
            .collect()
    }

    /// The debug type for one enum case `v` (variant `idx`): a struct composite
    /// over its (positional) fields, built on the case's payload layout, named by
    /// the case's source name. `None` if any field lacks a debug type.
    fn dbg_variant_case(
        &self,
        ty: Ty<'tcx>,
        idx: usize,
        v: &mir::VariantDef<'tcx>,
    ) -> Option<Attribute<'c>> {
        let payload = self.tys.variant_payload_of(ty, idx).ok()?;
        let field_attrs: Option<Vec<Attribute<'c>>> = v
            .fields
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let field_type = self.dbg_type(f)?;
                // Variant fields are positional; name them `fN`.
                let name = StringAttribute::new(self.context, &format!("f{i}"));
                Some(dbg::record_member(self.context, name, field_type))
            })
            .collect();
        let case_name = StringAttribute::new(self.context, self.program.symbol(v.name));
        Some(dbg::record_type(
            self.context,
            &field_attrs?,
            false,
            payload,
            case_name,
        ))
    }
}
