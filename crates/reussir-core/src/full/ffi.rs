//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//! FFI boundary rendering for `#[ffi(import)]` functions and opaque
//! `#[ffi(rust = ...)]` records.
//!
//! Monomorphization renders, per ground instance, a self-contained Rust
//! *texture* (compiled to bitcode by the `reussir-compile-polymorphic-ffi`
//! pass and linked into the final module) plus the boundary metadata the
//! codegen needs: the external boundary symbol, the native declaration, and
//! an import trampoline binding the two.
//!
//! The boundary is the trampoline convention (platform-independent — only
//! integers and pointers cross it): [`classify_trivial`] mirrors
//! `evaluateCABISignatureForC` in
//! `lib/Conversion/BasicOpsLowering/CABISignatureConversion.cpp`, and the
//! generated wrapper's `extern "C"` signature matches the shape the import
//! trampoline lowering calls. The two must agree; the e2e tests pin both.
//!
//! Boundary types (v1) and their Rust spellings:
//!
//! * scalars — `i8..i128`/`u8..u128`, `f32`/`f64`, `bool`, `char` (and
//!   `unit` as a return type);
//! * opaque `#[ffi]` records — the declared Rust path applied to the
//!   rendered arguments (`::reussir_rt::collections::vec::Vec<f64>`);
//! * shared Reussir records — a generated `#[repr(transparent)]` pointer
//!   wrapper named by the instance's v0 symbol, whose `Clone`/`Drop` bind to
//!   compiler-emitted `<sym>_ffi_acquire`/`<sym>_ffi_release` rc glue.
//!
//! Symbol scheme (all suffixes appended to complete v0 manglings, which is
//! injective — no valid v0 symbol is a prefix of another):
//!
//! * `<fn instance>` — the native function callers call;
//! * `<fn instance>_ffi` — the Rust-side boundary wrapper;
//! * `<record instance>_ffi_drop` — an opaque instance's drop hook;
//! * `<record instance>_ffi_acquire` / `_ffi_release` — Reussir rc glue.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rustc_hash::FxHashMap;

use crate::semi::ctxt::{DefaultCap, Record};
use crate::semi::ty::{DefId, FpTy, IntTy, Ty, TyKind};

/// A shared-record boundary wrapper collected while rendering: its Rust
/// declaration text and the instance type its rc glue operates on. Keyed by
/// the instance's v0 symbol in a `BTreeMap` for deterministic emission.
pub struct WrapperDecl<'tcx> {
    pub decl: String,
    pub ty: Ty<'tcx>,
}

/// Rendering context: the elaborated record table (capabilities, `#[ffi]`
/// templates, generic parameter names) and a symbol source for instances.
pub struct FfiCtx<'a, 'tcx> {
    pub records: &'a FxHashMap<DefId, Record<'tcx>>,
    /// Mangle a ground record instance to its v0 symbol.
    pub instance_symbol: &'a dyn Fn(DefId, &'tcx [Ty<'tcx>]) -> String,
}

impl<'a, 'tcx> FfiCtx<'a, 'tcx> {
    /// The Rust spelling of a boundary type. Shared-record wrappers the
    /// spelling depends on are collected into `decls`. `Err` carries a
    /// description of why the type cannot cross the boundary.
    pub fn rust_name(
        &self,
        ty: Ty<'tcx>,
        decls: &mut BTreeMap<String, WrapperDecl<'tcx>>,
    ) -> Result<String, String> {
        match *ty.kind() {
            TyKind::Int(IntTy::Signed(w)) => Ok(format!("i{w}")),
            TyKind::Int(IntTy::Unsigned(w)) => Ok(format!("u{w}")),
            TyKind::Fp(FpTy::Ieee(w @ (32 | 64))) => Ok(format!("f{w}")),
            TyKind::Fp(_) => Err("a non-IEEE float has no Rust spelling".into()),
            TyKind::Bool => Ok("bool".into()),
            TyKind::Char => Ok("char".into()),
            TyKind::Record { def, args, .. } => {
                let Some(record) = self.records.get(&def) else {
                    return Err("an unresolved record".into());
                };
                if let Some(template) = &record.ffi {
                    let mut name = template.clone();
                    if !args.is_empty() {
                        name.push('<');
                        for (i, &arg) in args.iter().enumerate() {
                            if i > 0 {
                                name.push_str(", ");
                            }
                            name.push_str(&self.rust_name(arg, decls)?);
                        }
                        name.push('>');
                    }
                    return Ok(name);
                }
                match record.default_cap {
                    DefaultCap::Shared => Ok(self.shared_wrapper(def, args, ty, decls)),
                    DefaultCap::Value => {
                        Err("a `[value]` record cannot cross the FFI boundary yet".into())
                    }
                    DefaultCap::Regional => {
                        Err("a regional record cannot cross the FFI boundary".into())
                    }
                }
            }
            TyKind::Unit => Err("`unit` only crosses the FFI boundary as a return type".into()),
            TyKind::Arc(_) => Err("an `Arc` coloring cannot cross the FFI boundary yet".into()),
            TyKind::Nullable(_) => Err("`Nullable` cannot cross the FFI boundary yet".into()),
            TyKind::Str => Err("`str` cannot cross the FFI boundary yet".into()),
            TyKind::Cell { .. } => Err("a cell cannot cross the FFI boundary".into()),
            TyKind::Array { .. } => Err("an array cannot cross the FFI boundary yet".into()),
            TyKind::Closure { .. } => Err("a closure cannot cross the FFI boundary yet".into()),
            TyKind::Bottom | TyKind::Generic(_) | TyKind::Hole(_) => {
                Err("the type is not ground".into())
            }
        }
    }

    /// The `#[repr(transparent)]` pointer wrapper for a shared Reussir
    /// record instance, registering its declaration (and thereby its rc-glue
    /// requirement) under the instance symbol.
    fn shared_wrapper(
        &self,
        def: DefId,
        args: &'tcx [Ty<'tcx>],
        ty: Ty<'tcx>,
        decls: &mut BTreeMap<String, WrapperDecl<'tcx>>,
    ) -> String {
        let sym = (self.instance_symbol)(def, args);
        if !decls.contains_key(&sym) {
            let mut d = String::new();
            let _ = write!(
                d,
                "#[repr(transparent)]\n\
                 pub struct {sym}(*mut ::std::ffi::c_void);\n\
                 unsafe extern \"C\" {{\n\
                 \x20   unsafe fn {sym}_ffi_acquire(this: *mut ::std::ffi::c_void);\n\
                 \x20   unsafe fn {sym}_ffi_release(this: *mut ::std::ffi::c_void);\n\
                 }}\n\
                 impl Clone for {sym} {{\n\
                 \x20   fn clone(&self) -> Self {{\n\
                 \x20       unsafe {{ {sym}_ffi_acquire(self.0) }};\n\
                 \x20       Self(self.0)\n\
                 \x20   }}\n\
                 }}\n\
                 impl Drop for {sym} {{\n\
                 \x20   fn drop(&mut self) {{\n\
                 \x20       unsafe {{ {sym}_ffi_release(self.0) }};\n\
                 \x20   }}\n\
                 }}\n"
            );
            decls.insert(sym.clone(), WrapperDecl { decl: d, ty });
        }
        sym
    }

}
