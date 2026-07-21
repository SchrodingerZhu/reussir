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

    /// Whether the type lowers to an LLVM integer or pointer at the
    /// boundary. Mirrors `isTrivialFFIType` in
    /// `CABISignatureConversion.cpp`: scalars that are integers (`iN`,
    /// `bool`, `char`) and rc pointers (opaque `#[ffi]` and shared records)
    /// qualify; floats and aggregates do not.
    pub fn integer_like(&self, ty: Ty<'tcx>) -> bool {
        match *ty.kind() {
            TyKind::Int(_) | TyKind::Bool | TyKind::Char => true,
            TyKind::Record { def, .. } => self.records.get(&def).is_some_and(|r| {
                r.ffi.is_some() || r.default_cap == DefaultCap::Shared
            }),
            _ => false,
        }
    }

    /// The trampoline triviality rule, mirroring `evaluateCABISignatureForC`
    /// (`CABISignatureConversion.cpp`): trivial iff the return is `unit` or
    /// integer-like and there are fewer than four parameters, all
    /// integer-like.
    pub fn classify_trivial(&self, params: &[Ty<'tcx>], ret: Ty<'tcx>) -> bool {
        let trivial_ret = matches!(*ret.kind(), TyKind::Unit) || self.integer_like(ret);
        let trivial_params = params.len() < 4 && params.iter().all(|&p| self.integer_like(p));
        trivial_ret && trivial_params
    }
}

/// A Rust spelling of `name` as a binding identifier: raw (`r#name`) when
/// the plain form could collide with a Rust keyword. `Err` for the few
/// identifiers Rust cannot spell at all.
pub fn rust_ident(name: &str) -> Result<String, String> {
    match name {
        "self" | "Self" | "super" | "crate" | "_" => Err(format!(
            "parameter name `{name}` cannot cross the FFI boundary"
        )),
        // `r#` is valid on any non-reserved identifier, so apply it
        // uniformly rather than tracking the keyword list.
        _ => Ok(format!("r#{name}")),
    }
}

/// Substitute `[:Name:]` generic placeholders in a foreign body with the
/// instance's rendered Rust spellings. Unknown keys are left verbatim (the
/// Rust compiler then reports them in context).
pub fn substitute_placeholders(body: &str, map: &FxHashMap<&str, String>) -> String {
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(start) = rest.find("[:") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find(":]") {
            Some(end) => {
                let key = &after[..end];
                match map.get(key) {
                    Some(value) => out.push_str(value),
                    None => {
                        out.push_str("[:");
                        out.push_str(key);
                        out.push_str(":]");
                    }
                }
                rest = &after[end + 2..];
            }
            None => {
                out.push_str("[:");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}
