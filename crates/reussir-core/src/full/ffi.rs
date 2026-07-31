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
            TyKind::Record { def, .. } => self
                .records
                .get(&def)
                .is_some_and(|r| r.ffi.is_some() || r.default_cap == DefaultCap::Shared),
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

/// The common texture head: linkage feature, lint silencing, the runtime
/// crate, and the file's foreign preludes (brace interiors spliced).
fn texture_head(preludes: &[&str]) -> String {
    let mut out = String::from(
        "#![feature(linkage)]\n#![allow(nonstandard_style, unused, unsafe_op_in_unsafe_fn)]\n\
         extern crate reussir_rt;\n",
    );
    for prelude in preludes {
        // A prelude block is `{ items... }`; splice the interior.
        let interior = prelude
            .strip_prefix('{')
            .and_then(|p| p.strip_suffix('}'))
            .unwrap_or(prelude);
        out.push_str(interior);
        out.push('\n');
    }
    out
}

/// One parameter of a generated wrapper: its Rust binding identifier and its
/// Rust type spelling.
pub struct WrapperParam {
    pub ident: String,
    pub rust_ty: String,
}

/// Render the boundary wrapper texture for one `#[ffi(import)]` function
/// instance. `ret` is `None` for a `unit` return; `ret_direct` says the
/// result crosses directly (`unit` or integer-like — mirrors
/// `hasReturnPtr`); `body` is the user's Rust block (braces included) with
/// generic placeholders already substituted.
#[allow(clippy::too_many_arguments)]
pub fn import_texture(
    preludes: &[&str],
    decls: &BTreeMap<String, WrapperDecl<'_>>,
    boundary: &str,
    params: &[WrapperParam],
    ret: Option<&str>,
    trivial: bool,
    ret_direct: bool,
    body: &str,
) -> String {
    let mut out = texture_head(preludes);
    for decl in decls.values() {
        out.push_str(&decl.decl);
    }
    let attrs = "#[linkage = \"weak_odr\"]\n#[unsafe(no_mangle)]\n";
    if trivial {
        let sig: Vec<String> = params
            .iter()
            .map(|p| format!("{}: {}", p.ident, p.rust_ty))
            .collect();
        let ret_ann = ret.map(|r| format!(" -> {r}")).unwrap_or_default();
        let _ = writeln!(
            out,
            "{attrs}pub unsafe extern \"C\" fn {boundary}({}){ret_ann} {body}",
            sig.join(", ")
        );
        return out;
    }
    // The nontrivial boundary: a leading return pointer when the result does
    // not cross directly, and every parameter packed into a `#[repr(C)]`
    // struct passed by pointer — the exact shape the import trampoline
    // lowering calls (`rewriteImport` in `BasicOpsLowering.cpp`).
    if !params.is_empty() {
        let fields: Vec<&str> = params.iter().map(|p| p.rust_ty.as_str()).collect();
        let _ = write!(
            out,
            "#[repr(C)]\nstruct __ReussirArgs({});\n",
            fields.join(", ")
        );
    }
    let mut sig = Vec::new();
    if !ret_direct {
        sig.push(format!(
            "__reussir_ret: *mut {}",
            ret.expect("an indirect return has a type")
        ));
    }
    if !params.is_empty() {
        sig.push("__reussir_args: *mut __ReussirArgs".to_owned());
    }
    let ret_ann = match (ret_direct, ret) {
        (true, Some(r)) => format!(" -> {r}"),
        _ => String::new(),
    };
    let _ = writeln!(
        out,
        "{attrs}pub unsafe extern \"C\" fn {boundary}({}){ret_ann} {{",
        sig.join(", ")
    );
    if !params.is_empty() {
        let binders: Vec<&str> = params.iter().map(|p| p.ident.as_str()).collect();
        let _ = writeln!(
            out,
            "    let __ReussirArgs({}) = unsafe {{ __reussir_args.read() }};",
            binders.join(", ")
        );
    }
    if ret_direct {
        let _ = write!(out, "    {body}\n}}\n");
    } else {
        let _ = write!(
            out,
            "    let __reussir_result: {} = {body};\n    \
             unsafe {{ __reussir_ret.write(__reussir_result) }};\n}}\n",
            ret.expect("an indirect return has a type")
        );
    }
    out
}

/// Render the drop-hook texture for one opaque `#[ffi]` record instance:
/// a wrapper that takes the foreign value by transparent pointer and drops
/// it. Called by the `rc.dec` lowering with the rc pointer.
pub fn drop_texture(
    decls: &BTreeMap<String, WrapperDecl<'_>>,
    hook: &str,
    rust_name: &str,
) -> String {
    let mut out = texture_head(&[]);
    for decl in decls.values() {
        out.push_str(&decl.decl);
    }
    let _ = write!(
        out,
        "#[linkage = \"weak_odr\"]\n#[unsafe(no_mangle)]\n\
         pub unsafe extern \"C\" fn {hook}(_this: {rust_name}) {{}}\n"
    );
    out
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
