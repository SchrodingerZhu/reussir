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

use crate::semi::ctxt::{DefaultCap, GenericInfo, Record};
use crate::semi::lang::LangItem;
use crate::semi::traits::TraitDb;
use crate::semi::ty::{DefId, FpTy, IntTy, Ty, TyKind};

/// Which Rust comparison capabilities a foreign container requires from a
/// shared Reussir type. The bits are normalized over the comparison
/// super-trait tower, so `Ord` includes all four capabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComparisonBridgeNeeds(u8);

impl ComparisonBridgeNeeds {
    const PARTIAL_EQ: u8 = 1 << 0;
    const EQ: u8 = 1 << 1;
    const PARTIAL_ORD: u8 = 1 << 2;
    const ORD: u8 = 1 << 3;

    fn insert_lang(&mut self, item: LangItem) {
        self.0 |= match item {
            LangItem::PartialEq => Self::PARTIAL_EQ,
            LangItem::Eq => Self::PARTIAL_EQ | Self::EQ,
            LangItem::PartialOrd => Self::PARTIAL_EQ | Self::PARTIAL_ORD,
            LangItem::Ord => Self::PARTIAL_EQ | Self::EQ | Self::PARTIAL_ORD | Self::ORD,
            _ => 0,
        };
    }

    pub fn union(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub fn partial_eq(self) -> bool {
        self.0 & Self::PARTIAL_EQ != 0
    }

    pub fn eq(self) -> bool {
        self.0 & Self::EQ != 0
    }

    pub fn partial_ord(self) -> bool {
        self.0 & Self::PARTIAL_ORD != 0
    }

    pub fn ord(self) -> bool {
        self.0 & Self::ORD != 0
    }
}

/// A shared-record boundary wrapper collected while rendering. Its declaration
/// is rendered only after all mentions have unioned their comparison needs.
/// Keyed by the instance's v0 symbol in a `BTreeMap` for deterministic output.
#[derive(Clone)]
pub struct WrapperDecl<'tcx> {
    pub symbol: String,
    pub ty: Ty<'tcx>,
    pub needs: ComparisonBridgeNeeds,
}

impl WrapperDecl<'_> {
    /// Render the existing one-pointer owner plus the comparison behavior its
    /// enclosing foreign types require. Standard Rust comparison traits live
    /// on `reussir_rt::bridge::Bridge<T>`; this inner type implements only the
    /// local bridge traits, preserving the orphan-rule boundary.
    pub fn render(&self) -> String {
        let sym = &self.symbol;
        let mut d = String::new();
        let _ = write!(
            d,
            "#[repr(transparent)]\n\
             pub struct {sym}(*mut ::std::ffi::c_void);\n\
             unsafe extern \"C\" {{\n\
             \x20   unsafe fn {sym}_ffi_acquire(this: *mut ::std::ffi::c_void);\n\
             \x20   unsafe fn {sym}_ffi_release(this: *mut ::std::ffi::c_void);\n"
        );
        if self.needs.partial_eq() {
            let _ = writeln!(
                d,
                "    unsafe fn {sym}_ffi_eq(lhs: *mut ::std::ffi::c_void, rhs: *mut ::std::ffi::c_void) -> u8;"
            );
        }
        if self.needs.partial_ord() && !self.needs.ord() {
            let _ = writeln!(
                d,
                "    unsafe fn {sym}_ffi_partial_cmp(lhs: *mut ::std::ffi::c_void, rhs: *mut ::std::ffi::c_void) -> i32;"
            );
        }
        if self.needs.ord() {
            let _ = writeln!(
                d,
                "    unsafe fn {sym}_ffi_cmp(lhs: *mut ::std::ffi::c_void, rhs: *mut ::std::ffi::c_void) -> i32;"
            );
        }
        let _ = write!(
            d,
            "}}\n\
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
        if self.needs.partial_eq() {
            let _ = write!(
                d,
                "impl ::reussir_rt::bridge::PartialEqBridge for {sym} {{\n\
                 \x20   fn eq_bridge(&self, other: &Self) -> bool {{\n\
                 \x20       unsafe {{ {sym}_ffi_eq(self.0, other.0) != 0 }}\n\
                 \x20   }}\n\
                 }}\n"
            );
        }
        if self.needs.eq() {
            let _ = writeln!(d, "impl ::reussir_rt::bridge::EqBridge for {sym} {{}}");
        }
        if self.needs.partial_ord() {
            let _ = write!(
                d,
                "impl ::reussir_rt::bridge::PartialOrdBridge for {sym} {{\n\
                 \x20   fn partial_cmp_bridge(&self, other: &Self) -> Option<::std::cmp::Ordering> {{\n"
            );
            if self.needs.ord() {
                let _ = writeln!(
                    d,
                    "        Some(<Self as ::reussir_rt::bridge::OrdBridge>::cmp_bridge(self, other))"
                );
            } else {
                let _ = write!(
                    d,
                    "        match unsafe {{ {sym}_ffi_partial_cmp(self.0, other.0) }} {{\n\
                     \x20           n if n < 0 => Some(::std::cmp::Ordering::Less),\n\
                     \x20           0 => Some(::std::cmp::Ordering::Equal),\n\
                     \x20           1 => Some(::std::cmp::Ordering::Greater),\n\
                     \x20           _ => None,\n\
                     \x20       }}\n"
                );
            }
            let _ = write!(d, "    }}\n}}\n");
        }
        if self.needs.ord() {
            let _ = write!(
                d,
                "impl ::reussir_rt::bridge::OrdBridge for {sym} {{\n\
                 \x20   fn cmp_bridge(&self, other: &Self) -> ::std::cmp::Ordering {{\n\
                 \x20       match unsafe {{ {sym}_ffi_cmp(self.0, other.0) }} {{\n\
                 \x20           n if n < 0 => ::std::cmp::Ordering::Less,\n\
                 \x20           0 => ::std::cmp::Ordering::Equal,\n\
                 \x20           _ => ::std::cmp::Ordering::Greater,\n\
                 \x20       }}\n\
                 \x20   }}\n\
                 }}\n"
            );
        }
        d
    }
}

/// Rendering context: the elaborated record table (capabilities, `#[ffi]`
/// templates, generic parameter names) and a symbol source for instances.
pub struct FfiCtx<'a, 'tcx> {
    pub records: &'a FxHashMap<DefId, Record<'tcx>>,
    pub traits: Option<&'a TraitDb<'tcx>>,
    pub generic_env: Option<&'a [GenericInfo]>,
    pub lang: &'a FxHashMap<DefId, LangItem>,
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
        self.rust_name_with_needs(ty, ComparisonBridgeNeeds::default(), decls)
    }

    /// The Rust spelling of the ground argument one *binder* was instantiated
    /// at, carrying that binder's own declared comparison bounds.
    ///
    /// A bounded container is not the only source of a comparison
    /// requirement: an `#[ffi(import)]` signature may bound its own generic
    /// (`fn same<T: PartialEq>(lhs: T, rhs: T) -> bool`), and its Rust body
    /// then compares the rendered values directly. Without the bridge those
    /// bodies fail inside `rustc` on a `Bridge<_>: PartialEq` bound instead of
    /// using the implementation the signature already promised.
    pub fn rust_name_for_generic(
        &self,
        ty: Ty<'tcx>,
        generic: crate::semi::ty::GenericId,
        decls: &mut BTreeMap<String, WrapperDecl<'tcx>>,
    ) -> Result<String, String> {
        self.rust_name_with_needs(ty, self.generic_bridge_needs(generic), decls)
    }

    fn rust_name_with_needs(
        &self,
        ty: Ty<'tcx>,
        needs: ComparisonBridgeNeeds,
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
                            let arg_needs = record
                                .ty_params
                                .get(i)
                                .map_or_else(ComparisonBridgeNeeds::default, |(_, gid)| {
                                    self.generic_bridge_needs(*gid)
                                });
                            name.push_str(&self.rust_name_with_needs(arg, arg_needs, decls)?);
                        }
                        name.push('>');
                    }
                    return Ok(name);
                }
                match record.default_cap {
                    DefaultCap::Shared => Ok(self.shared_wrapper(def, args, ty, needs, decls)),
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
            // `str` lowers to `{ptr, len}`; the runtime's `#[repr(C)]`
            // borrowed view is bit-identical. Every surface `str` today is a
            // `'static` global literal, so the spelled lifetime is sound —
            // and unlike `'_`, it is legal in the packed-args struct field
            // the nontrivial boundary generates.
            TyKind::Str => Ok("::reussir_rt::collections::string::Str<'static>".into()),
            TyKind::Cell { .. } => Err("a cell cannot cross the FFI boundary".into()),
            TyKind::Array { .. } => Err("an array cannot cross the FFI boundary yet".into()),
            TyKind::Tensor { .. } => Err("a tensor cannot cross the FFI boundary".into()),
            TyKind::Closure { .. } => Err("a closure cannot cross the FFI boundary yet".into()),
            TyKind::Bottom | TyKind::Generic(_) | TyKind::Hole(_) => {
                Err("the type is not ground".into())
            }
        }
    }

    /// Comparison lang items implied by the declared bounds on one generic.
    /// A custom bound such as `T: Sorted` is recognized when `Sorted: Ord`.
    fn generic_bridge_needs(&self, generic: crate::semi::ty::GenericId) -> ComparisonBridgeNeeds {
        let (Some(db), Some(info)) = (
            self.traits,
            self.generic_env.and_then(|env| env.get(generic.0 as usize)),
        ) else {
            return ComparisonBridgeNeeds::default();
        };
        let mut needs = ComparisonBridgeNeeds::default();
        for item in LangItem::CMP_TRAITS {
            let Some(target) = self
                .lang
                .iter()
                .find_map(|(&def, &bound)| (bound == item).then(|| db.trait_by_def(def)).flatten())
            else {
                continue;
            };
            if info.bounds.iter().any(|&have| db.implies(have, target)) {
                needs.insert_lang(item);
            }
        }
        needs
    }

    /// The `#[repr(transparent)]` pointer wrapper for a shared Reussir
    /// record instance, registering its declaration (and thereby its rc-glue
    /// requirement) under the instance symbol.
    fn shared_wrapper(
        &self,
        def: DefId,
        args: &'tcx [Ty<'tcx>],
        ty: Ty<'tcx>,
        needs: ComparisonBridgeNeeds,
        decls: &mut BTreeMap<String, WrapperDecl<'tcx>>,
    ) -> String {
        let sym = (self.instance_symbol)(def, args);
        decls
            .entry(sym.clone())
            .and_modify(|decl| decl.needs.union(needs))
            .or_insert_with(|| WrapperDecl {
                symbol: sym.clone(),
                ty,
                needs,
            });
        format!("::reussir_rt::bridge::Bridge<{sym}>")
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
        out.push_str(&decl.render());
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
        out.push_str(&decl.render());
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
