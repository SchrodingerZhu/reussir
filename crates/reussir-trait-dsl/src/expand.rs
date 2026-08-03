//! Validation and code generation for `builtin_traits!`.
//!
//! The generated code targets the registration contract of
//! `reussir-core::semi::traits` (paths spelled `crate::semi::…`, so the
//! macro only expands meaningfully inside that crate): caller-assigned dense
//! `TraitId`/`ImplId`s in declaration order, trait defs declared through the
//! `DefTable`'s trait namespace at the crate root, and a `Builtins` handle
//! struct whose snake-cased fields the checker references by name.

use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;

use crate::ast::{Input, Item};

/// The closed scalar vocabulary an impl clause may name, with the type
/// constructor each expands to.
fn scalar_ctor(ident: &syn::Ident) -> Option<TokenStream> {
    let ts = match ident.to_string().as_str() {
        "i8" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Signed(8))),
        "i16" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Signed(16))),
        "i32" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Signed(32))),
        "i64" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Signed(64))),
        "u8" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Unsigned(8))),
        "u16" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Unsigned(16))),
        "u32" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Unsigned(32))),
        "u64" => quote!(tcx.mk_int(crate::semi::ty::IntTy::Unsigned(64))),
        "f16" => quote!(tcx.mk_fp(crate::semi::ty::FpTy::Ieee(16))),
        "f32" => quote!(tcx.mk_fp(crate::semi::ty::FpTy::Ieee(32))),
        "f64" => quote!(tcx.mk_fp(crate::semi::ty::FpTy::Ieee(64))),
        "bf16" => quote!(tcx.mk_fp(crate::semi::ty::FpTy::BFloat16)),
        "f8" => quote!(tcx.mk_fp(crate::semi::ty::FpTy::Float8)),
        _ => return None,
    };
    Some(ts)
}

const KNOWN_SCALARS: &str = "i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bf16, f8";

pub(crate) fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let input: Input = syn::parse2(input)?;

    // Validation pass: names resolve backwards within the invocation, ids
    // are declaration-ordered, structural traits stay impl-less.
    let mut traits: Vec<&crate::ast::TraitDecl> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut impl_clauses: Vec<(&crate::ast::ImplDecl, usize)> = Vec::new();
    let mut seen_impl: HashMap<(String, String), ()> = HashMap::new();
    for item in &input.items {
        match item {
            Item::Trait(decl) => {
                let name = decl.name.to_string();
                if index.contains_key(&name) {
                    return Err(syn::Error::new_spanned(
                        &decl.name,
                        format!("trait `{name}` is declared more than once"),
                    ));
                }
                for sup in &decl.supertraits {
                    if !index.contains_key(&sup.to_string()) {
                        return Err(syn::Error::new_spanned(
                            sup,
                            format!(
                                "supertrait `{sup}` must be declared before `{name}` \
                                 (declaration order is the id assignment)"
                            ),
                        ));
                    }
                }
                if let Some(m) = decl.methods.first() {
                    return Err(syn::Error::new_spanned(
                        &m.name,
                        "trait method signatures are not yet supported by `builtin_traits!`",
                    ));
                }
                index.insert(name, traits.len());
                traits.push(decl);
            }
            Item::Impl(decl) => {
                let tname = decl.trait_name.to_string();
                let Some(&t) = index.get(&tname) else {
                    return Err(syn::Error::new_spanned(
                        &decl.trait_name,
                        format!("`impl` names undeclared trait `{tname}`"),
                    ));
                };
                if traits[t].structural {
                    return Err(syn::Error::new_spanned(
                        &decl.trait_name,
                        format!(
                            "structural trait `{tname}` is answered by the checker \
                             and cannot have impls"
                        ),
                    ));
                }
                for ty in &decl.self_tys {
                    if scalar_ctor(ty).is_none() {
                        return Err(syn::Error::new_spanned(
                            ty,
                            format!(
                                "unknown scalar `{ty}`; the builtin impl vocabulary is \
                                 {KNOWN_SCALARS}"
                            ),
                        ));
                    }
                    if seen_impl
                        .insert((tname.clone(), ty.to_string()), ())
                        .is_some()
                    {
                        return Err(syn::Error::new_spanned(
                            ty,
                            format!("duplicate impl of `{tname}` for `{ty}`"),
                        ));
                    }
                }
                impl_clauses.push((decl, t));
            }
        }
    }

    // Generation. Field order and id order are both declaration order.
    let field_idents: Vec<syn::Ident> = traits
        .iter()
        .map(|t| format_ident!("{}", t.name.to_string().to_snake_case()))
        .collect();
    let fields = traits.iter().zip(&field_idents).map(|(t, field)| {
        let docs = &t.docs;
        quote! {
            #(#docs)*
            pub #field: crate::semi::traits::TraitId,
        }
    });

    let trait_regs = traits
        .iter()
        .zip(&field_idents)
        .enumerate()
        .map(|(i, (t, field))| {
            let i = proc_macro2::Literal::u32_unsuffixed(i as u32);
            let name = t.name.to_string();
            let sealed = t.sealed;
            let supers = t.supertraits.iter().map(|sup| {
                let s = &field_idents[index[&sup.to_string()]];
                quote!(self_ref(#s))
            });
            quote! {
                let #field = crate::semi::traits::TraitId(#i);
                {
                    let key = interner.get_or_intern(#name);
                    let def = defs.declare_trait(key).expect("builtins precede user defs");
                    db.add_trait(crate::semi::traits::def::TraitDef {
                        id: #field,
                        def,
                        visibility: crate::surface::Visibility::Public,
                        sealed: #sealed,
                        self_param: self_g,
                        params: vec![],
                        supertraits: vec![#(#supers),*],
                        methods: vec![],
                        assoc_tys: vec![],
                        span: None,
                        file: reussir_syntax::source::FileId::ROOT,
                    });
                }
            }
        });

    let mut next_impl = 0u32;
    let impl_regs = impl_clauses.iter().flat_map(|(decl, t)| {
        let field = &field_idents[*t];
        decl.self_tys
            .iter()
            .map(|ty| {
                let ctor = scalar_ctor(ty).expect("validated above");
                let id = proc_macro2::Literal::u32_unsuffixed(next_impl);
                next_impl += 1;
                quote! {
                    {
                        let self_ty = #ctor;
                        db.add_impl(crate::semi::traits::def::ImplDef {
                            id: crate::semi::traits::ImplId(#id),
                            generics: vec![],
                            trait_ref: crate::semi::traits::TraitRef {
                                trait_id: #field,
                                args: vec![self_ty],
                            },
                            self_ty,
                            where_clauses: vec![],
                            methods: vec![],
                            span: None,
                            file: reussir_syntax::source::FileId::ROOT,
                        });
                    }
                }
            })
            .collect::<Vec<_>>()
    });

    Ok(quote! {
        /// Handles to the built-in traits, for the type checker to refer to
        /// by name. Generated by `builtin_traits!`; declaration order is the
        /// dense id order.
        pub struct Builtins {
            #(#fields)*
        }

        impl Builtins {
            /// Register the built-in traits and their primitive impls into
            /// `db`, declaring each trait in `defs`' trait namespace at the
            /// crate root — prelude visibility for bare references,
            /// shadowable by a same-named module-local trait.
            pub fn register<'tcx>(
                db: &mut crate::semi::traits::TraitDb<'tcx>,
                defs: &mut crate::semi::resolve::DefTable,
                interner: &mut impl reussir_syntax::Interner<reussir_syntax::kind::TokenKey>,
                tcx: &crate::semi::ty::TyCtxt<'tcx>,
            ) -> Builtins {
                // Every built-in trait has a single `Self` parameter, generic
                // #0 — deliberately UNALLOCATED in the elaborator's generics
                // table (the builtin defs predate it, and allocating would
                // shift every dump's `$n` numbering).
                let self_g = crate::semi::ty::GenericId(0);
                let self_ref = |trait_id| crate::semi::traits::TraitRef {
                    trait_id,
                    args: vec![tcx.mk_generic(self_g)],
                };
                let _ = &self_ref;
                #(#trait_regs)*
                #(#impl_regs)*
                Builtins { #(#field_idents),* }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn full_table() -> TokenStream {
        quote! {
            #[sealed] trait Num;
            #[sealed] trait Integral: Num;
            #[sealed] trait FloatingPoint: Num;
            #[sealed] trait PtrLike;
            /// Marker (auto) trait.
            #[sealed] #[structural] trait Sync;
            impl Integral for i8 | u8 | i16 | u16 | i32 | u32 | i64 | u64;
            impl FloatingPoint for f16 | f32 | f64 | bf16 | f8;
        }
    }

    fn expand_ok(ts: TokenStream) -> String {
        let out = expand(ts).expect("expansion");
        // The output must be a valid Rust item sequence.
        syn::parse2::<syn::File>(out.clone()).expect("parses as items");
        out.to_string()
    }

    fn expand_err(ts: TokenStream) -> String {
        expand(ts).expect_err("must fail").to_string()
    }

    #[test]
    fn expand_full_builtin_table() {
        let out = expand_ok(full_table());
        // Field order = declaration order, snake-cased.
        let num = out.find("pub num :").expect("num field");
        let integral = out.find("pub integral :").expect("integral field");
        let fp = out.find("pub floating_point :").expect("fp field");
        let ptr = out.find("pub ptr_like :").expect("ptr field");
        let sync = out.find("pub sync :").expect("sync field");
        assert!(num < integral && integral < fp && fp < ptr && ptr < sync);
        // Dense declaration-ordered ids: 5 traits, 13 impls (8 + 5).
        for i in 0..5u32 {
            assert!(out.contains(&format!("TraitId ({i})")), "TraitId({i})");
        }
        for i in 0..13u32 {
            assert!(out.contains(&format!("ImplId ({i})")), "ImplId({i})");
        }
        assert_eq!(out.matches("add_trait").count(), 5);
        assert_eq!(out.matches("add_impl").count(), 13);
        // Every builtin is sealed.
        assert_eq!(out.matches("sealed : true").count(), 5);
        // The impl order preserves the listed scalar order.
        let first = out.find("Signed (8)").expect("i8 impl");
        let last = out.find("Float8").expect("f8 impl");
        assert!(first < last);
    }

    #[test]
    fn supertrait_reference_resolves_in_order() {
        let out = expand_ok(quote! {
            trait A;
            trait B;
            trait C: A + B;
        });
        let c = out.find("TraitId (2)").expect("C");
        let refs = &out[c..];
        let a = refs.find("self_ref (a)").expect("A ref");
        let b = refs.find("self_ref (b)").expect("B ref");
        assert!(a < b, "supertrait order preserved");
    }

    #[test]
    fn doc_comments_forward_to_fields() {
        let out = expand_ok(quote! {
            /// The marker.
            trait Sync;
        });
        assert!(out.contains("The marker."));
    }

    #[test]
    fn forward_supertrait_is_an_error() {
        let msg = expand_err(quote! { trait A: B; trait B; });
        assert!(
            msg.contains("supertrait `B` must be declared before `A`"),
            "{msg}"
        );
    }

    #[test]
    fn impl_for_undeclared_trait_is_an_error() {
        let msg = expand_err(quote! { impl Show for i8; });
        assert!(msg.contains("undeclared trait `Show`"), "{msg}");
    }

    #[test]
    fn structural_trait_with_impls_is_an_error() {
        let msg = expand_err(quote! { #[structural] trait S; impl S for i8; });
        assert!(msg.contains("structural trait `S`"), "{msg}");
    }

    #[test]
    fn unknown_scalar_is_an_error() {
        let msg = expand_err(quote! { trait Num; impl Num for i7; });
        assert!(msg.contains("unknown scalar `i7`"), "{msg}");
        assert!(msg.contains("i64"), "{msg}");
    }

    #[test]
    fn duplicates_are_errors() {
        let msg = expand_err(quote! { trait A; trait A; });
        assert!(msg.contains("declared more than once"), "{msg}");
        let msg = expand_err(quote! { trait A; impl A for i8 | i8; });
        assert!(msg.contains("duplicate impl of `A` for `i8`"), "{msg}");
    }

    #[test]
    fn method_sigs_parse_but_are_rejected() {
        let msg = expand_err(quote! {
            trait Num { fn add(self: Self, other: Self) -> Self; }
        });
        assert!(msg.contains("not yet supported"), "{msg}");
    }

    #[test]
    fn unknown_attribute_is_an_error() {
        let msg = expand_err(quote! { #[frobnicate] trait A; });
        assert!(msg.contains("unknown attribute"), "{msg}");
    }
}
