//! A total structural substituter over `Ty` for trait machinery.
//!
//! Unlike `full::subst::subst_ty` — which panics on an unmapped generic
//! (monomorphization must ground everything) — this walker lets unmapped
//! generics pass through: conformance checking substitutes only the trait's
//! binder while the member's own generics stay rigid.

use rustc_hash::FxHashMap;

use crate::semi::ty::{GenericId, Ty, TyCtxt, TyKind};

pub fn replace_generics<'tcx>(
    tcx: &TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    map: &FxHashMap<GenericId, Ty<'tcx>>,
) -> Ty<'tcx> {
    match *ty.kind() {
        TyKind::Generic(g) => map.get(&g).copied().unwrap_or(ty),
        TyKind::Nullable(inner) => {
            let n = replace_generics(tcx, inner, map);
            if n == inner { ty } else { tcx.mk_nullable(n) }
        }
        TyKind::Arc(inner) => {
            let n = replace_generics(tcx, inner, map);
            if n == inner { ty } else { tcx.mk_arc(n) }
        }
        TyKind::Cell { elem, kind } => {
            let n = replace_generics(tcx, elem, map);
            if n == elem { ty } else { tcx.mk_cell(n, kind) }
        }
        TyKind::Record {
            def,
            ref args,
            flex,
        } => {
            let new: Vec<Ty> = args
                .iter()
                .map(|&a| replace_generics(tcx, a, map))
                .collect();
            if new.iter().zip(args.iter()).all(|(a, b)| a == b) {
                ty
            } else {
                tcx.mk_record(def, &new, flex)
            }
        }
        TyKind::Closure { ref params, ret } => {
            let new: Vec<Ty> = params
                .iter()
                .map(|&p| replace_generics(tcx, p, map))
                .collect();
            let r = replace_generics(tcx, ret, map);
            if r == ret && new.iter().zip(params.iter()).all(|(a, b)| a == b) {
                ty
            } else {
                tcx.mk_closure(&new, r)
            }
        }
        TyKind::Array { elem, ref dims } => {
            let n = replace_generics(tcx, elem, map);
            if n == elem { ty } else { tcx.mk_array(n, dims) }
        }
        _ => ty,
    }
}

/// Collect every generic occurring in `ty` — the E0207 constrainedness walk.
pub fn collect_generics_of(ty: Ty<'_>, out: &mut rustc_hash::FxHashSet<GenericId>) {
    match *ty.kind() {
        TyKind::Generic(g) => {
            out.insert(g);
        }
        TyKind::Nullable(inner) | TyKind::Arc(inner) => collect_generics_of(inner, out),
        TyKind::Cell { elem, .. } => collect_generics_of(elem, out),
        TyKind::Record { ref args, .. } => args.iter().for_each(|&a| collect_generics_of(a, out)),
        TyKind::Closure { ref params, ret } => {
            params.iter().for_each(|&p| collect_generics_of(p, out));
            collect_generics_of(ret, out);
        }
        TyKind::Array { elem, .. } => collect_generics_of(elem, out),
        _ => {}
    }
}
