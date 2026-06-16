//! Unification variables and the inference context.
//!
//! The metavariable store is `ena`'s in-place union-find. Each class of holes
//! carries a [`HoleValue`]: still-unknown (with a generalization [`Level`]) or
//! solved to an interned [`Ty`]. Unification merges and solves classes; `ena`'s
//! logged snapshots give speculative probing with correct rollback.
//!
//! Checking is unification-driven — the union-find *is* the substitution, so no
//! structural `subst` happens here. The one place a `generic ↦ type` mapping
//! appears is use-site instantiation ([`InferCtxt::instantiate`] /
//! [`InferCtxt::unify_instantiated`]), and that is fully lazy: a template's
//! generics are resolved through the instantiation on the fly, and a template is
//! materialized into the arena only when it must be assigned into a hole.

use std::marker::PhantomData;

use ena::unify::{InPlaceUnificationTable, NoError, UnifyKey, UnifyValue};
use rustc_hash::FxHashMap;

use crate::semi::ty::{GenericId, HoleId, Ty, TyCtxt, TyKind};

/// A generalization level (Rémy/OCaml-style rank). Entering a binder bumps the
/// level; generalization keeps only variables born at the current level, so
/// nothing escapes its scope. Threaded in from the start — retrofitting levels
/// later is painful.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Level(pub u32);

/// The `ena` unification key. It carries `'tcx` via `PhantomData` so its
/// associated `Value` can be a `'tcx`-bearing [`HoleValue`]; the real identity
/// is the [`HoleId`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct HoleKey<'tcx> {
    id: HoleId,
    _marker: PhantomData<&'tcx ()>,
}

fn key<'tcx>(id: HoleId) -> HoleKey<'tcx> {
    HoleKey {
        id,
        _marker: PhantomData,
    }
}

/// The value `ena` stores per equivalence class.
#[derive(Clone, Debug)]
pub enum HoleValue<'tcx> {
    Unknown { level: Level },
    Known(Ty<'tcx>),
}

impl<'tcx> UnifyKey for HoleKey<'tcx> {
    type Value = HoleValue<'tcx>;

    fn index(&self) -> u32 {
        self.id.0
    }

    fn from_index(index: u32) -> Self {
        key(HoleId(index))
    }

    fn tag() -> &'static str {
        "HoleKey"
    }
}

impl<'tcx> UnifyValue for HoleValue<'tcx> {
    // Two *known* classes never unify through here — callers shallow-resolve and
    // unify the underlying types structurally — so the merge is total.
    type Error = NoError;

    fn unify_values(a: &Self, b: &Self) -> Result<Self, Self::Error> {
        use HoleValue::*;
        Ok(match (a, b) {
            (Unknown { level: l1 }, Unknown { level: l2 }) => Unknown {
                level: (*l1).min(*l2),
            },
            (Known(t), Unknown { .. }) | (Unknown { .. }, Known(t)) => Known(*t),
            // Defensive fallback only; see the note above.
            (Known(t), Known(_)) => Known(*t),
        })
    }
}

/// A unification failure: the two (shallow-resolved) types that clashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mismatch<'tcx> {
    pub left: Ty<'tcx>,
    pub right: Ty<'tcx>,
}

/// A use-site instantiation: a definition's generics mapped to fresh holes.
/// Consulted lazily by [`InferCtxt::unify_instantiated`]; never eagerly applied.
pub struct Instantiation<'tcx> {
    map: FxHashMap<GenericId, Ty<'tcx>>,
}

impl<'tcx> Instantiation<'tcx> {
    /// Build an instantiation from explicit `generic ↦ type` pairs (the types
    /// may be concrete or holes).
    pub fn from_pairs(pairs: impl IntoIterator<Item = (GenericId, Ty<'tcx>)>) -> Self {
        Instantiation {
            map: pairs.into_iter().collect(),
        }
    }

    pub fn get(&self, generic: GenericId) -> Option<Ty<'tcx>> {
        self.map.get(&generic).copied()
    }
}

/// The inference context: the hole table, the current level, and the interner.
pub struct InferCtxt<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    table: InPlaceUnificationTable<HoleKey<'tcx>>,
    level: Level,
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn new(tcx: &'a TyCtxt<'tcx>) -> Self {
        InferCtxt {
            tcx,
            table: InPlaceUnificationTable::new(),
            level: Level(0),
        }
    }

    pub fn level(&self) -> Level {
        self.level
    }

    pub fn enter_level(&mut self) -> Level {
        self.level.0 += 1;
        self.level
    }

    pub fn exit_level(&mut self) {
        debug_assert!(
            self.level.0 > 0,
            "exit_level underflow: not inside a binder"
        );
        self.level.0 -= 1;
    }

    /// Allocate a fresh unification variable at the current level.
    pub fn new_hole(&mut self) -> HoleId {
        self.table
            .new_key(HoleValue::Unknown { level: self.level })
            .id
    }

    pub fn new_hole_ty(&mut self) -> Ty<'tcx> {
        let hole = self.new_hole();
        self.tcx.mk_hole(hole)
    }

    /// Follow solved holes one layer at a time. An unknown hole resolves to its
    /// class representative so equal holes compare equal.
    pub fn shallow_resolve(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.kind() {
            TyKind::Hole(h) => match self.table.probe_value(key(*h)) {
                HoleValue::Known(t) => self.shallow_resolve(t),
                HoleValue::Unknown { .. } => self.tcx.mk_hole(self.table.find(key(*h)).id),
            },
            _ => ty,
        }
    }

    /// Deeply substitute every solved hole; unknown holes stay as their
    /// representative. (Reads the union-find — this is zonking, not `subst`.)
    pub fn resolve(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.shallow_resolve(ty);
        match ty.kind() {
            TyKind::Record { path, args, flex } => {
                let resolved: Vec<Ty<'tcx>> = args.iter().map(|a| self.resolve(*a)).collect();
                self.tcx.mk(TyKind::Record {
                    path,
                    args: self.tcx.intern_tys(&resolved),
                    flex: *flex,
                })
            }
            TyKind::Closure { params, ret } => {
                let resolved: Vec<Ty<'tcx>> = params.iter().map(|p| self.resolve(*p)).collect();
                let ret = self.resolve(*ret);
                self.tcx.mk(TyKind::Closure {
                    params: self.tcx.intern_tys(&resolved),
                    ret,
                })
            }
            TyKind::Nullable(inner) => {
                let inner = self.resolve(*inner);
                self.tcx.mk_nullable(inner)
            }
            _ => ty,
        }
    }

    /// Unify two types, solving holes as needed.
    pub fn unify(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> Result<(), Mismatch<'tcx>> {
        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);
        if a == b {
            // Interned: identical handles are identical types.
            return Ok(());
        }
        match (a.kind(), b.kind()) {
            (TyKind::Hole(h1), TyKind::Hole(h2)) => {
                self.table.union(key(*h1), key(*h2));
                Ok(())
            }
            (TyKind::Hole(h), _) => self.solve(*h, b),
            (_, TyKind::Hole(h)) => self.solve(*h, a),

            // Flexivity is a coloring overlaid on the structural type; it is not
            // part of unification identity (it is resolved separately).
            (
                TyKind::Record {
                    path: p1, args: a1, ..
                },
                TyKind::Record {
                    path: p2, args: a2, ..
                },
            ) if p1 == p2 && a1.len() == a2.len() => {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    self.unify(*x, *y)?;
                }
                Ok(())
            }
            (
                TyKind::Closure {
                    params: p1,
                    ret: r1,
                },
                TyKind::Closure {
                    params: p2,
                    ret: r2,
                },
            ) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify(*x, *y)?;
                }
                self.unify(*r1, *r2)
            }
            (TyKind::Nullable(x), TyKind::Nullable(y)) => self.unify(*x, *y),

            (TyKind::Int(x), TyKind::Int(y)) if x == y => Ok(()),
            (TyKind::Fp(x), TyKind::Fp(y)) if x == y => Ok(()),
            (TyKind::Bool, TyKind::Bool)
            | (TyKind::Str, TyKind::Str)
            | (TyKind::Unit, TyKind::Unit) => Ok(()),
            (TyKind::Generic(x), TyKind::Generic(y)) if x == y => Ok(()),

            (TyKind::Bottom, _) | (_, TyKind::Bottom) => Ok(()),

            _ => Err(Mismatch { left: a, right: b }),
        }
    }

    /// Solve hole `h` to `ty`, after the occurs-check.
    fn solve(&mut self, h: HoleId, ty: Ty<'tcx>) -> Result<(), Mismatch<'tcx>> {
        if self.occurs(h, ty) {
            return Err(Mismatch {
                left: self.tcx.mk_hole(h),
                right: ty,
            });
        }
        self.table.union_value(key(h), HoleValue::Known(ty));
        Ok(())
    }

    /// Does `h`'s class occur within `ty`? Prevents infinite types.
    fn occurs(&mut self, h: HoleId, ty: Ty<'tcx>) -> bool {
        match self.shallow_resolve(ty).kind() {
            TyKind::Hole(h2) => self.table.find(key(h)) == self.table.find(key(*h2)),
            TyKind::Record { args, .. } => {
                for a in args.iter() {
                    if self.occurs(h, *a) {
                        return true;
                    }
                }
                false
            }
            TyKind::Closure { params, ret } => {
                for p in params.iter() {
                    if self.occurs(h, *p) {
                        return true;
                    }
                }
                self.occurs(h, *ret)
            }
            TyKind::Nullable(inner) => self.occurs(h, *inner),
            _ => false,
        }
    }

    /// Run `f` speculatively. On `Err`, every unification inside is rolled back;
    /// on `Ok` the changes are kept. This backs method resolution and the
    /// generic-argument ambiguity.
    pub fn probe<T, E>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, E>) -> Result<T, E> {
        let snapshot = self.table.snapshot();
        match f(self) {
            Ok(value) => {
                self.table.commit(snapshot);
                Ok(value)
            }
            Err(err) => {
                self.table.rollback_to(snapshot);
                Err(err)
            }
        }
    }

    /// Instantiate a definition's generics to fresh holes for one use site.
    pub fn instantiate(&mut self, generics: &[GenericId]) -> Instantiation<'tcx> {
        let mut map = FxHashMap::default();
        map.reserve(generics.len());
        for &g in generics {
            let hole = self.new_hole_ty();
            map.insert(g, hole);
        }
        Instantiation { map }
    }

    /// Unify an `ambient` type against a `template` read under `inst`. The
    /// template's generics are resolved lazily through `inst`; nothing is
    /// rebuilt except when a template must be assigned into a hole, where it is
    /// materialized via [`InferCtxt::materialize`].
    pub fn unify_instantiated(
        &mut self,
        template: Ty<'tcx>,
        inst: &Instantiation<'tcx>,
        ambient: Ty<'tcx>,
    ) -> Result<(), Mismatch<'tcx>> {
        // Map a leading template generic to its hole, then resolve both sides.
        let template = match template.kind() {
            TyKind::Generic(g) => inst.get(*g).unwrap_or(template),
            _ => template,
        };
        let template = self.shallow_resolve(template);
        let ambient = self.shallow_resolve(ambient);
        if template == ambient {
            return Ok(());
        }
        match (template.kind(), ambient.kind()) {
            (TyKind::Hole(h1), TyKind::Hole(h2)) => {
                self.table.union(key(*h1), key(*h2));
                Ok(())
            }
            // Assigning the template into a hole: materialize its generics now.
            (_, TyKind::Hole(h)) => {
                let solved = self.materialize(template, inst);
                self.solve(*h, solved)
            }
            (TyKind::Hole(h), _) => self.solve(*h, ambient),

            (
                TyKind::Record {
                    path: p1, args: a1, ..
                },
                TyKind::Record {
                    path: p2, args: a2, ..
                },
            ) if p1 == p2 && a1.len() == a2.len() => {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    self.unify_instantiated(*x, inst, *y)?;
                }
                Ok(())
            }
            (
                TyKind::Closure {
                    params: p1,
                    ret: r1,
                },
                TyKind::Closure {
                    params: p2,
                    ret: r2,
                },
            ) if p1.len() == p2.len() => {
                for (x, y) in p1.iter().zip(p2.iter()) {
                    self.unify_instantiated(*x, inst, *y)?;
                }
                self.unify_instantiated(*r1, inst, *r2)
            }
            (TyKind::Nullable(x), TyKind::Nullable(y)) => self.unify_instantiated(*x, inst, *y),

            (TyKind::Int(x), TyKind::Int(y)) if x == y => Ok(()),
            (TyKind::Fp(x), TyKind::Fp(y)) if x == y => Ok(()),
            (TyKind::Bool, TyKind::Bool)
            | (TyKind::Str, TyKind::Str)
            | (TyKind::Unit, TyKind::Unit) => Ok(()),
            (TyKind::Generic(x), TyKind::Generic(y)) if x == y => Ok(()),

            (TyKind::Bottom, _) | (_, TyKind::Bottom) => Ok(()),

            _ => Err(Mismatch {
                left: template,
                right: ambient,
            }),
        }
    }

    /// Instantiate a `template` type under `inst`: rebuild it replacing each
    /// generic by its binding. The bounded substitution that happens at a use
    /// site (a call/ctor/field access); never used for ordinary checking.
    pub fn instantiate_ty(&self, template: Ty<'tcx>, inst: &Instantiation<'tcx>) -> Ty<'tcx> {
        self.materialize(template, inst)
    }

    /// Rebuild `template` into the arena, replacing its generics by the bindings
    /// in `inst`. This is the *only* substitution, and only happens at
    /// instantiation boundaries.
    fn materialize(&self, template: Ty<'tcx>, inst: &Instantiation<'tcx>) -> Ty<'tcx> {
        match template.kind() {
            TyKind::Generic(g) => inst.get(*g).unwrap_or(template),
            TyKind::Record { path, args, flex } => {
                let built: Vec<Ty<'tcx>> =
                    args.iter().map(|a| self.materialize(*a, inst)).collect();
                self.tcx.mk(TyKind::Record {
                    path,
                    args: self.tcx.intern_tys(&built),
                    flex: *flex,
                })
            }
            TyKind::Closure { params, ret } => {
                let built: Vec<Ty<'tcx>> =
                    params.iter().map(|p| self.materialize(*p, inst)).collect();
                let ret = self.materialize(*ret, inst);
                self.tcx.mk(TyKind::Closure {
                    params: self.tcx.intern_tys(&built),
                    ret,
                })
            }
            TyKind::Nullable(inner) => {
                let inner = self.materialize(*inner, inst);
                self.tcx.mk_nullable(inner)
            }
            _ => template,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::ty::IntTy;
    use crate::with_tcx;

    #[test]
    fn solve_var_then_resolve() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let h = ic.new_hole_ty();
            let i32 = tcx.mk_int(IntTy::Signed(32));
            ic.unify(h, i32).unwrap();
            assert_eq!(ic.resolve(h), i32);
        });
    }

    #[test]
    fn interned_types_are_pointer_equal() {
        with_tcx(|tcx| {
            let a = tcx.mk_record(
                "List",
                &[tcx.mk_int(IntTy::Signed(32))],
                crate::semi::ty::Capability::Rigid,
            );
            let b = tcx.mk_record(
                "List",
                &[tcx.mk_int(IntTy::Signed(32))],
                crate::semi::ty::Capability::Rigid,
            );
            // Structurally equal -> the very same handle.
            assert_eq!(a, b);
            assert!(std::ptr::eq(a.kind(), b.kind()));
        });
    }

    #[test]
    fn linked_vars_share_a_solution() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let a = ic.new_hole_ty();
            let b = ic.new_hole_ty();
            ic.unify(a, b).unwrap();
            ic.unify(b, tcx.mk_bool()).unwrap();
            assert_eq!(ic.resolve(a), tcx.mk_bool());
        });
    }

    #[test]
    fn occurs_check_rejects_infinite_type() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let h = ic.new_hole_ty();
            let recursive = tcx.mk_nullable(h);
            assert!(ic.unify(h, recursive).is_err());
        });
    }

    #[test]
    fn probe_rolls_back_on_err() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let h = ic.new_hole_ty();
            let _: Result<(), ()> = ic.probe(|ic| {
                ic.unify(h, tcx.mk_bool()).unwrap();
                Err(())
            });
            assert!(matches!(ic.resolve(h).kind(), TyKind::Hole(_)));
        });
    }

    #[test]
    fn lazy_instantiation_identity() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            // Template `fn id<T>(T) -> T` as a closure over generic #0.
            let t = tcx.mk_generic(GenericId(0));
            let id_ty = tcx.mk_closure(&[t], t);
            let inst = ic.instantiate(&[GenericId(0)]);

            let i32 = tcx.mk_int(IntTy::Signed(32));
            let use_ty = tcx.mk_closure(&[i32], ic.new_hole_ty());
            ic.unify_instantiated(id_ty, &inst, use_ty).unwrap();

            // The return hole flowed to i32 via the shared instantiation var.
            let TyKind::Closure { ret, .. } = use_ty.kind() else {
                panic!("expected closure");
            };
            assert_eq!(ic.resolve(*ret), i32);
        });
    }

    #[test]
    fn lazy_instantiation_materializes_into_hole() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let t = tcx.mk_generic(GenericId(0));
            let inst = ic.instantiate(&[GenericId(0)]);

            // Solve the instantiation var to i32 via the argument...
            let i32 = tcx.mk_int(IntTy::Signed(32));
            ic.unify_instantiated(t, &inst, i32).unwrap();

            // ...then a `Nullable<T>` template assigned into a hole materializes
            // as `Nullable<i32>`.
            let template = tcx.mk_nullable(t);
            let hole = ic.new_hole_ty();
            ic.unify_instantiated(template, &inst, hole).unwrap();
            assert_eq!(ic.resolve(hole), tcx.mk_nullable(i32));
        });
    }
}
