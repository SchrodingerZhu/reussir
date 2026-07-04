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
    ///
    /// Operational semantics. Let σ be the hole substitution (the `ena`
    /// union-find) and ⟦τ⟧ = `shallow_resolve` τ (follow solved holes one layer,
    /// map an unknown hole to its class representative). `resolve` realizes the
    /// judgment σ ⊢ τ ⇓ τ′ ("τ zonks to τ′"), a read-only pass that leaves σ
    /// unchanged. K ranges over the structural constructors with children —
    /// Record, Closure, Nullable — and the rebuilt node is re-interned:
    ///
    /// ```text
    ///   ⟦τ⟧ = K(τ₁ … τₙ)      σ ⊢ τᵢ ⇓ τ′ᵢ   (1 ≤ i ≤ n)
    /// ─────────────────────────────────────────────────── (ζ-congr)
    ///                σ ⊢ τ ⇓ K(τ′₁ … τ′ₙ)
    ///
    ///   ⟦τ⟧ = κ      (scalar, Generic, or unknown hole: no children)
    /// ───────────────────────────────────────────────────────────── (ζ-atom)
    ///                        σ ⊢ τ ⇓ κ
    /// ```
    pub fn resolve(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.shallow_resolve(ty);
        match ty.kind() {
            TyKind::Record { def, args, flex } => {
                let resolved: Vec<Ty<'tcx>> = args.iter().map(|a| self.resolve(*a)).collect();
                self.tcx.mk(TyKind::Record {
                    def: *def,
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
    ///
    /// Operational semantics. With σ the hole substitution (the `ena`
    /// union-find) and ⟦τ⟧ = `shallow_resolve` τ, `unify` realizes the
    /// state-threading judgment `σ ⊢ a ≡ b ⊣ σ′` ("under σ, a and b unify,
    /// yielding σ′").
    ///
    /// Both sides are shallow-resolved first. α, β range over holes; κ over the
    /// by-value atoms (Int, Fp, Bool, Str, Unit, Generic); ⊥ over Bottom. We
    /// write `σ, α↦τ` for solving a hole and `σ, α~β` for unioning two hole
    /// classes; ✗ is failure (a `Mismatch`). K (in `congr`) must agree on head
    /// and arity: Record (path + args — the flexivity coloring is *not* part of
    /// identity, so it is ignored here), Closure (params + ret), Nullable.
    ///
    /// ```text
    ///    ⟦a⟧ = ⟦b⟧                    ⟦a⟧ = α    ⟦b⟧ = β    α ≠ β
    /// ───────────────── (refl)     ───────────────────────────── (union)
    ///   σ ⊢ a ≡ b ⊣ σ                   σ ⊢ a ≡ b ⊣ σ, α~β
    ///
    ///   ⟦a⟧ = α    ⟦b⟧ = τ    τ ≠ hole    α ∉ τ
    /// ────────────────────────────────────────── (solve)   (+ symmetric)
    ///            σ ⊢ a ≡ b ⊣ σ, α↦τ
    ///
    ///   ⟦a⟧ = K(a₁ … aₙ)    ⟦b⟧ = K(b₁ … bₙ)    σ₀ = σ
    ///   σᵢ₋₁ ⊢ aᵢ ≡ bᵢ ⊣ σᵢ    (1 ≤ i ≤ n)
    /// ──────────────────────────────────────────────── (congr)
    ///                σ ⊢ a ≡ b ⊣ σₙ
    ///
    ///   ⟦a⟧ = ⊥  or  ⟦b⟧ = ⊥           ⟦a⟧, ⟦b⟧ otherwise distinct
    /// ────────────────────── (bottom)  ───────────────────────────── (clash)
    ///    σ ⊢ a ≡ b ⊣ σ                       σ ⊢ a ≡ b ⊣ ✗
    /// ```
    ///
    /// The (solve) side condition α ∉ τ is the occurs-check; its failure is
    /// itself a ✗. Rule precedence follows the match order: (refl), (union),
    /// (solve), (congr), (bottom), then (clash) as the catch-all.
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
                    def: p1, args: a1, ..
                },
                TyKind::Record {
                    def: p2, args: a2, ..
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
            | (TyKind::Char, TyKind::Char)
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

    /// Unify a definition's type — the **template** — read through an
    /// instantiation, against an **ambient** type from the surrounding inference
    /// state, solving holes as needed.
    ///
    /// Terminology:
    /// * **template** `t`: a type taken from a *definition's* signature — a
    ///   parameter, a return, a constructor field — still written in that
    ///   definition's own generic parameters (`TyKind::Generic`). It is the
    ///   polymorphic schema being used at one site.
    /// * **`inst`** (θ): the *use-site instantiation* mapping each of the
    ///   definition's generics to a fresh hole (or an explicit type argument)
    ///   for this use. The template's generics are read *through* θ.
    /// * **ambient** `a`: the type the surrounding context demands — the actual
    ///   argument's type, the expected type — already expressed in the caller's
    ///   own inference state (its holes and its *own* rigid generics).
    ///
    /// The point is **laziness**: instead of substituting θ through the whole
    /// template up front (allocating an instantiated copy) and then unifying, we
    /// thread θ down as we descend, mapping a generic to its hole only at the
    /// node under inspection. The template is fully rebuilt (`materialize`) in
    /// exactly one case — when it must be *stored* as a hole's solution — because
    /// a value recorded in the table is read later with no θ in hand, so it must
    /// be self-contained (every `Generic` already replaced).
    ///
    /// Operational semantics. With σ the hole substitution, θ = `inst`,
    /// ⟦τ⟧ = `shallow_resolve` τ, θ▸τ the one-step head mapping of a generic
    /// (θ▸(Generic g) = θ(g) when g ∈ θ, else τ unchanged), and ⌈τ⌉θ =
    /// `materialize` τ under θ (the full eager substitution), write the
    /// normalized sides t̂ = ⟦θ▸t⟧ and â = ⟦a⟧. `unify_instantiated` realizes
    /// `σ; θ ⊢ t ⋈ a ⊣ σ′`:
    ///
    /// ```text
    ///    t̂ = â                      t̂ = α    â = β    α ≠ β
    /// ───────────────── (refl)    ───────────────────────────── (union)
    ///  σ;θ ⊢ t ⋈ a ⊣ σ               σ;θ ⊢ t ⋈ a ⊣ σ, α~β
    ///
    ///   â = β    t̂ ≠ hole    β ∉ ⌈t̂⌉θ
    /// ──────────────────────────────────── (materialize)
    ///        σ;θ ⊢ t ⋈ a ⊣ σ, β ↦ ⌈t̂⌉θ
    ///
    ///   t̂ = α    â ≠ hole    α ∉ â
    /// ──────────────────────────────── (solve)
    ///        σ;θ ⊢ t ⋈ a ⊣ σ, α↦â
    ///
    ///   t̂ = K(t₁ … tₙ)    â = K(a₁ … aₙ)    σ₀ = σ
    ///   σᵢ₋₁;θ ⊢ tᵢ ⋈ aᵢ ⊣ σᵢ    (1 ≤ i ≤ n)
    /// ──────────────────────────────────────────────── (congr)
    ///                σ;θ ⊢ t ⋈ a ⊣ σₙ
    ///
    ///   t̂ = ⊥  or  â = ⊥           t̂, â otherwise distinct
    /// ────────────────────── (bottom)  ───────────────────────── (clash)
    ///   σ;θ ⊢ t ⋈ a ⊣ σ                 σ;θ ⊢ t ⋈ a ⊣ ✗
    /// ```
    ///
    /// The asymmetry is the whole idea: θ rides only the template side (the
    /// (congr) premises keep θ on the left), and the sole full substitution ⌈·⌉θ
    /// is in (materialize), when the template crosses into a hole. Precedence
    /// follows the match order — (refl), (union), (materialize), (solve),
    /// (congr), (bottom), (clash) — so the two hole cases are disjoint: both
    /// holes ⇒ (union); only `â` a hole ⇒ (materialize); only `t̂` a hole ⇒
    /// (solve). `unify` is exactly this judgment with θ empty.
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
                    def: p1, args: a1, ..
                },
                TyKind::Record {
                    def: p2, args: a2, ..
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
            | (TyKind::Char, TyKind::Char)
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
            TyKind::Record { def, args, flex } => {
                let built: Vec<Ty<'tcx>> =
                    args.iter().map(|a| self.materialize(*a, inst)).collect();
                self.tcx.mk(TyKind::Record {
                    def: *def,
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
    use crate::semi::ty::{DefId, IntTy};
    use crate::with_tcx;

    /// A distinct record `DefId` for tests (no resolution pass).
    fn def(n: u32) -> DefId {
        DefId(n)
    }

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
                def(1),
                &[tcx.mk_int(IntTy::Signed(32))],
                crate::semi::ty::Flexivity::Rigid,
            );
            let b = tcx.mk_record(
                def(1),
                &[tcx.mk_int(IntTy::Signed(32))],
                crate::semi::ty::Flexivity::Rigid,
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

    // ----- mismatches (the `clash` rule) -----

    #[test]
    fn scalar_clash_reports_both_sides() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let i32 = tcx.mk_int(IntTy::Signed(32));
            let boolean = tcx.mk_bool();
            let err = ic.unify(i32, boolean).unwrap_err();
            assert_eq!(err.left, i32);
            assert_eq!(err.right, boolean);
        });
    }

    #[test]
    fn record_head_and_arity_clash() {
        with_tcx(|tcx| {
            use crate::semi::ty::Flexivity::Irrelevant;
            let mut ic = InferCtxt::new(tcx);
            let i32 = tcx.mk_int(IntTy::Signed(32));
            // Different head constructor.
            let list = tcx.mk_record(def(1), &[i32], Irrelevant);
            let option = tcx.mk_record(def(2), &[i32], Irrelevant);
            assert!(ic.unify(list, option).is_err());
            // Same head, different arity.
            let pair1 = tcx.mk_record(def(3), &[i32], Irrelevant);
            let pair2 = tcx.mk_record(def(3), &[i32, i32], Irrelevant);
            assert!(ic.unify(pair1, pair2).is_err());
        });
    }

    #[test]
    fn nested_clash_reports_the_innermost_pair() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let i32 = tcx.mk_int(IntTy::Signed(32));
            let boolean = tcx.mk_bool();
            let f1 = tcx.mk_closure(&[i32], i32);
            let f2 = tcx.mk_closure(&[boolean], i32);
            // The congruence walk surfaces the innermost clash (the parameter
            // types), not the enclosing closures.
            let err = ic.unify(f1, f2).unwrap_err();
            assert_eq!(err.left, i32);
            assert_eq!(err.right, boolean);
        });
    }

    #[test]
    fn clash_resolves_through_a_solved_hole() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let h = ic.new_hole_ty();
            let i32 = tcx.mk_int(IntTy::Signed(32));
            ic.unify(h, i32).unwrap();
            // `h` is now i32; clashing with bool reports the *resolved* left
            // side, not the hole.
            let err = ic.unify(h, tcx.mk_bool()).unwrap_err();
            assert_eq!(err.left, i32);
            assert_eq!(err.right, tcx.mk_bool());
        });
    }

    #[test]
    fn bottom_absorbs_without_clash() {
        with_tcx(|tcx| {
            let mut ic = InferCtxt::new(tcx);
            let bottom = tcx.mk(TyKind::Bottom);
            let i32 = tcx.mk_int(IntTy::Signed(32));
            // Error recovery: Bottom unifies with anything, either orientation.
            assert!(ic.unify(bottom, i32).is_ok());
            assert!(ic.unify(i32, bottom).is_ok());
        });
    }
}
