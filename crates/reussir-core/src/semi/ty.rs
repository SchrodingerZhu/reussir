//! The type representation, interned for O(1) comparison.
//!
//! Every type node is hash-consed into a bump arena ([`TyCtxt`]), so a [`Ty`] is
//! just a pointer and equality/hashing are by identity. Two structurally equal
//! types are the *same* handle, which makes the unifier's equality checks and
//! the instance database's self-type matching pointer comparisons.
//!
//! [`TyKind::Generic`] is a rigid type parameter in scope; [`TyKind::Hole`] is a
//! unification variable (see [`crate::semi::infer`]). The arena lifetime `'tcx` brands
//! every handle, so no type can escape the [`stumpalo::Arena::with_scope`] that
//! created it.
//!
//! Sketch: this is specifically the *Semi*-phase type (it carries generics and
//! holes). The monomorphized *Full* type is a different representation; see the
//! crate-root note about the eventual `semi::`/`full::` split.

use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr;

use rustc_hash::FxHashMap;
use stumpalo::ArenaRef;

/// Identifies a resolved top-level item (a record or function). Globally unique
/// and path-aware (see [`crate::semi::resolve`]); two same-named records in
/// different modules get distinct `DefId`s, hence distinct nominal types.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DefId(pub u32);

/// Identifies a rigid type parameter (`<T>`) in scope.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct GenericId(pub u32);

/// Identifies a unification variable.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct HoleId(pub u32);

/// The per-use memory capability ("flexivity") a value carries: `Flex` is
/// mutable but cannot be materialized out of its region, `Rigid` is immutable
/// but materializable (the frozen form that escapes a region), `Regional` is
/// the unrefined regional form, and `Irrelevant` is the non-regional default.
///
/// These do **not** form a total order — `Flex` and `Rigid` are incomparable
/// (different axes: mutability vs materializability) — so there is no
/// subsumption helper. Capability-as-a-bound is deferred until its semantics are
/// settled; for now this is purely the coloring stored on [`TyKind::Record`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Capability {
    Irrelevant,
    Regional,
    Flex,
    Rigid,
}

/// A width-tagged integer type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntTy {
    Signed(u16),
    Unsigned(u16),
}

/// A floating-point type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FpTy {
    /// IEEE 754 binary float of the given width (16, 32, 64).
    Ieee(u16),
    BFloat16,
    Float8,
}

/// The structure of a type. Compound variants hold interned children
/// (`Ty<'tcx>` and arena slices), so `TyKind` is `Copy` and its derived
/// `Eq`/`Hash` bottom out in pointer comparison — exactly the hash-cons key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TyKind<'tcx> {
    /// A user-defined nominal type applied to arguments, at a capability. `def`
    /// is the record's resolved [`DefId`], so the nominal identity is path-aware
    /// (distinct modules ⇒ distinct types) rather than by-name.
    Record {
        def: DefId,
        args: &'tcx [Ty<'tcx>],
        flex: Capability,
    },
    Int(IntTy),
    Fp(FpTy),
    Bool,
    Str,
    Unit,
    Closure {
        params: &'tcx [Ty<'tcx>],
        ret: Ty<'tcx>,
    },
    Nullable(Ty<'tcx>),
    /// A rigid type parameter in scope.
    Generic(GenericId),
    /// A unification variable.
    Hole(HoleId),
    /// The error-recovery type; unifies with anything.
    Bottom,
}

/// An interned type: a pointer into the arena. `Copy`, with identity
/// equality/hashing.
#[derive(Clone, Copy)]
pub struct Ty<'tcx>(&'tcx TyKind<'tcx>);

impl<'tcx> Ty<'tcx> {
    /// The structure behind this handle.
    pub fn kind(self) -> &'tcx TyKind<'tcx> {
        self.0
    }

    /// The capability a value of this type carries, if any. Only nominal
    /// records carry one today.
    pub fn capability(self) -> Option<Capability> {
        match self.0 {
            TyKind::Record { flex, .. } => Some(*flex),
            _ => None,
        }
    }
}

impl PartialEq for Ty<'_> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}

impl Eq for Ty<'_> {}

impl Hash for Ty<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self.0, state);
    }
}

impl fmt::Debug for Ty<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

/// The interning context: the arena plus the hash-cons table. Carry one of
/// these (by shared reference) wherever types are built.
pub struct TyCtxt<'tcx> {
    arena: &'tcx ArenaRef<'tcx>,
    intern: RefCell<FxHashMap<TyKind<'tcx>, Ty<'tcx>>>,
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn new(arena: &'tcx ArenaRef<'tcx>) -> Self {
        TyCtxt {
            arena,
            intern: RefCell::new(FxHashMap::default()),
        }
    }

    /// Intern a kind, returning its canonical handle.
    pub fn mk(&self, kind: TyKind<'tcx>) -> Ty<'tcx> {
        if let Some(&ty) = self.intern.borrow().get(&kind) {
            return ty;
        }
        let allocated: &'tcx TyKind<'tcx> = self.arena.alloc(kind);
        let ty = Ty(allocated);
        self.intern.borrow_mut().insert(kind, ty);
        ty
    }

    /// Intern a slice of (already-interned) types into the arena.
    pub fn intern_tys(&self, tys: &[Ty<'tcx>]) -> &'tcx [Ty<'tcx>] {
        if tys.is_empty() {
            return &[];
        }
        self.arena.alloc_slice_copy(tys)
    }

    pub fn mk_int(&self, int: IntTy) -> Ty<'tcx> {
        self.mk(TyKind::Int(int))
    }

    pub fn mk_fp(&self, fp: FpTy) -> Ty<'tcx> {
        self.mk(TyKind::Fp(fp))
    }

    pub fn mk_bool(&self) -> Ty<'tcx> {
        self.mk(TyKind::Bool)
    }

    pub fn mk_str(&self) -> Ty<'tcx> {
        self.mk(TyKind::Str)
    }

    pub fn mk_unit(&self) -> Ty<'tcx> {
        self.mk(TyKind::Unit)
    }

    pub fn mk_generic(&self, generic: GenericId) -> Ty<'tcx> {
        self.mk(TyKind::Generic(generic))
    }

    pub fn mk_hole(&self, hole: HoleId) -> Ty<'tcx> {
        self.mk(TyKind::Hole(hole))
    }

    pub fn mk_nullable(&self, inner: Ty<'tcx>) -> Ty<'tcx> {
        self.mk(TyKind::Nullable(inner))
    }

    pub fn mk_record(&self, def: DefId, args: &[Ty<'tcx>], flex: Capability) -> Ty<'tcx> {
        let args = self.intern_tys(args);
        self.mk(TyKind::Record { def, args, flex })
    }

    pub fn mk_closure(&self, params: &[Ty<'tcx>], ret: Ty<'tcx>) -> Ty<'tcx> {
        let params = self.intern_tys(params);
        self.mk(TyKind::Closure { params, ret })
    }

    /// Allocate a value into the arena, returning a shared arena reference. Used
    /// by the Full MIR to keep its (arena-allocated, `Copy`) node tree alongside
    /// the types it references — these are *not* interned, just arena-owned.
    pub fn alloc<T>(&self, value: T) -> &'tcx T {
        self.arena.alloc(value)
    }

    /// Allocate a slice of `Copy` values into the arena.
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &'tcx [T] {
        if slice.is_empty() {
            return &[];
        }
        self.arena.alloc_slice_copy(slice)
    }
}
