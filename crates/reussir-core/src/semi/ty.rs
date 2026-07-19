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

/// The per-use *flexivity* a value carries — its regional memory coloring:
/// `Flex` is mutable but cannot be materialized out of its region, `Rigid` is
/// immutable but materializable (the frozen form that escapes a region),
/// `Regional` is the unrefined regional form, and `Irrelevant` is the
/// non-regional default.
///
/// This names the regional-coloring axis **only**; it is deliberately *not*
/// called a "capability", to keep it distinct from rc-management (value vs
/// shared vs regional, decided per record from its declaration). The two
/// interact in exactly one place — a region-managed record needs rc only when
/// its flexivity is `Rigid` (frozen/escaped); see `full::ownership`.
///
/// The colorings form a **refinement tree** — each one refines (is more specific
/// than) its parent:
///
/// ```text
/// Unknown (absent — a non-regional or not-yet-resolved head; see `flexivity()`)
/// ├─ Irrelevant   (a value/shared record: carries no regional coloring)
/// └─ Regional     (unrefined: may still resolve to flex or rigid)
///    ├─ Flex      (region-local, mutable)
///    └─ Rigid     (frozen, materializable)
/// ```
///
/// Two colorings are **compatible** ([`compatible`](Self::compatible)) when one
/// refines to the other — i.e. they are comparable, a directed path connects
/// them in the tree. The only incompatibilities are the sibling pairs: `Flex`
/// versus `Rigid` (a mutable region-local value is not a frozen one), and
/// `Irrelevant` versus a regional coloring. This is a compatibility *check*, not
/// full lattice unification: it does not rewrite a `Regional` head to the more
/// refined `Flex`/`Rigid` it meets.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Flexivity {
    Irrelevant,
    Regional,
    Flex,
    Rigid,
}

impl Flexivity {
    /// This coloring's parent in the refinement tree, or `None` for a child of
    /// the (absent) `Unknown` root.
    fn parent(self) -> Option<Flexivity> {
        match self {
            Flexivity::Irrelevant | Flexivity::Regional => None,
            Flexivity::Flex | Flexivity::Rigid => Some(Flexivity::Regional),
        }
    }

    /// Whether `self` is `other` or refines from it — `self` lies on the path
    /// from the root down to `other` (equivalently, `other` is an ancestor of
    /// `self`, or they are equal).
    fn refines_from(self, other: Flexivity) -> bool {
        self == other || self.parent().is_some_and(|p| p.refines_from(other))
    }

    /// Whether two colorings are compatible: one refines to the other, so a
    /// directed path connects them in the [refinement tree](Flexivity). `Flex`
    /// and `Rigid` (incomparable siblings), and `Irrelevant` versus a regional
    /// coloring, are the only incompatible pairs.
    pub fn compatible(self, other: Flexivity) -> bool {
        self.refines_from(other) || other.refines_from(self)
    }
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

/// The synchronization discipline of a [`TyKind::Cell`], mirroring the
/// dialect's `CellKind` lattice (thread-safety design §4). The kind decides
/// whether the cell's box is atomically counted, whether the cell is `Sync`,
/// which element types it admits, and which `core::intrinsic::cell`
/// operations apply.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CellKind {
    /// Surface `Cell<T>`: whole-element `get`/`set`, no synchronization.
    Plain,
    /// Surface `RefCell<T>`: adds the guarded `rmw` move-out region and the
    /// `in_use` flag observation.
    Exclusive,
    /// Surface `Atomic<T>`: an inline atomic arithmetic scalar; accesses are
    /// atomic loads/stores/rmw.
    Atomic,
    /// Surface `Mutex<T>`: accesses run inside a mutex critical section.
    Mutex,
    /// Surface `FlatLock<T>`: accesses run inside a flat-combining lock
    /// critical section (the body may execute on the combining thread).
    Flatlock,
    /// Surface `RwLock<T>`: reads (`get`/`rdlock`) share the read lock,
    /// writes (`set`/`rmw`) take it exclusively.
    Rwlock,
}

impl CellKind {
    /// The surface type-constructor name, also used by the textual IR.
    pub fn surface_name(self) -> &'static str {
        match self {
            CellKind::Plain => "Cell",
            CellKind::Exclusive => "RefCell",
            CellKind::Atomic => "Atomic",
            CellKind::Mutex => "Mutex",
            CellKind::Flatlock => "FlatLock",
            CellKind::Rwlock => "RwLock",
        }
    }

    /// Parse a surface / textual-IR type-constructor name.
    pub fn parse(name: &str) -> Option<CellKind> {
        match name {
            "Cell" => Some(CellKind::Plain),
            "RefCell" => Some(CellKind::Exclusive),
            "Atomic" => Some(CellKind::Atomic),
            "Mutex" => Some(CellKind::Mutex),
            "FlatLock" => Some(CellKind::Flatlock),
            "RwLock" => Some(CellKind::Rwlock),
            _ => None,
        }
    }

    /// Whether the cell is a synchronization primitive: born in an atomic
    /// shared box and `Sync` by itself (given its element bound).
    pub fn is_sync(self) -> bool {
        matches!(
            self,
            CellKind::Atomic | CellKind::Mutex | CellKind::Flatlock | CellKind::Rwlock
        )
    }

    /// Whether accesses are guarded by a lock. Lock-guarded cells require a
    /// `Sync` element: the lock only protects the stored slot, while clones
    /// of the element leave the critical section and are used concurrently.
    pub fn is_lock(self) -> bool {
        matches!(
            self,
            CellKind::Mutex | CellKind::Flatlock | CellKind::Rwlock
        )
    }
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
        flex: Flexivity,
    },
    Int(IntTy),
    Fp(FpTy),
    Bool,
    Str,
    Char,
    Unit,
    Closure {
        params: &'tcx [Ty<'tcx>],
        ret: Ty<'tcx>,
    },
    /// A statically shaped multidimensional array of `Plain` elements,
    /// reference counted as a whole; the extents are part of the type
    /// (`[f64; 512, 512]`). See issue #344.
    Array {
        elem: Ty<'tcx>,
        dims: &'tcx [u64],
    },
    /// A shared mutable cell. The cell box is always reference-counted; `T`
    /// may itself contain reference-counted ownership. The [`CellKind`]
    /// mirrors the dialect's lattice and decides both the synchronization
    /// discipline and which `core::intrinsic::cell` operations apply.
    Cell {
        elem: Ty<'tcx>,
        kind: CellKind,
    },
    Nullable(Ty<'tcx>),
    /// An atomically reference-counted coloring of a `[shared]` record:
    /// `Arc<X>` is the same nominal `X` behind a box whose refcount is
    /// adjusted with atomic operations, so the value may cross threads. Only
    /// the rc discipline differs — reads (projection, matching) see `X`.
    Arc(Ty<'tcx>),
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

    /// The flexivity (regional coloring) a value of this type carries, if any.
    /// Only nominal records carry one today.
    pub fn flexivity(self) -> Option<Flexivity> {
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

    pub fn mk_char(&self) -> Ty<'tcx> {
        self.mk(TyKind::Char)
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

    pub fn mk_array(&self, elem: Ty<'tcx>, dims: &[u64]) -> Ty<'tcx> {
        let dims = self.alloc_slice(dims);
        self.mk(TyKind::Array { elem, dims })
    }

    pub fn mk_cell(&self, elem: Ty<'tcx>, kind: CellKind) -> Ty<'tcx> {
        self.mk(TyKind::Cell { elem, kind })
    }

    pub fn mk_nullable(&self, inner: Ty<'tcx>) -> Ty<'tcx> {
        self.mk(TyKind::Nullable(inner))
    }

    pub fn mk_arc(&self, inner: Ty<'tcx>) -> Ty<'tcx> {
        self.mk(TyKind::Arc(inner))
    }

    pub fn mk_record(&self, def: DefId, args: &[Ty<'tcx>], flex: Flexivity) -> Ty<'tcx> {
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

#[cfg(test)]
mod tests {
    use super::Flexivity::{Flex, Irrelevant, Regional, Rigid};

    #[test]
    fn flexivity_compatibility_follows_the_refinement_tree() {
        // Reflexive, and symmetric.
        for x in [Irrelevant, Regional, Flex, Rigid] {
            assert!(x.compatible(x), "{x:?} should be compatible with itself");
            for y in [Irrelevant, Regional, Flex, Rigid] {
                assert_eq!(x.compatible(y), y.compatible(x), "symmetry {x:?}/{y:?}");
            }
        }
        // Regional is the parent of Flex and Rigid, so it refines to either.
        assert!(Regional.compatible(Flex));
        assert!(Regional.compatible(Rigid));
        // Siblings do not refine to each other.
        assert!(!Flex.compatible(Rigid));
        // Irrelevant is in a separate subtree from the regional colorings.
        assert!(!Irrelevant.compatible(Regional));
        assert!(!Irrelevant.compatible(Flex));
        assert!(!Irrelevant.compatible(Rigid));
    }
}
