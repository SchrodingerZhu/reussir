//! Language items: the declarations the compiler itself must know.
//!
//! A lang item ties a name the compiler wires into checking (a numeric
//! tower trait, an operator trait, the `Ordering` enum) to an actual
//! declaration. The declaration comes from one of two places, in
//! precedence order:
//!
//! 1. **Source or a loaded interface** carrying `#[lang("…")]` — how the
//!    `core` package hands the compiler its declarations, complete with
//!    real source locations for diagnostics, debug info, and the LSP.
//! 2. **The compiler's fallback table** (`builtin_traits!`) — what a
//!    session gets when no `core` is compiled against: unit tests, bare
//!    `rrc file.rr`, the REPL today. Fallback items carry no source
//!    location until their declarations migrate into `core`.
//!
//! The wired numeric/marker tower is fallback-registered in every session
//! for now; migrated items will suppress their fallback entry instead
//! (a later PR in this series).

use rustc_hash::FxHashMap;

use crate::semi::traits::TraitId;
use crate::semi::traits::builtins::Builtins;
use crate::semi::ty::DefId;

/// A `core::intrinsic` operation, as a lang item: the marker on the
/// prototype `core` declares for it, so the operation has a real
/// declaration site (locations for diagnostics, debug info, the LSP)
/// while checking and lowering stay wired in the compiler.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IntrinsicItem {
    Math(crate::intrinsic::MathFn),
    Array(crate::intrinsic::ArrayFn),
    Cell(crate::intrinsic::CellFn),
}

impl IntrinsicItem {
    pub fn family(self) -> &'static str {
        match self {
            IntrinsicItem::Math(_) => "math",
            IntrinsicItem::Array(_) => "array",
            IntrinsicItem::Cell(_) => "cell",
        }
    }

    pub fn op_name(self) -> &'static str {
        match self {
            IntrinsicItem::Math(f) => f.as_str(),
            IntrinsicItem::Array(f) => f.as_str(),
            IntrinsicItem::Cell(f) => f.as_str(),
        }
    }

    fn from_parts(family: &str, name: &str) -> Option<IntrinsicItem> {
        match family {
            "math" => crate::intrinsic::MathFn::parse(name).map(IntrinsicItem::Math),
            "array" => crate::intrinsic::ArrayFn::parse(name).map(IntrinsicItem::Array),
            "cell" => crate::intrinsic::CellFn::parse(name).map(IntrinsicItem::Cell),
            _ => None,
        }
    }
}

/// Every language item the compiler recognizes. Closed: an unknown
/// `#[lang("…")]` name is a diagnostic, so the set is versioned with the
/// compiler.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LangItem {
    /// The numeric/marker tower — wired, fallback-registered today.
    Num,
    Integral,
    FloatingPoint,
    PtrLike,
    Sync,
    /// The comparison tower — declared by `core` (a later PR in this
    /// series); the slots exist so its source can land without a compiler
    /// release in between.
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    /// `core::cmp::Ordering`, the result of `Ord::cmp`.
    Ordering,
    /// A `core::intrinsic` operation's prototype
    /// (`#[lang("core::intrinsic::math::sqrt")]`).
    Intrinsic(IntrinsicItem),
}

/// What kind of declaration a lang item must be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LangItemKind {
    Trait,
    Record,
    Function,
}

impl LangItem {
    /// The wired numeric/marker tower, in `Builtins` field order — the
    /// items whose [`LangItems`] fields repoint when a declaration binds.
    pub const TOWER: [LangItem; 5] = [
        LangItem::Num,
        LangItem::Integral,
        LangItem::FloatingPoint,
        LangItem::PtrLike,
        LangItem::Sync,
    ];

    /// The fixed (non-intrinsic) items, for enumeration.
    pub const FIXED: [LangItem; 10] = [
        LangItem::Num,
        LangItem::Integral,
        LangItem::FloatingPoint,
        LangItem::PtrLike,
        LangItem::Sync,
        LangItem::PartialEq,
        LangItem::Eq,
        LangItem::PartialOrd,
        LangItem::Ord,
        LangItem::Ordering,
    ];

    /// The attribute spelling: `#[lang("partial_eq")]`,
    /// `#[lang("core::intrinsic::math::sqrt")]`.
    pub fn name(self) -> String {
        match self {
            LangItem::Num => "num".into(),
            LangItem::Integral => "integral".into(),
            LangItem::FloatingPoint => "floating_point".into(),
            LangItem::PtrLike => "ptr_like".into(),
            LangItem::Sync => "sync".into(),
            LangItem::PartialEq => "partial_eq".into(),
            LangItem::Eq => "eq".into(),
            LangItem::PartialOrd => "partial_ord".into(),
            LangItem::Ord => "ord".into(),
            LangItem::Ordering => "ordering".into(),
            LangItem::Intrinsic(op) => {
                format!("core::intrinsic::{}::{}", op.family(), op.op_name())
            }
        }
    }

    pub fn from_name(name: &str) -> Option<LangItem> {
        if let Some(rest) = name.strip_prefix("core::intrinsic::") {
            let (family, op) = rest.split_once("::")?;
            return IntrinsicItem::from_parts(family, op).map(LangItem::Intrinsic);
        }
        Self::FIXED.into_iter().find(|item| item.name() == name)
    }

    /// The comparison tower's traits, in declaration (and fallback
    /// registration) order. `Ordering` is a record, not part of this set.
    pub const CMP_TRAITS: [LangItem; 4] = [
        LangItem::PartialEq,
        LangItem::Eq,
        LangItem::PartialOrd,
        LangItem::Ord,
    ];

    pub fn is_cmp_trait(self) -> bool {
        Self::CMP_TRAITS.contains(&self)
    }

    pub fn kind(self) -> LangItemKind {
        match self {
            LangItem::Ordering => LangItemKind::Record,
            LangItem::Intrinsic(_) => LangItemKind::Function,
            _ => LangItemKind::Trait,
        }
    }
}

/// The session's lang-item registry: the wired tower ids plus every
/// `#[lang]`-declared item. Lives on the elaborator; declarations are
/// insertion-ordered so the REPL checkpoint is a plain length and a
/// rejected batch retracts exactly its own (the same truncate-and-retain
/// scheme every other elaborator table uses).
pub struct LangItems {
    pub num: TraitId,
    pub integral: TraitId,
    pub floating_point: TraitId,
    pub ptr_like: TraitId,
    /// Marker (auto) trait: a value may be reached from any thread,
    /// concurrently; answered structurally, never by impl search.
    pub sync: TraitId,
    /// The construction-time builtin ids, in [`LangItem::TOWER`] order —
    /// retained after the wired fields repoint, so bare-name resolution
    /// can recognize (and substitute) a builtin hit.
    builtin: [TraitId; 5],
    declared: Vec<(LangItem, DefId)>,
}

impl LangItems {
    /// A registry over the fallback-registered tower.
    pub fn new(fallback: Builtins) -> Self {
        LangItems {
            num: fallback.num,
            integral: fallback.integral,
            floating_point: fallback.floating_point,
            ptr_like: fallback.ptr_like,
            sync: fallback.sync,
            builtin: [
                fallback.num,
                fallback.integral,
                fallback.floating_point,
                fallback.ptr_like,
                fallback.sync,
            ],
            declared: Vec::new(),
        }
    }

    /// Point a wired tower field at `id` — the declared trait (`core`'s)
    /// takes over from the construction builtin, so everything the checker
    /// wires through the field (literal bounds, arithmetic obligations,
    /// defaulting, the structural `Sync` answer, the `Num ⊴ PartialOrd`
    /// edge) follows the declaration.
    pub(crate) fn repoint(&mut self, item: LangItem, id: TraitId) {
        match item {
            LangItem::Num => self.num = id,
            LangItem::Integral => self.integral = id,
            LangItem::FloatingPoint => self.floating_point = id,
            LangItem::PtrLike => self.ptr_like = id,
            LangItem::Sync => self.sync = id,
            _ => {}
        }
    }

    /// The wired fields, in [`LangItem::TOWER`] order — the REPL
    /// checkpoint's snapshot (a rejected batch that repointed them must
    /// restore the builtins).
    pub(crate) fn fields(&self) -> [TraitId; 5] {
        [
            self.num,
            self.integral,
            self.floating_point,
            self.ptr_like,
            self.sync,
        ]
    }

    pub(crate) fn restore_fields(&mut self, fields: [TraitId; 5]) {
        [
            self.num,
            self.integral,
            self.floating_point,
            self.ptr_like,
            self.sync,
        ] = fields;
    }

    /// Substitute a bare-name hit on a *construction builtin* with the
    /// declared tower item, when one is bound: `T: Num` keeps resolving at
    /// the crate root in plain sessions, and resolves to `core`'s
    /// declaration the moment an interface (or the batch itself) binds it.
    /// Module-local shadows are their own defs and pass through untouched.
    pub fn override_builtin(&self, def: DefId, db: &crate::semi::traits::TraitDb<'_>) -> DefId {
        LangItem::TOWER
            .iter()
            .zip(self.builtin)
            .find(|&(_, builtin_id)| db.trait_def(builtin_id).def == def)
            .and_then(|(&item, _)| self.get(item))
            .unwrap_or(def)
    }

    /// Bind `item` to `def`. `Err` returns the prior declaration for the
    /// duplicate diagnostic (re-declaring the *same* def is idempotent —
    /// the same interface loaded twice).
    pub fn declare(&mut self, item: LangItem, def: DefId) -> Result<(), DefId> {
        match self.get(item) {
            None => {
                self.declared.push((item, def));
                Ok(())
            }
            Some(prior) if prior == def => Ok(()),
            Some(prior) => Err(prior),
        }
    }

    /// The declared def of `item`, if any source/interface declared it.
    pub fn get(&self, item: LangItem) -> Option<DefId> {
        self.declared
            .iter()
            .find(|&&(i, _)| i == item)
            .map(|&(_, def)| def)
    }

    /// Every `#[lang]`-declared binding, def-keyed — the serialization
    /// view (`.rri` emission re-prints the markers; the wired fallback
    /// tower is compiler-provided and never serializes).
    pub fn declared_by_def(&self) -> FxHashMap<DefId, LangItem> {
        self.declared
            .iter()
            .map(|&(item, def)| (def, item))
            .collect()
    }

    /// The lang item bound to `def`, if any.
    pub fn item_of(&self, def: DefId) -> Option<LangItem> {
        self.declared
            .iter()
            .find(|&&(_, d)| d == def)
            .map(|&(item, _)| item)
    }

    /// Whether `def` is a bound intrinsic prototype — a location anchor,
    /// not a callable value (calls to its path are intercepted as the
    /// intrinsic itself).
    pub fn is_intrinsic_def(&self, def: DefId) -> bool {
        self.declared
            .iter()
            .any(|&(item, d)| d == def && matches!(item, LangItem::Intrinsic(_)))
    }

    /// The checkpoint counter, in the elaborator's truncate-and-retain
    /// scheme.
    pub(crate) fn len(&self) -> usize {
        self.declared.len()
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.declared.truncate(len);
    }
}
