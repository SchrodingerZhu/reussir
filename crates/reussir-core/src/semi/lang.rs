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
}

/// What kind of declaration a lang item must be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LangItemKind {
    Trait,
    Record,
}

impl LangItem {
    pub const ALL: [LangItem; 10] = [
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

    /// The attribute spelling: `#[lang("partial_eq")]`.
    pub fn name(self) -> &'static str {
        match self {
            LangItem::Num => "num",
            LangItem::Integral => "integral",
            LangItem::FloatingPoint => "floating_point",
            LangItem::PtrLike => "ptr_like",
            LangItem::Sync => "sync",
            LangItem::PartialEq => "partial_eq",
            LangItem::Eq => "eq",
            LangItem::PartialOrd => "partial_ord",
            LangItem::Ord => "ord",
            LangItem::Ordering => "ordering",
        }
    }

    pub fn from_name(name: &str) -> Option<LangItem> {
        Self::ALL.into_iter().find(|item| item.name() == name)
    }

    pub fn kind(self) -> LangItemKind {
        match self {
            LangItem::Ordering => LangItemKind::Record,
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
            declared: Vec::new(),
        }
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

    /// The checkpoint counter, in the elaborator's truncate-and-retain
    /// scheme.
    pub(crate) fn len(&self) -> usize {
        self.declared.len()
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.declared.truncate(len);
    }
}
