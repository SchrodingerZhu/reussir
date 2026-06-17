//! Name resolution: globally-unique item identities ([`DefId`]) with
//! fully-qualified paths, and a scope that maps references to them.
//!
//! Surface code names items by bare or type-qualified token paths (`Foo`,
//! `Enum::Variant`); within a module those names are not unique across the whole
//! program once modules enter the picture. Resolution assigns every top-level
//! item a [`DefId`] and records its **fully-qualified path** (the enclosing
//! module segments followed by the item name), so two same-named records in
//! different modules are genuinely distinct — `TyKind::Record` carries the
//! `DefId`, not the bare name.
//!
//! Records (the *type* namespace) and functions (the *value* namespace) are
//! resolved separately, mirroring the surface, but share one dense `DefId`
//! space so a single id keys the type/HIR layers.
//!
//! Today every program is one flat module, so a path is just `[name]` and the
//! scope is one map; the representation already carries module prefixes so
//! nested modules and cross-unit resolution slot in without reshaping callers.

use reussir_syntax::kind::{Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::semi::ty::DefId;

/// Which namespace an item lives in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind {
    Record,
    Function,
}

/// A fully-qualified item path: the enclosing module segments, then the item
/// name as the final segment. Stored per [`DefId`] for display and mangling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedPath(pub Vec<TokenKey>);

impl QualifiedPath {
    /// The item's own (unqualified) name — the last segment.
    pub fn name(&self) -> TokenKey {
        *self.0.last().expect("a qualified path is never empty")
    }

    /// Render as `a::b::Name` using `resolver` to expand each segment.
    pub fn display(&self, resolver: &dyn Resolver<TokenKey>) -> String {
        self.0
            .iter()
            .map(|&seg| resolver.resolve(seg))
            .collect::<Vec<_>>()
            .join("::")
    }
}

#[derive(Clone, Debug)]
pub struct DefInfo {
    pub path: QualifiedPath,
    pub kind: DefKind,
}

/// The resolution registry: every item's [`DefInfo`] (indexed by [`DefId`]) plus
/// the current module's name→def scopes.
#[derive(Default)]
pub struct DefTable {
    defs: Vec<DefInfo>,
    /// The enclosing module path (empty = crate root) that items qualify under.
    module: Vec<TokenKey>,
    /// Type-namespace scope (records) and value-namespace scope (functions).
    types: FxHashMap<TokenKey, DefId>,
    values: FxHashMap<TokenKey, DefId>,
}

impl DefTable {
    pub fn new() -> Self {
        Self::default()
    }

    fn declare(
        &mut self,
        scope: impl Fn(&mut Self) -> &mut FxHashMap<TokenKey, DefId>,
        name: TokenKey,
        kind: DefKind,
    ) -> Option<DefId> {
        if scope(self).contains_key(&name) {
            return None; // name clash in this scope
        }
        let id = DefId(self.defs.len() as u32);
        let mut path = self.module.clone();
        path.push(name);
        self.defs.push(DefInfo {
            path: QualifiedPath(path),
            kind,
        });
        scope(self).insert(name, id);
        Some(id)
    }

    /// Declare a record in the current module's type namespace. `None` on clash.
    pub fn declare_record(&mut self, name: TokenKey) -> Option<DefId> {
        self.declare(|s| &mut s.types, name, DefKind::Record)
    }

    /// Declare a function in the current module's value namespace. `None` on
    /// clash.
    pub fn declare_function(&mut self, name: TokenKey) -> Option<DefId> {
        self.declare(|s| &mut s.values, name, DefKind::Function)
    }

    /// Resolve a type reference (a record name) in scope.
    pub fn resolve_record(&self, name: TokenKey) -> Option<DefId> {
        self.types.get(&name).copied()
    }

    /// Resolve a value reference (a function name) in scope.
    pub fn resolve_function(&self, name: TokenKey) -> Option<DefId> {
        self.values.get(&name).copied()
    }

    pub fn info(&self, def: DefId) -> &DefInfo {
        &self.defs[def.0 as usize]
    }

    pub fn path(&self, def: DefId) -> &QualifiedPath {
        &self.defs[def.0 as usize].path
    }
}

#[cfg(test)]
mod tests {
    use reussir_syntax::kind::InternKey;

    use super::*;

    /// A distinct interned key for a name in tests (no parser).
    fn k(n: u32) -> TokenKey {
        TokenKey::try_from_u32(n).expect("nonzero key")
    }

    #[test]
    fn declares_and_resolves_in_separate_namespaces() {
        let mut t = DefTable::new();
        let foo = t.declare_record(k(1)).expect("fresh");
        let bar = t.declare_function(k(2)).expect("fresh");
        assert_eq!(t.resolve_record(k(1)), Some(foo));
        assert_eq!(t.resolve_function(k(2)), Some(bar));
        // The type and value namespaces are independent.
        assert_eq!(t.resolve_function(k(1)), None);
        assert_eq!(t.resolve_record(k(2)), None);
        // A record and a function may share a name; they get distinct defs.
        let ty = t.declare_record(k(3)).expect("fresh");
        let val = t.declare_function(k(3)).expect("fresh");
        assert_ne!(ty, val);
        // The qualified path's final segment is the item's own name.
        assert_eq!(t.path(foo).name(), k(1));
        assert_eq!(t.info(bar).kind, DefKind::Function);
    }

    #[test]
    fn rejects_a_same_namespace_clash() {
        let mut t = DefTable::new();
        assert!(t.declare_record(k(1)).is_some());
        assert!(t.declare_record(k(1)).is_none());
    }
}
