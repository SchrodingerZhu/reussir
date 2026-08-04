//! Name resolution: globally-unique item identities ([`DefId`]) with
//! fully-qualified paths, and module-aware scopes that map references to them.
//!
//! Surface code names items by bare or qualified token paths (`Foo`,
//! `Enum::Variant`, `utils::math::double`). Resolution assigns every top-level
//! item a [`DefId`] and records its **fully-qualified path** (the enclosing
//! module segments followed by the item name), so two same-named records in
//! different modules are genuinely distinct — `TyKind::Record` carries the
//! `DefId`, not the bare name.
//!
//! Records (the *type* namespace) and functions (the *value* namespace) are
//! resolved separately, mirroring the surface, but share one dense `DefId`
//! space so a single id keys the type/HIR layers.
//!
//! The scopes are keyed by **full path**: a bare reference resolves in the
//! current module first and falls back to the crate root; a qualified
//! reference is tried absolute, then relative to the current module, then
//! relative to the package root (the reference-frontend rules). The `root::` /
//! `super::` keywords are classified by the *caller* (the elaborator owns the
//! text of interned segments) into [`PathSeg`]s before resolution.

use reussir_syntax::kind::{Resolver, TokenKey};
use rustc_hash::FxHashMap;

use crate::semi::ty::DefId;

/// Which namespace an item lives in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind {
    Record,
    Function,
    Trait,
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

/// One classified segment of a reference path: the `root`/`super` module
/// keywords, or an ordinary name. Classification is textual (the segments are
/// interned tokens), so the elaborator performs it and resolution stays
/// key-only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathSeg {
    /// `root::` — the package root (the first segment of the current module).
    Root,
    /// `super::` — the parent module (a leading run may pop several levels).
    Super,
    /// An ordinary module/item name.
    Name(TokenKey),
}

#[derive(Clone, Debug)]
pub struct DefInfo {
    pub path: QualifiedPath,
    pub kind: DefKind,
}

/// The resolution registry: every item's [`DefInfo`] (indexed by [`DefId`])
/// plus the full-path-keyed namespaces and the current module.
#[derive(Default)]
pub struct DefTable {
    defs: Vec<DefInfo>,
    /// The module path items currently declare under and bare references
    /// resolve against (empty = crate root; `[pkg, sub]` inside a package).
    module: Vec<TokenKey>,
    /// Type-namespace scope (records), value-namespace scope (functions),
    /// and trait-namespace scope, keyed by the item's full qualified path.
    types: FxHashMap<Box<[TokenKey]>, DefId>,
    values: FxHashMap<Box<[TokenKey]>, DefId>,
    traits: FxHashMap<Box<[TokenKey]>, DefId>,
}

impl DefTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Switch the module items declare under and references resolve against.
    pub fn set_module(&mut self, module: Vec<TokenKey>) {
        self.module = module;
    }

    /// The current module path (empty = crate root).
    pub fn module(&self) -> &[TokenKey] {
        &self.module
    }

    fn declare(
        &mut self,
        scope: impl Fn(&mut Self) -> &mut FxHashMap<Box<[TokenKey]>, DefId>,
        name: TokenKey,
        kind: DefKind,
    ) -> Option<DefId> {
        let mut path = self.module.clone();
        path.push(name);
        let key: Box<[TokenKey]> = path.clone().into_boxed_slice();
        if scope(self).contains_key(&key) {
            return None; // name clash in this module's namespace
        }
        let id = DefId(self.defs.len() as u32);
        self.defs.push(DefInfo {
            path: QualifiedPath(path),
            kind,
        });
        scope(self).insert(key, id);
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

    /// Declare a trait in the current module's trait namespace. `None` on
    /// clash.
    pub fn declare_trait(&mut self, name: TokenKey) -> Option<DefId> {
        self.declare(|s| &mut s.traits, name, DefKind::Trait)
    }

    /// Look up an exact fully-qualified path in the type namespace.
    pub fn lookup_record(&self, path: &[TokenKey]) -> Option<DefId> {
        self.types.get(path).copied()
    }

    /// Look up an exact fully-qualified path in the value namespace.
    pub fn lookup_function(&self, path: &[TokenKey]) -> Option<DefId> {
        self.values.get(path).copied()
    }

    /// Look up an exact fully-qualified path in the trait namespace.
    pub fn lookup_trait(&self, path: &[TokenKey]) -> Option<DefId> {
        self.traits.get(path).copied()
    }

    /// Resolve a *bare* type reference: the current module first, then the
    /// crate root.
    pub fn resolve_record(&self, name: TokenKey) -> Option<DefId> {
        self.resolve_bare(&self.types, name)
    }

    /// Resolve a *bare* value reference: the current module first, then the
    /// crate root.
    pub fn resolve_function(&self, name: TokenKey) -> Option<DefId> {
        self.resolve_bare(&self.values, name)
    }

    /// Resolve a *bare* trait reference: the current module first, then the
    /// crate root (where the builtin traits live, giving them prelude
    /// visibility that a same-named module-local trait shadows).
    pub fn resolve_trait(&self, name: TokenKey) -> Option<DefId> {
        self.resolve_bare(&self.traits, name)
    }

    fn resolve_bare(
        &self,
        scope: &FxHashMap<Box<[TokenKey]>, DefId>,
        name: TokenKey,
    ) -> Option<DefId> {
        let mut path = self.module.clone();
        path.push(name);
        if let Some(&id) = scope.get(path.as_slice()) {
            return Some(id);
        }
        scope.get([name].as_slice()).copied()
    }

    /// Resolve a *qualified* type reference (see [`Self::resolve_qualified`]).
    pub fn resolve_record_path(&self, segs: &[PathSeg], name: TokenKey) -> Option<DefId> {
        self.resolve_qualified(&self.types, segs, name)
    }

    /// Resolve a *qualified* value reference (see [`Self::resolve_qualified`]).
    pub fn resolve_function_path(&self, segs: &[PathSeg], name: TokenKey) -> Option<DefId> {
        self.resolve_qualified(&self.values, segs, name)
    }

    /// Resolve a *qualified* trait reference (see [`Self::resolve_qualified`]).
    pub fn resolve_trait_path(&self, segs: &[PathSeg], name: TokenKey) -> Option<DefId> {
        self.resolve_qualified(&self.traits, segs, name)
    }

    /// Resolve `segs::name` against the current module:
    ///
    /// * A path led by `root::` resolves against the package root (the first
    ///   segment of the current module); one led by `super::`s pops that many
    ///   trailing segments off the current module. The keyword run must be a
    ///   prefix; popping past the root fails.
    /// * Any other path is tried **absolute**, then **relative to the current
    ///   module**, then **relative to the package root** — first hit wins.
    fn resolve_qualified(
        &self,
        scope: &FxHashMap<Box<[TokenKey]>, DefId>,
        segs: &[PathSeg],
        name: TokenKey,
    ) -> Option<DefId> {
        let plain = |segs: &[PathSeg]| -> Option<Vec<TokenKey>> {
            segs.iter()
                .map(|s| match s {
                    PathSeg::Name(k) => Some(*k),
                    // `root`/`super` past the leading position do not nest.
                    PathSeg::Root | PathSeg::Super => None,
                })
                .collect()
        };
        match segs {
            [PathSeg::Root, rest @ ..] => {
                let rest = plain(rest)?;
                let mut path: Vec<TokenKey> = self.module.first().copied().into_iter().collect();
                path.extend(rest);
                path.push(name);
                scope.get(path.as_slice()).copied()
            }
            [PathSeg::Super, ..] => {
                let supers = segs.iter().take_while(|s| **s == PathSeg::Super).count();
                let rest = plain(&segs[supers..])?;
                // Each `super` pops one level; popping past the package root
                // is unresolvable.
                if supers > self.module.len() {
                    return None;
                }
                let mut path = self.module[..self.module.len() - supers].to_vec();
                path.extend(rest);
                path.push(name);
                scope.get(path.as_slice()).copied()
            }
            _ => {
                let segs = plain(segs)?;
                let mut absolute = segs.clone();
                absolute.push(name);
                if let Some(&id) = scope.get(absolute.as_slice()) {
                    return Some(id);
                }
                let mut in_module = self.module.clone();
                in_module.extend_from_slice(&segs);
                in_module.push(name);
                if let Some(&id) = scope.get(in_module.as_slice()) {
                    return Some(id);
                }
                let root = self.module.first().copied()?;
                let mut in_root = vec![root];
                in_root.extend_from_slice(&segs);
                in_root.push(name);
                scope.get(in_root.as_slice()).copied()
            }
        }
    }

    /// Every record name (the item's own, unqualified name) in the type
    /// namespace. Order is unspecified; used to build "did you mean"
    /// suggestions on a failed lookup.
    pub fn record_names(&self) -> impl Iterator<Item = TokenKey> + '_ {
        self.types
            .keys()
            .map(|p| *p.last().expect("paths are never empty"))
    }

    /// Every function name (the item's own, unqualified name) in the value
    /// namespace. Order is unspecified; used to build "did you mean"
    /// suggestions on a failed lookup.
    pub fn function_names(&self) -> impl Iterator<Item = TokenKey> + '_ {
        self.values
            .keys()
            .map(|p| *p.last().expect("paths are never empty"))
    }

    /// Every trait name (the item's own, unqualified name) in the trait
    /// namespace, for "did you mean" suggestions.
    pub fn trait_names(&self) -> impl Iterator<Item = TokenKey> + '_ {
        self.traits
            .keys()
            .map(|p| *p.last().expect("paths are never empty"))
    }

    /// The number of declared defs — a checkpoint for [`DefTable::truncate`].
    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Retract every def declared at index `>= len`, removing its path from
    /// the owning namespace scope. The REPL uses this to roll an elaboration
    /// attempt back atomically; it relies on `declare` rejecting clashes, so a
    /// path in a scope always maps to the def that introduced it.
    pub fn truncate(&mut self, len: usize) {
        while self.defs.len() > len {
            let id = DefId((self.defs.len() - 1) as u32);
            let info = self.defs.pop().expect("len checked above");
            let scope = match info.kind {
                DefKind::Record => &mut self.types,
                DefKind::Function => &mut self.values,
                DefKind::Trait => &mut self.traits,
            };
            let removed = scope.remove(info.path.0.as_slice());
            debug_assert_eq!(removed, Some(id), "scope must point at the popped def");
        }
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

    fn name(n: u32) -> PathSeg {
        PathSeg::Name(k(n))
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

    #[test]
    fn same_name_in_two_modules_is_distinct() {
        let mut t = DefTable::new();
        t.set_module(vec![k(10), k(11)]); // pkg::a
        let in_a = t.declare_function(k(1)).expect("fresh");
        t.set_module(vec![k(10), k(12)]); // pkg::b
        let in_b = t.declare_function(k(1)).expect("fresh");
        assert_ne!(in_a, in_b);
        // A bare reference resolves in the *current* module.
        assert_eq!(t.resolve_function(k(1)), Some(in_b));
        t.set_module(vec![k(10), k(11)]);
        assert_eq!(t.resolve_function(k(1)), Some(in_a));
    }

    #[test]
    fn qualified_paths_resolve_absolute_module_relative_and_root_relative() {
        let mut t = DefTable::new();
        // pkg::utils::math::double
        t.set_module(vec![k(10), k(20), k(21)]);
        let double = t.declare_function(k(1)).expect("fresh");
        // From pkg::lib (module [pkg]): `utils::math::double` resolves
        // root-relatively (and module-relatively — [pkg]+segs — identically).
        t.set_module(vec![k(10)]);
        assert_eq!(
            t.resolve_function_path(&[name(20), name(21)], k(1)),
            Some(double)
        );
        // Absolute (full path spelled out).
        assert_eq!(
            t.resolve_function_path(&[name(10), name(20), name(21)], k(1)),
            Some(double)
        );
        // From a sibling module, root-relative still lands.
        t.set_module(vec![k(10), k(30)]);
        assert_eq!(
            t.resolve_function_path(&[name(20), name(21)], k(1)),
            Some(double)
        );
        assert_eq!(t.resolve_function_path(&[name(99)], k(1)), None);
    }

    #[test]
    fn root_and_super_resolve_against_the_current_module() {
        let mut t = DefTable::new();
        // pkg::models::seed
        t.set_module(vec![k(10), k(40)]);
        let seed = t.declare_function(k(2)).expect("fresh");
        // pkg::utils::offset
        t.set_module(vec![k(10), k(20)]);
        let offset = t.declare_function(k(3)).expect("fresh");

        // From pkg::utils::math: `super::offset`, `super::super::models::seed`,
        // `root::models::seed`.
        t.set_module(vec![k(10), k(20), k(21)]);
        assert_eq!(
            t.resolve_function_path(&[PathSeg::Super], k(3)),
            Some(offset)
        );
        assert_eq!(
            t.resolve_function_path(&[PathSeg::Super, PathSeg::Super, name(40)], k(2)),
            Some(seed)
        );
        assert_eq!(
            t.resolve_function_path(&[PathSeg::Root, name(40)], k(2)),
            Some(seed)
        );
        // Popping past the package root fails.
        assert_eq!(
            t.resolve_function_path(
                &[
                    PathSeg::Super,
                    PathSeg::Super,
                    PathSeg::Super,
                    PathSeg::Super
                ],
                k(2)
            ),
            None
        );
    }

    #[test]
    fn trait_namespace_is_independent() {
        let mut t = DefTable::new();
        let name = k(1);
        let r = t.declare_record(name).expect("record");
        let f = t.declare_function(name).expect("function");
        let tr = t.declare_trait(name).expect("trait");
        assert!(r != f && f != tr && r != tr);
        assert_eq!(t.resolve_trait(name), Some(tr));
        // A second trait of the same name in the same module clashes.
        assert!(t.declare_trait(name).is_none());
    }

    #[test]
    fn truncate_retracts_a_trait_def() {
        let mut t = DefTable::new();
        let name = k(1);
        let live = t.len();
        let tr = t.declare_trait(name).expect("trait");
        assert_eq!(t.lookup_trait(&[name]), Some(tr));
        t.truncate(live);
        assert_eq!(t.lookup_trait(&[name]), None);
        // The retracted name is reusable.
        assert!(t.declare_trait(name).is_some());
    }

    #[test]
    fn truncate_retracts_later_defs() {
        let mut t = DefTable::new();
        let a = t.declare_record(k(1)).expect("fresh");
        let cp = t.len();
        t.declare_record(k(2)).expect("fresh");
        t.declare_function(k(3)).expect("fresh");
        t.truncate(cp);
        // Defs before the checkpoint survive; later ones are gone from both
        // the registry and their namespace scopes.
        assert_eq!(t.resolve_record(k(1)), Some(a));
        assert_eq!(t.resolve_record(k(2)), None);
        assert_eq!(t.resolve_function(k(3)), None);
        assert_eq!(t.len(), cp);
        // A retracted name can be declared again.
        assert!(t.declare_record(k(2)).is_some());
    }
}
