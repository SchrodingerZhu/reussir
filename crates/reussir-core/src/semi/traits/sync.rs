//! Structural `Sync`: the single thread-safety auto trait.
//!
//! `Sync(τ)` holds when a value of type `τ` may be reached — aliased, read,
//! cloned, stored, dropped — from any thread concurrently. It subsumes both of
//! Rust's `Send` and `Sync` (duplicable rc handles collapse transfer into
//! sharing); the rules live in `docs/design/thread-safety.md` §2.2 and this
//! module is their implementation.
//!
//! The check is **structural** (an auto trait: derived, never written by the
//! user) and **coinductive**: re-encountering an `Arc` inner that is already
//! on the current checking path is success, so recursive arc'd structures
//! (`Arc<Node>` lists/trees) are well-formed — a genuine violation always
//! surfaces at a finite unfolding on some non-cyclic path.
//!
//! Two queries are exposed:
//!
//! - [`sync_verdict`]: is `τ` `Sync`? A bare `[shared]` record never is — its
//!   own box count is not atomic — no matter what its members are.
//! - [`wf_arc`]: is `Arc<X>` well-formed? The stratified rule: the arc
//!   supplies atomicity for exactly the one box it wraps, so every *member*
//!   of `X` (element for arrays, argument type for closures) must already be
//!   `Sync` on its own. This runs *after* the kind gate
//!   ([`crate::semi::ty_eval::arc_inner_rejection`]) has admitted `X` as a
//!   shared rc box.
//!
//! A failed check carries a [`Witness`]: the member path to the offending
//! leaf and the reason, so diagnostics can say *which* member broke the type
//! instead of shrugging at the root.
//!
//! Record lookup is abstracted behind [`SyncEnv`] because the two callers
//! substitute differently: the elaborator instantiates member types with
//! [`crate::semi::infer::InferCtxt::instantiate_ty`], monomorphization with
//! [`crate::full::subst::subst_ty`]. An env may also answer "not known yet"
//! (unresolved def, fields not populated) — the verdict is then [`Blocked`]
//! and the caller defers to the monomorphization backstop, where every type
//! is ground and every record complete.
//!
//! [`Blocked`]: SyncVerdict::Blocked

use rustc_hash::FxHashSet;

use crate::semi::ty::{DefId, Ty, TyKind};

/// Record information the checker needs, provided by each phase.
pub trait SyncEnv<'tcx> {
    /// The record's declared capability; `None` when the def is unknown
    /// (error recovery) — the verdict blocks.
    fn default_cap(&self, def: DefId) -> Option<crate::semi::ctxt::DefaultCap>;

    /// The record's members instantiated at `args`, labeled for diagnostics
    /// (`name` / `.0` / `Variant.0`). `None` when the fields are not
    /// populated yet — the verdict blocks and the mono backstop re-checks.
    fn members(&self, def: DefId, args: &'tcx [Ty<'tcx>]) -> Option<Vec<(String, Ty<'tcx>)>>;

    /// The defs of record heads referenced anywhere in `def`'s declared
    /// member types (through nullables, value members, type arguments, …) —
    /// the def-level reference graph the recursive-group (SCC) computation
    /// walks. `None` when the fields are not populated yet.
    fn member_record_defs(&self, def: DefId) -> Option<Vec<DefId>>;
}

/// Collect the def of every record head appearing anywhere in `ty`, for the
/// [`SyncEnv::member_record_defs`] implementations: the def-level reference
/// graph the recursive-group computation walks.
pub fn collect_record_defs<'tcx>(ty: Ty<'tcx>, out: &mut Vec<DefId>) {
    match *ty.kind() {
        TyKind::Record { def, args, .. } => {
            out.push(def);
            for &arg in args {
                collect_record_defs(arg, out);
            }
        }
        TyKind::Arc(inner) | TyKind::Nullable(inner) => collect_record_defs(inner, out),
        TyKind::Cell { elem, .. } | TyKind::Array { elem, .. } => collect_record_defs(elem, out),
        TyKind::Closure { params, ret } => {
            for &p in params {
                collect_record_defs(p, out);
            }
            collect_record_defs(ret, out);
        }
        _ => {}
    }
}

/// The recursive group of `root`: every def `D` such that `root` reaches `D`
/// and `D` reaches `root` through the def-level member-reference graph
/// (`root` itself included). §3.3's SCC promotion re-colors exactly these
/// under an `Arc`. Type arguments are ignored at this level, so polymorphic
/// recursion collapses onto its def. Returns `None` (blocked) when any
/// reachable def's fields are not populated yet.
pub fn recursive_group<'tcx>(
    env: &dyn SyncEnv<'tcx>,
    root: DefId,
) -> Option<FxHashSet<DefId>> {
    // Forward reachability from `root`.
    let mut forward = FxHashSet::default();
    let mut stack = vec![root];
    while let Some(def) = stack.pop() {
        if !forward.insert(def) {
            continue;
        }
        for next in env.member_record_defs(def)? {
            stack.push(next);
        }
    }
    // Keep the defs that reach back: walk each candidate's forward set for
    // `root`. Record graphs are small; quadratic is fine.
    let mut group = FxHashSet::default();
    for &candidate in &forward {
        let mut seen = FxHashSet::default();
        let mut stack = vec![candidate];
        let mut reaches_root = false;
        while let Some(def) = stack.pop() {
            if def == root {
                reaches_root = true;
                break;
            }
            if !seen.insert(def) {
                continue;
            }
            for next in env.member_record_defs(def)? {
                stack.push(next);
            }
        }
        if reaches_root {
            group.insert(candidate);
        }
    }
    group.insert(root);
    Some(group)
}

/// Why a type is not `Sync`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotSyncReason {
    /// A bare shared rc box (record, array, or closure): its refcount is not
    /// atomic, so two threads bumping it race.
    NonAtomicBox,
    /// A regional object; regions and threads do not mix (for now).
    Regional,
    /// A plain `Cell`: whole-element stores with no synchronization.
    PlainCell,
    /// A `RefCell`: the in-use guard flag is not atomic.
    ExclusiveCell,
}

impl NotSyncReason {
    pub fn describe(self) -> &'static str {
        match self {
            NotSyncReason::NonAtomicBox => {
                "a plain `[shared]` rc box whose refcount is not atomic"
            }
            NotSyncReason::Regional => "a regional object",
            NotSyncReason::PlainCell => "a plain `Cell`, which is unsynchronized",
            NotSyncReason::ExclusiveCell => "a `RefCell`, whose in-use guard is not atomic",
        }
    }
}

/// The member path to a `!Sync` leaf, for diagnostics.
#[derive(Clone, Debug)]
pub struct Witness<'tcx> {
    /// Member labels from the checked root down to the leaf, e.g.
    /// `["next", "Cons.1"]`. Empty when the root itself is the leaf.
    pub path: Vec<String>,
    /// The offending type.
    pub leaf: Ty<'tcx>,
    pub reason: NotSyncReason,
}

impl Witness<'_> {
    /// Render as a diagnostic fragment: `` member `a.b` is <reason> `` (or
    /// `it is <reason>` for the root).
    pub fn describe(&self) -> String {
        if self.path.is_empty() {
            format!("it is {}", self.reason.describe())
        } else {
            format!(
                "member `{}` is {}",
                self.path.join("."),
                self.reason.describe()
            )
        }
    }
}

#[derive(Clone, Debug)]
pub enum SyncVerdict<'tcx> {
    Sync,
    NotSync(Witness<'tcx>),
    /// The answer depends on a generic, a hole, or a not-yet-known record —
    /// defer; monomorphization re-checks every ground instantiation.
    Blocked,
}

/// Is `τ` `Sync`?
pub fn sync_verdict<'tcx>(env: &dyn SyncEnv<'tcx>, ty: Ty<'tcx>) -> SyncVerdict<'tcx> {
    let mut cx = Cx {
        env,
        on_path: FxHashSet::default(),
        path: Vec::new(),
        group: FxHashSet::default(),
    };
    cx.check(ty)
}

/// Is `Arc<inner>` well-formed — every member of `inner` `Sync`? Run after
/// the kind gate has admitted `inner` as a shared rc box; inners the gate
/// rejects report there and yield `Sync` here to avoid double diagnostics.
pub fn wf_arc<'tcx>(env: &dyn SyncEnv<'tcx>, inner: Ty<'tcx>) -> SyncVerdict<'tcx> {
    let mut cx = Cx {
        env,
        on_path: FxHashSet::default(),
        path: Vec::new(),
        group: FxHashSet::default(),
    };
    cx.on_path.insert(inner);
    cx.wf(inner)
}

struct Cx<'e, 'tcx> {
    env: &'e dyn SyncEnv<'tcx>,
    /// Arc inners currently being wf-checked on this path: a re-encounter is
    /// a cycle and coinductively succeeds.
    on_path: FxHashSet<Ty<'tcx>>,
    path: Vec<String>,
    /// The recursive group of the shared record whose members are currently
    /// being folded under an `Arc` (§3.3): a bare shared member whose def is
    /// in the group is checked *as its arc'd version* — the automatic SCC
    /// promotion — instead of refuting as a plain box. Empty outside an arc'd
    /// fold, and cleared while folding a `[value]` member's own members: a
    /// value intermediate can escape the box by value, where nothing carries
    /// the coloring, so its same-group links must be written as explicit
    /// `Arc` members instead.
    group: FxHashSet<DefId>,
}

impl<'tcx> Cx<'_, 'tcx> {
    fn refute(&self, leaf: Ty<'tcx>, reason: NotSyncReason) -> SyncVerdict<'tcx> {
        SyncVerdict::NotSync(Witness {
            path: self.path.clone(),
            leaf,
            reason,
        })
    }

    /// Fold a labeled member list: the first definite failure wins; a block
    /// only surfaces when nothing fails outright.
    fn fold(
        &mut self,
        members: impl IntoIterator<Item = (String, Ty<'tcx>)>,
    ) -> SyncVerdict<'tcx> {
        let mut blocked = false;
        for (label, ty) in members {
            self.path.push(label);
            let verdict = self.check(ty);
            self.path.pop();
            match verdict {
                SyncVerdict::Sync => {}
                SyncVerdict::Blocked => blocked = true,
                refuted @ SyncVerdict::NotSync(_) => return refuted,
            }
        }
        if blocked {
            SyncVerdict::Blocked
        } else {
            SyncVerdict::Sync
        }
    }

    fn check(&mut self, ty: Ty<'tcx>) -> SyncVerdict<'tcx> {
        use crate::semi::ctxt::DefaultCap;
        match *ty.kind() {
            TyKind::Int(_)
            | TyKind::Fp(_)
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Unit
            // `str` values are immutable byte references; the frontend type
            // does not carry a lifescope (yet), so global and local strs are
            // not distinguished here — see the design doc's str caveat.
            | TyKind::Str
            // Error recovery: don't cascade.
            | TyKind::Bottom => SyncVerdict::Sync,
            TyKind::Generic(_) | TyKind::Hole(_) => SyncVerdict::Blocked,
            TyKind::Nullable(inner) => self.check(inner),
            TyKind::Cell { exclusive, .. } => self.refute(
                ty,
                if exclusive {
                    NotSyncReason::ExclusiveCell
                } else {
                    NotSyncReason::PlainCell
                },
            ),
            // Bare shared rc boxes: never `Sync`, regardless of contents —
            // shareability is carried by the box coloring (`Arc`), not the
            // nominal type.
            TyKind::Array { .. } | TyKind::Closure { .. } => {
                self.refute(ty, NotSyncReason::NonAtomicBox)
            }
            TyKind::Record { def, args, .. } => match self.env.default_cap(def) {
                None => SyncVerdict::Blocked,
                // A bare shared member of the current arc'd recursive group
                // promotes (§3.3): it is checked as its own `Arc`, and the
                // coinductive guard closes the cycle. Outside the group the
                // bare box refutes — shareability is carried by the coloring.
                Some(DefaultCap::Shared) if self.group.contains(&def) => {
                    if !self.on_path.insert(ty) {
                        return SyncVerdict::Sync;
                    }
                    let verdict = self.wf(ty);
                    self.on_path.remove(&ty);
                    verdict
                }
                Some(DefaultCap::Shared) => self.refute(ty, NotSyncReason::NonAtomicBox),
                Some(DefaultCap::Regional) => self.refute(ty, NotSyncReason::Regional),
                // A `[value]` record is inline storage: `Sync` iff its
                // members are. Its fold runs *outside* the promotion group —
                // a value intermediate can leave the box by value, where
                // nothing carries the coloring.
                Some(DefaultCap::Value) => match self.env.members(def, args) {
                    None => SyncVerdict::Blocked,
                    Some(members) => {
                        let saved = std::mem::take(&mut self.group);
                        let verdict = self.fold(members);
                        self.group = saved;
                        verdict
                    }
                },
            },
            TyKind::Arc(inner) => {
                // Coinduction: an arc inner already being checked on this
                // path is a cycle — assume it holds; a real violation shows
                // up at a finite unfolding elsewhere.
                if !self.on_path.insert(inner) {
                    return SyncVerdict::Sync;
                }
                let verdict = self.wf(inner);
                self.on_path.remove(&inner);
                verdict
            }
        }
    }

    /// The `Arc` member rule. Anything the kind gate rejects as an arc inner
    /// (scalars, cells, value/regional records, nested arcs, nullables) is
    /// reported by the gate at its own site; here those yield `Sync` so one
    /// error is emitted, not two.
    fn wf(&mut self, inner: Ty<'tcx>) -> SyncVerdict<'tcx> {
        use crate::semi::ctxt::DefaultCap;
        match *inner.kind() {
            TyKind::Record { def, args, .. } => match self.env.default_cap(def) {
                Some(DefaultCap::Shared) => match self.env.members(def, args) {
                    None => SyncVerdict::Blocked,
                    Some(members) => {
                        // The members fold under this record's recursive
                        // group: same-group bare shared links promote (§3.3).
                        let Some(group) = recursive_group(self.env, def) else {
                            return SyncVerdict::Blocked;
                        };
                        let saved = std::mem::replace(&mut self.group, group);
                        let verdict = self.fold(members);
                        self.group = saved;
                        verdict
                    }
                },
                None => SyncVerdict::Blocked,
                // Value/regional inners: the kind gate's error.
                Some(_) => SyncVerdict::Sync,
            },
            // Array elements and closure arguments are not record-member
            // positions, so the SCC promotion does not apply inside them —
            // their fold runs outside any group.
            TyKind::Array { elem, .. } => {
                let saved = std::mem::take(&mut self.group);
                self.path.push("the array element".to_owned());
                let verdict = self.check(elem);
                self.path.pop();
                self.group = saved;
                verdict
            }
            // Argument types only: applied arguments become captures, so the
            // type-level bound covers every later partial application. The
            // creation-site check of *environment* captures lands with arc'd
            // closure construction.
            TyKind::Closure { params, .. } => {
                let saved = std::mem::take(&mut self.group);
                let verdict = self.fold(
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| (format!("closure argument {i}"), p)),
                );
                self.group = saved;
                verdict
            }
            TyKind::Generic(_) | TyKind::Hole(_) | TyKind::Bottom => SyncVerdict::Blocked,
            // Everything else is the kind gate's error.
            _ => SyncVerdict::Sync,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::ctxt::DefaultCap;
    use crate::semi::ty::{Flexivity, IntTy, Ty, TyCtxt};
    use crate::with_tcx;
    use rustc_hash::FxHashMap;

    /// A synthetic record table: `def ↦ (cap, members)`. Members are stored
    /// pre-instantiated (the tests use no generics).
    #[derive(Default)]
    struct TestEnv<'tcx> {
        records: FxHashMap<DefId, (DefaultCap, Option<Vec<(String, Ty<'tcx>)>>)>,
    }

    impl<'tcx> SyncEnv<'tcx> for TestEnv<'tcx> {
        fn default_cap(&self, def: DefId) -> Option<DefaultCap> {
            self.records.get(&def).map(|(cap, _)| *cap)
        }
        fn members(&self, def: DefId, _args: &'tcx [Ty<'tcx>]) -> Option<Vec<(String, Ty<'tcx>)>> {
            self.records.get(&def).and_then(|(_, m)| m.clone())
        }
        fn member_record_defs(&self, def: DefId) -> Option<Vec<DefId>> {
            let (_, members) = self.records.get(&def)?;
            let members = members.as_ref()?;
            let mut out = Vec::new();
            for (_, ty) in members {
                collect_record_defs(*ty, &mut out);
            }
            Some(out)
        }
    }

    fn rec<'tcx>(tcx: &TyCtxt<'tcx>, def: u32) -> Ty<'tcx> {
        tcx.mk_record(DefId(def), &[], Flexivity::Irrelevant)
    }

    #[test]
    fn scalars_are_sync() {
        with_tcx(|tcx: &TyCtxt| {
            let env = TestEnv::default();
            for ty in [tcx.mk_int(IntTy::Unsigned(64)), tcx.mk_bool(), tcx.mk_str()] {
                assert!(matches!(sync_verdict(&env, ty), SyncVerdict::Sync));
            }
        });
    }

    #[test]
    fn bare_shared_record_is_never_sync() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            // Even with all-Sync members: shareability is the box coloring's.
            env.records.insert(
                DefId(0),
                (
                    DefaultCap::Shared,
                    Some(vec![("a".into(), tcx.mk_int(IntTy::Signed(32)))]),
                ),
            );
            let verdict = sync_verdict(&env, rec(tcx, 0));
            let SyncVerdict::NotSync(w) = verdict else {
                panic!("expected NotSync, got {verdict:?}");
            };
            assert_eq!(w.reason, NotSyncReason::NonAtomicBox);
            assert!(w.path.is_empty());
        });
    }

    #[test]
    fn value_record_folds_members_with_path() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            // [value] Mid { leaf: Leaf }, [shared] Leaf {} — the witness
            // names the nested member path.
            env.records
                .insert(DefId(1), (DefaultCap::Shared, Some(vec![])));
            let leaf = rec(tcx, 1);
            env.records.insert(
                DefId(2),
                (DefaultCap::Value, Some(vec![("leaf".into(), leaf)])),
            );
            let mid = rec(tcx, 2);
            env.records.insert(
                DefId(3),
                (DefaultCap::Value, Some(vec![("m".into(), mid)])),
            );
            let SyncVerdict::NotSync(w) = sync_verdict(&env, rec(tcx, 3)) else {
                panic!("expected NotSync");
            };
            assert_eq!(w.path, vec!["m".to_owned(), "leaf".to_owned()]);
            assert_eq!(w.reason, NotSyncReason::NonAtomicBox);
        });
    }

    #[test]
    fn arc_wf_requires_sync_members() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            // [shared] Pair { a: i32 } — Arc<Pair> is WF and Sync.
            env.records.insert(
                DefId(0),
                (
                    DefaultCap::Shared,
                    Some(vec![("a".into(), tcx.mk_int(IntTy::Signed(32)))]),
                ),
            );
            let pair = rec(tcx, 0);
            assert!(matches!(wf_arc(&env, pair), SyncVerdict::Sync));
            assert!(matches!(
                sync_verdict(&env, tcx.mk_arc(pair)),
                SyncVerdict::Sync
            ));

            // [shared] Holder { p: Pair } — a bare shared member refutes.
            env.records.insert(
                DefId(1),
                (DefaultCap::Shared, Some(vec![("p".into(), pair)])),
            );
            let SyncVerdict::NotSync(w) = wf_arc(&env, rec(tcx, 1)) else {
                panic!("expected NotSync");
            };
            assert_eq!(w.path, vec!["p".to_owned()]);
        });
    }

    #[test]
    fn recursive_arc_is_coinductively_wf() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            // [shared] Node { v: i64, next: Nullable<Arc<Node>> } — checking
            // wf_arc(Node) re-encounters Arc<Node>; the cycle succeeds.
            let node = rec(tcx, 0);
            env.records.insert(
                DefId(0),
                (
                    DefaultCap::Shared,
                    Some(vec![
                        ("v".into(), tcx.mk_int(IntTy::Signed(64))),
                        ("next".into(), tcx.mk_nullable(tcx.mk_arc(node))),
                    ]),
                ),
            );
            assert!(matches!(wf_arc(&env, node), SyncVerdict::Sync));
            assert!(matches!(
                sync_verdict(&env, tcx.mk_arc(node)),
                SyncVerdict::Sync
            ));
        });
    }

    #[test]
    fn cells_refute_with_their_kind() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            let i32t = tcx.mk_int(IntTy::Signed(32));
            env.records.insert(
                DefId(0),
                (
                    DefaultCap::Shared,
                    Some(vec![("c".into(), tcx.mk_cell(i32t, false))]),
                ),
            );
            let SyncVerdict::NotSync(w) = wf_arc(&env, rec(tcx, 0)) else {
                panic!("expected NotSync");
            };
            assert_eq!(w.reason, NotSyncReason::PlainCell);
            assert_eq!(w.path, vec!["c".to_owned()]);
            assert!(matches!(
                sync_verdict(&env, tcx.mk_cell(i32t, true)),
                SyncVerdict::NotSync(Witness {
                    reason: NotSyncReason::ExclusiveCell,
                    ..
                })
            ));
        });
    }

    #[test]
    fn incomplete_records_block() {
        with_tcx(|tcx: &TyCtxt| {
            let mut env = TestEnv::default();
            env.records.insert(DefId(0), (DefaultCap::Shared, None));
            assert!(matches!(wf_arc(&env, rec(tcx, 0)), SyncVerdict::Blocked));
            // Unknown def: also blocked (error recovery).
            assert!(matches!(
                sync_verdict(&env, rec(tcx, 9)),
                SyncVerdict::Blocked
            ));
        });
    }
}
