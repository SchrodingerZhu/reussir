//! The package export closure: what `--emit rri` ships, computed over the
//! elaborated HIR. See `docs/design/rri.md`.
//!
//! An `.rri` interface must let a foreign package type-check against this one
//! and monomorphize its generics locally. That fixes the contents exactly:
//!
//! - every `pub` **generic** ships with its full body (the foreign compilation
//!   re-instantiates it), and so does every generic those bodies transitively
//!   call, `pub` or not;
//! - every ground function the shipped surface mentions — `pub` grounds, and
//!   any ground a shipped body calls — ships as a bodyless **prototype**: its
//!   symbol stays externally linkable in this package's artifacts (the
//!   [`mono_exports`](super::mono::mono_exports) guarantee from the linkage
//!   side, which is *derived from this closure* so the two cannot drift);
//! - every **record** the shipped signatures, bodies, or shipped record fields
//!   mention ships in full (layout resolution downstream needs fields),
//!   closed to a fixed point over field types;
//! - **strings** shipped bodies reference travel with them.
//!
//! Trampolines (`#[main]`, `extern "C" trampoline`) never ship: they are this
//! package's own link surface and its monomorphization roots, not something a
//! consumer may re-instantiate.

use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use crate::semi::ctxt::RecordFields;
use crate::semi::hir::ExprKind;
use crate::semi::ty::{DefId, Ty, TyKind};
use crate::surface::Visibility;
use crate::utils::string::StringToken;

use super::mono::{MonoInput, for_each_expr};

/// The `.rri` format integer, bumped on any change to the textual grammar or
/// its meaning. The header's `producer` string (the exact producing rrc
/// version) gates the rest while the format is unstable.
pub const RRI_FORMAT: u32 = 1;

/// The defs and ancillary facts an `.rri` interface ships. All sets are over
/// the producing elaboration's ids; emission resolves them back through the
/// same tables it prints.
#[derive(Debug, Default)]
pub struct ExportClosure {
    /// Functions shipped whole — signature and body (or `ffi` body): the
    /// `pub` generics and every generic their bodies transitively call.
    pub bodies: FxHashSet<DefId>,
    /// Functions shipped as bodyless prototypes: `pub` ground functions, plus
    /// every ground function a shipped body calls. At load time a bodyless
    /// prototype means *external declaration* — the symbol resolves against
    /// this package's compiled artifacts.
    pub protos: FxHashSet<DefId>,
    /// Records shipped in full, closed over field types.
    pub records: FxHashSet<DefId>,
    /// String literals referenced by shipped bodies.
    pub strings: FxHashSet<StringToken>,
}

/// Compute the export closure of an elaborated program.
///
/// Seeds: `pub` generics (into [`ExportClosure::bodies`], walked) and `pub`
/// grounds (into [`ExportClosure::protos`], not walked — their bodies are
/// compiled here and only the symbol crosses). Walking a body classifies each
/// call target the same way and collects the types and strings it mentions;
/// the record set then closes over field types.
///
/// Only `elaborated` — this package's own items — seeds or joins the closure:
/// imported functions (`MonoInput::externs`) are a dependency's surface, and
/// a consumer never re-exports its dependency.
pub fn export_closure(input: &MonoInput<'_, '_>) -> ExportClosure {
    let by_def: rustc_hash::FxHashMap<DefId, &crate::semi::hir::Function<'_>> =
        input.elaborated.iter().map(|f| (f.def, f)).collect();
    let mut closure = ExportClosure::default();
    let mut tys: FxHashSet<Ty<'_>> = FxHashSet::default();

    let mut queue: VecDeque<DefId> = VecDeque::new();
    for f in input.elaborated {
        if f.visibility != Visibility::Public {
            continue;
        }
        if f.generics.is_empty() {
            closure.protos.insert(f.def);
        } else if closure.bodies.insert(f.def) {
            queue.push_back(f.def);
        }
    }
    while let Some(def) = queue.pop_front() {
        let Some(func) = by_def.get(&def) else {
            continue;
        };
        for (_, _, t) in &func.params {
            tys.insert(*t);
        }
        tys.insert(func.return_ty);
        let Some(body) = func.body.as_ref() else {
            // A generic `#[ffi(import)]`: its texture is the body; nothing to
            // walk, the signature types above are its whole reach.
            continue;
        };
        for_each_expr(body, &mut |e| {
            tys.insert(e.ty);
            match &e.kind {
                ExprKind::GlobalStr(s) => {
                    closure.strings.insert(*s);
                }
                ExprKind::FuncCall { target, .. } => {
                    let Some(callee) = by_def.get(target) else {
                        return;
                    };
                    if callee.generics.is_empty() {
                        closure.protos.insert(*target);
                    } else if closure.bodies.insert(*target) {
                        queue.push_back(*target);
                    }
                }
                _ => {}
            }
        });
    }
    // Prototype signatures are surface too: a consumer type-checks calls
    // against them, and lowering a call inside an instantiated body needs
    // the same types the producer saw.
    for def in &closure.protos {
        let Some(func) = by_def.get(def) else {
            continue;
        };
        for (_, _, t) in &func.params {
            tys.insert(*t);
        }
        tys.insert(func.return_ty);
    }
    // Close records over the collected types, then over field types to a
    // fixed point (the loader resolves layouts through those fields).
    let mut record_queue: VecDeque<DefId> = VecDeque::new();
    for t in tys {
        note_records(t, &mut closure.records, &mut record_queue);
    }
    while let Some(def) = record_queue.pop_front() {
        let Some(rec) = input.records.get(&def) else {
            continue;
        };
        let member_tys: Vec<Ty<'_>> = match &rec.fields {
            Some(RecordFields::Named(fs)) => fs.iter().map(|(_, t, _, _)| *t).collect(),
            Some(RecordFields::Unnamed(fs)) => fs.iter().map(|(t, _, _)| *t).collect(),
            Some(RecordFields::Variants(vs)) => {
                vs.iter().flat_map(|v| v.fields.iter().copied()).collect()
            }
            Some(RecordFields::Opaque) | None => Vec::new(),
        };
        for t in member_tys {
            note_records(t, &mut closure.records, &mut record_queue);
        }
    }
    closure
}

/// Note every record a type's structure mentions, queueing fresh ones for the
/// field-closure pass.
fn note_records<'tcx>(ty: Ty<'tcx>, records: &mut FxHashSet<DefId>, queue: &mut VecDeque<DefId>) {
    match *ty.kind() {
        TyKind::Record { def, args, .. } => {
            if records.insert(def) {
                queue.push_back(def);
            }
            for &a in args {
                note_records(a, records, queue);
            }
        }
        TyKind::Closure { params, ret } => {
            for &p in params {
                note_records(p, records, queue);
            }
            note_records(ret, records, queue);
        }
        TyKind::Array { elem, .. } | TyKind::Cell { elem, .. } => {
            note_records(elem, records, queue);
        }
        TyKind::Nullable(inner) | TyKind::Arc(inner) => {
            note_records(inner, records, queue);
        }
        TyKind::Int(_)
        | TyKind::Fp(_)
        | TyKind::Bool
        | TyKind::Str
        | TyKind::Char
        | TyKind::Unit
        | TyKind::Generic(_)
        | TyKind::Hole(_)
        | TyKind::Bottom => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::elaborate;
    use crate::{surface, with_tcx};

    /// The closure over one mixed program, all four facts at once: `pub`
    /// generics and the generics they call ship bodies; grounds the shipped
    /// bodies call — and `pub` grounds themselves — ship prototypes; records
    /// reach through prototype signatures and close over fields; strings ride
    /// with the bodies. Ground bodies are never walked, so the closure stops
    /// at the ground boundary: `api` → `bar` ships `bar` as a prototype, but
    /// `baz` — reached only from `bar`'s body, which is compiled here — stays
    /// home (and, through the derived `mono_exports`, may stay `internal`).
    /// Same for a `pub` ground's private callee, and the unreachable helper
    /// ships nowhere.
    #[test]
    fn export_closure_selects_the_shippable_surface() {
        let source = "
            struct Internal { value: i64 }
            struct Nested { i: Internal }
            fn helper(n: Nested) -> i64 { n.i.value }
            fn deep(x: i64) -> i64 { x + 1 }
            fn mid<T : Num>(x: T) -> i64 {
                let s = \"tag\";
                deep(2) + helper(Nested { i: Internal { value: 3 } })
            }
            fn baz(x: i64) -> i64 { x + 5 }
            fn bar(x: i64) -> i64 { baz(x) }
            fn from_ground(x: i64) -> i64 { x + 3 }
            fn unreachable_helper(x: i64) -> i64 { x + 4 }
            pub fn api<T : Num>(x: T) -> i64 { mid(x) + bar(1) }
            pub fn ground(x: i64) -> i64 { from_ground(x) }
        ";
        with_tcx(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(
                !elab.has_errors(),
                "elaboration errors: {:#?}",
                elab.reports
            );
            let closure = export_closure(&elab.mono_input());
            let names = |set: &FxHashSet<DefId>| -> Vec<String> {
                let mut v: Vec<String> = set
                    .iter()
                    .map(|d| elab.defs.path(*d).display(elab.resolver))
                    .collect();
                v.sort();
                v
            };
            assert_eq!(names(&closure.bodies), ["api", "mid"]);
            assert_eq!(names(&closure.protos), ["bar", "deep", "ground", "helper"]);
            assert_eq!(names(&closure.records), ["Internal", "Nested"]);
            assert_eq!(closure.strings.len(), 1, "{:?}", closure.strings);

            // The printed interface: header first, closure only, prototypes
            // bodyless, functions in path order — and it re-enters through
            // the ordinary HIR parser with the header surfaced.
            let strings: Vec<_> = elab
                .strings
                .entries()
                .into_iter()
                .filter(|(t, _)| closure.strings.contains(t))
                .collect();
            let text = crate::semi::hir::print::Printer::new(&elab.defs, elab.resolver)
                .with_interface(crate::semi::hir::print::InterfaceEmit {
                    format: RRI_FORMAT,
                    package: "demo",
                    producer: "rrc-test",
                    bodies: &closure.bodies,
                    protos: &closure.protos,
                    records: &closure.records,
                    file_root: None,
                })
                .program(&elab.elaborated, &strings, &elab.records, &[]);
            assert!(
                text.starts_with("interface 1 package \"demo\" producer \"rrc-test\";"),
                "{text}"
            );
            assert!(text.contains("fn #deep(v0 (x): i64) -> i64;"), "{text}");
            assert!(
                text.contains("pub fn #ground(v0 (x): i64) -> i64;"),
                "{text}"
            );
            assert!(!text.contains("unreachable_helper"), "{text}");
            assert!(!text.contains("from_ground"), "{text}");
            let api = text.find("fn #api").expect("api printed");
            let mid = text.find("fn #mid").expect("mid printed");
            assert!(api < mid, "functions not in path order:\n{text}");

            let parsed =
                crate::semi::hir::build::parse_program(tcx, &text).expect("interface re-parses");
            let header = parsed.header.expect("header surfaced");
            assert_eq!(header.format, RRI_FORMAT);
            assert_eq!(header.package, "demo");
            assert_eq!(header.producer, "rrc-test");
            assert_eq!(parsed.funcs.len(), 6);
            let body_of = |name: &str| {
                parsed
                    .funcs
                    .iter()
                    .find(|f| parsed.defs.path(f.def).display(&parsed.names) == name)
                    .unwrap_or_else(|| panic!("{name} missing"))
                    .body
                    .is_some()
            };
            assert!(body_of("api") && body_of("mid"));
            assert!(!body_of("deep") && !body_of("ground") && !body_of("helper"));
            assert!(!body_of("bar"), "ground bodies never ship");
        });
    }
}
