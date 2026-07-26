# `--emit rri`: the reduced HIR interface

Second PR of the cross-package HIR-index arc (#451, opened by #467). An `.rri`
file is the interface a compiled Reussir package presents to downstream
packages: everything a consumer needs to type-check against it and to
monomorphize its generics locally, and nothing else. It is the analogue of
rustc's `.rmeta` — except it is not a new format at all: an rri is a valid
textual HIR document, produced by the existing printer and consumed by the
existing parser, filtered down to the export closure and fronted by a
versioned header.

## Why reduced, and why HIR

Monomorphization needs bodies. `full/mono.rs` instantiates a generic by
substituting into its elaborated HIR `Function` — so whatever crosses the
package boundary must carry generic bodies in a form `MonoInput` can be
rebuilt from. That path is already proven end to end: `--emit hir` output
re-enters through `hir::build::parse_program`, and the round-trip test in
`mono.rs` asserts mono-from-reparsed-HIR is byte-identical to
mono-from-original.

Reduced, because the full dump ships every private body. The reduction:

- keeps private implementation out of the artifact — an edit to a private
  ground function's *body* leaves the interface's semantic content unchanged,
  which is what lets `rene` skip rebuilding dependents (see Determinism);
- makes the access-control story explicit: what is in the rri is exactly what
  a foreign instantiation may touch.

#467 already did the linkage half of this contract: `mono::mono_exports`
computes the private ground functions reachable from `pub` generic bodies and
keeps their symbols externally linkable (`LinkagePolicy::Aot`), and foreign
re-instantiations dedup as `weak_odr`. The rri is the other half: the set of
things it ships is computed by the *same traversal*, so the two cannot drift.

## Contents: the export closure

Computed post-elaboration over `elab.elaborated` / `elab.records`, sharing
`mono_exports`' call-target walk (factor the traversal, do not duplicate it).

Seeds:

- every `pub` **generic** function — shipped with its full body;
- every `pub` **ground** function — shipped as a prototype (`fn #p(...) -> T;`).

Closure, walking only the bodies that ship:

- a **generic** callee (any visibility) ships with its body and is walked in
  turn — a downstream instantiation re-instantiates it locally;
- a **ground** callee (any visibility) ships as a prototype — its symbol is
  external in this package's artifact, exactly the #467 `mono_exports`
  guarantee. Ground bodies never ship, `pub` or not;
- every **record** mentioned by a shipped signature, body, or shipped record
  field ships in full (fields included — the consumer's mono must resolve
  layouts), closed to fixed point mirroring `close_records_over_fields`;
- **strings** referenced by shipped bodies ship (table filtered and
  reindexed);
- **transform anchors/scripts** referenced by shipped bodies ship;
- **ffi imports and the preludes they need** ship for shipped defs — a `pub`
  generic `#[ffi(import)]` (polymorphic FFI) is part of the surface; the
  consumer's build generates and compiles its Rust textures per instance.
  (Format-complete now; the `--extern` PR decides when instantiating them is
  actually supported.)

Excluded: trampolines and `#[main]` (the package's own C-ABI link surface,
not cross-package callable — and mono roots must not leak: a consumer never
re-instantiates this package's roots), and everything private outside the
closure.

**The load-bearing invariant:** every bodyless, non-ffi function in an rri
names a symbol the producing package's compiled artifact exports with
external linkage. Loading gives `body: None` a second meaning the consumer
pipeline needs anyway: *external declaration* — not seeded as a definition
root in mono, lowered as a MIR declaration, resolved at link time against the
producing package's artifact.

## Format

A valid `.hir` document under the existing lalrpop grammar, plus one new
leading header item:

```
interface 1 package "demo" producer "rrc 0.5.0 (7d9c4d3)";
```

- `1` is the format integer, bumped on any grammar change;
- `package` names the package the interface describes; `--extern name=path`
  must match it;
- `producer` is the exact producing-rrc version string. While the HIR grammar
  is unstable the consumer refuses **any** mismatch of either field (rustc
  does the same with `.rmeta`); relax when the format stabilizes.

The header is a grammar item (parsed, not sniffed by the driver), rejected in
plain `.hir` re-entry and required first in `.rri`.

**Spans stay.** The rri is printed in the sources form
(`Printer::with_sources`): file table plus per-item/per-node spans, so
diagnostics raised while instantiating an upstream body point into the
upstream package's sources. File-table paths are rewritten
**package-root-relative** (the `SourceCache` may hold absolute paths;
absolute paths in the artifact would break reproducibility across checkouts).

**Determinism.** Records are already printed sorted by qualified path;
functions are likewise sorted by qualified path in rri mode (elaboration
order leaks declaration order otherwise; plain `--emit hir` keeps its current
order to avoid churning existing expectations). One honest trade-off of
keeping spans: a private-body edit *above* an exported item shifts its byte
offsets, so the rri file changes even though its semantic content did not.
`rene` still fingerprints dependents on the rri digest — coarser than ideal,
never wrong. If this proves to matter, a later refinement can hash a
span-stripped normalization; not in v1.

**Paths are package-rooted.** `module_path` roots every file's module at the
package name (`[pkg, foo]`), and the printer emits qualified paths, so rri
items are addressable as `#pkg::mod::item` with no collision against consumer
defs. DefIds are never part of the wire format — `parse_program` rebuilds a
fresh `DefTable`; paths are the stable identity, and the shared v0 mangling
(`Mangler::mangle_instance`) makes consumer-emitted instance symbols coincide
with any the producer already emitted (`weak_odr` dedups at link).

## Record visibility (prerequisite)

Records and enums currently have no visibility — the whole package is one
unit, so it never mattered. Cross-package it does: the rri must ship private
records in the closure (layout), but consumer *source* must not name them.

So the arc grows a preparatory step: surface `pub struct` / `pub enum`, a
visibility field on `Record`/record HIR (printed `pub`, defaulting private
like functions), and the private-in-public check — a `pub` function's
signature and a `pub` record's fields may only mention `pub` records
(Rust's E0446 discipline). This is enforced at elaboration always, not only
when emitting rri; in-repo tests and examples grow `pub` on the records their
public surfaces mention. Shipping it as its own PR keeps the churn reviewable
apart from the rri machinery.

Access control then falls out of the existing representation: the parsed
`visibility` governs whether *consumer-source resolution* may reference a
loaded def, while references inside loaded bodies resolve within the parsed
unit and bypass the check — a shipped private generic is instantiable through
a `pub` caller but never nameable.

## Driver wiring

- `Stage::Rri`, ordered between `Hir` and `Mir`; `is_input()` is false
  (consumption is `--extern`, a different axis than pipeline re-entry — an
  rri is not a completable program: it has no mono roots). The ordering
  makes `.mir`-and-later inputs refuse it through the existing
  pipeline-runs-forward check — no generic bodies left to witness
  reachability, the same reason #467 does not serialize `mono_exported`.
- Emission hooks at the existing `target == Stage::Hir` branch in
  `frontend_package`: full elaboration in hand, compute the closure, print
  the filtered slices. **Package mode only**: the header names the package
  it describes, and only `--package-root`/`--package-name` carry that name
  authoritatively — a bare file or a plain `.hir` dump does not (and
  `--package-name` on a `.hir` input already means "treat as package
  source"), so both are refused.
- A plain-`.hir` re-entry that finds an interface header is refused: the
  prototypes have no bodies to compile and the mono roots stayed home.
- `--emit rri`, extension `.rri`, `-o` respected (including `-o -`). No
  other new flags in this PR.
- v1 emits rri in its own invocation. A combined `--emit staticlib` +
  side-interface flag (rustc's `--emit metadata,link` shape) would save one
  elaboration per dependency; deferred until rene wiring shows it matters.

## Consumer contract (forward-looking, later PRs)

Recorded here so the format cannot paint them into a corner:

- `--extern name=path.rri`: header gate (format int + producer string +
  package name), `parse_program` into the shared `TyCtxt`, defs offered to
  resolution under visibility.
- **Re-anchoring the file table** (debug-info fidelity): the rri's paths are
  package-root-relative on purpose — byte-stability across checkouts is what
  rene fingerprints — but a consumer that monomorphizes a loaded body emits
  that instance's locations, and a bare `lib.rr` resolved against the
  *consumer's* build context would detach the debug info to the wrong place.
  This is DWARF's `comp_dir` split: relative names in the artifact, base
  directory supplied by the environment. `--extern` therefore carries a
  per-extern source root alongside the interface path (rene supplies it for
  free — its path dependencies know each dep's package root); the loader
  joins file-table entries onto it before building the consumer-side source
  cache, so diagnostics and the DWARF of locally-emitted instances point at
  the producing package's real sources. With no root given, entries load as
  unfetchable virtual files — locations degrade to name-only, never resolve
  against the consumer's cwd.
- `MonoInput` gains the loaded functions/records/strings/ffi/transform
  tables; the one mono change is bodyless-ground handling (declaration, not
  definition root).
- Linking: `--link-lib` / rene passes the producing package's artifact;
  rene's `dependencies` map (recorded-only today) starts resolving — bake
  each dep to `staticlib` + `rri`, feed dependents `--extern` + `--link-lib`,
  fingerprint dependents on the rri digest.
- COFF comdat closes the arc (`weak_odr` needs comdat on COFF).

## Testing

- **reussir-core unit** (beside the `mono_exports` test): closure selection —
  `pub` generic ships with body; private generic callee ships with body;
  private ground callee ships as prototype; `pub` ground ships as prototype
  (no body); unreachable private items absent; record field closure reaches
  nested records; strings filtered to shipped bodies.
- **driver round-trip**: emitted rri parses via `parse_program`; reprint is a
  fixed point (rri → parse → print → identical text).
- **header gate unit**: format/producer/package mismatch refused; header
  rejected in plain `.hir`.
- **lit**: a package mixing pub/private, generic/ground → `--emit rri` →
  FileCheck: header line; `pub fn` generic body present; prototypes end in
  `;`; `NOT:` on a private unreachable symbol and on any ground body; record
  `pub` markers.
- Record-visibility PR carries its own diagnostics lit (private-in-public).

## PR sequence

1. Record visibility + private-in-public check (surface, HIR grammar, tests).
2. `--emit rri` (header item, closure computation shared with
   `mono_exports`, filtered sources-form printing, relative file table).
3. `--extern` loading + access control.
4. Cross-package monomorphization + `--link-lib` + rene dependency wiring.
5. COFF comdat.
