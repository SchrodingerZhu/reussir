# Whole-program devirtualization for closures

## The model

A closure box is `{refcnt, vtable, cursor, payload…}` behind an rc pointer;
its vtable is three slots, `{drop, clone, evaluate}`. The slot ABIs depend on
nothing beyond the closure's **return type**:

- `evaluate` is `fn(rc<box>) -> c` — `closure.eval` is verified fully-applied
  and the outlined function reads **every** argument from the payload;
- `drop` is `fn(rc) -> void` and `clone` is `fn(rc) -> rc` — signature-free.

`closure.apply`/`closure.uniqify`/`closure.clone` never change the vtable —
only the static type (one fewer leading parameter) and the cursor. The return
type is therefore the one component of the static type that every value
backed by a given vtable agrees on, and the strongest fact a call site can
assert about the vtable it is about to dispatch through. That yields the
whole id scheme: **one type id per closure return type**
(`reussir.closure.wpd.<b62(blake3(print(c)))>`, built in `ClosureWpd.h`;
identical return types produce identical ids in any module). Captures fold in
for free — the frontend materializes a capture as a pre-applied leading
parameter, so a capturing closure shares the id of every closure with the
same visible return type.

This maps directly onto LLVM's stock WPD contract:

- each closure vtable global (a `constant` with the three function pointers
  at offsets 0/8/16) carries `!type !{i64 0, !"<id>"}` plus translation-unit
  `!vcall_visibility`;
- each indirect slot call site asserts its operand's id with
  `llvm.type.test` + `llvm.assume`, **on the very vtable-pointer SSA value
  the slot load and call hang off**. This discipline is load-bearing:
  `WholeProgramDevirt` discovers devirtualizable calls by a pure def-use walk
  from the tested pointer (constant-offset GEPs → load → call, dominance
  required), with no aliasing reasoning — testing a separate load of the
  same slot would silently disconnect the assertion from the call.

WPD devirtualizes per (type id, slot byte offset), so drop (0), clone (8)
and evaluate (16) fold independently.

## Speculative devirtualization — sound on any world

The backend runs `WholeProgramDevirtPass(nullptr, nullptr,
/*DevirtSpeculatively=*/true)`. In this mode a single-implementation family
is not folded unconditionally: WPD **versions the call site** — compare the
loaded function pointer against the sole implementation, direct call (which
then inlines) on the likely arm, the original indirect call as the fallback.
A vtable the module has never seen simply takes the fallback, so the
transform needs no closed-world assumption: `closure-wpd` is a compile-time
knob, not a soundness gate. Partitioned builds (`--codegen-units N`) and
even JIT increments could opt in — each module devirtualizes the families it
sees whole and leaves the rest indirect. Speculative mode also structurally
skips branch funnels and virtual constant propagation, neither of which
survives instruction selection outside the LTO pipelines (a funnel emitted
here would abort codegen: "all llvm.icall.branch.funnel operands must refer
to the same GlobalValue").

The `!vcall_visibility` stamp is not required by speculative mode; it is
kept because it is free and leaves an unconditional closed-world WPD mode —
no guard, but single-CGU-only and genuinely unsound cross-module — one
pipeline switch away should the guard ever show up in profiles.

## Pipeline

`rrc` (on by default at `-O aggressive`/`-O size`; `--no-closure-wpd` opts
out) → `LoweringOptions::closure_wpd` → the basic-ops lowering prologue
stamps each outlined vtable op with its id (`reussir.wpd.type_id`) and marks
the module (`reussir.wpd`) → the vtable conversion pattern carries the id
onto the vtable global; the dispatch conversion patterns (eval/clone/drop)
emit a `reussir.closure.wpd_test` directly on the vtable pointer they just
loaded → at `translateModuleToLLVMIR`, the dialect's LLVM translation
interface — the LLVM dialect can express neither `!type` metadata nor
`llvm.type.test`'s metadata-string operand — stamps the metadata and lowers
each `wpd_test` into `llvm.type.test` + `llvm.assume` → the backend LLVM
pipeline (`reussirRunBackendLLVMPipeline`, shared by the AOT compiler and
the JIT) runs speculative `WholeProgramDevirt` before the per-module
pipeline (so guarded direct calls inline), then `LowerTypeTests` in
drop-Assume mode so no `llvm.type.test` reaches instruction selection.
Modules lowered without `closure-wpd` (REPL/JIT increments among them) carry
no type tests and keep their pipeline unchanged.

Because the type test is emitted on the same SSA load the call consumes, the
pipeline needs no GVN run to connect them, and because speculative mode
never emits funnels, no `cl::opt` surgery is needed either — the whole
integration is two stock passes.

## What the scheme trades away

Ids are keyed by return type only, so single-impl devirt fires iff the
module has exactly one closure per return type (speculative mode also
disregards trivial empty `void` slot functions when collecting targets). A
finer family scheme (per-suffix signature tiers with a dataflow that
strengthens eval sites past the family root) can recover modules where
several closures share a return type but a richer static suffix isolates
one implementation; it was tried (PRs #372/#373), costs ~10× the machinery,
and is an incremental layer on top of this design if benchmarks ever demand
it.

## Invariant canary

Everything above rests on the `evaluate` ABI reading all arguments from the
payload. `tests/integration/conversion/closure_wpd_ids.mlir` pins both that
signature and the id scheme (two closures with different signatures but the
same return type share one id); if evaluation ever passes remaining
arguments in registers, the slot ABI depends on more than the return type
and the per-return-type ids must be reworked.
