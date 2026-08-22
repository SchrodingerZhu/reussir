# Non-linear FFI usage instrumentation

The `reussir-instrument-nonlinear-ffi` pass guards every FFI-import call that
consumes an rc'd `ffi_object` or `array` argument with a reference-count
check. A count other than one at the boundary calls the runtime entry point
`__reussir_report_nonlinear_usage(file, line, col)`, which prints the
call's source location on stderr.

Instrumentation is off by default. It is diagnostic output and never changes
program semantics: a shared value still crosses the boundary, the foreign side
still copies on write, and the result is identical.

## Why the count matters at the boundary

The FFI boundary follows the owned calling convention: every call consumes its
rc arguments (`docs/design/polymorphic-ffi.md`, "Ownership: the boundary
consumes"). When the caller still needs the value afterwards, the Perceus
analysis retains it (`reussir.rc.inc`) before the call, so the count the
callee observes is greater than one. For copy-on-write foreign structures —
`reussir_rt::collections::vec::Vec` and everything else built on the foreign
`Rc` header — that is exactly the case where an "in-place" update silently
clones the whole payload. The instrumentation makes that cliff observable at
its source location.

## Entry points

```sh
reussir-opt input.mlir --reussir-instrument-nonlinear-ffi
rrc input.rr -o demo.o --instrument-nonlinear-ffi
```

`rene` profiles spell it positively:

```toml
[profiles.dev]
instrument_nonlinear_ffi = true
```

## What the pass emits

FFI call sites are ordinary `func.call`s whose callee is the body-less native
declaration a `reussir.trampoline import` names as its `target`. For each such
call, and for each argument whose type is `!reussir.rc` of an `ffi_object` or
`array` payload (region-managed capabilities are skipped — their boxes carry
no ordinary count), the pass inserts before the call:

```mlir
%count  = reussir.rc.fetch (%arg : ...) : index
%shared = arith.cmpi ne, %count, %c1
%cold   = reussir.expect(%shared : i1, false) : i1
scf.if %cold {
  llvm.call @__reussir_report_nonlinear_usage(%file, %line, %col)
}
```

The file name is a content-addressed (`Blake3Symbol.h`), NUL-terminated
`llvm.mlir.global`; line and column come from the call's `FileLineColLoc`
(resolved through fused/name/call-site locations, zero when absent). The
report path is `reussir.expect`'ed cold.

## Pipeline position

The pass runs after `reussir-token-reuse` and the `rc-create-sink`/
`rc-create-fusion` pair and before `reussir-compile-polymorphic-ffi` — after
every pass that can change a reference count, so the observed count is the
final one, and before `scf` is lowered away. Running it earlier would be
wrong twice over: `inc-dec-cancellation` treats `reussir.rc.is_unique` as a
barrier but not `reussir.rc.fetch`, so an early fetch could observe a count a
cancelled pair later invalidates; and an early check would perturb the very
optimizations whose residue it is meant to report.

## Runtime

`__reussir_report_nonlinear_usage` lives in `reussir-rt`
(`src/instrument.rs`) and is registered in the JIT symbol table
(`src/symbols.rs`). It never aborts; it writes one line per firing check to
stderr.
