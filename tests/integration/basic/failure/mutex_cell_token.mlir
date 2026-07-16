// RUN: %reussir-opt %s -verify-diagnostics

// A token-attached mutex create is verified at parse time, before any pass can
// declare a pass-level dialect dependency: checking the token against the box
// layout sizes the cell as `!sync.mutex<i64>`, so the reussir dialect itself
// must load the sync dialect (`dependentDialects` in ReussirDialect.td) — the
// verifier used to crash here instead of diagnosing the mismatched token. The
// token is deliberately undersized so the check stays independent of the
// host's default data layout.
func.func @mutex_create_with_bad_token(%value: i64)
    -> !reussir.rc<!reussir.cell<i64 mutex> atomic> {
  %token = reussir.token.alloc : !reussir.token<align: 1, size: 1>
  // expected-error @+1 {{token type must match RC cell box layout}}
  %cell = reussir.cell.create value(%value : i64)
    token(%token : !reussir.token<align: 1, size: 1>)
    : !reussir.rc<!reussir.cell<i64 mutex> atomic>
  return %cell : !reussir.rc<!reussir.cell<i64 mutex> atomic>
}
