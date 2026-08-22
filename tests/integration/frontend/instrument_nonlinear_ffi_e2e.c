//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for instrument_nonlinear_ffi_e2e.rr. `main` returns `f64`, which is
// nontrivial at the trampoline boundary, so the export takes a leading
// return pointer. The instrumentation must not change the result: both
// `total` calls see the same vector, copy-on-write notwithstanding.

extern void reussir_main(double *ret);

int main(void) {
  double result;
  reussir_main(&result);
  return result == 3.0 ? 0 : 1;
}
