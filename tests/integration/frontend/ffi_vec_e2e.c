//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for ffi_vec_e2e.rr. `main` returns `f64`, which is nontrivial at
// the trampoline boundary, so the export takes a leading return pointer.

extern void reussir_main(double *ret);

int main(void) {
  double result;
  reussir_main(&result);
  return result == 3.75 ? 0 : 1;
}
