//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for ffi_vec_record_e2e.rr: an i64 return is trivial at the
// trampoline boundary, so the export forwards directly.

#include <stdint.h>

extern int64_t reussir_main(void);

int main(void) { return reussir_main() == 42 ? 0 : 1; }
