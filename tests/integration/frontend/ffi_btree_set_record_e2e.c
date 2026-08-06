//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for ffi_btree_set_record_e2e.rr. The result encodes the old set's
// first key and length followed by the new set's length and first key:
// old={2,4}, new={1,2,3,4} => 2 2 4 1.

#include <stdint.h>

extern int64_t reussir_main(void);

int main(void) { return reussir_main() == 2241 ? 0 : 1; }
