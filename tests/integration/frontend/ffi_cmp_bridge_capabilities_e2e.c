//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for ffi_cmp_bridge_capabilities_e2e.rr. The Reussir entry point
// returns 42 only after all exact-capability and four-state partial-order
// checks have passed through Rust.

#include <stdint.h>

extern int64_t reussir_main(void);

int main(void) { return reussir_main() == 42 ? 0 : 1; }
