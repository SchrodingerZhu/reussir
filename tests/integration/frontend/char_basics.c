//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for char_basics.rr: eight independent properties, one decimal
// digit each.

extern long long reussir_main(void);

int main(void) { return reussir_main() == 11111111 ? 0 : 1; }
