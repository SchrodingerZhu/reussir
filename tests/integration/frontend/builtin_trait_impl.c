//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for builtin_trait_impl.rr: `main` computes 5*2 + (5*2)*2 = 30
// through the u64 impl of a local trait.

extern unsigned long long reussir_main(void);

int main(void) { return reussir_main() == 30 ? 0 : 1; }
