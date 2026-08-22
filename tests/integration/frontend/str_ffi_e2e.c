//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
// Driver for str_ffi_e2e.rr: echo(1)*10000 + len(6)*1000 + byte 'r'(114).

extern unsigned long long reussir_main(void);

int main(void) { return reussir_main() == 16114 ? 0 : 1; }
