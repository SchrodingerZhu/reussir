//===-- ReussirOps.td - Reussir dialect operations ---------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reussir dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIROPS_TD
#define REUSSIR_IR_REUSSIROPS_TD

include "Reussir/IR/ReussirDialect.td"
include "Reussir/IR/ReussirAttrs.td"
include "mlir/IR/BuiltinAttributes.td"

class ReussirOp<string mnemonic, list<Trait> traits>
  : Op<ReussirDialect, mnemonic, traits>;

#endif // REUSSIR_IR_REUSSIROPS_TD
