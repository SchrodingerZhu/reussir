//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Reussir dialect and its core functionality.
///
//===----------------------------------------------------------------------===//

// The dependent-dialect loading generated into the dialect constructor
// (`dependentDialects` in ReussirDialect.td) needs the sync dialect's
// declaration.
#include "Sync/IR/SyncDialect.h"

#include "Reussir/IR/ReussirDialect.h"

#include "Reussir/IR/ReussirDialect.cpp.inc"

namespace reussir {

void ReussirDialect::initialize() {
  // Register the builtin Attributes.
  registerAttributes();
  // Register the builtin Types.
  registerTypes();
  // Register the builtin Operations.
  registerOperations();
  // Register interfaces.
  registerInterfaces();
}
} // namespace reussir
