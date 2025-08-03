//===-- ReussirOps.cpp - Reussir operations implementation ------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the operations used in the Reussir dialect.
//
//===----------------------------------------------------------------------===//
#include <mlir/IR/OpImplementation.h>

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.cpp.inc"

namespace reussir {
///===----------------------------------------------------------------------===//
// Reussir Dialect Operations Registration
//===----------------------------------------------------------------------===//
void ReussirDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Reussir/IR/ReussirOps.cpp.inc"
      >();
}
} // namespace reussir
