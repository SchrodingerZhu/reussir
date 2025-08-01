//===-- ReussirDialect.cpp - Reussir dialect implementation -----*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the Reussir dialect and its core functionality.
//
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"

#include "Reussir/IR/ReussirDialect.cpp.inc"

namespace reussir {

void ReussirDialect::initialize() {
  // Register the builtin Attributes.
  registerAttributes();
  // Register the builtin Types.
  registerTypes();
}

mlir::Attribute ReussirDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                               mlir::Type type) const {
  return {};
}

void ReussirDialect::printAttribute(mlir::Attribute attr,
                                    mlir::DialectAsmPrinter &p) const {}

void ReussirDialect::registerAttributes() {}

} // namespace reussir
