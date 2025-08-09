//===-- ReussirAttrs.cpp - Reussir attributes implementation ----*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the attributes used in the Reussir dialect.
//
//===----------------------------------------------------------------------===//
#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirDialect.h"

#include "Reussir/IR/ReussirAttrs.cpp.inc"

namespace reussir {

//===----------------------------------------------------------------------===//
// ReussirDialect Attributes Registration
//===----------------------------------------------------------------------===//
mlir::Attribute ReussirDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                               mlir::Type type) const {
  return {};
}

void ReussirDialect::printAttribute(mlir::Attribute attr,
                                    mlir::DialectAsmPrinter &p) const {}

void ReussirDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Reussir/IR/ReussirAttrs.cpp.inc"
      >();
}
} // namespace reussir
