//===-- DropExpansion.h - Reussir drop expansion ------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides expansion patterns for Reussir drop operations.
//
//===----------------------------------------------------------------------===//
#ifndef REUSSIR_CONVERSION_DROPEXPANSION_H
#define REUSSIR_CONVERSION_DROPEXPANSION_H

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace reussir {

#define GEN_PASS_DECL_REUSSIRDROPEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

void populateDropExpansionConversionPatterns(mlir::RewritePatternSet &patterns);

} // namespace reussir

#endif // REUSSIR_CONVERSION_DROPEXPANSION_H
