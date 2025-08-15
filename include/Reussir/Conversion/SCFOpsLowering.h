//===-- SCFOpsLowering.h - Reussir SCF ops lowering ------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides lowering patterns for Reussir SCF operations.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_CONVERSION_SCFOPSLOWERING_H
#define REUSSIR_CONVERSION_SCFOPSLOWERING_H

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace reussir {

#define GEN_PASS_DECL_REUSSIRSCFOPSLOWERINGPASS
#include "Reussir/Conversion/Passes.h.inc"

void populateSCFOpsLoweringConversionPatterns(
    mlir::RewritePatternSet &patterns);

} // namespace reussir

#endif // REUSSIR_CONVERSION_SCFOPSLOWERING_H
