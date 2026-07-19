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
/// This header file provides expansion patterns for Reussir acquire and drop
/// operations.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_CONVERSION_ACQUIREDROPEXPANSION_H
#define REUSSIR_CONVERSION_ACQUIREDROPEXPANSION_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

#define GEN_PASS_DECL_REUSSIRACQUIREDROPEXPANSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

void populateAcquireDropExpansionConversionPatterns(
    mlir::RewritePatternSet &patterns, bool outlineRecord);

} // namespace reussir

#endif // REUSSIR_CONVERSION_ACQUIREDROPEXPANSION_H
