//===-- BasicOpsLowering.h - Reussir basic ops lowering --------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides lowering patterns for basic Reussir operations.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_CONVERSION_BASICOPSLOWERING_H
#define REUSSIR_CONVERSION_BASICOPSLOWERING_H

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/Conversion/TypeConverter.h"

namespace reussir {

#define GEN_PASS_DECL_REUSSIRBASICOPSLOWERINGPASS
#include "Reussir/Conversion/Passes.h.inc"

void populateBasicOpsLoweringToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);
} // namespace reussir

#endif // REUSSIR_CONVERSION_BASICOPSLOWERING_H
