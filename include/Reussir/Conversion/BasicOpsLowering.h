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

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/BuiltinOps.h>

#include "Reussir/Conversion/TypeConverter.h"

namespace reussir {

#define GEN_PASS_DECL_REUSSIRBASICOPSLOWERINGPASS
#define GEN_PASS_DECL_REUSSIRDEBUGINFOCONVERSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

void populateBasicOpsLoweringToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void registerReussirBasicOpsLoweringInterface(mlir::DialectRegistry &registry);

void lowerFusedDBGAttributeInLocations(mlir::ModuleOp moduleOp);
} // namespace reussir

#endif // REUSSIR_CONVERSION_BASICOPSLOWERING_H
