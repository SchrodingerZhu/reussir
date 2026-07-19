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
/// This header file provides patterns for canceling adjacent increment and
/// decrement operations in Reussir.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_TRANSFORMATION_INCDECCANCELLATION_H
#define REUSSIR_TRANSFORMATION_INCDECCANCELLATION_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

#define GEN_PASS_DECL_REUSSIRINCDECCANCELLATIONPASS
#include "Reussir/Transformation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// IncDecCancellationPass
//===----------------------------------------------------------------------===//
//
// This pass only cancels out subfield increment and decrement operations
// locally.
//
//===----------------------------------------------------------------------===//
llvm::LogicalResult runIncDecCancellation(mlir::func::FuncOp func);

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_INCDECCANCELLATION_H
