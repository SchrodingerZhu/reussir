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
/// This header file provides patterns for inferring variant tags in Reussir
/// drop and acquire operations.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_TRANSFORMATION_INFERVARIANTTAG_H
#define REUSSIR_TRANSFORMATION_INFERVARIANTTAG_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

#define GEN_PASS_DECL_REUSSIRINFERVARIANTTAGPASS
#include "Reussir/Transformation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Tag inference
//===----------------------------------------------------------------------===//
// A variant record reference has a known tag if:
//
// 1. it is dominated by a coercion operation whose operand has a must-alias
//    relation to the target reference. In this case, the reference tag is
//    inferred from the coercion tag.
// 2. the reference is inside a record dispatch region and the region parent
//   (the dispatch operation) is applied to a reference that is a must-alias to
//   the target reference. In this case, the reference tag is inferred from the
//   dispatch tag of the corresponding region.
//
// For now, we only care about the above two cases, and we only attach the tag
// attribute to drop operations. Ideally, the analysis can be made in a dataflow
// framework, but we assume functional DAG flow so we can just go through
// regions.
//
// To run the analysis, we first initialize the domainence analysis and the
// alias analysis on the function. We collect all the coercion operations.
// At each block, we select out all the coercion operations that dominate the
// block, recording the references and tags.  Additionally, we check if the
// block has a record dispatch operation as parent where the region of current
// block has a unique tag. If so, we also add the corresponding reference and
// tag into consideration. References and their tags are maintained in a
// DenseMap.
//
// Inside the block, we walk through each operation of interest (drop and
// acquire operations) and check if it has a must-alias relation to any
// reference within the context. (We assume consistency if there are multiple
// matches. Hence, we only select the first match.) If so, we attach the variant
// attribute to the drop or acquire operation.
//===----------------------------------------------------------------------===//

void runTagInference(mlir::func::FuncOp func);

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_INFERVARIANTTAG_H
