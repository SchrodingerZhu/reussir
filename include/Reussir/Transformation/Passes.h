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
/// This header file provides the definitions for transformation passes used in
/// the Reussir dialect.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_TRANSFORMATION_PASSES_H
#define REUSSIR_TRANSFORMATION_PASSES_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace reussir {
#define GEN_PASS_DECL
#include "Reussir/Transformation/Passes.h.inc"

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_PASSES_H
