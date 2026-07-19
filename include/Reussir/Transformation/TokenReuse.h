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
/// This header file provides the definition for the token reuse pass.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_TRANSFORMATION_TOKENREUSE_H
#define REUSSIR_TRANSFORMATION_TOKENREUSE_H

#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DECL_REUSSIRTOKENREUSEPASS
#include "Reussir/Transformation/Passes.h.inc"

} // namespace reussir

#endif // REUSSIR_TRANSFORMATION_TOKENREUSE_H
