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
/// This file declares the interfaces for the Reussir dialect.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIRINTERFACES_H
#define REUSSIR_IR_REUSSIRINTERFACES_H

#include "Reussir/IR/ReussirTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

/// Include the generated interface declarations.
#include "Reussir/IR/ReussirInterfaces.h.inc"

#endif // REUSSIR_IR_REUSSIRINTERFACES_H
