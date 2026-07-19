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
/// This header file defines the Reussir dialect and its core functionality.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_IR_REUSSIRDIALECT_H
#define REUSSIR_IR_REUSSIRDIALECT_H

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dialect.h>

#include "Reussir/IR/ReussirDialect.h.inc"

namespace reussir {
// TODO: this does really work since we need to link external modules
// void registerReussirDialectTranslation(mlir::DialectRegistry &registry);
} // namespace reussir

#endif // REUSSIR_IR_REUSSIRDIALECT_H
