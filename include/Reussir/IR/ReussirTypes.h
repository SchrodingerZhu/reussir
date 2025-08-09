//===-- ReussirTypes.h - Reussir dialect types ------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides the definitions for types used in the Reussir
// dialect.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_IR_REUSSIRTYPES_H
#define REUSSIR_IR_REUSSIRTYPES_H

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LLVM.h>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypeDetails.h"

namespace reussir {
std::optional<std::pair<
    llvm::TypeSize,
    uint64_t>> inline deriveCompoundSizeAndAlignment(llvm::ArrayRef<mlir::Type>
                                                         members,
                                                     const mlir::DataLayout
                                                         &dataLayout);
bool isNonNullPointerType(mlir::Type type);
mlir::Type getProjectedType(mlir::Type type);
} // namespace reussir

#define GET_TYPEDEF_CLASSES
#include "Reussir/IR/ReussirOpsTypes.h.inc"

#endif // REUSSIR_IR_REUSSIRTYPES_H
