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
/// This header file provides type conversion utilities for the Reussir dialect.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_CONVERSION_TYPECONVERTER_H
#define REUSSIR_CONVERSION_TYPECONVERTER_H

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

mlir::LowerToLLVMOptions getReussirToLLVMOptions(mlir::ModuleOp op);

void populateReussirToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);
} // namespace reussir

#endif // REUSSIR_CONVERSION_TYPECONVERTER_H
