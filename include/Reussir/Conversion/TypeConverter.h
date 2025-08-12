//===-- TypeConverter.h - Reussir type conversion utilities ---*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides type conversion utilities for the Reussir dialect.
//
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

class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp op);

  std::optional<llvm::LogicalResult>
  convertRecordType(RecordType type,
                    llvm::SmallVectorImpl<mlir::Type> &results);

  const mlir::DataLayout &getDataLayout() const { return dataLayout; }

private:
  mlir::DataLayout dataLayout;
};
} // namespace reussir

#endif // REUSSIR_CONVERSION_TYPECONVERTER_H
