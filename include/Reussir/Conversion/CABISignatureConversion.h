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
/// This header defines helpers for evaluating C ABI-facing trampoline
/// signatures from internal LLVM-level function types.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace reussir {

struct CABISignature {
  bool isTrivial = false;
  mlir::Type abiReturnType;
  llvm::SmallVector<mlir::Type> abiParamTypes;
  bool hasReturnPtr = false;
  unsigned returnPtrIndex = 0;
  mlir::Type returnStorageType;
  bool hasPackedArgs = false;
  unsigned packedArgsIndex = 0;
  mlir::LLVM::LLVMStructType packedArgsType;
};

bool isTrivialFFIType(mlir::Type type);

CABISignature evaluateCABISignatureForC(mlir::Type returnType,
                                        llvm::ArrayRef<mlir::Type> paramTypes);

} // namespace reussir
