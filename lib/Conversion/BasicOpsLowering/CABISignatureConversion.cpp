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
/// This file implements C ABI signature conversion.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/CABISignatureConversion.h"

namespace reussir {
namespace {

static bool isScalarIntegerLike(mlir::Type type) {
  return mlir::isa<mlir::IntegerType>(type) ||
         mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

} // namespace

bool isTrivialFFIType(mlir::Type type) { return isScalarIntegerLike(type); }

CABISignature evaluateCABISignatureForC(mlir::Type returnType,
                                        llvm::ArrayRef<mlir::Type> paramTypes) {
  CABISignature sig;
  sig.abiReturnType = returnType;
  sig.returnStorageType = returnType;
  auto ptrType = mlir::LLVM::LLVMPointerType::get(returnType.getContext());
  const bool trivialReturn = mlir::isa<mlir::LLVM::LLVMVoidType>(returnType) ||
                             isTrivialFFIType(returnType);
  const bool trivialParams =
      paramTypes.size() < 4 &&
      llvm::all_of(
          paramTypes, [](mlir::Type type) { return isTrivialFFIType(type); });

  sig.isTrivial = trivialReturn && trivialParams;
  if (sig.isTrivial) {
    sig.abiParamTypes.append(paramTypes.begin(), paramTypes.end());
    return sig;
  }

  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(returnType) &&
      !isTrivialFFIType(returnType)) {
    sig.hasReturnPtr = true;
    sig.returnPtrIndex = 0;
    sig.abiReturnType = mlir::LLVM::LLVMVoidType::get(returnType.getContext());
    sig.abiParamTypes.push_back(ptrType);
  }

  if (!paramTypes.empty()) {
    sig.hasPackedArgs = true;
    sig.packedArgsIndex = sig.abiParamTypes.size();
    sig.packedArgsType = mlir::LLVM::LLVMStructType::getLiteral(
        returnType.getContext(), paramTypes);
    sig.abiParamTypes.push_back(ptrType);
  }

  return sig;
}

} // namespace reussir
