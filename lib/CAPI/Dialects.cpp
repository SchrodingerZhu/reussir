//===-- Dialects.cpp - Reussir dialect C API ------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Dialects.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "Reussir/IR/ReussirDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Reussir, reussir, ::reussir::ReussirDialect)

void reussirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<::reussir::ReussirDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::math::MathDialect, mlir::ub::UBDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  mlir::MLIRContext *ctx = unwrap(context);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
}
