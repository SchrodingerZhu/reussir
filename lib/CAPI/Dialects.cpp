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
/// This file implements the Reussir dialect C API.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Dialects.h"

// Complete mlir::DialectVersion before MLIR C API headers reach
// BytecodeWriter.h; MSVC's STL rejects destroying unique_ptr<T> when T is only
// forward declared.
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/TransformOps/FuncTransformOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/TransformOps/DialectExtension.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/TransformOps/SCFTransformOps.h>
#include <mlir/Dialect/Transform/IR/TransformDialect.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "Reussir/Conversion/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Sync/IR/SyncDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Reussir, reussir,
                                      ::reussir::ReussirDialect)

namespace {
// Populates a registry with the Reussir dialect plus the minimal set of
// upstream dialects, ConvertToLLVM interfaces and translations the lowering
// pipeline relies on.
void populateReussirRegistry(mlir::DialectRegistry &registry) {
  // Only the dialects the Reussir lowering pipeline actually operates on. Using
  // the minimal set (rather than registerAllDialects/registerAllExtensions)
  // keeps unrelated dialects such as complex/tensor/bufferization out of the
  // context, so the ConvertToLLVM pass never trips over their unimplemented
  // promised interfaces.
  registry.insert<::reussir::ReussirDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::math::MathDialect, mlir::ub::UBDialect,
                  mlir::func::FuncDialect, mlir::cf::ControlFlowDialect,
                  mlir::ptr::PtrDialect, mlir::sync::SyncDialect>();

  // Since MLIR 17 the func dialect's DialectInlinerInterface lives in a
  // separately registered extension; without it the inliner pass resolves no
  // `func.call` sites and silently inlines nothing.
  mlir::func::registerInlinerExtension(registry);

  // The transform dialect carries user-authored schedules (issue #349): the
  // lowering pipeline can run `transform-interpreter` at named anchors over
  // scripts supplied by the driver (`rrc --transform-script`). Register the
  // dialect plus the extensions schedules need — func/structured matching
  // (`transform.structured.match`) and the scf loop transforms
  // (`transform.loop.*`). This keeps the minimal-dialect policy above: none
  // of it (nor the dialects the extensions declare) is loaded into a context
  // until a schedule actually parses transform ops, so pipelines without
  // scripts see the same loaded-dialect set as before.
  registry.insert<mlir::transform::TransformDialect>();
  mlir::func::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);

  // Register the ConvertToLLVMPatternInterface for exactly the dialects that
  // can reach the ConvertToLLVM pass, plus Reussir's own lowering interface.
  // These are the conversion-interface registration APIs (distinct from the
  // dialects themselves); without them a loaded dialect's promised interface is
  // fatal.
  ::reussir::registerReussirBasicOpsLoweringInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);

  // Translations needed to lower the LLVM dialect to LLVM IR.
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
}
} // namespace

void reussirPopulateRegistry(MlirDialectRegistry registry) {
  populateReussirRegistry(*unwrap(registry));
}

void reussirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  populateReussirRegistry(registry);
  mlir::MLIRContext *ctx = unwrap(context);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
}
