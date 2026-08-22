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
/// This file implements the Reussir non-linear FFI usage instrumentation:
/// before every FFI-import call that consumes an rc'd `ffi_object` or `array`
/// argument, insert a reference-count check that reports the call site to the
/// runtime when the count is not one.
///
//===----------------------------------------------------------------------===//

#include <string>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h>

#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

namespace reussir {

#define GEN_PASS_DEF_REUSSIRINSTRUMENTNONLINEARFFIPASS
#include "Reussir/Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kReportFunctionName =
    "__reussir_report_nonlinear_usage";

// The source position a location resolves to, for the runtime report. Walks
// through fused, named, and call-site locations to the first file location
// with a valid line (same policy as the debug-info conversion).
struct SourcePos {
  llvm::StringRef file;
  unsigned line = 0;
  unsigned col = 0;
};

SourcePos extractSourcePos(mlir::Location loc) {
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc))
    return {fileLoc.getFilename().getValue(), fileLoc.getLine(),
            fileLoc.getColumn()};
  if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location inner : fusedLoc.getLocations()) {
      SourcePos pos = extractSourcePos(inner);
      if (pos.line != 0)
        return pos;
    }
  }
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc))
    return extractSourcePos(nameLoc.getChildLoc());
  if (auto callSiteLoc = llvm::dyn_cast<mlir::CallSiteLoc>(loc))
    return extractSourcePos(callSiteLoc.getCallee());
  return {};
}

mlir::LLVM::LLVMFuncOp getOrCreateReportFunction(mlir::ModuleOp module) {
  if (auto existing =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(kReportFunctionName))
    return existing;
  mlir::MLIRContext *ctx = module.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto fnType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(ctx), {ptrType, i32Type, i32Type});
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(module.getBody());
  auto func = mlir::LLVM::LLVMFuncOp::create(
      builder, mlir::UnknownLoc::get(ctx), kReportFunctionName, fnType);
  func.setLinkage(mlir::LLVM::Linkage::External);
  return func;
}

// The address of a content-addressed, NUL-terminated global holding `file`,
// so the name crosses the boundary as a single C string.
mlir::Value materializeFileName(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::ModuleOp module, llvm::StringRef file) {
  std::string payload = file.str();
  payload.push_back('\0');
  std::string globalName =
      mangledBlake3Symbol("REUSSIR_NONLINEAR_FILE", payload);
  if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
    auto i8Type = mlir::IntegerType::get(builder.getContext(), 8);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, payload.size());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    mlir::LLVM::GlobalOp::create(builder, loc, arrayType,
                                 /*isConstant=*/true,
                                 mlir::LLVM::Linkage::LinkonceODR, globalName,
                                 builder.getStringAttr(payload));
  }
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return mlir::LLVM::AddressOfOp::create(builder, loc, ptrType, globalName);
}

// Before `call`, checks `operand`'s reference count and reports the call's
// source position to the runtime when the count is not one. The report path
// is `reussir.expect`'ed cold.
void instrumentOperand(mlir::ModuleOp module, mlir::func::CallOp call,
                       mlir::Value operand) {
  mlir::OpBuilder builder(call);
  mlir::Location loc = call.getLoc();
  mlir::Value count =
      ReussirRcFetchOp::create(builder, loc, operand).getRefCount();
  mlir::Value one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value shared = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, count, one);
  mlir::Value cold =
      ReussirExpectOp::create(builder, loc, shared, false).getLikely();
  auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{}, cold,
                                      /*addThenBlock=*/true,
                                      /*addElseBlock=*/false);
  builder.setInsertionPointToStart(ifOp.thenBlock());
  SourcePos pos = extractSourcePos(loc);
  mlir::LLVM::LLVMFuncOp reportFn = getOrCreateReportFunction(module);
  mlir::Value filePtr = materializeFileName(builder, loc, module, pos.file);
  auto i32Type = builder.getI32Type();
  mlir::Value line = mlir::LLVM::ConstantOp::create(
      builder, loc, i32Type, builder.getI32IntegerAttr(pos.line));
  mlir::Value col = mlir::LLVM::ConstantOp::create(
      builder, loc, i32Type, builder.getI32IntegerAttr(pos.col));
  mlir::LLVM::CallOp::create(builder, loc, reportFn,
                             mlir::ValueRange{filePtr, line, col});
  mlir::scf::YieldOp::create(builder, loc);
}

//===----------------------------------------------------------------------===//
// InstrumentNonlinearFFIPass
//===----------------------------------------------------------------------===//

struct InstrumentNonlinearFFIPass
    : public impl::ReussirInstrumentNonlinearFFIPassBase<
          InstrumentNonlinearFFIPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // FFI call sites are ordinary `func.call`s whose callee is the native
    // declaration an import trampoline names as its target.
    llvm::DenseSet<llvm::StringRef> importTargets;
    module.walk([&](ReussirTrampolineOp op) {
      if (op.getDirection() == TrampolineDirection::Import)
        importTargets.insert(op.getTarget());
    });
    if (importTargets.empty())
      return;

    llvm::SmallVector<mlir::func::CallOp> calls;
    module.walk([&](mlir::func::CallOp call) {
      if (importTargets.contains(call.getCallee()))
        calls.push_back(call);
    });

    for (mlir::func::CallOp call : calls) {
      for (mlir::Value operand : call.getArgOperands()) {
        // Region-managed boxes do not carry an ordinary reference count.
        auto rcType = llvm::dyn_cast<RcType>(operand.getType());
        if (!rcType || rcType.isRegional())
          continue;
        if (!llvm::isa<FFIObjectType, ArrayType>(rcType.getElementType()))
          continue;
        instrumentOperand(module, call, operand);
      }
    }
  }
};

} // namespace

} // namespace reussir
