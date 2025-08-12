#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Pass/Pass.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"

namespace reussir {
#define GEN_PASS_DEF_REUSSIRBASICOPSLOWERINGPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {} // namespace

//===----------------------------------------------------------------------===//
// Runtime Functions
//===----------------------------------------------------------------------===//

namespace {
void addRuntimeFunction(mlir::Block *body, llvm::StringRef name,
                        llvm::ArrayRef<mlir::Type> inputs,
                        llvm::ArrayRef<mlir::Type> outputs) {
  mlir::MLIRContext *ctx = body->getParentOp()->getContext();
  mlir::FunctionType type = mlir::FunctionType::get(ctx, inputs, outputs);
  mlir::func::FuncOp func =
      mlir::func::FuncOp::create(mlir::UnknownLoc::get(ctx), name, type);
  func.setPrivate();
  body->push_front(func);
}

void addRuntimeFunctions(mlir::ModuleOp module,
                         const LLVMTypeConverter &converter) {
  mlir::MLIRContext *ctx = module.getContext();
  mlir::Block *body = module.getBody();
  auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto indexType = converter.getIndexType();
  addRuntimeFunction(body, "__reussir_freeze_flex_object", {llvmPtrType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_cleanup_region", {llvmPtrType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_acquire_rigid_object", {llvmPtrType}, {});
  addRuntimeFunction(body, "__reussir_release_rigid_object", {llvmPtrType}, {});
  addRuntimeFunction(body, "__reussir_allocate", {indexType, indexType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_deallocate",
                     {llvmPtrType, indexType, indexType}, {});
  addRuntimeFunction(body, "__reussir_reallocate",
                     {llvmPtrType, indexType, indexType, indexType, indexType},
                     {});
  // currently this will abort execution after printing the message and
  // stacktrace. No unwinding is attempted yet.
  addRuntimeFunction(body, "__reussir_panic", {llvmPtrType}, {});
}
} // namespace

//===----------------------------------------------------------------------===//
// BasicOpsLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct BasicOpsLoweringPass
    : public impl::ReussirBasicOpsLoweringPassBase<BasicOpsLoweringPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::LLVMConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(getOperation());
    populateBasicOpsLoweringToLLVMConversionPatterns(converter, patterns);
    addRuntimeFunctions(getOperation(), converter);
    target.addIllegalDialect<ReussirDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateBasicOpsLoweringToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {}
} // namespace reussir
