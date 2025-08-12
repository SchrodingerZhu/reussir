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

namespace {
struct BasicOpsLoweringPass
    : public impl::ReussirBasicOpsLoweringPassBase<BasicOpsLoweringPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::LLVMConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(getOperation());
    populateBasicOpsLoweringToLLVMConversionPatterns(converter, patterns);
    target.addIllegalDialect<ReussirDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateBasicOpsLoweringToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  llvm_unreachable("Not implemented");
}
} // namespace reussir
