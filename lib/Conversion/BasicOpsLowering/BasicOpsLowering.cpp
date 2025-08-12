#include <llvm/ADT/ArrayRef.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
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

namespace {

struct ReussirTokenAllocConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenAllocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // Get the token type and extract alignment and size
    TokenType tokenType = op.getToken().getType();
    uint64_t alignment = tokenType.getAlign();
    uint64_t size = tokenType.getSize();

    // Create constants for alignment and size
    auto alignConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexAttr(alignment));
    auto sizeConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexAttr(size));

    // Get the LLVM pointer type for the result
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(ctx);

    // Create the runtime function call
    auto funcOp = rewriter.create<mlir::func::CallOp>(
        loc, "__reussir_allocate", mlir::TypeRange{llvmPtrType},
        mlir::ValueRange{alignConst, sizeConst});

    // Replace the original operation with the function call result
    rewriter.replaceOp(op, funcOp.getResult(0));

    return mlir::success();
  }
};

struct ReussirTokenFreeConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenFreeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenFreeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Get the token operand (already converted to LLVM pointer)
    mlir::Value tokenPtr = adaptor.getToken();

    // Get the token type and extract alignment and size
    TokenType tokenType = llvm::dyn_cast<TokenType>(op.getToken().getType());
    if (!tokenType)
      return op.emitOpError("token operand must be of TokenType");

    uint64_t alignment = tokenType.getAlign();
    uint64_t size = tokenType.getSize();

    // Create constants for alignment and size
    auto alignConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexAttr(alignment));
    auto sizeConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexAttr(size));

    // Replace the original operation with the runtime function call
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, "__reussir_deallocate", mlir::TypeRange{}, // No return type
        mlir::ValueRange{tokenPtr, alignConst, sizeConst});

    return mlir::success();
  }
};

struct ReussirTokenReinterpretConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenReinterpretOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenReinterpretOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For reinterpret, we just use the input token directly since it's already
    // converted to an LLVM pointer by the type converter
    rewriter.replaceOp(op, adaptor.getToken());
    return mlir::success();
  }
};

} // namespace

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
  auto indexType = mlir::IndexType::get(ctx);
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
    mlir::populateFuncToLLVMFuncOpConversionPattern(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
    addRuntimeFunctions(getOperation(), converter);
    target.addIllegalDialect<mlir::func::FuncDialect,
                             mlir::arith::ArithDialect>();
    target.addIllegalOp<ReussirTokenAllocOp, ReussirTokenFreeOp,
                        ReussirTokenReinterpretOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateBasicOpsLoweringToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<ReussirTokenAllocConversionPattern>(converter,
                                                   patterns.getContext());
  patterns.add<ReussirTokenFreeConversionPattern>(converter,
                                                  patterns.getContext());
  patterns.add<ReussirTokenReinterpretConversionPattern>(converter,
                                                         patterns.getContext());
}
} // namespace reussir
