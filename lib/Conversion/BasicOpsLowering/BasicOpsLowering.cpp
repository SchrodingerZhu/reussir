#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
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
#include "Reussir/IR/ReussirEnumAttrs.h"
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

struct ReussirTokenReallocConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenReallocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenReallocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TokenType inputTokenType =
        llvm::TypeSwitch<mlir::Type, TokenType>(op.getToken().getType())
            .Case<TokenType>([](TokenType type) { return type; })
            .Case<NullableType>([](NullableType type) {
              return llvm::cast<TokenType>(type.getPtrTy());
            })
            .Default([](mlir::Type type) -> TokenType {
              llvm::report_fatal_error("Unexpected token type");
            });
    TokenType outputTokenType = op.getRealloced().getType();
    size_t oldAlign = inputTokenType.getAlign();
    size_t oldSize = inputTokenType.getSize();
    size_t newAlign = outputTokenType.getAlign();
    size_t newSize = outputTokenType.getSize();
    mlir::Value oldAlignVal = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(oldAlign));
    mlir::Value oldSizeVal = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(oldSize));
    mlir::Value newAlignVal = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(newAlign));
    mlir::Value newSizeVal = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getIndexAttr(newSize));
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, "__reussir_reallocate", mlir::TypeRange{llvmPtrType},
        mlir::ValueRange{adaptor.getToken(), oldAlignVal, oldSizeVal,
                         newAlignVal, newSizeVal});
    return mlir::success();
  }
};

struct ReussirRefStoreConversionPattern
    : public mlir::OpConversionPattern<ReussirRefStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the converted operands (reference pointer and value)
    mlir::Value refPtr = adaptor.getRef();
    mlir::Value value = adaptor.getValue();

    // Create LLVM store operation
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, value, refPtr);

    return mlir::success();
  }
};

struct ReussirRefSpilledConversionPattern
    : public mlir::OpConversionPattern<ReussirRefSpilledOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefSpilledOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    // Get the value to spill (already converted by the type converter)
    mlir::Value value = adaptor.getValue();

    auto converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    auto valueType = converter->convertType(op.getValue().getType());
    auto llvmPtrType = converter->convertType(op.getSpilled().getType());
    auto alignment =
        converter->getDataLayout().getTypePreferredAlignment(valueType);

    // Allocate stack space using llvm.alloca
    auto convertedIndexType = converter->getIndexType();
    auto constantArraySize = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(convertedIndexType, 1));
    auto allocaOp = rewriter.create<mlir::LLVM::AllocaOp>(
        loc, llvmPtrType, valueType, constantArraySize, alignment);

    // Store the value to the allocated space
    rewriter.create<mlir::LLVM::StoreOp>(loc, value, allocaOp);

    // Return the pointer to the allocated space
    rewriter.replaceOp(op, allocaOp);

    return mlir::success();
  }
};

struct ReussirRecordCompoundConversionPattern
    : public mlir::OpConversionPattern<ReussirRecordCompoundOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRecordCompoundOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get the record type and convert it to LLVM struct type
    RecordType recordType = op.getCompound().getType();
    mlir::Type llvmStructType = converter->convertType(recordType);

    if (!llvmStructType)
      return op.emitOpError("failed to convert record type to LLVM type");

    // Create an undef value of the struct type
    auto undefOp = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmStructType);

    // Get the field values (already converted by the type converter)
    auto fieldValues = adaptor.getFields();

    // Insert each field using insertvalue
    mlir::Value result = undefOp;
    for (size_t i = 0; i < fieldValues.size(); ++i)
      result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result,
                                                          fieldValues[i], i);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReussirRecordVariantConversionPattern
    : public mlir::OpConversionPattern<ReussirRecordVariantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRecordVariantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get the record type and convert it to LLVM struct type
    RecordType recordType = op.getVariant().getType();
    mlir::Type llvmStructType = converter->convertType(recordType);

    if (!llvmStructType)
      return op.emitOpError("failed to convert record type to LLVM type");
    auto indexType = converter->getIndexType();
    // Get the tag and value (already converted by the type converter)
    mlir::Value tag = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(indexType, op.getTag().getZExtValue()));
    mlir::Value value = adaptor.getValue();

    // Get the preferred alignment for the struct type
    auto alignment =
        converter->getDataLayout().getTypePreferredAlignment(llvmStructType);
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(indexType, 1));
    // Allocate stack space for the struct
    auto allocaOp = rewriter.create<mlir::LLVM::AllocaOp>(
        loc, ptrType, llvmStructType, one, alignment);

    // Get a pointer to the tag field (index 0) and store the tag
    auto tagPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, llvmStructType, allocaOp,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    rewriter.create<mlir::LLVM::StoreOp>(loc, tag, tagPtr);

    // Get a pointer to the value field (index 1) and store the value
    auto valuePtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, llvmStructType, allocaOp,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
    rewriter.create<mlir::LLVM::StoreOp>(loc, value, valuePtr);

    // Load the complete struct from the allocated space
    auto result =
        rewriter.create<mlir::LLVM::LoadOp>(loc, llvmStructType, allocaOp);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReussirRefLoadConversionPattern
    : public mlir::OpConversionPattern<ReussirRefLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type pointeeTy = op.getResult().getType();
    mlir::Type llvmPointeeTy = getTypeConverter()->convertType(pointeeTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, llvmPointeeTy,
                                                    adaptor.getRef());
    return mlir::success();
  }
};

struct ReussirReferenceProjectConversionPattern
    : public mlir::OpConversionPattern<ReussirRefProjectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefProjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get the reference pointer (already converted by the type converter)
    mlir::Value refPtr = adaptor.getRef();

    // Get the index value
    auto indexType = converter->getIndexType();
    mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(indexType, op.getIndex().getZExtValue()));

    // Get the result type (should be a pointer type after conversion)
    mlir::Type resultType = converter->convertType(op.getProjected().getType());
    auto llvmPtrType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(resultType);
    if (!llvmPtrType)
      return op.emitOpError("projected result must be an LLVM pointer type");

    // Get the element type that the reference points to
    RefType refType = op.getRef().getType();
    mlir::Type elementType = converter->convertType(refType.getElementType());

    // Create GEP operation to get the field pointer
    auto gepOp = rewriter.create<mlir::LLVM::GEPOp>(
        loc, llvmPtrType, elementType, refPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, index});

    rewriter.replaceOp(op, gepOp);
    return mlir::success();
  }
};

struct ReussirNullableCheckConversionPattern
    : public mlir::OpConversionPattern<ReussirNullableCheckOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableCheckOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type llvmPtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value nullable = adaptor.getNullable();
    mlir::Value nullConstant =
        rewriter.create<mlir::LLVM::ZeroOp>(op.getLoc(), llvmPtrType);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::eq, nullable, nullConstant);
    return mlir::success();
  }
};

struct ReussirNullableCreateConversionPattern
    : public mlir::OpConversionPattern<ReussirNullableCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if the operation has input. If so, replace it directly with adaptor
    // value. Otherwise, create a new null value.
    if (op.getPtr())
      rewriter.replaceOp(op, adaptor.getPtr());
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
          op, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
    return mlir::success();
  }
};

struct ReussirRcIncConversionPattern
    : public mlir::OpConversionPattern<ReussirRcIncOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcIncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    RcType rcPtrTy = op.getRcPtr().getType();
    if (rcPtrTy.getCapability() == Capability::value ||
        rcPtrTy.getCapability() == Capability::flex ||
        rcPtrTy.getCapability() == Capability::field)
      return op.emitOpError("unsupported capability");

    // If it is rigid, we directly emit __reussir_acquire_rigid_object
    if (rcPtrTy.getCapability() == Capability::rigid) {
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, "__reussir_acquire_rigid_object", mlir::TypeRange{},
          mlir::ValueRange{adaptor.getRcPtr()});
      return mlir::success();
    }

    RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
    // GEP [0].1
    auto convertedBoxType = getTypeConverter()->convertType(rcBoxType);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto refcntPtr = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), llvmPtrType, convertedBoxType, adaptor.getRcPtr(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    auto indexType = static_cast<const LLVMTypeConverter *>(getTypeConverter())
                         ->getIndexType();
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(indexType, 1));
    mlir::Value oldRefCnt;
    if (rcPtrTy.getAtomicKind() == AtomicKind::normal) {
      oldRefCnt = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), indexType,
                                                      refcntPtr);
      auto newRefCnt = rewriter.create<mlir::arith::AddIOp>(
          op.getLoc(), indexType, oldRefCnt, one);
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), newRefCnt, refcntPtr);
    } else {
      oldRefCnt = rewriter.create<mlir::LLVM::AtomicRMWOp>(
          op.getLoc(), mlir::LLVM::AtomicBinOp::add, refcntPtr, one,
          mlir::LLVM::AtomicOrdering::monotonic);
    }
    auto geOne = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(), mlir::LLVM::ICmpPredicate::uge, oldRefCnt, one);
    rewriter.create<mlir::LLVM::AssumeOp>(op.getLoc(), geOne);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReussirRcCreateOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    RcType rcPtrTy = op.getRcPtr().getType();
    RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
    if (rcBoxType.isRegional())
      return op->emitError("TODO: regional rc create");
    if (rcPtrTy.getAtomicKind() == AtomicKind::atomic)
      return op->emitError("TODO: atomic rc create");
    auto convertedBoxType = getTypeConverter()->convertType(rcBoxType);
    auto indexType = static_cast<const LLVMTypeConverter *>(getTypeConverter())
                         ->getIndexType();
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(indexType, 1));
    auto refcntPtr = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), llvmPtrType, convertedBoxType, adaptor.getToken(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    auto elementPtr = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), llvmPtrType, convertedBoxType, adaptor.getToken(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), one, elementPtr);
    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), adaptor.getValue(),
                                         refcntPtr);
    rewriter.replaceOp(op, adaptor.getToken());
    return mlir::success();
  }
};

struct ReussirRcBorrowOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcBorrowOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcBorrowOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    RcType rcPtrTy = op.getRcPtr().getType();
    RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, llvmPtrType, getTypeConverter()->convertType(rcBoxType),
        adaptor.getRcPtr(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, rcBoxType.getElementIndex()});
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
                     {llvmPtrType});
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
    target.addIllegalOp<
        ReussirTokenAllocOp, ReussirTokenFreeOp, ReussirTokenReinterpretOp,
        ReussirTokenReallocOp, ReussirRefLoadOp, ReussirRefStoreOp,
        ReussirRefSpilledOp, ReussirNullableCheckOp, ReussirNullableCreateOp,
        ReussirRcIncOp, ReussirRcCreateOp, ReussirRcBorrowOp,
        ReussirRecordCompoundOp, ReussirRecordVariantOp, ReussirRefProjectOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void populateBasicOpsLoweringToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<
      ReussirTokenAllocConversionPattern, ReussirTokenFreeConversionPattern,
      ReussirTokenReinterpretConversionPattern,
      ReussirTokenReallocConversionPattern, ReussirRefLoadConversionPattern,
      ReussirRefStoreConversionPattern, ReussirRefSpilledConversionPattern,
      ReussirNullableCheckConversionPattern,
      ReussirNullableCreateConversionPattern, ReussirRcIncConversionPattern,
      ReussirRcCreateOpConversionPattern, ReussirRcBorrowOpConversionPattern,
      ReussirRecordCompoundConversionPattern,
      ReussirRecordVariantConversionPattern,
      ReussirReferenceProjectConversionPattern>(converter,
                                                patterns.getContext());
}
} // namespace reussir
