//===-- BasicOpsLowering.cpp - Reussir basic ops lowering impl --*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/DebugInfo/DWARF/DWARFAttribute.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/Conversion/Blake3Symbol.h"
#include "Reussir/Conversion/CABISignatureConversion.h"
#include "Reussir/Conversion/TypeConverter.h"
#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Transformation/SpecialPointerTag.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"

namespace reussir {
#define GEN_PASS_DEF_REUSSIRBASICOPSLOWERINGPASS
#define GEN_PASS_DEF_REUSSIRCONVERTTOLLVMPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Special pointer tag support
//===----------------------------------------------------------------------===//
// When `reussir-special-pointer-tag` ran (module carries kSpecialPtrTagAttr),
// a value of a taggable shared rc<variant> type may be an unboxed immediate
// pointing at a per-tag *dummy box* — a well-formed shared box header the
// hot accesses land in, so refcount loads/increments, uniqueness reads, and
// even the variant-tag read through a tagged value lower exactly as without
// the scheme: plain loads, zero guards. The attribute's value selects one of
// two encodings, differing only in how the cold guards recognize immediates:
//   * `tbi` (aarch64): top byte = tag + 1, low bits = the dummy address;
//     the hardware ignores the top byte on data accesses. The dummy count is
//     initialized to 2 and guards test the pointer's top byte. Foreign code
//     may decode `tag = top - 1` without dereferencing.
//   * `immortal` (any target, including 32-bit ones like wasm32): the
//     immediate *is* the dummy address — no pointer bit tricks. For a w-bit
//     refcount word the dummy count is initialized to 3 * 2^(w-2) and guards
//     test refcount magnitude: a real box can never reach the 2^(w-1)
//     threshold (each reference occupies >= 4 bytes of address space, so
//     fewer than 2^(w-2) can exist). Foreign code sees a layout-compatible
//     box.
// Two invariants keep both encodings sound:
//   * the dummy's refcount is pinned above the shared/unique decision point,
//     so an immediate's decrement always takes the *shared* branch (no field
//     drop, no token, no free of the dummy) and `rc.is_unique` answers
//     false. In the immortal encoding on targets narrower than 64 bits,
//     `rc.inc`'s store is steered away from a recognized dummy as well —
//     2^(w-2) unbalanced increments would otherwise be reachable and wrap
//     the count. (With w = 64 that is > 4 * 10^18 increments: unreachable,
//     so the increment stays guard-free on 64-bit targets.)
//   * `rc.set` — the one decrementing store — steers a recognized-immediate
//     access to a separate scratch word (an address select off the critical
//     path), so the dummy count can never decay to 1.
// The only other guard is `rc.assume_unique`, which must neutralize its
// assumption for immediates (an `assume(false)` would be unsound).

enum class TagEncoding { None, TBI, Immortal };

// The immortal encoding's dummy refcount initializer and recognition
// threshold for a `width`-bit refcount word.
uint64_t immortalRefCount(unsigned width) {
  return (uint64_t(3) << (width - 2)) & llvm::maskTrailingOnes<uint64_t>(width);
}
uint64_t immortalThreshold(unsigned width) {
  return uint64_t(1) << (width - 1);
}

TagEncoding specialPtrTagEncoding(mlir::Operation *op) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return TagEncoding::None;
  auto attr = module->getAttrOfType<mlir::StringAttr>(kSpecialPtrTagAttr);
  if (!attr)
    return TagEncoding::None;
  return attr.getValue() == kSpecialPtrTagImmortal ? TagEncoding::Immortal
                                                   : TagEncoding::TBI;
}

// The per-tag dummy box a tagged immediate points at: `{refcount, tag}`,
// sized to the target's index width. Loads and increments through a tagged
// value land here and observe a well-formed shared box header carrying the
// immediate's own tag; the refcount stays pinned because `rc.set` never
// writes it (see above). The dummy mirrors the fused box header exactly:
// an i32 count and the i32 tag sharing one 8-byte word.
mlir::LLVM::GlobalOp tagDummyBox(mlir::ModuleOp module, mlir::Location loc,
                                 uint64_t tag, TagEncoding encoding,
                                 mlir::OpBuilder &builder) {
  // Content-addressed v0 name, like every other lowering-generated global
  // (namespace + the decimal tag as payload).
  std::string name =
      mangledBlake3Symbol("REUSSIR_TAG_DUMMY", llvm::Twine(tag).str());
  auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name);
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    // The fused box header: {i32 count, i32 tag} in one 8-byte word.
    auto countTy = mlir::IntegerType::get(module.getContext(), 32);
    auto arrTy = mlir::LLVM::LLVMArrayType::get(countTy, 2);
    uint64_t refCount =
        encoding == TagEncoding::Immortal ? immortalRefCount(32) : 2;
    auto init = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({2}, countTy),
        llvm::ArrayRef<llvm::APInt>{llvm::APInt(32, refCount),
                                    llvm::APInt(32, tag)});
    global = mlir::LLVM::GlobalOp::create(
        builder, loc, arrTy,
        /*isConstant=*/false, mlir::LLVM::Linkage::Internal, name, init);
  }
  return global;
}

// `count >= immortalThreshold` — the value under the immortal encoding is a
// dummy box (a real refcount can never reach the threshold).
mlir::Value isImmortalCount(mlir::Value count, mlir::Location loc,
                            mlir::OpBuilder &builder) {
  auto intTy = llvm::cast<mlir::IntegerType>(count.getType());
  auto threshold = mlir::LLVM::ConstantOp::create(
      builder, loc, intTy,
      builder.getIntegerAttr(intTy, immortalThreshold(intTy.getWidth())));
  return mlir::LLVM::ICmpOp::create(
      builder, loc, mlir::LLVM::ICmpPredicate::uge, count, threshold);
}

// The per-module scratch word guarded accesses are steered to.
mlir::Value tagScratchAddr(mlir::ModuleOp module, mlir::Location loc,
                           mlir::OpBuilder &builder) {
  std::string kName = mangledBlake3Symbol("REUSSIR_TAG_SCRATCH", "");
  auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(kName);
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto i64Ty = builder.getI64Type();
    global = mlir::LLVM::GlobalOp::create(
        builder, loc, i64Ty, /*isConstant=*/false,
        mlir::LLVM::Linkage::Internal, kName, builder.getI64IntegerAttr(0));
  }
  return mlir::LLVM::AddressOfOp::create(builder, loc, global);
}

// `ptr`'s top byte as an i64 (zero for any real heap pointer).
mlir::Value topByteOf(mlir::Value ptr, mlir::Location loc,
                      mlir::OpBuilder &builder) {
  auto i64Ty = builder.getI64Type();
  auto raw = mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, ptr);
  auto c56 = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                            builder.getI64IntegerAttr(56));
  return mlir::LLVM::LShrOp::create(builder, loc, raw, c56);
}

// `top != 0` — the value is a tagged immediate.
mlir::Value isTaggedImmediate(mlir::Value topByte, mlir::Location loc,
                              mlir::OpBuilder &builder) {
  auto zero = mlir::LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                             builder.getI64IntegerAttr(0));
  return mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::ne,
                                    topByte, zero);
}

// Steer `addr` to the scratch word when `isTagged` holds.
mlir::Value guardedAddr(mlir::Operation *op, mlir::Value isTagged,
                        mlir::Value addr, mlir::Location loc,
                        mlir::OpBuilder &builder) {
  auto scratch =
      tagScratchAddr(op->getParentOfType<mlir::ModuleOp>(), loc, builder);
  return mlir::LLVM::SelectOp::create(builder, loc, isTagged, scratch, addr);
}

// Skip a guarded refcount store entirely behind an unlikely branch. The
// select-steered form makes the store *address* data-depend on the freshly
// loaded count, so the store — and everything queued behind it — serializes
// against the load on every rc op (measured: IPC halves on nbe-shaped
// traversals, #325). A predicted-not-taken branch turns that data dependency
// into a control dependency the core speculates past, and skipping the store
// outright also means the dummy box is never written: no scratch word, no
// wrap-around accounting on this path.
void emitGuardedStore(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, mlir::Value value, mlir::Value addr,
                      mlir::Value skipCond) {
  mlir::Block *condBlock = rewriter.getInsertionBlock();
  mlir::Block *contBlock =
      rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
  mlir::Block *storeBlock = rewriter.createBlock(
      contBlock->getParent(), mlir::Region::iterator(contBlock));
  rewriter.setInsertionPointToEnd(storeBlock);
  mlir::LLVM::StoreOp::create(rewriter, loc, value, addr);
  mlir::LLVM::BrOp::create(rewriter, loc, contBlock);
  rewriter.setInsertionPointToEnd(condBlock);
  auto condBr = mlir::LLVM::CondBrOp::create(rewriter, loc, skipCond,
                                             contBlock, storeBlock);
  condBr->setAttr("branch_weights", rewriter.getDenseI32ArrayAttr({1, 2000}));
  rewriter.setInsertionPointToStart(contBlock);
}

void ensureRuntimeFunctions(mlir::ModuleOp module,
                            const mlir::LLVMTypeConverter &converter);

bool hasIndexedFieldAttr(mlir::Operation *op, llvm::StringRef attrName,
                         int64_t index) {
  auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(attrName);
  if (!attr)
    return false;
  for (int64_t value : attr.asArrayRef())
    if (value == index)
      return true;
  return false;
}

bool shouldSkipFieldStore(mlir::Operation *op, int64_t index) {
  return hasIndexedFieldAttr(op, "skipFields", index) ||
         hasIndexedFieldAttr(op, "holeFields", index);
}

mlir::DataLayout getDataLayout(const mlir::LLVMTypeConverter &converter,
                               mlir::Operation *op) {
  if (const auto *analysis = converter.getDataLayoutAnalysis())
    return analysis->getAtOrAbove(op);
  return mlir::DataLayout::closest(op);
}

template <typename Op>
void addLifetimeOrInvariantOp(
    mlir::OpBuilder &rewriter, mlir::Location loc,
    [[maybe_unused]] mlir::Type type, mlir::Value value,
    [[maybe_unused]] const mlir::LLVMTypeConverter &converter,
    [[maybe_unused]] mlir::Operation *scopeOp) {
  // The lifetime/invariant intrinsics take no size argument.
  Op::create(rewriter, loc, value);
}

struct ReussirPanicConversionPattern
    : public mlir::OpConversionPattern<ReussirPanicOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirPanicOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto indexType = converter->getIndexType();

    // Get the panic message
    llvm::StringRef message = op.getMessage();

    // Create a stable symbol name for the global string using BLAKE3.
    std::string globalName =
        mangledBlake3Symbol("REUSSIR_PANIC_MESSAGE", message);

    // Get or create the global string constant
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto existingGlobal =
        moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalName);

    if (!existingGlobal) {
      // Create an LLVM array type for the string (no null terminator needed)
      auto i8Type = mlir::IntegerType::get(rewriter.getContext(), 8);
      auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, message.size());

      // Create the string attribute
      auto stringAttr = rewriter.getStringAttr(message);

      // Insert global at module level
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      mlir::LLVM::GlobalOp::create(rewriter, loc, arrayType,
                                   /*isConstant=*/true,
                                   mlir::LLVM::Linkage::LinkonceODR, globalName,
                                   stringAttr);
    }

    // Get address of the global string
    auto strPtr =
        mlir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrType, globalName);

    // Create the length constant
    auto lenVal = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, message.size()));

    // Call __reussir_panic(ptr, len) - this function does not return
    auto panicFunc =
        moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__reussir_panic");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, panicFunc, mlir::ValueRange{strPtr, lenVal});

    return mlir::success();
  }
};

struct ReussirTokenAllocConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenAllocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);

    // Get the token type and extract alignment and size
    TokenType tokenType = op.getToken().getType();
    uint64_t alignment = tokenType.getAlign();
    uint64_t size = tokenType.getSize();
    auto indexType = converter->getIndexType();

    // Create constants for alignment and size
    auto alignConst = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, alignment));
    auto sizeConst = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, size));

    // Create the runtime function call
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto allocFunc =
        moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__reussir_allocate");
    auto funcOp = mlir::LLVM::CallOp::create(
        rewriter, loc, allocFunc, mlir::ValueRange{alignConst, sizeConst});

    // Replace the original operation with the function call result
    rewriter.replaceOp(op, funcOp.getResult());

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
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);

    // Get the token operand (already converted to LLVM pointer)
    mlir::Value tokenPtr = adaptor.getToken();

    // Get the token type and extract alignment and size
    TokenType tokenType = llvm::dyn_cast<TokenType>(op.getToken().getType());
    if (!tokenType) {
      auto nullableType = llvm::dyn_cast<NullableType>(op.getToken().getType());
      if (nullableType)
        tokenType = llvm::dyn_cast<TokenType>(nullableType.getPtrTy());
    }
    if (!tokenType)
      return op.emitOpError(
          "token operand must be of TokenType or NullableTokenType");

    uint64_t alignment = tokenType.getAlign();
    uint64_t size = tokenType.getSize();
    auto indexType = converter->getIndexType();

    // Create constants for alignment and size
    auto alignConst = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, alignment));
    auto sizeConst = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, size));

    // Replace the original operation with the runtime function call
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto deallocFunc =
        moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__reussir_deallocate");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, deallocFunc, mlir::ValueRange{tokenPtr, alignConst, sizeConst});

    return mlir::success();
  }
};

struct ReussirTokenReinterpretConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenReinterpretOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenReinterpretOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For reinterpret, we just use the input token directly since it's
    // already converted to an LLVM pointer by the type converter
    rewriter.replaceOp(op, adaptor.getToken());
    return mlir::success();
  }
};

struct ReussirRcReinterpretConversionPattern
    : public mlir::OpConversionPattern<ReussirRcReinterpretOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcReinterpretOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For reinterpret, we just use the input token directly since it's
    // already converted to an LLVM pointer by the type converter
    rewriter.replaceOp(op, adaptor.getRcPtr());
    return mlir::success();
  }
};

struct ReussirRcFetchConversionPattern
    : public mlir::OpConversionPattern<ReussirRcFetchOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcFetchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto indexTy =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter())
            ->getIndexType();
    mlir::Location loc = op.getLoc();
    // No guard for tagged immediates: TBI masks the top byte, so the load
    // hits the dummy box's refcount, which is pinned >= 2 — exactly what
    // routes the expanded decrement to its shared branch and answers
    // `is_unique` with false (see the special-pointer-tag notes above).
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        rewriter, loc, rewriter.getI32Type(), adaptor.getRcPtr());
    mlir::Value widened =
        llvm::cast<mlir::IntegerType>(indexTy).getWidth() > 32
            ? mlir::LLVM::ZExtOp::create(rewriter, loc, indexTy, loaded)
                  .getResult()
            : loaded;
    rewriter.replaceOp(op, widened);
    return mlir::success();
  }
};

struct ReussirRcSetConversionPattern
    : public mlir::OpConversionPattern<ReussirRcSetOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcSetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value addr = adaptor.getRcPtr();
    mlir::Value narrowCount =
        adaptor.getRefCount().getType().getIntOrFloatBitWidth() > 32
            ? mlir::LLVM::TruncOp::create(rewriter, op.getLoc(),
                                          rewriter.getI32Type(),
                                          adaptor.getRefCount())
                  .getResult()
            : adaptor.getRefCount();
    // `rc.set` is the one store that can lower a refcount, so it must not
    // reach the dummy box (its count would eventually decay to 1 and send an
    // immediate down the unique branch). Skip a recognized-immediate store
    // behind an unlikely branch: the dummy's count is then never written at
    // all, and the store issues speculatively on the (overwhelmingly common)
    // real-box path instead of address-depending on the loaded count. TBI
    // recognizes immediates by the pointer's top byte, the immortal encoding
    // by the magnitude of the count being stored (the dummy's decremented
    // count still clears the threshold: it starts at 3 * 2^(w-2) and never
    // actually decreases, since this very gating keeps every decrementing
    // store away from it).
    TagEncoding encoding = specialPtrTagEncoding(op);
    if (encoding != TagEncoding::None &&
        op.getRcPtr().getType().mayCarrySpecialPointerTag()) {
      mlir::Value tagged;
      if (encoding == TagEncoding::TBI) {
        mlir::Value top = topByteOf(addr, op.getLoc(), rewriter);
        tagged = isTaggedImmediate(top, op.getLoc(), rewriter);
      } else {
        tagged = isImmortalCount(narrowCount, op.getLoc(), rewriter);
      }
      emitGuardedStore(rewriter, op.getLoc(), narrowCount, addr, tagged);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, narrowCount, addr);
    return mlir::success();
  }
};

struct ReussirExpectConversionPattern
    : public mlir::OpConversionPattern<ReussirExpectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirExpectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto expected = mlir::arith::ConstantIntOp::create(
        rewriter, op.getLoc(), static_cast<int64_t>(op.getExpected()), 1);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
        op, adaptor.getCondition().getType(),
        rewriter.getStringAttr("llvm.expect"),
        mlir::ValueRange{adaptor.getCondition(), expected.getResult()});
    return mlir::success();
  }
};

struct ReussirTokenReallocConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenReallocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenReallocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
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
    auto indexType = converter->getIndexType();
    mlir::Value oldAlignVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(indexType, oldAlign));
    mlir::Value oldSizeVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(indexType, oldSize));
    mlir::Value newAlignVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(indexType, newAlign));
    mlir::Value newSizeVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(indexType, newSize));
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto reallocFunc =
        moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>("__reussir_reallocate");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, reallocFunc,
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

//===----------------------------------------------------------------------===//
// Constructor contexts lower to a {root, hole} pointer pair with a null hole
// denoting the empty context. Extend and apply are compiled branch-free: the
// store that plugs the hole is steered to the per-module scratch word when
// the context is empty (the value lands in the root/result select instead),
// mirroring the special-pointer-tag scratch steering.
//===----------------------------------------------------------------------===//

struct ReussirCctxEmptyConversionPattern
    : public mlir::OpConversionPattern<ReussirCctxEmptyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCctxEmptyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto structTy = getTypeConverter()->convertType(op.getCtx().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, structTy);
    return mlir::success();
  }
};

struct ReussirCctxExtendConversionPattern
    : public mlir::OpConversionPattern<ReussirCctxExtendOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCctxExtendOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto structTy = getTypeConverter()->convertType(op.getNewCtx().getType());

    mlir::Value root = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getCtx(), llvm::ArrayRef<int64_t>{0});
    mlir::Value hole = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getCtx(), llvm::ArrayRef<int64_t>{1});
    mlir::Value null = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    mlir::Value isEmpty = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, hole, null);
    mlir::Value scratch = tagScratchAddr(
        op->getParentOfType<mlir::ModuleOp>(), loc, rewriter);
    mlir::Value addr =
        mlir::LLVM::SelectOp::create(rewriter, loc, isEmpty, scratch, hole);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getRcPtr(), addr);
    mlir::Value newRoot = mlir::LLVM::SelectOp::create(
        rewriter, loc, isEmpty, adaptor.getRcPtr(), root);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, structTy);
    result = mlir::LLVM::InsertValueOp::create(rewriter, loc, result, newRoot,
                                               llvm::ArrayRef<int64_t>{0});
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, result, adaptor.getHole(), llvm::ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReussirCctxApplyConversionPattern
    : public mlir::OpConversionPattern<ReussirCctxApplyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirCctxApplyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    mlir::Value root = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getCtx(), llvm::ArrayRef<int64_t>{0});
    mlir::Value hole = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getCtx(), llvm::ArrayRef<int64_t>{1});
    mlir::Value null = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    mlir::Value isEmpty = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, hole, null);
    mlir::Value scratch = tagScratchAddr(
        op->getParentOfType<mlir::ModuleOp>(), loc, rewriter);
    mlir::Value addr =
        mlir::LLVM::SelectOp::create(rewriter, loc, isEmpty, scratch, hole);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getValue(), addr);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
        op, isEmpty, adaptor.getValue(), root);
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

    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto dataLayout = getDataLayout(*converter, op.getOperation());

    auto valueType = converter->convertType(op.getValue().getType());
    auto llvmPtrType = converter->convertType(op.getSpilled().getType());
    auto alignment = dataLayout.getTypePreferredAlignment(valueType);

    // Allocate stack space using llvm.alloca
    auto convertedIndexType = converter->getIndexType();
    auto constantArraySize = mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(convertedIndexType, 1));
    auto allocaOp = mlir::LLVM::AllocaOp::create(
        rewriter, loc, llvmPtrType, valueType, constantArraySize, alignment);

    // Store the value to the allocated space
    mlir::LLVM::StoreOp::create(rewriter, loc, value, allocaOp);
    mlir::LLVM::InvariantStartOp::create(
        rewriter, loc, dataLayout.getTypeABIAlignment(valueType), allocaOp);
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
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the record type and convert it to LLVM struct type
    RecordType recordType = op.getCompound().getType();
    mlir::Type llvmStructType = converter->convertType(recordType);

    if (!llvmStructType)
      return op.emitOpError("failed to convert record type to LLVM type");

    mlir::MLIRContext *ctx = rewriter.getContext();

    // Create an opaque pointer type for the alloca and GEP operations
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);

    // Create a constant '1' for the alloca array size
    mlir::Value one = mlir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    // Allocate the struct on the stack
    mlir::Value alloca = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                                      llvmStructType, one);
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeStartOp>(
        rewriter, loc, llvmStructType, alloca, *converter, op.getOperation());

    // Get the field values (already converted by the type converter)
    auto fieldValues = adaptor.getFields();

    // Insert each field using GEP + store
    auto dataLayout = getDataLayout(*converter, op.getOperation());
    for (size_t i = 0; i < fieldValues.size(); ++i) {
      // GEP indices:
      // 0 -> Dereference the base pointer (step into the allocated element)
      // then the packed physical position of the i-th logical member
      llvm::SmallVector<mlir::LLVM::GEPArg, 2> gepArgs{
          0, static_cast<int32_t>(recordType.getPhysicalMemberIndex(
                 dataLayout, static_cast<uint32_t>(i)))};

      mlir::Value fieldPtr = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, llvmStructType, alloca, gepArgs);

      mlir::LLVM::StoreOp::create(rewriter, loc, fieldValues[i], fieldPtr);
    }

    // Load the final populated struct value
    mlir::Value result =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmStructType, alloca);
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeEndOp>(
        rewriter, loc, llvmStructType, alloca, *converter, op.getOperation());

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ReussirRecordExtractConversionPattern
    : public mlir::OpConversionPattern<ReussirRecordExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRecordExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the LLVM types for the struct and the extracted field
    mlir::Type llvmStructType = adaptor.getRecord().getType();
    mlir::Type llvmResultType =
        converter->convertType(op.getResult().getType());

    if (!llvmResultType)
      return op.emitOpError("failed to convert result type to LLVM type");

    // Create an opaque pointer type for memory operations
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);

    // Constant '1' for the alloca array size
    mlir::Value one = mlir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    // 1. Spill: Allocate memory for the struct on the stack
    mlir::Value alloca = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                                      llvmStructType, one);

    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeStartOp>(
        rewriter, loc, llvmStructType, alloca, *converter, op.getOperation());

    // 2. Store the entire struct value into the allocated memory
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getRecord(), alloca);

    // 3. GEP: Calculate the memory address of the specific field, mapping
    // the logical member index to its packed physical position.
    int32_t fieldIndex = static_cast<int32_t>(op.getIndex().getZExtValue());
    if (auto recordType =
            llvm::dyn_cast<RecordType>(op.getRecord().getType());
        recordType && recordType.isCompound()) {
      auto dataLayout = getDataLayout(*converter, op.getOperation());
      fieldIndex = static_cast<int32_t>(recordType.getPhysicalMemberIndex(
          dataLayout, static_cast<uint32_t>(fieldIndex)));
    }
    llvm::SmallVector<mlir::LLVM::GEPArg, 2> gepArgs{0, fieldIndex};

    mlir::Value fieldPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrType, llvmStructType, alloca, gepArgs);

    // 4. Load: Read the individual field value back into an SSA register
    mlir::Value result =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmResultType, fieldPtr);

    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeEndOp>(
        rewriter, loc, llvmStructType, alloca, *converter, op.getOperation());

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
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the record type and convert it to LLVM struct type
    RecordType recordType = op.getVariant().getType();
    auto llvmStructType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
        converter->convertType(recordType));

    if (!llvmStructType)
      return op.emitOpError("failed to convert record type to LLVM type");
    auto indexType = converter->getIndexType();
    // Get the tag (at the record's minimal tag width) and value (already
    // converted by the type converter)
    mlir::Value tag = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::IntegerAttr::get(recordType.getTagType(),
                               op.getTag().getZExtValue()));
    mlir::Value value = adaptor.getValue();

    // Get the preferred alignment for the struct type
    auto dataLayout = getDataLayout(*converter, op.getOperation());
    auto alignment = dataLayout.getTypePreferredAlignment(llvmStructType);
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto one = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, 1));
    // Allocate stack space for the struct
    auto allocaOp = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrType, llvmStructType, one, alignment);
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeStartOp>(
        rewriter, loc, llvmStructType, allocaOp, *converter, op.getOperation());
    // Store the tag (past the count slot for fused-header variants).
    bool fused = recordType.hasFusedHeader();
    int32_t tagField = fused ? 1 : 0;
    int32_t valueField = fused ? 2 : 1;
    auto tagPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrType, llvmStructType, allocaOp,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, tagField});
    mlir::LLVM::StoreOp::create(rewriter, loc, tag, tagPtr);

    // Store the payload value when the variant has one.
    if (llvmStructType.getSubelementIndexMap()->size() >
        static_cast<size_t>(valueField)) {
      auto valuePtr = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, llvmStructType, allocaOp,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, valueField});
      mlir::LLVM::StoreOp::create(rewriter, loc, value, valuePtr);
    }
    // Load the complete struct from the allocated space
    auto result =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmStructType, allocaOp);
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeEndOp>(
        rewriter, loc, llvmStructType, allocaOp, *converter, op.getOperation());
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
    auto loadOp = rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, llvmPointeeTy, adaptor.getRef());
    if (op.getInvariantGroup())
      loadOp.setInvariantGroup(true);
    return mlir::success();
  }
};

struct ReussirRefToMemrefConversionPattern
    : public mlir::OpConversionPattern<ReussirRefToMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefToMemrefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto viewType = llvm::dyn_cast<mlir::MemRefType>(op.getView().getType());
    if (!viewType)
      return op.emitOpError("view result must be a memref");
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, loc, *converter, viewType, adaptor.getRef(),
        adaptor.getRef());
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct ReussirRefFromMemrefConversionPattern
    : public mlir::OpConversionPattern<ReussirRefFromMemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefFromMemrefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    mlir::Type resultType =
        getTypeConverter()->convertType(op.getRef().getType());
    auto llvmPtrType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(resultType);
    if (!llvmPtrType)
      return op.emitOpError("ref result must lower to an LLVM pointer");

    auto viewType = llvm::dyn_cast<mlir::MemRefType>(op.getView().getType());
    if (!viewType)
      return op.emitOpError("view input must be a memref");

    mlir::Value ptr = mlir::LLVM::getStridedElementPtr(
        rewriter, loc, *converter, viewType, adaptor.getView(), {});
    if (ptr.getType() != llvmPtrType)
      ptr = mlir::LLVM::BitcastOp::create(rewriter, loc, llvmPtrType, ptr);
    rewriter.replaceOp(op, ptr);
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
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the reference pointer (already converted by the type converter)
    mlir::Value refPtr = adaptor.getRef();
    // Get the result type (should be a pointer type after conversion)
    mlir::Type resultType = converter->convertType(op.getProjected().getType());
    auto llvmPtrType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(resultType);
    if (!llvmPtrType)
      return op.emitOpError("projected result must be an LLVM pointer type");

    // Get the element type that the reference points to
    RefType refType = op.getRef().getType();
    mlir::Type elementType = converter->convertType(refType.getElementType());

    // Logical member indices map to packed physical struct fields.
    int32_t fieldIndex = static_cast<int32_t>(op.getIndex().getZExtValue());
    if (auto recordType =
            llvm::dyn_cast<RecordType>(refType.getElementType());
        recordType && recordType.isCompound()) {
      auto dataLayout = getDataLayout(*converter, op.getOperation());
      fieldIndex = static_cast<int32_t>(recordType.getPhysicalMemberIndex(
          dataLayout, static_cast<uint32_t>(fieldIndex)));
    }

    // Create GEP operation to get the field pointer
    auto gepOp = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, elementType, refPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, fieldIndex});

    rewriter.replaceOp(op, gepOp);
    return mlir::success();
  }
};

struct ReussirArrayProjectConversionPattern
    : public mlir::OpConversionPattern<ReussirArrayProjectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirArrayProjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto viewType = llvm::cast<mlir::MemRefType>(op.getView().getType());
    auto extent = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::IntegerAttr::get(converter->getIndexType(),
                               viewType.getShape().front()));
    auto inBounds = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::ult, adaptor.getIndex(),
        extent.getResult());
    mlir::LLVM::AssumeOp::create(rewriter, loc, inBounds);

    if (viewType.getRank() == 1) {
      mlir::Type resultType =
          converter->convertType(op.getProjected().getType());
      auto llvmPtrType =
          llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(resultType);
      if (!llvmPtrType)
        return op.emitOpError("projected result must lower to an LLVM pointer");

      auto elementPtr = mlir::LLVM::getStridedElementPtr(
          rewriter, loc, *converter, viewType, adaptor.getView(),
          mlir::ValueRange{adaptor.getIndex()});
      if (elementPtr.getType() != llvmPtrType)
        elementPtr = mlir::LLVM::BitcastOp::create(rewriter, loc, llvmPtrType,
                                                   elementPtr);
      rewriter.replaceOp(op, elementPtr);
      return mlir::success();
    }

    auto resultMemRefType =
        llvm::cast<mlir::MemRefType>(op.getProjected().getType());
    mlir::Type resultType = converter->convertType(resultMemRefType);
    mlir::MemRefDescriptor srcDesc(adaptor.getView());
    auto resultDesc = mlir::MemRefDescriptor::poison(rewriter, loc, resultType);
    resultDesc.setAllocatedPtr(rewriter, loc,
                               srcDesc.allocatedPtr(rewriter, loc));
    resultDesc.setAlignedPtr(rewriter, loc, srcDesc.alignedPtr(rewriter, loc));

    auto offset = srcDesc.offset(rewriter, loc);
    auto stride0 = srcDesc.stride(rewriter, loc, 0);
    auto delta =
        mlir::arith::MulIOp::create(rewriter, loc, adaptor.getIndex(), stride0);
    auto newOffset = mlir::arith::AddIOp::create(rewriter, loc, offset, delta);
    resultDesc.setOffset(rewriter, loc, newOffset);

    for (int64_t i = 0, e = resultMemRefType.getRank(); i < e; ++i) {
      resultDesc.setSize(rewriter, loc, i, srcDesc.size(rewriter, loc, i + 1));
      resultDesc.setStride(rewriter, loc, i,
                           srcDesc.stride(rewriter, loc, i + 1));
    }

    rewriter.replaceOp(op, mlir::Value(resultDesc));
    return mlir::success();
  }
};

struct ReussirArrayViewConversionPattern
    : public mlir::OpConversionPattern<ReussirArrayViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirArrayViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto viewType = llvm::dyn_cast<mlir::MemRefType>(op.getView().getType());
    if (!viewType)
      return op.emitOpError(
          "tensor array.view must be bufferized before lowering basic ops");
    ArrayType arrayType = llvm::cast<ArrayType>(
        llvm::cast<RefType>(op.getRef().getType()).getElementType());
    mlir::Type llvmArrayType = converter->convertType(arrayType);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    llvm::SmallVector<mlir::LLVM::GEPArg> zeroIndices(arrayType.getRank() + 1,
                                                      0);
    auto elementPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, llvmPtrType, llvmArrayType,
                                  adaptor.getRef(), zeroIndices);
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, loc, *converter, viewType, elementPtr, elementPtr);
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct ReussirRecordTagConversionPattern
    : public mlir::OpConversionPattern<ReussirRecordTagOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRecordTagOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the reference pointer (already converted by the type converter)
    mlir::Value refPtr = adaptor.getVariant();

    // Get the element type that the reference points to
    RefType refType = op.getVariant().getType();
    RecordType recordType =
        llvm::dyn_cast<RecordType>(refType.getElementType());
    if (!recordType)
      return op.emitOpError("expected a record type");
    mlir::Type elementType = converter->convertType(recordType);

    // Get the index type for the result
    auto indexType = converter->getIndexType();

    // The discriminant field: past the count slot for fused-header
    // variants, first field otherwise.
    int32_t tagField = recordType.hasFusedHeader() ? 1 : 0;
    auto tagPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto tagPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, tagPtrType, elementType, refPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, tagField});

    // Load the tag value at the record's minimal tag width and widen it to
    // the op's index result. Under the special-pointer-tag scheme a tagged
    // immediate scrutinee needs no decode: the load lands in its per-tag
    // dummy box (TBI masks the top byte) and reads the real tag — so the
    // match hot path is identical with and without the scheme.
    mlir::IntegerType tagType = recordType.getTagType();
    mlir::Value narrowTag =
        mlir::LLVM::LoadOp::create(rewriter, loc, tagType, tagPtr);
    mlir::Value tagValue =
        tagType.getWidth() <
                llvm::cast<mlir::IntegerType>(indexType).getWidth()
            ? mlir::LLVM::ZExtOp::create(rewriter, loc, indexType, narrowTag)
                  .getResult()
            : narrowTag;
    rewriter.replaceOp(op, tagValue);

    // Assume that the tag is always in bounds
    auto numberMembers = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::IntegerAttr::get(indexType, recordType.getMembers().size()));
    auto tagInRange = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::ult, tagValue,
        numberMembers);
    mlir::LLVM::AssumeOp::create(rewriter, loc, tagInRange);
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
        mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), llvmPtrType);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::ne, nullable, nullConstant);
    return mlir::success();
  }
};

struct ReussirNullableCreateConversionPattern
    : public mlir::OpConversionPattern<ReussirNullableCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if the operation has input. If so, replace it directly with
    // adaptor value. Otherwise, create a new null value.
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
      auto converter =
          static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
      ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      auto acquireFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "__reussir_acquire_rigid_object");
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, acquireFunc, mlir::ValueRange{adaptor.getRcPtr()});
      return mlir::success();
    }
    mlir::Value refcntPtr;
    // If inner element type is a FFI object, we do not reuse gep to expose
    // the struct.
    if (auto eleTy = mlir::dyn_cast<FFIObjectType>(rcPtrTy.getElementType())) {
      refcntPtr = adaptor.getRcPtr();
    } else {
      RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
      // GEP [0].1
      auto convertedBoxType = getTypeConverter()->convertType(rcBoxType);
      auto llvmPtrType =
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      refcntPtr = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), llvmPtrType, convertedBoxType,
          adaptor.getRcPtr(), llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    }
    // Usually no guard for tagged immediates: the increment lands on the
    // dummy box's refcount — a benign write that only ever grows it (rc.set,
    // the sole decrementing store, is steered away). The exception is the
    // immortal encoding on refcount words narrower than 64 bits, where the
    // dummy could actually be grown past wrap-around within a program's
    // lifetime: there the increment's store is steered to the scratch word
    // instead (the load stays plain, so nothing on the read path changes).
    auto countType = rewriter.getI32Type();
    TagEncoding encoding = specialPtrTagEncoding(op);
    bool steerNarrowImmortal = encoding == TagEncoding::Immortal &&
                               rcPtrTy.mayCarrySpecialPointerTag();
    auto one = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(countType, 1));
    mlir::Value oldRefCnt;
    if (rcPtrTy.getAtomicKind() == AtomicKind::normal) {
      oldRefCnt = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), countType,
                                             refcntPtr);
      auto newRefCnt = mlir::arith::AddIOp::create(rewriter, op.getLoc(),
                                                   countType, oldRefCnt, one);
      if (steerNarrowImmortal) {
        // Same rationale as `rc.set`: skip the dummy's store behind an
        // unlikely branch rather than address-selecting it into the scratch
        // word, so the store doesn't data-depend on the loaded count.
        mlir::Value immortal =
            isImmortalCount(oldRefCnt, op.getLoc(), rewriter);
        emitGuardedStore(rewriter, op.getLoc(), newRefCnt, refcntPtr,
                         immortal);
      } else {
        mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), newRefCnt,
                                    refcntPtr);
      }
    } else {
      oldRefCnt = mlir::LLVM::AtomicRMWOp::create(
          rewriter, op.getLoc(), mlir::LLVM::AtomicBinOp::add, refcntPtr, one,
          mlir::LLVM::AtomicOrdering::monotonic);
    }
    // Valid for immediates too: the dummy box's count starts at 2 and only
    // ever grows, so `old >= 1` holds on that path as well.
    mlir::Value geOne = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::uge, oldRefCnt, one);
    mlir::LLVM::AssumeOp::create(rewriter, op.getLoc(), geOne);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

namespace {
struct RcCreateStorageInfo {
  mlir::Value token;
  mlir::Value elementPtr;
  bool regional;
  // Non-null when the pattern must store the refcount AFTER initializing the
  // element: with a fused header a whole-value store would clobber the
  // count slot it overlays.
  mlir::Value countPtr;
};

// Stores `1` (i32) to a box's refcount slot.
void storeInitialRcCount(mlir::Value countPtr, mlir::Location loc,
                         mlir::ConversionPatternRewriter &rewriter) {
  auto one = mlir::arith::ConstantOp::create(
      rewriter, loc, mlir::IntegerAttr::get(rewriter.getI32Type(), 1));
  mlir::LLVM::StoreOp::create(rewriter, loc, one, countPtr);
}

template <typename OpT, typename AdaptorT>
mlir::FailureOr<RcCreateStorageInfo>
initializeRcCreateStorage(OpT op, AdaptorT adaptor,
                          const mlir::TypeConverter *typeConverter,
                          mlir::ConversionPatternRewriter &rewriter) {
  RcType rcPtrTy = op.getRcPtr().getType();
  RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
  if (rcPtrTy.getAtomicKind() == AtomicKind::atomic)
    return op->emitError("TODO: atomic rc create"), mlir::failure();

  auto convertedBoxType = typeConverter->convertType(rcBoxType);
  auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto token = adaptor.getToken();

  if (!rcBoxType.isRegional()) {
    mlir::Value countPtr;
    if (!op.getSkipRc())
      countPtr = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    auto elementPtr = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            0, static_cast<int32_t>(rcBoxType.getElementIndex())});
    return RcCreateStorageInfo{token, elementPtr, false, countPtr};
  }

  if (!op.getToken())
    return op->emitError("token is required but not provided"), mlir::failure();

  auto regionPtr = adaptor.getRegion();
  auto tailPtr =
      mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), llvmPtrType, regionPtr);
  auto null = mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), llvmPtrType);

  mlir::Value vtable = null.getRes();
  if (op.needsVTable()) {
    if (!op.getVtable())
      return op->emitError("vtable is required but not provided"),
             mlir::failure();
    vtable = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(), llvmPtrType,
                                             *op.getVtable());
  }

  auto statePtr = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
  auto nextPtr = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
  auto vtablePtr = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  auto elementPtr = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), llvmPtrType, convertedBoxType, token,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
  mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), null, statePtr);
  mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), tailPtr, nextPtr);
  auto vtableStore =
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), vtable, vtablePtr);
  vtableStore.setInvariantGroup(true);
  mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), token, regionPtr);
  return RcCreateStorageInfo{token, elementPtr, true, /*countPtr=*/{}};
}
} // namespace

struct ReussirRcCreateOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto storage =
        initializeRcCreateStorage(op, adaptor, getTypeConverter(), rewriter);
    if (mlir::failed(storage))
      return mlir::failure();
    auto objectStore = mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), storage->elementPtr);
    if (!storage->regional)
      objectStore.setInvariantGroup(true);
    if (storage->countPtr)
      storeInitialRcCount(storage->countPtr, op.getLoc(), rewriter);
    rewriter.replaceOp(op, storage->token);
    return mlir::success();
  }
};

struct ReussirRcCreateCompoundOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcCreateCompoundOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateCompoundOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto storage =
        initializeRcCreateStorage(op, adaptor, getTypeConverter(), rewriter);
    if (mlir::failed(storage))
      return mlir::failure();

    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto llvmRecordType = converter->convertType(op.getRecordType());
    auto dataLayout = getDataLayout(*converter, op.getOperation());
    llvm::SmallVector<mlir::Value> results;
    results.push_back(storage->token);
    for (auto [index, field] : llvm::enumerate(adaptor.getFields())) {
      bool isHoleField = hasIndexedFieldAttr(op.getOperation(), "holeFields",
                                             static_cast<int64_t>(index));
      bool skipStore =
          isHoleField ||
          shouldSkipFieldStore(op.getOperation(), static_cast<int64_t>(index));
      if (skipStore && !isHoleField)
        continue;
      auto fieldPtr = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), llvmPtrType, llvmRecordType,
          storage->elementPtr,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{
              0, static_cast<int32_t>(op.getRecordType().getPhysicalMemberIndex(
                     dataLayout, static_cast<uint32_t>(index)))});
      if (isHoleField)
        results.push_back(fieldPtr);
      if (skipStore)
        continue;
      auto store =
          mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), field, fieldPtr);
      store.setInvariantGroup(true);
    }
    if (storage->countPtr)
      storeInitialRcCount(storage->countPtr, op.getLoc(), rewriter);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ReussirRcTaggedConversionPattern
    : public mlir::OpConversionPattern<ReussirRcTaggedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcTaggedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // The immediate is the per-tag dummy box's address; refcount,
    // uniqueness, and tag accesses through the value hit the dummy box
    // unguarded and observe a well-formed shared header. The TBI encoding
    // additionally sets the top byte to `tag + 1` (masked by the hardware
    // on data accesses) so foreign code and future inhabited-variant
    // tagging can decode from the pointer; the immortal encoding uses the
    // plain address.
    mlir::Location loc = op.getLoc();
    auto indexTy = llvm::cast<mlir::IntegerType>(
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter())
            ->getIndexType());
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    TagEncoding encoding = specialPtrTagEncoding(op);
    auto dummy =
        tagDummyBox(op->getParentOfType<mlir::ModuleOp>(), loc,
                    op.getTag().getZExtValue(), encoding, rewriter);
    mlir::Value base = mlir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy,
                                                       dummy.getSymName());
    if (encoding != TagEncoding::TBI) {
      rewriter.replaceOp(op, base);
      return mlir::success();
    }
    // The TBI encoding manipulates a 64-bit pointer's top byte; it is
    // meaningless on narrower targets (use the immortal encoding there).
    if (indexTy.getWidth() != 64)
      return op.emitOpError(
          "the TBI encoding requires a 64-bit target; use the "
          "arch-independent encoding instead");
    mlir::Value baseInt =
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, indexTy, base);
    uint64_t encoded = (op.getTag().getZExtValue() + 1) << 56;
    mlir::Value tagBits = mlir::LLVM::ConstantOp::create(
        rewriter, loc, indexTy,
        rewriter.getIntegerAttr(indexTy, static_cast<int64_t>(encoded)));
    mlir::Value imm = mlir::LLVM::OrOp::create(rewriter, loc, baseInt, tagBits);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(op, ptrTy, imm);
    return mlir::success();
  }
};

struct ReussirRcCreateVariantOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcCreateVariantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcCreateVariantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto storage =
        initializeRcCreateStorage(op, adaptor, getTypeConverter(), rewriter);
    if (mlir::failed(storage))
      return mlir::failure();

    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto llvmVariantType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
        converter->convertType(op.getRecordType()));
    if (!llvmVariantType)
      return op.emitOpError("failed to convert record type to LLVM type");

    bool fused = op.getRecordType().hasFusedHeader();
    int32_t tagField = fused ? 1 : 0;
    int32_t payloadField = fused ? 2 : 1;
    auto tag = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(op.getRecordType().getTagType(),
                               op.getTag().getZExtValue()));
    auto tagPtr = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(), llvmPtrType, llvmVariantType,
        storage->elementPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, tagField});
    auto tagStore =
        mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), tag, tagPtr);
    tagStore.setInvariantGroup(true);

    llvm::SmallVector<mlir::Value> results;
    results.push_back(storage->token);
    if (llvmVariantType.getSubelementIndexMap()->size() >
        static_cast<size_t>(payloadField)) {
      auto payloadPtr = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), llvmPtrType, llvmVariantType,
          storage->elementPtr,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, payloadField});
      if (op.getValue()) {
        auto payloadStore = mlir::LLVM::StoreOp::create(
            rewriter, op.getLoc(), adaptor.getValue(), payloadPtr);
        payloadStore.setInvariantGroup(true);
      } else {
        auto payloadType = llvm::cast<RecordType>(
            op.getRecordType().getMembers()[op.getTag().getZExtValue()]);
        auto llvmPayloadType = converter->convertType(payloadType);
        auto dataLayout = getDataLayout(*converter, op.getOperation());
        for (auto [index, field] : llvm::enumerate(adaptor.getFields())) {
          bool isHoleField = hasIndexedFieldAttr(
              op.getOperation(), "holeFields", static_cast<int64_t>(index));
          bool skipStore =
              isHoleField || shouldSkipFieldStore(op.getOperation(),
                                                  static_cast<int64_t>(index));
          if (skipStore && !isHoleField)
            continue;
          auto fieldPtr = mlir::LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, llvmPayloadType, payloadPtr,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{
                  0, static_cast<int32_t>(payloadType.getPhysicalMemberIndex(
                         dataLayout, static_cast<uint32_t>(index)))});
          if (isHoleField)
            results.push_back(fieldPtr);
          if (skipStore)
            continue;
          auto store = mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), field,
                                                   fieldPtr);
          store.setInvariantGroup(true);
        }
      }
    }

    if (storage->countPtr)
      storeInitialRcCount(storage->countPtr, op.getLoc(), rewriter);
    rewriter.replaceOp(op, results);
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
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            0, static_cast<int>(rcBoxType.getElementIndex())});
    return mlir::success();
  }
};

struct ReussirNullableCoerceConversionPattern
    : public mlir::OpConversionPattern<ReussirNullableCoerceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirNullableCoerceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For nullable coerce, we just use the input nullable directly since both
    // nullable and non-nullable pointers are LLVM pointers after conversion
    rewriter.replaceOp(op, adaptor.getNullable());
    return mlir::success();
  }
};

struct ReussirRecordCoerceConversionPattern
    : public mlir::OpConversionPattern<ReussirRecordCoerceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRecordCoerceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto dataLayout = mlir::DataLayout::closest(op.getOperation());

    // Get the variant reference pointer (already converted by the type
    // converter)
    mlir::Value variantPtr = adaptor.getVariant();

    // Get the element type that the reference points to (the variant record)
    RefType refType = op.getVariant().getType();
    auto elementType = llvm::cast<mlir::LLVM::LLVMStructType>(
        converter->convertType(refType.getElementType()));
    auto variantRecord =
        llvm::cast<RecordType>(refType.getElementType());
    int32_t payloadField = variantRecord.hasFusedHeader() ? 2 : 1;
    bool variantHasNonZeroChild =
        elementType.getSubelementIndexMap()->size() >
        static_cast<size_t>(payloadField);

    // Get the result type (should be a pointer type after conversion)
    mlir::Type resultType = converter->convertType(op.getCoerced().getType());

    // The payload field follows the header (count slot + tag for fused
    // variants, just the tag otherwise).
    if (variantHasNonZeroChild)
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
          op, resultType, elementType, variantPtr,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, payloadField});
    else
      rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, resultType);
    return mlir::success();
  }
};

/*
Create a global constant
pub struct VTable {
    pub drop: Option<unsafe extern "C" fn(*mut u8)>,
    pub scan_instrs: *const PackedInstr,
    pub size: usize,
    pub alignment: usize,
}
*/
struct ReussirRegionVTableOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRegionVTableOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRegionVTableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::LLVM::LLVMPointerType llvmPtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    mlir::Type indexType = converter->getIndexType();
    mlir::LLVM::LLVMStructType vtableType =
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {llvmPtrType, llvmPtrType, indexType, indexType});
    std::string arrayName =
        (llvm::Twine("_RNv") + op.getSymName().ltrim("_R") + "11scan_instrs")
            .str();
    reussir::RecordType recordType =
        llvm::dyn_cast<reussir::RecordType>(op.getType());
    auto dataLayout = mlir::DataLayout::closest(op.getOperation());

    mlir::LLVM::GlobalOp scannerOp;
    if (recordType) {
      llvm::SmallVector<int32_t> buffer;
      recordType.emitScannerInstructions(buffer, dataLayout, {});
      mlir::LLVM::LLVMArrayType arrayType = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(rewriter.getContext(), 32), buffer.size());
      auto dataAttr = rewriter.getI32TensorAttr(buffer);
      scannerOp = mlir::LLVM::GlobalOp::create(
          rewriter, op.getLoc(), arrayType, /*isConstant=*/true,
          mlir::LLVM::Linkage::LinkonceODR, arrayName, dataAttr);
    }
    mlir::LLVM::GlobalOp vtableOp = mlir::LLVM::GlobalOp::create(
        rewriter, op.getLoc(), vtableType, /*isConstant=*/true,
        mlir::LLVM::Linkage::LinkonceODR, op.getSymName(), nullptr);
    mlir::Block *initBlock =
        rewriter.createBlock(&vtableOp.getInitializerRegion());
    rewriter.setInsertionPointToEnd(initBlock);
    mlir::Value scannerPtr, dropPtr;
    if (scannerOp) {
      scannerPtr = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(),
                                                   llvmPtrType, arrayName);
    } else {
      scannerPtr =
          mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), llvmPtrType);
    }
    if (op.getDrop()) {
      dropPtr = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(),
                                                llvmPtrType, *op.getDrop());
    } else {
      dropPtr = mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), llvmPtrType);
    }
    auto sizeVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(indexType,
                               dataLayout.getTypeSize(op.getType())));
    auto alignVal = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(indexType,
                               dataLayout.getTypeABIAlignment(op.getType())));
    auto undef = mlir::LLVM::UndefOp::create(rewriter, op.getLoc(), vtableType);
    auto withDrop = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), undef, dropPtr, llvm::ArrayRef<int64_t>{0});
    auto withScanner = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), withDrop, scannerPtr,
        llvm::ArrayRef<int64_t>{1});
    auto withSize =
        mlir::LLVM::InsertValueOp::create(rewriter, op.getLoc(), withScanner,
                                          sizeVal, llvm::ArrayRef<int64_t>{2});
    auto withAlign = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), withSize, alignVal, llvm::ArrayRef<int64_t>{3});
    mlir::LLVM::ReturnOp::create(rewriter, op.getLoc(), withAlign);
    rewriter.replaceOp(op, vtableOp);
    return mlir::success();
  }
};

struct ReussirClosureVtableOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureVtableOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureVtableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::LLVM::LLVMPointerType llvmPtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    // Create LLVM struct type for vtable: { void*, void*, void* }
    // representing { drop, clone, evaluate } function pointers
    mlir::LLVM::LLVMStructType vtableType =
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {llvmPtrType, llvmPtrType, llvmPtrType});

    // Create the global vtable
    mlir::LLVM::GlobalOp vtableOp = mlir::LLVM::GlobalOp::create(
        rewriter, op.getLoc(), vtableType, /*isConstant=*/true,
        mlir::LLVM::Linkage::Internal, op.getSymName(), nullptr);

    // Create initializer block
    mlir::Block *initBlock =
        rewriter.createBlock(&vtableOp.getInitializerRegion());
    rewriter.setInsertionPointToEnd(initBlock);

    // Get addresses of the three functions
    auto dropPtr = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(),
                                                   llvmPtrType, op.getDrop());
    auto clonePtr = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(),
                                                    llvmPtrType, op.getClone());
    auto funcPtr = mlir::LLVM::AddressOfOp::create(rewriter, op.getLoc(),
                                                   llvmPtrType, op.getFunc());

    // Build the struct { drop, clone, evaluate }
    auto undef = mlir::LLVM::UndefOp::create(rewriter, op.getLoc(), vtableType);
    auto withDrop = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), undef, dropPtr,
        llvm::ArrayRef<int64_t>{ClosureType::VTABLE_DROP_INDEX});
    auto withClone = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), withDrop, clonePtr,
        llvm::ArrayRef<int64_t>{ClosureType::VTABLE_CLONE_INDEX});
    auto withFunc = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), withClone, funcPtr,
        llvm::ArrayRef<int64_t>{ClosureType::VTABLE_EVALUATE_INDEX});

    mlir::LLVM::ReturnOp::create(rewriter, op.getLoc(), withFunc);
    rewriter.replaceOp(op, vtableOp);
    return mlir::success();
  }
};

struct ReussirClosureCreateOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Only handle outlined closures (vtable must be present, body must be
    // empty)
    if (!op.isOutlined())
      return op.emitOpError("closure create must be outlined before lowering");

    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    // Get the RcBox<ClosureBox> type
    RcBoxType rcBoxType = op.getRcClosureBoxType();
    auto convertedRcBoxType = converter->convertType(rcBoxType);

    // Token is the pointer to RcBox<ClosureBox>
    mlir::Value tokenPtr = adaptor.getToken();

    // 1. Assign refcnt (GEP[0, 0]) to 1
    auto refcntPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, convertedRcBoxType, tokenPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    storeInitialRcCount(refcntPtr, loc, rewriter);

    // 2. Assign vtable (GEP[0, 1, 0]) to address of the vtable
    auto vtablePtrSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, convertedRcBoxType, tokenPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, ClosureBoxType::VTABLE_INDEX});
    auto vtableAddr = mlir::LLVM::AddressOfOp::create(
        rewriter, loc, llvmPtrType, *op.getVtable());
    auto vtableStore =
        mlir::LLVM::StoreOp::create(rewriter, loc, vtableAddr, vtablePtrSlot);
    vtableStore.setInvariantGroup(true);

    // 3. Assign cursor (GEP[0, 1, 1]) to the value of GEP[0, 1, 2]
    //    (cursor points to the start of payload area)
    auto cursorSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, convertedRcBoxType, tokenPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1,
                                           ClosureBoxType::ARG_CURSOR_INDEX});
    auto payloadStart = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, convertedRcBoxType, tokenPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 2});
    mlir::LLVM::StoreOp::create(rewriter, loc, payloadStart, cursorSlot);

    // Replace with the token pointer (which is now a valid RcPtr<Closure>)
    rewriter.replaceOp(op, tokenPtr);
    return mlir::success();
  }
};

struct ReussirRcDecOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcDecOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcDecOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // simply check if this is decreasing a rigid object. If so, convert it to
    // function call. otherwise return failure.
    RcType rcPtrTy = op.getRcPtr().getType();
    if (rcPtrTy.getCapability() == Capability::rigid) {
      auto converter =
          static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
      ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      auto releaseFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          "__reussir_release_rigid_object");
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, releaseFunc, mlir::ValueRange{adaptor.getRcPtr()});
      return mlir::success();
    }
    if (auto eleTy = mlir::dyn_cast<FFIObjectType>(rcPtrTy.getElementType())) {
      mlir::FlatSymbolRefAttr cleanupHook = eleTy.getCleanupHook();
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, cleanupHook, mlir::TypeRange{},
          mlir::ValueRange{adaptor.getRcPtr()});
      return mlir::success();
    }
    if (auto closureTy =
            mlir::dyn_cast<ClosureType>(rcPtrTy.getElementType())) {
      mlir::Location loc = op.getLoc();
      auto llvmPtrType =
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

      // Get the RcBox<ClosureBox> type
      RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
      auto convertedBoxType = getTypeConverter()->convertType(rcBoxType);

      // GEP [0, 1, VTABLE_INDEX] to get the vtable pointer slot
      auto vtablePtr = mlir::LLVM::GEPOp::create(
          rewriter, loc, llvmPtrType, convertedBoxType, adaptor.getRcPtr(),
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1,
                                             ClosureBoxType::VTABLE_INDEX});

      // Load the vtable pointer
      auto vtable =
          mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, vtablePtr);
      vtable.setInvariantGroup(true);

      // Create LLVM struct type for vtable: { void*, void*, void* }
      // representing { drop, clone, evaluate } function pointers
      mlir::LLVM::LLVMStructType vtableType =
          mlir::LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(), {llvmPtrType, llvmPtrType, llvmPtrType});

      // GEP [0, VTABLE_DROP_INDEX] to get the drop function pointer
      auto dropPtr = mlir::LLVM::GEPOp::create(
          rewriter, loc, llvmPtrType, vtableType, vtable,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                             ClosureType::VTABLE_DROP_INDEX});

      // Load the drop function pointer
      auto dropFunc =
          mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, dropPtr);
      dropFunc.setInvariantGroup(true);

      // Create function type for drop: void (*)(void*)
      auto voidType = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
      auto funcType =
          mlir::LLVM::LLVMFunctionType::get(voidType, {llvmPtrType});

      // Call the drop function with the RC pointer as argument
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, funcType, mlir::ValueRange{dropFunc, adaptor.getRcPtr()});
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct ReussirRcFreezeOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcFreezeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcFreezeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Call the runtime function __reussir_freeze_flex_object
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto freezeFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
        "__reussir_freeze_flex_object");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, freezeFunc, mlir::ValueRange{adaptor.getRcPtr()});
    return mlir::success();
  }
};

struct ReussirRcIsUniqueOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcIsUniqueOp> {
  using OpConversionPattern::OpConversionPattern;

  // Builds `refcount == 1`; when `outCount` is non-null it receives the
  // loaded refcount so callers can post-process the condition (see
  // rc.assume_unique's immortal neutralization).
  static mlir::Value
  buildUniquenessCondition(mlir::Location loc, RcType rcPtrTy,
                           mlir::Value rcPtr,
                           const mlir::TypeConverter *typeConverter,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value *outCount = nullptr) {
    RcBoxType rcBoxType = rcPtrTy.getInnerBoxType();
    auto convertedBoxType = typeConverter->convertType(rcBoxType);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    auto refcntPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, convertedBoxType, rcPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    // No guard for tagged immediates: the load reads the dummy box's
    // refcount, pinned above the unique/shared decision point, so
    // uniqueness is naturally false there.
    auto countType = rewriter.getI32Type();
    auto refcnt =
        mlir::LLVM::LoadOp::create(rewriter, loc, countType, refcntPtr);
    if (outCount)
      *outCount = refcnt;
    auto one = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(countType, 1));
    mlir::Value isOne = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, refcnt, one);
    return isOne;
  }

  mlir::LogicalResult
  matchAndRewrite(ReussirRcIsUniqueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    RcType rcPtrTy = op.getRcPtr().getType();
    auto isUnique = buildUniquenessCondition(
        op.getLoc(), rcPtrTy, adaptor.getRcPtr(), getTypeConverter(), rewriter);
    rewriter.replaceOp(op, isUnique);
    return mlir::success();
  }
};

struct ReussirRcAssumeUniqueOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRcAssumeUniqueOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRcAssumeUniqueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // NOTE: for a tagged immediate the condition reads the dummy box's
    // refcount and is `false`; asserting `assume(false)` would be unsound,
    // so neutralize the assumption on that path. TBI recognizes immediates
    // by the pointer's top byte, the immortal encoding by the loaded
    // count's magnitude.
    RcType rcPtrTy = op.getRcPtr().getType();
    mlir::Value count;
    mlir::Value isUnique =
        ReussirRcIsUniqueOpConversionPattern::buildUniquenessCondition(
            op.getLoc(), rcPtrTy, adaptor.getRcPtr(), getTypeConverter(),
            rewriter, &count);
    TagEncoding encoding = specialPtrTagEncoding(op);
    if (encoding != TagEncoding::None && rcPtrTy.mayCarrySpecialPointerTag()) {
      mlir::Value tagged;
      if (encoding == TagEncoding::TBI) {
        mlir::Value top = topByteOf(adaptor.getRcPtr(), op.getLoc(), rewriter);
        tagged = isTaggedImmediate(top, op.getLoc(), rewriter);
      } else {
        tagged = isImmortalCount(count, op.getLoc(), rewriter);
      }
      auto vTrue = mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                                  rewriter.getI1Type(),
                                                  rewriter.getBoolAttr(true));
      isUnique = mlir::LLVM::SelectOp::create(rewriter, op.getLoc(), tagged,
                                              vTrue, isUnique);
    }
    mlir::LLVM::AssumeOp::create(rewriter, op.getLoc(), isUnique);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReussirRegionCleanupOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRegionCleanupOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRegionCleanupOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Call the runtime function __reussir_freeze_flex_object
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    ensureRuntimeFunctions(op->getParentOfType<mlir::ModuleOp>(), *converter);
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto cleanupFunc = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
        "__reussir_cleanup_region");
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, cleanupFunc, mlir::ValueRange{adaptor.getRegion()});
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeEndOp>(
        rewriter, op.getLoc(), ptrType, adaptor.getRegion(), *converter,
        op.getOperation());
    return mlir::success();
  }
};

struct ReussirRegionCreateOpConversionPattern
    : public mlir::OpConversionPattern<ReussirRegionCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRegionCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Call the runtime function __reussir_create_region
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto one = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(converter->getIndexType(), 1));
    auto alloca = rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, ptrType, ptrType, one);
    addLifetimeOrInvariantOp<mlir::LLVM::LifetimeStartOp>(
        rewriter, op.getLoc(), ptrType, alloca, *converter, op.getOperation());
    auto nullValue = mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrType);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), nullValue, alloca);
    return mlir::success();
  }
};

struct ReussirClosureApplyOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureApplyOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  mlir::Value emitPointerAlign(mlir::Value ptr, mlir::Type type,
                               mlir::OpBuilder &builder,
                               mlir::Operation *scopeOp) const {
    const mlir::LLVMTypeConverter *converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto alignment =
        getDataLayout(*converter, scopeOp).getTypeABIAlignment(type);
    if (alignment <= 1)
      return ptr;
    auto addr = mlir::LLVM::PtrToIntOp::create(builder, ptr.getLoc(),
                                               converter->getIndexType(), ptr);
    auto mask = mlir::arith::ConstantOp::create(
        builder, ptr.getLoc(),
        mlir::IntegerAttr::get(converter->getIndexType(), alignment - 1));
    auto zero = mlir::arith::ConstantOp::create(
        builder, ptr.getLoc(),
        mlir::IntegerAttr::get(converter->getIndexType(), 0));
    auto negAddr = mlir::arith::SubIOp::create(
        builder, ptr.getLoc(), converter->getIndexType(), zero, addr);
    auto offset = mlir::arith::AndIOp::create(
        builder, ptr.getLoc(), converter->getIndexType(), negAddr, mask);
    auto alignedAddr = mlir::arith::AddIOp::create(
        builder, ptr.getLoc(), converter->getIndexType(), addr, offset,
        mlir::arith::IntegerOverflowFlags::nuw);
    return mlir::LLVM::IntToPtrOp::create(builder, ptr.getLoc(), ptr.getType(),
                                          alignedAddr.getResult());
  }

  mlir::Value emitPointerBump(mlir::Value ptr, mlir::Type type,
                              mlir::OpBuilder &builder,
                              mlir::Operation *scopeOp) const {
    const mlir::LLVMTypeConverter *converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto size = getDataLayout(*converter, scopeOp).getTypeSize(type);
    if (size == 0)
      return ptr;
    auto addr = mlir::LLVM::PtrToIntOp::create(builder, ptr.getLoc(),
                                               converter->getIndexType(), ptr);
    auto sizeVal = mlir::arith::ConstantOp::create(
        builder, ptr.getLoc(),
        mlir::IntegerAttr::get(converter->getIndexType(), size));
    auto newAddr = mlir::arith::AddIOp::create(
        builder, ptr.getLoc(), converter->getIndexType(), addr, sizeVal,
        mlir::arith::IntegerOverflowFlags::nuw);
    return mlir::LLVM::IntToPtrOp::create(builder, ptr.getLoc(), ptr.getType(),
                                          newAddr.getResult());
  }

public:
  mlir::LogicalResult
  matchAndRewrite(ReussirClosureApplyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto rcClosureBox = op.getType().getInnerBoxType();
    auto structType = typeConverter->convertType(rcClosureBox);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    // This operation assumes that the closure is already uniquely owned.
    // First, use GEP[0, 1, ARG_CURSOR_INDEX] and load op to read the cursor
    // RcBox {
    //    size_t refcnt;
    //    {
    //        void* vtable;
    //        void* cursor;
    //        // trailing payload
    //    };
    // }
    auto cursorPtr = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(), llvmPtrType, structType, adaptor.getClosure(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1,
                                           ClosureBoxType::ARG_CURSOR_INDEX});
    auto cursor = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), llvmPtrType,
                                             cursorPtr);

    // Second, align the cursor to current input type's TypeABIAlignment.
    auto inputType = typeConverter->convertType(op.getArg().getType());
    auto alignedCursor =
        emitPointerAlign(cursor, inputType, rewriter, op.getOperation());

    // Third, copy the input value to the aligned cursor.
    auto payloadRef = mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(), adaptor.getArg(), alignedCursor);
    payloadRef.setInvariantGroup(true);

    // Fourth, bump the cursor by the input type's size.
    auto newCursor =
        emitPointerBump(alignedCursor, inputType, rewriter, op.getOperation());

    // Fifth, store the new cursor back to the RcBox.
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), newCursor, cursorPtr);

    // Return the same closure pointer (in-place modification)
    rewriter.replaceOp(op, adaptor.getClosure());
    return mlir::success();
  }
};

struct ReussirClosureCloneOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureCloneOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureCloneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the RC closure box type and convert it to LLVM struct type
    auto rcClosureBox = op.getType().getInnerBoxType();
    auto structType = converter->convertType(rcClosureBox);

    // First, GEP [0, 1, VTABLE_INDEX] to get the vtable pointer
    auto vtablePtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getClosure(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, ClosureBoxType::VTABLE_INDEX});

    // Load the vtable
    auto vtable =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, vtablePtr);
    vtable.setInvariantGroup(true);

    // Create LLVM struct type for vtable: { void*, void*, void* }
    // representing { drop, clone, evaluate } function pointers
    mlir::LLVM::LLVMStructType vtableType =
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {llvmPtrType, llvmPtrType, llvmPtrType});

    // Second, GEP [0, VTABLE_CLONE_INDEX] to get the clone function pointer
    auto clonePtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, vtableType, vtable,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, ClosureType::VTABLE_CLONE_INDEX});

    // Load the clone function pointer
    auto cloneFunc =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, clonePtr);
    cloneFunc.setInvariantGroup(true);

    // Create function type for clone: void* (*)(void*)
    auto funcType =
        mlir::LLVM::LLVMFunctionType::get(llvmPtrType, {llvmPtrType});

    // Call the clone function with the RC pointer as argument
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, funcType,
        mlir::ValueRange{cloneFunc.getResult(), adaptor.getClosure()});
    return mlir::success();
  }
};

struct ReussirClosureEvalOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureEvalOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureEvalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    // Get the RC closure box type and convert it to LLVM struct type
    RcType rcClosureType = op.getClosure().getType();
    auto rcClosureBox = rcClosureType.getInnerBoxType();
    auto structType = converter->convertType(rcClosureBox);

    // GEP [0, 1, VTABLE_INDEX] to get the vtable pointer slot
    auto vtablePtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getClosure(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, ClosureBoxType::VTABLE_INDEX});

    // Load the vtable pointer
    auto vtable =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, vtablePtr);
    vtable.setInvariantGroup(true);

    // Create LLVM struct type for vtable: { void*, void*, void* }
    // representing { drop, clone, evaluate } function pointers
    mlir::LLVM::LLVMStructType vtableType =
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {llvmPtrType, llvmPtrType, llvmPtrType});

    // GEP [0, VTABLE_EVALUATE_INDEX] to get the evaluate function pointer
    auto evalPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, vtableType, vtable,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                           ClosureType::VTABLE_EVALUATE_INDEX});

    // Load the evaluate function pointer
    auto evalFunc =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, evalPtr);
    evalFunc.setInvariantGroup(true);

    // Determine if the closure has a return value
    mlir::Type resultType =
        op.getResult() ? op.getResult().getType() : mlir::Type{};
    if (resultType) {
      // Closure has a return value: T (*)(void*)
      mlir::Type llvmResultType = converter->convertType(resultType);
      auto funcType =
          mlir::LLVM::LLVMFunctionType::get(llvmResultType, {llvmPtrType});
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, funcType, mlir::ValueRange{evalFunc, adaptor.getClosure()});
    } else {
      // Closure has no return value: void (*)(void*)
      auto voidType = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
      auto funcType =
          mlir::LLVM::LLVMFunctionType::get(voidType, {llvmPtrType});
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, funcType, mlir::ValueRange{evalFunc, adaptor.getClosure()});
    }
    return mlir::success();
  }
};

struct ReussirClosureInspectPayloadOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureInspectPayloadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureInspectPayloadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the ClosureBox type from the reference
    RefType refType = op.getClosureBoxRef().getType();
    ClosureBoxType closureBoxType =
        llvm::cast<ClosureBoxType>(refType.getElementType());

    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto structType = getTypeConverter()->convertType(closureBoxType);

    // GEP[0, N+2] where N is the payload index
    // ClosureBox layout: { vtable (0), cursor (1), payload... (2+) }
    int64_t gepIndex =
        static_cast<int64_t>(op.getPayloadIndex().getZExtValue()) + 2;
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        op, llvmPtrType, structType, adaptor.getClosureBoxRef(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, gepIndex});
    return mlir::success();
  }
};

struct ReussirClosureCursorOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureCursorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureCursorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the ClosureBox type from the reference
    RefType refType = op.getClosureBoxRef().getType();
    ClosureBoxType closureBoxType =
        llvm::cast<ClosureBoxType>(refType.getElementType());

    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto structType = getTypeConverter()->convertType(closureBoxType);

    // GEP[0, 1] to access the cursor field
    // ClosureBox layout: { vtable (0), cursor (1), payload... (2+) }
    auto cursor =
        mlir::LLVM::GEPOp::create(rewriter, op->getLoc(), llvmPtrType,
                                  structType, adaptor.getClosureBoxRef(),
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{
                                      0, ClosureBoxType::ARG_CURSOR_INDEX});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, llvmPtrType, cursor);
    return mlir::success();
  }
};

struct ReussirClosureTransferOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureTransferOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureTransferOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto indexType = converter->getIndexType();

    // Get the ClosureBox type from the source reference
    RefType srcRefType = op.getSrc().getType();
    ClosureBoxType closureBoxType =
        llvm::cast<ClosureBoxType>(srcRefType.getElementType());
    auto structType = converter->convertType(closureBoxType);
    // 1. Extract the cursor pointer from src
    // GEP[0, 1] to access the cursor field
    auto cursorPtrSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getSrc(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                           ClosureBoxType::ARG_CURSOR_INDEX});
    auto cursor =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, cursorPtrSlot);

    // 2. Compute the offset from src to cursor
    // Convert pointers to integers
    mlir::Value srcInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, loc, indexType, adaptor.getSrc());
    mlir::Value dstInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, loc, indexType, adaptor.getDst());
    mlir::Value cursorInt =
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, indexType, cursor);

    // Compute offset: cursor - src
    mlir::Value fullOffset =
        mlir::arith::SubIOp::create(rewriter, loc, cursorInt, srcInt,
                                    mlir::arith::IntegerOverflowFlags::nsw |
                                        mlir::arith::IntegerOverflowFlags::nuw);
    mlir::Value ptrSize = mlir::arith::ConstantIntOp::create(
        rewriter, loc, indexType,
        getDataLayout(*converter, op.getOperation())
            .getTypeSize(llvmPtrType)
            .getFixedValue());
    mlir::Value offset =
        mlir::arith::SubIOp::create(rewriter, loc, fullOffset, ptrSize,
                                    mlir::arith::IntegerOverflowFlags::nsw |
                                        mlir::arith::IntegerOverflowFlags::nuw);
    mlir::Value srcPlusPtrSize =
        mlir::arith::AddIOp::create(rewriter, loc, srcInt, ptrSize,
                                    mlir::arith::IntegerOverflowFlags::nsw |
                                        mlir::arith::IntegerOverflowFlags::nuw);
    mlir::Value src = mlir::LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrType,
                                                     srcPlusPtrSize);
    mlir::Value dstPlusPtrSize =
        mlir::arith::AddIOp::create(rewriter, loc, dstInt, ptrSize,
                                    mlir::arith::IntegerOverflowFlags::nsw |
                                        mlir::arith::IntegerOverflowFlags::nuw);
    mlir::Value dst = mlir::LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrType,
                                                     dstPlusPtrSize);
    // Call llvm.memcpy intrinsic
    auto dataLayout = getDataLayout(*converter, op.getOperation());
    auto closureSizeFixed =
        dataLayout.getTypeSize(closureBoxType).getFixedValue();
    auto ptrSizeFixed = dataLayout.getTypeSize(llvmPtrType).getFixedValue();
    if (closureSizeFixed < ptrSizeFixed)
      llvm::report_fatal_error("Closure size is smaller than pointer size");
    auto size = closureSizeFixed - ptrSizeFixed;
    if (size > 32)
      mlir::LLVM::MemcpyOp::create(rewriter, loc, dst, src, offset, false);
    else {
      // directly copy the whole closure box
      mlir::IntegerAttr sizeAttr = mlir::IntegerAttr::get(indexType, size);
      mlir::LLVM::MemcpyInlineOp::create(rewriter, loc, dst, src, sizeAttr,
                                         false);
    }
    // (explicity transfer vtable)
    auto vtablePtrSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getSrc(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, ClosureBoxType::VTABLE_INDEX});
    auto vtable =
        mlir::LLVM::LoadOp::create(rewriter, loc, llvmPtrType, vtablePtrSlot);
    vtable.setInvariantGroup(true);
    auto targetVtablePtrSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getDst(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, ClosureBoxType::VTABLE_INDEX});
    auto targetVtable =
        mlir::LLVM::StoreOp::create(rewriter, loc, vtable, targetVtablePtrSlot);
    targetVtable.setInvariantGroup(true);

    // Update cursor to (dst + offset)
    auto newCursor = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, rewriter.getI8Type(), adaptor.getDst(),
        fullOffset, mlir::LLVM::GEPNoWrapFlags::inbounds);

    // Get the cursor slot in the destination closure box
    auto dstCursorSlot = mlir::LLVM::GEPOp::create(
        rewriter, loc, llvmPtrType, structType, adaptor.getDst(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                           ClosureBoxType::ARG_CURSOR_INDEX});

    // Store the new cursor to the destination's cursor slot
    mlir::LLVM::StoreOp::create(rewriter, loc, newCursor, dstCursorSlot);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReussirClosureInstantiateOpConversionPattern
    : public mlir::OpConversionPattern<ReussirClosureInstantiateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirClosureInstantiateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    // Set RC value to 1 and return the original pointer
    auto one = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::IntegerAttr::get(converter->getIndexType(), 1));
    auto refcntPtr = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(), llvmPtrType,
        converter->convertType(
            op.getClosureBoxRc().getType().getInnerBoxType()),
        adaptor.getToken(), llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), one, refcntPtr);
    rewriter.replaceOp(op, adaptor.getToken());
    return mlir::success();
  }
};

struct ReussirStrGlobalOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::StringRef payload = op.getPayload();

    // Create an LLVM array type for the string (including null terminator)
    auto i8Type = mlir::IntegerType::get(rewriter.getContext(), 8);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, payload.size() + 1);

    // Create the string attribute with null terminator
    std::string payloadWithNull = payload.str();
    payloadWithNull.push_back('\0');
    auto stringAttr = rewriter.getStringAttr(payloadWithNull);

    // Create LLVM global op
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        // TODO: add target info in module attributes and use OnceDOR to
        // allow linker to merge globals with the same name
        op, arrayType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
        op.getSymName(), stringAttr);

    return mlir::success();
  }
};

struct ReussirStrLiteralOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrLiteralOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrLiteralOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto indexType = converter->getIndexType();

    // The result type is a struct { ptr, index } representing { data, len }
    auto structType = mlir::LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {llvmPtrType, indexType});

    // Look up the global string to get its length
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto globalOp =
        moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(op.getSymName());
    if (!globalOp)
      return op.emitOpError("referenced global string not found");

    // Get the string length from the array type (minus null terminator)
    auto arrayType =
        llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(globalOp.getType());
    if (!arrayType)
      return op.emitOpError("global is not an array type");
    size_t strLen = arrayType.getNumElements() - 1; // exclude null terminator

    // Get address of the global string
    auto strPtr = mlir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrType,
                                                  op.getSymName());

    // Create the length constant
    auto lenVal = mlir::arith::ConstantOp::create(
        rewriter, loc, mlir::IntegerAttr::get(indexType, strLen));

    // Build the struct { ptr, len }
    auto undef = mlir::LLVM::UndefOp::create(rewriter, loc, structType);
    auto withPtr = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, undef, strPtr, llvm::ArrayRef<int64_t>{0});
    auto withLen = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, withPtr, lenVal, llvm::ArrayRef<int64_t>{1});

    rewriter.replaceOp(op, withLen);
    return mlir::success();
  }
};

struct ReussirStrCastOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getGlobalStr());
    return mlir::success();
  }
};

struct ReussirStrLenOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrLenOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrLenOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Extract the length field (index 1) from the string struct
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getStr(), llvm::ArrayRef<int64_t>{1});
    return mlir::success();
  }
};

struct ReussirStrUnsafeByteAtOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrUnsafeByteAtOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrUnsafeByteAtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto i8Type = rewriter.getI8Type();

    // Extract the pointer field (index 0) from the string struct
    auto ptr = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getStr(), llvm::ArrayRef<int64_t>{0});

    // Calculate the address of the character
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto charPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType, i8Type, ptr,
                                  mlir::ValueRange{adaptor.getIndex()});

    // Load the character
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, i8Type, charPtr);

    return mlir::success();
  }
};

struct ReussirStrUnsafeStartWithOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrUnsafeStartWithOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrUnsafeStartWithOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    llvm::StringRef prefix = op.getPrefix();
    size_t len = prefix.size();

    // Extract the pointer field (index 0) from the string struct
    auto ptr = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getStr(), llvm::ArrayRef<int64_t>{0});

    // Use memcmp for all prefix lengths as LLVM optimizes it well.
    // 1. Create a deduplicated global string for the prefix.
    std::string globalName =
        mangledBlake3Symbol("REUSSIR_STRING_PREFIX", prefix);

    auto existingGlobal = module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName);
    if (!existingGlobal) {
      auto i8Type = rewriter.getI8Type();
      auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, prefix.size());
      auto stringAttr = rewriter.getStringAttr(prefix);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      mlir::LLVM::GlobalOp::create(rewriter, loc, arrayType, true,
                                   mlir::LLVM::Linkage::Internal, globalName,
                                   stringAttr);
    }

    // 2. Get address of global string
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto prefixPtr =
        mlir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrType, globalName);

    // 3. Declare memcmp if needed
    // declare i32 @memcmp(ptr, ptr, i64)
    if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("memcmp")) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto i32Type = rewriter.getI32Type();
      auto i64Type = rewriter.getI64Type();
      auto fnType = mlir::LLVM::LLVMFunctionType::get(
          i32Type, {llvmPtrType, llvmPtrType, i64Type});
      mlir::LLVM::LLVMFuncOp::create(rewriter, loc, "memcmp", fnType);
    }

    // 4. Call memcmp
    auto lenVal = mlir::arith::ConstantIntOp::create(rewriter, loc, len, 64);
    auto call = mlir::LLVM::CallOp::create(
        rewriter, loc, rewriter.getI32Type(),
        mlir::SymbolRefAttr::get(rewriter.getContext(), "memcmp"),
        mlir::ValueRange{ptr, prefixPtr, lenVal});

    // 5. Compare result == 0
    auto zero = mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    auto res = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, call.getResult(), zero);
    rewriter.replaceOp(op, res);

    return mlir::success();
  }
};

struct ReussirStrSliceOpConversionPattern
    : public mlir::OpConversionPattern<ReussirStrSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirStrSliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();
    auto startOffset = adaptor.getOffset();

    // 1. Get length and pointer
    auto ptr = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getStr(), llvm::ArrayRef<int64_t>{0});
    auto len = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getStr(), llvm::ArrayRef<int64_t>{1});

    // 2. Check bounds
    // offset > len ?
    auto outOfBounds = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::ugt, startOffset, len);

    // 3. Select new length and pointer adjustment
    // If outOfBounds, newLen = 0, newPtr = ptr (or undefined)
    // Actually, safest is:
    // adjustment = min(offset, len)
    // newLen = len - adjustment
    // newPtr = ptr + adjustment

    // adjustment = min(offset, len)
    // umin is not directly available in LLVM dialect as a single op usually?
    // Use select.
    auto adjustment = mlir::LLVM::SelectOp::create(rewriter, loc, outOfBounds,
                                                   len, startOffset);
    auto newLen = mlir::LLVM::SubOp::create(rewriter, loc, len, adjustment);

    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto newPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType, rewriter.getI8Type(),
                                  ptr, mlir::ValueRange{adjustment});

    // 4. Create new struct
    auto structType = getTypeConverter()->convertType(op.getType());
    auto newStr = mlir::LLVM::UndefOp::create(rewriter, loc, structType);
    auto s1 = mlir::LLVM::InsertValueOp::create(rewriter, loc, newStr, newPtr,
                                                llvm::ArrayRef<int64_t>{0});
    auto s2 = mlir::LLVM::InsertValueOp::create(rewriter, loc, s1, newLen,
                                                llvm::ArrayRef<int64_t>{1});

    rewriter.replaceOp(op, s2.getResult());
    return mlir::success();
  }
};

struct ReussirRefDiffConversionPattern
    : public mlir::OpConversionPattern<ReussirRefDiffOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefDiffOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto indexType = converter->getIndexType();

    // Convert base and target pointers to integers
    mlir::Value baseInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, op.getLoc(), indexType, adaptor.getBase());
    mlir::Value targetInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, op.getLoc(), indexType, adaptor.getTarget());

    // Compute difference: target - base
    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
        op, targetInt, baseInt, mlir::arith::IntegerOverflowFlags::nsw);
    return mlir::success();
  }
};

struct ReussirRefCmpConversionPattern
    : public mlir::OpConversionPattern<ReussirRefCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto indexType = converter->getIndexType();

    // Convert lhs and rhs pointers to integers
    mlir::Value lhsInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, op.getLoc(), indexType, adaptor.getLhs());
    mlir::Value rhsInt = mlir::LLVM::PtrToIntOp::create(
        rewriter, op.getLoc(), indexType, adaptor.getRhs());

    // Perform the comparison
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, op.getPredicate(),
                                                     lhsInt, rhsInt);
    return mlir::success();
  }
};

struct ReussirRefMemcpyConversionPattern
    : public mlir::OpConversionPattern<ReussirRefMemcpyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirRefMemcpyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto dataLayout = getDataLayout(*converter, op.getOperation());

    // Get the element type and its size
    RefType srcType = op.getSrc().getType();
    mlir::Type elementType = converter->convertType(srcType.getElementType());
    size_t size = dataLayout.getTypeSize(elementType);

    // Create LLVM memcpy intrinsic (non-overlapping, so isVolatile = false)
    rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyInlineOp>(
        op, adaptor.getDst(), adaptor.getSrc(),
        rewriter.getIntegerAttr(converter->getIndexType(), size),
        /*isVolatile=*/false);
    return mlir::success();
  }
};

struct ReussirTrampolineOpConversionPattern
    : public mlir::OpConversionPattern<ReussirTrampolineOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTrampolineOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::SymbolTable symTable = mlir::SymbolTable::getNearestSymbolTable(op);
    auto converter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto ptrType = mlir::LLVM::LLVMPointerType::get(op.getContext());
    mlir::Operation *funcOp = symTable.lookup(op.getTarget());
    if (!funcOp)
      return mlir::failure();
    // if start is not "C" ABI, return failure
    if (op.getAbiName() != "C")
      return mlir::failure();

    mlir::LLVM::LLVMFunctionType llvmFuncTy;
    if (auto llvmFuncOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(funcOp))
      llvmFuncTy = llvmFuncOp.getFunctionType();
    else if (auto mlirFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(funcOp)) {
      auto funcTy = mlirFuncOp.getFunctionType();
      // Check if we need signature conversion
      mlir::TypeConverter::SignatureConversion signatureConversion(
          funcTy.getNumInputs());
      llvmFuncTy = llvm::dyn_cast_if_present<mlir::LLVM::LLVMFunctionType>(
          converter->convertFunctionSignature(funcTy, /*isVariadic=*/false,
                                              /*useBarePtrCallConv=*/false,
                                              signatureConversion));
    }
    if (!llvmFuncTy)
      return mlir::failure();

    auto llvmRetTy = llvmFuncTy.getReturnType();
    auto llvmParamTys = llvmFuncTy.getParams();
    const bool isImport =
        op.getDirection() == reussir::TrampolineDirection::Import;
    CABISignature cabiSig;
    if (isImport) {
      cabiSig.isTrivial = true;
      cabiSig.abiReturnType = llvmRetTy;
      cabiSig.abiParamTypes.append(llvmParamTys.begin(), llvmParamTys.end());
    } else {
      cabiSig = evaluateCABISignatureForC(llvmRetTy, llvmParamTys);
    }

    auto trampolineTy = mlir::LLVM::LLVMFunctionType::get(
        cabiSig.abiReturnType, cabiSig.abiParamTypes);
    auto trampoline = mlir::LLVM::LLVMFuncOp::create(
        rewriter, op.getLoc(), op.getSymName(), trampolineTy);

    mlir::Block *entry = trampoline.addEntryBlock(rewriter);
    rewriter.setInsertionPointToStart(entry);

    llvm::SmallVector<mlir::Value> targetArgs;
    if (isImport) {
      targetArgs.append(trampoline.getArguments().begin(),
                        trampoline.getArguments().end());
    } else if (cabiSig.hasPackedArgs) {
      auto packedArgsPtr = trampoline.getArgument(cabiSig.packedArgsIndex);
      for (const auto &[idx, paramType] : llvm::enumerate(llvmParamTys)) {
        llvm::SmallVector<mlir::LLVM::GEPArg, 2> gepArgs{
            0, static_cast<int32_t>(idx)};
        auto fieldPtr = mlir::LLVM::GEPOp::create(
            rewriter, op.getLoc(), ptrType, cabiSig.packedArgsType,
            packedArgsPtr, gepArgs);
        targetArgs.push_back(mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                                        paramType, fieldPtr));
      }
    } else {
      auto trampolineArgs = trampoline.getArguments();
      auto firstTargetArg = cabiSig.hasReturnPtr ? 1u : 0u;
      targetArgs.append(trampolineArgs.begin() + firstTargetArg,
                        trampolineArgs.end());
    }

    mlir::Operation *callOp;

    if (auto llvmFuncOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(funcOp)) {
      callOp = mlir::LLVM::CallOp::create(rewriter, op.getLoc(), llvmFuncOp,
                                          targetArgs);
    } else if (auto mlirFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(funcOp)) {
      callOp = mlir::func::CallOp::create(rewriter, op.getLoc(), mlirFuncOp,
                                          targetArgs);
    } else {
      return mlir::failure();
    }

    if (cabiSig.hasReturnPtr && !isImport) {
      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(), callOp->getResult(0),
          trampoline.getArgument(cabiSig.returnPtrIndex));
      mlir::LLVM::ReturnOp::create(rewriter, op.getLoc(), mlir::ValueRange{});
    } else {
      if (llvm::isa<mlir::LLVM::LLVMVoidType>(cabiSig.abiReturnType))
        mlir::LLVM::ReturnOp::create(rewriter, op.getLoc(), mlir::ValueRange{});
      else
        mlir::LLVM::ReturnOp::create(rewriter, op.getLoc(),
                                     callOp->getResults());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};
struct ReussirTokenLaunderOpConversionPattern
    : public mlir::OpConversionPattern<ReussirTokenLaunderOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReussirTokenLaunderOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newSSA =
        rewriter.replaceOpWithNewOp<mlir::LLVM::LaunderInvariantGroupOp>(
            op, adaptor.getToken());
    /*
     * Why we emit llvm.assume after token launder (full repro):
     *
     * Input IR used for analysis:
     *
     *   ; ModuleID = 'caseC_reload_after_store_full.ll'
     *   source_filename = "caseC_reload_after_store_full.ll"
     *
     *   define void @store_reload_invariant(ptr %p) {
     *   entry:
     *     %x  = load i64, ptr %p, align 8, !invariant.group !0
     *     %p8 = getelementptr i8, ptr %p, i64 8
     *     %y  = load i64, ptr %p8, align 8, !invariant.group !0
     *
     *     %q  = call ptr @llvm.launder.invariant.group.p0(ptr %p)
     *     %eq = icmp eq ptr %q, %p
     *     call void @llvm.assume(i1 %eq)
     *
     *     %q8 = getelementptr i8, ptr %q, i64 8
     *
     *     call void @use_i64(i64 %x)
     *     call void @use_i64(i64 %y)
     *
     *     store i64 %x, ptr %q, align 8, !invariant.group !0
     *     %x_reload = load i64, ptr %q, align 8, !invariant.group !0
     *     call void @use_i64(i64 %x_reload)
     *
     *     %y_plus_1 = add i64 %y, 1
     *     store i64 %y_plus_1, ptr %q8, align 8, !invariant.group !0
     *     %y_reload = load i64, ptr %q8, align 8, !invariant.group !0
     *     call void @use_i64(i64 %y_reload)
     *     ret void
     *   }
     *
     *   declare ptr @llvm.launder.invariant.group.p0(ptr)
     *   declare void @llvm.assume(i1 noundef)
     *   declare void @use_i64(i64)
     *   !0 = !{}
     *
     * opt -S -O3 output with assume:
     *
     *   define void @store_reload_invariant(ptr captures(address) %p)
     *   local_unnamed_addr {
     *   entry:
     *     %x = load i64, ptr %p, align 8, !invariant.group !0
     *     %p8 = getelementptr i8, ptr %p, i64 8
     *     %y = load i64, ptr %p8, align 8, !invariant.group !0
     *     %q = tail call ptr @llvm.launder.invariant.group.p0(ptr nonnull %p)
     *     %eq = icmp eq ptr %q, %p
     *     tail call void @llvm.assume(i1 %eq)
     *     tail call void @use_i64(i64 %x)
     *     tail call void @use_i64(i64 %y)
     *     tail call void @use_i64(i64 %x)
     *     %y_plus_1 = add i64 %y, 1
     *     store i64 %y_plus_1, ptr %p8, align 8, !invariant.group !0
     *     tail call void @use_i64(i64 %y_plus_1)
     *     ret void
     *   }
     *
     * opt -S -O3 output without assume (remove `%eq` + `llvm.assume` first):
     *
     *   define void @store_reload_invariant(ptr captures(none) %p)
     *   local_unnamed_addr {
     *   entry:
     *     %x = load i64, ptr %p, align 8, !invariant.group !0
     *     %p8 = getelementptr i8, ptr %p, i64 8
     *     %y = load i64, ptr %p8, align 8, !invariant.group !0
     *     %q = tail call ptr @llvm.launder.invariant.group.p0(ptr nonnull %p)
     *     %q8 = getelementptr i8, ptr %q, i64 8
     *     tail call void @use_i64(i64 %x)
     *     tail call void @use_i64(i64 %y)
     *     store i64 %x, ptr %q, align 8, !invariant.group !0
     *     tail call void @use_i64(i64 %x)
     *     %y_plus_1 = add i64 %y, 1
     *     store i64 %y_plus_1, ptr %q8, align 8, !invariant.group !0
     *     tail call void @use_i64(i64 %y_plus_1)
     *     ret void
     *   }
     *
     * Analysis:
     * - `llvm.launder.invariant.group` intentionally creates a fresh SSA name.
     * - Without an explicit equality fact, O3 keeps stores on `%q/%q8`.
     * - With `llvm.assume(%q == %p)`, O3 can reconnect `%q` to `%p`, fold the
     *   memory updates back to `%p/%p8`, and eliminate the redundant `%x`
     * store.
     * - Therefore this assume is required to recover alias/value propagation
     *   power after the launder boundary.
     */
    auto ptrEq = mlir::LLVM::ICmpOp::create(rewriter, op.getLoc(),
                                            mlir::LLVM::ICmpPredicate::eq,
                                            newSSA, adaptor.getToken());
    mlir::LLVM::AssumeOp::create(rewriter, op.getLoc(), ptrEq);
    return mlir::success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Runtime Functions
//===----------------------------------------------------------------------===//

namespace {
mlir::LLVM::LLVMFuncOp addRuntimeFunction(mlir::Block *body,
                                          llvm::StringRef name,
                                          llvm::ArrayRef<mlir::Type> inputs,
                                          llvm::ArrayRef<mlir::Type> outputs) {
  auto module = mlir::cast<mlir::ModuleOp>(body->getParentOp());
  if (auto existing = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return existing;
  mlir::MLIRContext *ctx = module.getContext();
  auto type = mlir::LLVM::LLVMFunctionType::get(
      outputs.empty() ? mlir::LLVM::LLVMVoidType::get(ctx) : outputs.front(),
      inputs);
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(body);
  mlir::LLVM::LLVMFuncOp func = mlir::LLVM::LLVMFuncOp::create(
      builder, mlir::UnknownLoc::get(ctx), name, type);
  func.setLinkage(mlir::LLVM::Linkage::External);
  return func;
}

void ensureRuntimeFunctions(mlir::ModuleOp module,
                            const mlir::LLVMTypeConverter &converter) {
  mlir::MLIRContext *ctx = module.getContext();
  mlir::Block *body = module.getBody();
  auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto indexType = converter.getIndexType();
  addRuntimeFunction(body, "__reussir_freeze_flex_object", {llvmPtrType},
                     {llvmPtrType});
  addRuntimeFunction(body, "__reussir_cleanup_region", {llvmPtrType}, {});
  addRuntimeFunction(body, "__reussir_acquire_rigid_object", {llvmPtrType}, {});
  addRuntimeFunction(body, "__reussir_release_rigid_object", {llvmPtrType}, {});
  auto allocFunc = addRuntimeFunction(body, "__reussir_allocate",
                                      {indexType, indexType}, {llvmPtrType});
  auto deallocFunc = addRuntimeFunction(
      body, "__reussir_deallocate", {llvmPtrType, indexType, indexType}, {});
  // Add common runtime attributes.
  mlir::OpBuilder builder(ctx);
  auto passthroughAlloc = builder.getStrArrayAttr(
      {"mustprogress", "nounwind", "willreturn", "nocallback"});

  allocFunc->setAttr("passthrough", passthroughAlloc);
  allocFunc.setArgAttr(0, "llvm.allocalign", builder.getUnitAttr());
  allocFunc.setResultAttr(0, "llvm.noalias", builder.getUnitAttr());
  allocFunc.setResultAttr(0, "llvm.noundef", builder.getUnitAttr());
  allocFunc.setArgAttr(0, "llvm.noundef", builder.getUnitAttr());
  allocFunc.setArgAttr(1, "llvm.noundef", builder.getUnitAttr());

  auto passthroughDealloc = builder.getStrArrayAttr(
      {"mustprogress", "nounwind", "willreturn", "nocallback"});
  deallocFunc->setAttr("passthrough", passthroughDealloc);
  deallocFunc.setArgAttr(0, "llvm.nocapture", builder.getUnitAttr());
  deallocFunc.setArgAttr(0, "llvm.allocptr", builder.getUnitAttr());
  deallocFunc.setArgAttr(1, "llvm.allocalign", builder.getUnitAttr());
  deallocFunc.setArgAttr(0, "llvm.noundef", builder.getUnitAttr());
  deallocFunc.setArgAttr(1, "llvm.noundef", builder.getUnitAttr());
  deallocFunc.setArgAttr(2, "llvm.noundef", builder.getUnitAttr());

  auto reallocFunc = addRuntimeFunction(
      body, "__reussir_reallocate",
      {llvmPtrType, indexType, indexType, indexType, indexType}, {llvmPtrType});
  auto passthroughRealloc = builder.getStrArrayAttr(
      {"mustprogress", "nounwind", "willreturn", "nocallback"});
  reallocFunc->setAttr("passthrough", passthroughRealloc);
  reallocFunc.setArgAttr(0, "llvm.allocptr", builder.getUnitAttr());
  reallocFunc.setArgAttr(3, "llvm.allocalign", builder.getUnitAttr());
  reallocFunc.setResultAttr(0, "llvm.noundef", builder.getUnitAttr());
  reallocFunc.setArgAttr(0, "llvm.noundef", builder.getUnitAttr());
  reallocFunc.setArgAttr(1, "llvm.noundef", builder.getUnitAttr());
  reallocFunc.setArgAttr(2, "llvm.noundef", builder.getUnitAttr());
  reallocFunc.setArgAttr(3, "llvm.noundef", builder.getUnitAttr());
  reallocFunc.setArgAttr(4, "llvm.noundef", builder.getUnitAttr());
  // currently this will abort execution after printing the message and
  // stacktrace. No unwinding is attempted yet.
  addRuntimeFunction(body, "__reussir_panic", {llvmPtrType, indexType}, {});
}

// Debug info conversion logic has been moved to DbgInfoConversion.cpp
} // namespace

//===----------------------------------------------------------------------===//
// BasicOpsLoweringPass
//===----------------------------------------------------------------------===//

namespace {
template <typename DialectT>
void populateLLVMInterfaceForDialect(mlir::MLIRContext *context,
                                     mlir::ConversionTarget &target,
                                     mlir::LLVMTypeConverter &typeConverter,
                                     mlir::RewritePatternSet &patterns) {
  auto *dialect = context->getOrLoadDialect<DialectT>();
  auto *interface = dialect->template getRegisteredInterface<
      mlir::ConvertToLLVMPatternInterface>();
  if (!interface)
    return;
  interface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                     patterns);
}

struct ReussirConvertToLLVMPatternInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(mlir::MLIRContext *context) const override {
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    context->getOrLoadDialect<mlir::math::MathDialect>();
    context->getOrLoadDialect<mlir::memref::MemRefDialect>();
    context->getOrLoadDialect<mlir::ub::UBDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const override {
    populateReussirToLLVMTypeConversions(typeConverter);
    populateLLVMInterfaceForDialect<mlir::arith::ArithDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateLLVMInterfaceForDialect<mlir::cf::ControlFlowDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateLLVMInterfaceForDialect<mlir::func::FuncDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateLLVMInterfaceForDialect<mlir::math::MathDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateLLVMInterfaceForDialect<mlir::memref::MemRefDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateLLVMInterfaceForDialect<mlir::ub::UBDialect>(
        patterns.getContext(), target, typeConverter, patterns);
    populateBasicOpsLoweringToLLVMConversionPatterns(typeConverter, patterns);
    target.addIllegalOp<
        ReussirPanicOp, ReussirExpectOp, ReussirCctxEmptyOp, ReussirCctxExtendOp,
        ReussirCctxApplyOp, ReussirTokenAllocOp, ReussirTokenFreeOp,
        ReussirTokenReinterpretOp, ReussirTokenReallocOp, ReussirRefLoadOp,
        ReussirRefStoreOp, ReussirRefSpilledOp, ReussirRefToMemrefOp,
        ReussirRefFromMemrefOp, ReussirRefDiffOp, ReussirRefCmpOp,
        ReussirRefMemcpyOp, ReussirNullableCheckOp, ReussirNullableCreateOp,
        ReussirNullableCoerceOp, ReussirRcIncOp, ReussirRcCreateOp,
        ReussirRcCreateCompoundOp, ReussirRcCreateVariantOp, ReussirRcTaggedOp,
        ReussirRcDecOp, ReussirRcBorrowOp, ReussirRcIsUniqueOp,
        ReussirRcAssumeUniqueOp, ReussirRecordCompoundOp,
        ReussirRecordVariantOp, ReussirRefProjectOp, ReussirArrayProjectOp,
        ReussirArrayViewOp, ReussirRecordTagOp, ReussirRecordExtractOp,
        ReussirRecordCoerceOp, ReussirRegionVTableOp, ReussirRcFreezeOp,
        ReussirRegionCleanupOp, ReussirRegionCreateOp, ReussirRcReinterpretOp,
        ReussirClosureApplyOp, ReussirClosureCloneOp, ReussirClosureEvalOp,
        ReussirClosureInspectPayloadOp, ReussirClosureCursorOp,
        ReussirClosureInstantiateOp, ReussirClosureVtableOp,
        ReussirClosureCreateOp, ReussirRcFetchOp, ReussirRcSetOp,
        ReussirStrGlobalOp, ReussirStrLiteralOp, ReussirStrCastOp,
        ReussirStrLenOp, ReussirStrUnsafeByteAtOp, ReussirStrUnsafeStartWithOp,
        ReussirStrSliceOp, ReussirTrampolineOp, ReussirTokenLaunderOp>();
  }
};

struct BasicOpsLoweringPass
    : public impl::ReussirBasicOpsLoweringPassBase<BasicOpsLoweringPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::LLVMTypeConverter converter(moduleOp.getContext(),
                                      getReussirToLLVMOptions(moduleOp));
    ensureRuntimeFunctions(moduleOp, converter);
    // Fused debug-info attributes are converted to LLVM DI by the separate
    // `reussir-lowering-debug-info` pass, which runs after `convert-to-llvm`
    // so the functions are already `llvm.func` (a function's DI only survives
    // translation once its `llvm.func` carries a `DISubprogram`).
  }
};

// The upstream `convert-to-llvm` with one change: the `LLVMTypeConverter` is
// built from the module's stamped data layout (`getReussirToLLVMOptions`),
// index width included, instead of default options. Every converter Reussir
// constructs itself derives the index width that way — most visibly
// `ensureRuntimeFunctions` above, which declares `__reussir_allocate` and
// friends at the index type — so driving the actual conversion with a
// default (64-bit index) converter breaks 32-bit-pointer targets: the
// declarations say `i32` while the emitted call sites say `i64`.
struct ConvertToLLVMPass
    : public impl::ReussirConvertToLLVMPassBase<ConvertToLLVMPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = moduleOp.getContext();
    mlir::LLVMTypeConverter converter(context,
                                      getReussirToLLVMOptions(moduleOp));
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // Collect the `ConvertToLLVMPatternInterface` patterns of every *loaded*
    // dialect, exactly as the upstream pass's static mode does. Walking the
    // module's ops instead (`populateConversionTargetFromOperation`) misses
    // dialects that appear only in *types* — e.g. a REPL module whose
    // externalized declarations carry `!reussir.rc` signatures but no
    // reussir ops would lose the Reussir type conversions.
    for (mlir::Dialect *dialect : context->getLoadedDialects()) {
      if (auto *iface =
              llvm::dyn_cast<mlir::ConvertToLLVMPatternInterface>(dialect))
        iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                       patterns);
    }
    if (failed(mlir::applyPartialConversion(moduleOp, target,
                                            std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void registerReussirBasicOpsLoweringInterface(mlir::DialectRegistry &registry) {
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  registry.addExtension(
      +[](mlir::MLIRContext *context, ReussirDialect *dialect) {
        dialect->addInterfaces<ReussirConvertToLLVMPatternInterface>();
      });
}

void populateBasicOpsLoweringToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<
      ReussirExpectConversionPattern,
      ReussirCctxEmptyConversionPattern, ReussirCctxExtendConversionPattern,
      ReussirCctxApplyConversionPattern,
      ReussirTokenAllocConversionPattern, ReussirTokenFreeConversionPattern,
      ReussirTokenReinterpretConversionPattern,
      ReussirTokenReallocConversionPattern, ReussirRefLoadConversionPattern,
      ReussirRefToMemrefConversionPattern,
      ReussirRefFromMemrefConversionPattern, ReussirRefStoreConversionPattern,
      ReussirRefSpilledConversionPattern, ReussirRefDiffConversionPattern,
      ReussirRefCmpConversionPattern, ReussirRefMemcpyConversionPattern,
      ReussirNullableCheckConversionPattern,
      ReussirNullableCreateConversionPattern,
      ReussirNullableCoerceConversionPattern, ReussirRcIncConversionPattern,
      ReussirRcTaggedConversionPattern, ReussirRcDecOpConversionPattern,
      ReussirRcCreateOpConversionPattern,
      ReussirRcCreateCompoundOpConversionPattern,
      ReussirRcCreateVariantOpConversionPattern,
      ReussirRcBorrowOpConversionPattern, ReussirRcIsUniqueOpConversionPattern,
      ReussirRcAssumeUniqueOpConversionPattern,
      ReussirRecordCompoundConversionPattern,
      ReussirRecordExtractConversionPattern,
      ReussirRecordVariantConversionPattern,
      ReussirReferenceProjectConversionPattern,
      ReussirArrayProjectConversionPattern, ReussirArrayViewConversionPattern,
      ReussirRecordTagConversionPattern, ReussirRecordCoerceConversionPattern,
      ReussirRegionVTableOpConversionPattern,
      ReussirRcFreezeOpConversionPattern,
      ReussirRegionCleanupOpConversionPattern,
      ReussirRegionCreateOpConversionPattern,
      ReussirClosureApplyOpConversionPattern,
      ReussirClosureCloneOpConversionPattern,
      ReussirClosureEvalOpConversionPattern,
      ReussirClosureInspectPayloadOpConversionPattern,
      ReussirClosureCursorOpConversionPattern,
      ReussirClosureTransferOpConversionPattern,
      ReussirClosureInstantiateOpConversionPattern,
      ReussirClosureVtableOpConversionPattern,
      ReussirClosureCreateOpConversionPattern,
      ReussirRcReinterpretConversionPattern, ReussirRcFetchConversionPattern,
      ReussirRcSetConversionPattern, ReussirStrGlobalOpConversionPattern,
      ReussirStrLiteralOpConversionPattern, ReussirPanicConversionPattern,
      ReussirStrCastOpConversionPattern, ReussirStrLenOpConversionPattern,
      ReussirStrUnsafeByteAtOpConversionPattern,
      ReussirStrUnsafeStartWithOpConversionPattern,
      ReussirStrSliceOpConversionPattern, ReussirTrampolineOpConversionPattern,
      ReussirTokenLaunderOpConversionPattern>(converter, patterns.getContext());
}
} // namespace reussir
