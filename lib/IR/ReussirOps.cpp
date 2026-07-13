//===-- ReussirOps.cpp - Reussir operations implementation ------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the operations used in the Reussir dialect.
//
//===----------------------------------------------------------------------===//
#include <array>
#include <optional>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Transformation/SpecialPointerTag.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/DenseSet.h>

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.cpp.inc"

namespace reussir {
namespace {

static mlir::MemRefType getArrayViewMemRefType(ArrayType arrayType) {
  return mlir::MemRefType::get(arrayType.getShape(),
                               arrayType.getElementType());
}

static mlir::RankedTensorType getArrayViewTensorType(ArrayType arrayType) {
  return mlir::RankedTensorType::get(arrayType.getShape(),
                                     arrayType.getElementType());
}

static mlir::LogicalResult verifyZeroRankMemRefType(mlir::Operation *op,
                                                    mlir::Type type,
                                                    mlir::Type elementType,
                                                    llvm::StringRef valueName) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 0)
    return op->emitOpError(valueName) << " must be a zero-rank memref";
  if (memrefType.getElementType() != elementType)
    return op->emitOpError(valueName)
           << " element type mismatch: expected " << elementType << ", got "
           << memrefType.getElementType();
  return mlir::success();
}

static mlir::LogicalResult verifyArrayViewType(mlir::Operation *op,
                                               mlir::Type type,
                                               ArrayType arrayType,
                                               llvm::StringRef valueName) {
  if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type)) {
    if (memrefType != getArrayViewMemRefType(arrayType))
      return op->emitOpError(valueName)
             << " type mismatch: expected " << getArrayViewMemRefType(arrayType)
             << ", got " << memrefType;
    return mlir::success();
  }

  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    if (tensorType != getArrayViewTensorType(arrayType))
      return op->emitOpError(valueName)
             << " type mismatch: expected " << getArrayViewTensorType(arrayType)
             << ", got " << tensorType;
    return mlir::success();
  }

  return op->emitOpError(valueName)
         << " must be a statically shaped memref or tensor";
}

mlir::LogicalResult verifyRcCreateLikeOp(mlir::Operation *op, RcType rcType,
                                         mlir::Type valueType,
                                         mlir::Value token,
                                         mlir::Value region) {
  if (valueType != rcType.getElementType())
    return op->emitOpError("value type must match RC element type, ")
           << "value type: " << valueType
           << ", RC element type: " << rcType.getElementType();
  Capability expectedCap = region == nullptr ? reussir::Capability::shared
                                             : reussir::Capability::flex;
  if (rcType.getCapability() != expectedCap)
    return op->emitOpError("RC type capability must be ")
           << stringifyCapability(expectedCap) << ", but got "
           << stringifyCapability(rcType.getCapability());

  if (!token)
    return mlir::success();

  // The op's own `TokenAcceptor::getTokenType()` is the single source of
  // truth for the box's layout — under per-constructor box sizing a
  // variant construction expects `header + arm[tag]`, not the uniform
  // max-arm width, and this check is what holds the
  // allocated == token == freed size invariant together.
  auto acceptor = llvm::cast<TokenAcceptor>(op);
  TokenType expected = acceptor.getTokenType();
  TokenType tokenType = llvm::cast<TokenType>(token.getType());
  if (tokenType.getAlign() != expected.getAlign())
    return op->emitOpError("token alignment must match RC type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", RC type alignment: " << expected.getAlign();
  if (tokenType.getSize() != expected.getSize())
    return op->emitOpError("token size must match RC type size, ")
           << "token size: " << tokenType.getSize()
           << ", RC type size: " << expected.getSize();
  return mlir::success();
}

mlir::LogicalResult
verifyRcCreateLikeSymbolUses(mlir::Operation *op, mlir::Value region,
                             mlir::Type valueType,
                             mlir::FlatSymbolRefAttr vtableAttr,
                             mlir::SymbolTableCollection &symbolTable) {
  if (!vtableAttr)
    return mlir::success();

  if (!region)
    return op->emitOpError(
        "when vtable is provided, region argument must also exist");

  auto vtableOp = symbolTable.lookupNearestSymbolFrom<ReussirRegionVTableOp>(
      op, vtableAttr);
  if (!vtableOp)
    return op->emitOpError("vtable symbol not found: ") << vtableAttr;

  mlir::Type vtableType = vtableOp.getTypeAttr().getValue();
  if (vtableType != valueType)
    return op->emitOpError(
               "vtable type attribute must match value input type, ")
           << "vtable type: " << vtableType << ", value type: " << valueType;
  return mlir::success();
}

mlir::LogicalResult verifyCompoundFields(mlir::Operation *op,
                                         RecordType recordType,
                                         mlir::ValueRange fields) {
  if (!recordType)
    return op->emitOpError("RC element type must be a record type");
  if (!recordType.getComplete())
    return op->emitOpError("cannot assemble incomplete compound record");
  if (!recordType.isCompound())
    return op->emitOpError("RC element type must be a compound record");
  if (recordType.getMembers().size() != fields.size())
    return op->emitOpError("number of fields must match number of members");
  for (auto [field, member, memberCapability] : llvm::zip(
           fields, recordType.getMembers(), recordType.getMemberIsField())) {
    mlir::Type projectedType =
        reussir::getProjectedType(member, memberCapability, Capability::flex);
    if (projectedType != field.getType())
      return op->emitOpError("field type must match projected member type, ")
             << "field type: " << field.getType()
             << ", projected member type: " << projectedType;
  }
  return mlir::success();
}

mlir::LogicalResult verifySkippedFields(mlir::Operation *op,
                                        size_t fieldCount) {
  auto skippedFields = op->getAttrOfType<mlir::DenseI64ArrayAttr>("skipFields");
  if (!skippedFields)
    return mlir::success();

  llvm::DenseSet<int64_t> seen;
  for (int64_t index : skippedFields.asArrayRef()) {
    if (index < 0 || static_cast<size_t>(index) >= fieldCount)
      return op->emitOpError("skipFields index out of bounds: ") << index;
    if (!seen.insert(index).second)
      return op->emitOpError("skipFields indices must be unique");
  }
  return mlir::success();
}

mlir::LogicalResult verifyHoleFields(mlir::Operation *op,
                                     mlir::TypeRange fieldTypes,
                                     mlir::TypeRange holeTypes) {
  auto holeFields = op->getAttrOfType<mlir::DenseI64ArrayAttr>("holeFields");
  if (!holeFields) {
    if (!holeTypes.empty())
      return op->emitOpError("hole results require holeFields attribute");
    return mlir::success();
  }

  if (static_cast<size_t>(holeFields.size()) != holeTypes.size())
    return op->emitOpError("hole result count must match holeFields count");

  llvm::DenseSet<int64_t> seen;
  for (auto [index, holeType] : llvm::zip(holeFields.asArrayRef(), holeTypes)) {
    if (index < 0 || static_cast<size_t>(index) >= fieldTypes.size())
      return op->emitOpError("holeFields index out of bounds: ") << index;
    if (!seen.insert(index).second)
      return op->emitOpError("holeFields indices must be unique");
    auto expectedHoleType =
        HoleType::get(op->getContext(), fieldTypes[static_cast<size_t>(index)]);
    if (holeType != expectedHoleType)
      return op->emitOpError("hole type must match selected field type, got: ")
             << holeType << ", expected: " << expectedHoleType;
  }
  return mlir::success();
}

} // namespace

///===----------------------------------------------------------------------===//
// ReussirTokenReinterpretOp
//===----------------------------------------------------------------------===//
// ReinterpretOp verification
mlir::LogicalResult ReussirTokenReinterpretOp::verify() {
  TokenType tokenType = getToken().getType();
  RefType resultType = getReinterpreted().getType();
  mlir::Type elementType = resultType.getElementType();
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(elementType);
  auto size = dataLayout.getTypeSize(elementType);
  if (!size.isFixed())
    return emitOpError("reinterpreted type must have a fixed size");
  if (tokenType.getAlign() != alignment)
    return emitOpError(
               "token alignment must match reinterpreted type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", element alignment: " << alignment;
  if (tokenType.getSize() != size)
    return emitOpError("token size must match reinterpreted type size, ")
           << "token size:  " << tokenType.getSize()
           << ", element size: " << size.getFixedValue();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir RC Operation
//===----------------------------------------------------------------------===//
// RcIncOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcIncOp::verify() {
  RcType RcType = getRcPtr().getType();
  if (RcType.getCapability() == reussir::Capability::flex)
    return emitOpError("cannot increase reference count of a flex RC type");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcReinterpretOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcReinterpretOp::verify() {
  RcType rcType = getRcPtr().getType();
  TokenType tokenType = getReinterpreted().getType();

  // Get the RC box type for the RC pointer
  RcBoxType rcBoxType = rcType.getInnerBoxType();

  // Get the data layout to compute alignment and size
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  auto size = dataLayout.getTypeSize(rcBoxType);

  if (!size.isFixed())
    return emitOpError("RC box type must have a fixed size");

  // Check that token alignment matches RC box alignment
  if (tokenType.getAlign() != alignment)
    return emitOpError("token alignment must match RC box alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", RC box alignment: " << alignment;

  // Check that token size matches RC box size. Under per-constructor box
  // sizing a fused-header variant box may be any arm's cell: a
  // destructuring dec reinterprets it at the statically known arm size, and
  // an unpinned dec at a *dynamic* size (`token<align, ?>`) carried at
  // runtime — accept both for such variants.
  if (tokenType.getSize() != size) {
    if (auto recordType = llvm::dyn_cast<RecordType>(rcType.getElementType());
        recordType && recordType.isVariant() && recordType.getComplete() &&
        rcBoxType.isHeaderFused() && !recordType.getFixed()) {
      if (tokenType.isDynamicSize())
        return mlir::success();
      for (size_t tag = 0; tag < recordType.getMembers().size(); ++tag)
        if (tokenType.getSize() ==
            recordType.getVariantArmAllocSize(dataLayout, tag))
          return mlir::success();
    }
    return emitOpError("token size must match RC box size, ")
           << "token size: " << tokenType.getSize()
           << ", RC box size: " << size.getFixedValue();
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// RcDecOp destructuring helpers
//===----------------------------------------------------------------------===//
std::pair<reussir::RecordType, mlir::Value>
ReussirRcDecOp::destructuredPayloadAndRef(mlir::OpBuilder &builder) {
  RcType rcType = getRcPtr().getType();
  auto recordType = llvm::cast<RecordType>(rcType.getElementType());
  auto refType = builder.getType<RefType>(
      recordType, Capability::unspecified, rcType.getAtomicKind());
  mlir::Value ref =
      ReussirRcBorrowOp::create(builder, getLoc(), refType, getRcPtr());
  if (!isVariantDestructuring())
    return {recordType, ref};
  int64_t tag = getDestructureTagAttr().getInt();
  auto payload = llvm::cast<RecordType>(recordType.getMembers()[tag]);
  auto payloadRefType = builder.getType<RefType>(
      payload, Capability::unspecified, rcType.getAtomicKind());
  mlir::Value coerced = ReussirRecordCoerceOp::create(
      builder, getLoc(), payloadRefType, builder.getIndexAttr(tag), ref);
  return {payload, coerced};
}

void ReussirRcDecOp::rematerializeBoundRetains(mlir::OpBuilder &builder) {
  if (getBoundMembersAttr().empty())
    return;
  RcType rcType = getRcPtr().getType();
  auto [payload, payloadRef] = destructuredPayloadAndRef(builder);
  for (int64_t idx : getBoundMembersAttr().asArrayRef()) {
    auto projectedTy =
        getProjectedType(payload.getMembers()[idx],
                         payload.getMemberIsField()[idx],
                         Capability::unspecified);
    auto projectedRefTy = builder.getType<RefType>(
        projectedTy, Capability::unspecified, rcType.getAtomicKind());
    mlir::Value slot = ReussirRefProjectOp::create(
        builder, getLoc(), projectedRefTy, payloadRef,
        builder.getIndexAttr(idx));
    // A shared member retains through a count bump on the loaded pointer; a
    // value-record member retains its embedded children through the slot's
    // acquire (expanded member-wise later); anything trivially copyable
    // needs nothing.
    if (llvm::isa<RcType>(projectedTy)) {
      mlir::Value member = ReussirRefLoadOp::create(builder, getLoc(),
                                                    projectedTy, slot);
      ReussirRcIncOp::create(builder, getLoc(), member);
    } else if (!isTriviallyCopyable(projectedTy)) {
      ReussirRefAcquireOp::create(builder, getLoc(), slot);
    }
  }
}

//===----------------------------------------------------------------------===//
// RcDecOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcDecOp::verify() {
  RcType RcType = getRcPtr().getType();
  if (RcType.getCapability() == reussir::Capability::flex)
    return emitOpError("cannot decrease reference count of a flex RC type");
  if (getDestructureTagAttr() && !getBoundMembersAttr())
    return emitOpError("destructureTag requires boundMembers");
  if (isDestructuring()) {
    if (RcType.getAtomicKind() != AtomicKind::normal)
      return emitOpError("destructuring decrement requires a nonatomic box");
    if (RcType.isRegional())
      return emitOpError("destructuring decrement must not be regional");
    auto recordType = llvm::dyn_cast<RecordType>(RcType.getElementType());
    if (!recordType || !recordType.getComplete())
      return emitOpError(
          "destructuring decrement requires a complete record box");
    RecordType payload = recordType;
    if (isVariantDestructuring()) {
      if (!recordType.isVariant())
        return emitOpError(
            "tagged destructuring decrement requires a variant box");
      int64_t tag = getDestructureTagAttr().getInt();
      if (tag < 0 ||
          static_cast<size_t>(tag) >= recordType.getMembers().size())
        return emitOpError("destructure tag out of range: ") << tag;
      payload = llvm::dyn_cast<RecordType>(recordType.getMembers()[tag]);
      if (!payload || !payload.getComplete())
        return emitOpError("destructured arm must have a complete payload");
    } else if (!recordType.isCompound()) {
      return emitOpError(
          "tagless destructuring decrement requires a compound box");
    }
    for (int64_t index : getBoundMembersAttr().asArrayRef())
      if (index < 0 ||
          static_cast<size_t>(index) >= payload.getMembers().size())
        return emitOpError("bound member index out of range: ") << index;
  }
  if (RcType.getCapability() == reussir::Capability::rigid) {
    if (getNullableToken() != nullptr)
      return emitOpError("rigid RC decrement cannot return a token");
    return mlir::success();
  }
  if (getNullableToken() == nullptr)
    return mlir::success();
  if (!shouldProduceToken())
    return emitOpError("this RC decrement cannot produce a token");
  NullableType nullableType = getNullableToken().getType();
  TokenType tokenType = llvm::dyn_cast<TokenType>(nullableType.getPtrTy());
  if (!tokenType)
    return emitOpError("nullable token must be of TokenType");
  // `getTokenType()` is the single source of truth for the box's layout.
  // Under per-constructor box sizing a fused-header variant dec's
  // token is valid at any of: the exact arm cell (a pinned/destructuring
  // dec), the dynamic runtime size (`token<align, ?>`, an unpinned dec), or
  // the uniform max-arm width — and a preceding inc/dec-cancellation may add
  // or drop the destructuring tag after the token was typed, so accept the
  // whole valid set rather than the single current `getTokenType()`.
  TokenType expected = getTokenType();
  if (tokenType.getAlign() != expected.getAlign())
    return emitOpError("token alignment must match managed type alignment, ")
           << "token alignment: " << tokenType.getAlign()
           << ", element alignment: " << expected.getAlign();

  if (tokenType.getSize() != expected.getSize()) {
    auto dataLayout = mlir::DataLayout::closest(getOperation());
    RcBoxType rcBoxType = RcType.getInnerBoxType();
    if (auto recordType = llvm::dyn_cast<RecordType>(RcType.getElementType());
        recordType && recordType.isVariant() && recordType.getComplete() &&
        rcBoxType.isHeaderFused() && !recordType.getFixed()) {
      if (tokenType.isDynamicSize())
        return mlir::success();
      for (size_t tag = 0; tag < recordType.getMembers().size(); ++tag)
        if (tokenType.getSize() ==
            recordType.getVariantArmAllocSize(dataLayout, tag))
          return mlir::success();
    }
    return emitOpError("token size must match managed type size, ")
           << "token size: " << tokenType.getSize()
           << ", element size: " << expected.getSize();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcDecOp TokenProducer Interface
//===----------------------------------------------------------------------===//
bool ReussirRcDecOp::shouldProduceToken() {
  RcType rcType = getRcPtr().getType();
  // Only shared capability RC pointers produce tokens
  return rcType.getCapability() == reussir::Capability::shared &&
         !mlir::isa<FFIObjectType, ClosureType>(rcType.getElementType());
}

//===----------------------------------------------------------------------===//
// RcDecOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult
ReussirRcDecOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  RcType rcType = getRcPtr().getType();
  if (auto eleTy = mlir::dyn_cast<FFIObjectType>(rcType.getElementType())) {
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
        getOperation(), eleTy.getCleanupHook());
    if (!funcOp)
      return emitOpError("cleanup hook not found: ") << eleTy.getCleanupHook();
    if (funcOp.getFunctionType().getNumInputs() != 1 ||
        funcOp.getFunctionType().getNumResults() != 0 ||
        funcOp.getFunctionType().getInput(0) != rcType)
      return emitOpError(
          "cleanup hook must be a function with one argument and no return "
          "type, and the argument type must match the RC element type");
  }
  return mlir::success();
}

bool ReussirRcDecOp::isNullable() {
  // Always returns nullable tokens
  return true;
}

TokenType ReussirRcDecOp::getTokenType() {
  RcType rcType = getRcPtr().getType();
  RcBoxType rcBoxType = rcType.getInnerBoxType();

  // Compute token type as in verify
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  // Per-constructor box sizing (default; `fixed` variants opt out). For a
  // fused-header
  // variant box, the unique-path token — the cell handed to reuse or freed —
  // must match the arm actually allocated:
  //   * a *destructuring* dec knows the arm the pattern match consumed, so
  //     its token is that arm's exact per-constructor cell (static);
  //   * any other variant dec cannot know the arm statically, so its token
  //     has a *dynamic* size (`token<align, ?>`) carried at runtime — its
  //     free/realloc read the exact size, sound on any allocator.
  if (auto recordType = llvm::dyn_cast<RecordType>(rcType.getElementType());
      recordType && recordType.isVariant() && recordType.getComplete() &&
      rcBoxType.isHeaderFused() && !recordType.getFixed()) {
    if (isVariantDestructuring()) {
      llvm::TypeSize armSize = recordType.getVariantArmAllocSize(
          dataLayout, getDestructureTagAttr().getInt());
      return TokenType::get(getContext(), alignment, armSize.getFixedValue());
    }
    // A variant whose *token-producing* arms are all the same size stays a
    // *static* token even when unpinned: that size is exact for every box
    // the dec can ever free, so it reuses and frees normally. Only a
    // genuinely non-uniform variant needs the dynamic (runtime-carried)
    // size. Crucially, an arm only counts if a box of it can exist: under
    // the special-pointer-tag scheme (module layout info stamped by the
    // pass, which runs before token instantiation) a nullary arm of a
    // taggable type is constructed as an unboxed immediate — it never
    // allocates and its decrement never yields a token — so its phantom
    // size must not force the dynamic path. E.g. `Tree { Branch(...), Leaf }`
    // has exactly one real arm and keeps a static token, preserving in-place
    // reuse instead of a runtime realloc per node (the 2x rbtree regression
    // this rule fixes).
    mlir::ModuleOp module = (*this)->getParentOfType<mlir::ModuleOp>();
    bool nullaryImmediates = module && module->hasAttr(kSpecialPtrTagAttr) &&
                             rcType.mayCarrySpecialPointerTag();
    // An arm cannot produce a token exactly when the scheme is enabled and
    // its constructions are rewritten to immediates — which for a taggable
    // type (`mayCarrySpecialPointerTag`, folded into `nullaryImmediates`
    // above and guaranteeing the tag-slot range at the type level) is
    // simply every *nullary* arm. Same predicate as the rewrite pattern, so
    // the two can never drift apart; encoding-agnostic (the arch/encoding
    // choice was resolved when the pass stamped the module attribute).
    auto producesToken = [&](size_t tag) {
      return !(nullaryImmediates && recordType.isNullaryArm(tag));
    };
    std::optional<uint64_t> tokenArmSize;
    bool uniform = true;
    for (size_t tag = 0, n = recordType.getMembers().size(); tag < n; ++tag) {
      if (!producesToken(tag))
        continue;
      uint64_t armSize =
          recordType.getVariantArmAllocSize(dataLayout, tag).getFixedValue();
      if (!tokenArmSize) {
        tokenArmSize = armSize;
      } else if (*tokenArmSize != armSize) {
        uniform = false;
        break;
      }
    }
    if (uniform && tokenArmSize)
      return TokenType::get(getContext(), alignment, *tokenArmSize);
    if (!tokenArmSize) {
      // Every arm is an immediate: no token is ever produced at runtime; the
      // declared type is moot — keep the uniform box size (always static).
      auto boxSize = dataLayout.getTypeSize(rcBoxType);
      return TokenType::get(getContext(), alignment, boxSize.getFixedValue());
    }
    return TokenType::getDynamic(getContext(), alignment);
  }
  auto size = dataLayout.getTypeSize(rcBoxType);

  return TokenType::get(getContext(), alignment, size.getFixedValue());
}

mlir::Value ReussirRcDecOp::getProducedValue() { return getNullableToken(); }

mlir::LogicalResult
ReussirRcDecOp::replaceWithProduced(mlir::PatternRewriter &builder) {
  // Return early if already produced a token
  if (getNullableToken() != nullptr)
    return mlir::failure();

  mlir::OpBuilder::InsertionGuard guard(builder);

  // Compute the token type
  TokenType tokenType = getTokenType();
  NullableType nullableTokenType = NullableType::get(getContext(), tokenType);

  // Create a new RcDecOp with the token result
  builder.setInsertionPoint(getOperation());
  builder.replaceOpWithNewOp<ReussirRcDecOp>(getOperation(),
                                             nullableTokenType, // result type
                                             getRcPtr()         // operand
  );

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// RcCreateOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcCreateOp::verify() {
  return verifyRcCreateLikeOp(getOperation(), getRcPtr().getType(),
                              getValue().getType(), getToken(), getRegion());
}

//===----------------------------------------------------------------------===//
// RcCreateOp TokenAcceptor Interface
//===----------------------------------------------------------------------===//
TokenType ReussirRcCreateOp::getTokenType() {
  auto rcBoxType = RcBoxType::get(getContext(), getValue().getType(),
                                  getRegion() != nullptr);
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  // Per-constructor box sizing (default; `fixed` variants opt out): tokens
  // are attached
  // while constructions are still the unfused `rc.create(record.variant)`
  // chain (RcCreateFusion runs after token instantiation and reuse), so
  // the static arm is read off the variant producer: the box only needs
  // `header + arm[tag]`. The fused `rc.create_variant` this later becomes
  // reports the same size, keeping the token verifier's invariant.
  if (auto recordType = llvm::dyn_cast<RecordType>(getValue().getType());
      recordType && recordType.isVariant() && recordType.getComplete() &&
      rcBoxType.isHeaderFused() && !getRegion() && !recordType.getFixed()) {
    if (auto variant = llvm::dyn_cast_if_present<ReussirRecordVariantOp>(
            getValue().getDefiningOp())) {
      llvm::TypeSize armSize = recordType.getVariantArmAllocSize(
          dataLayout, variant.getTag().getZExtValue());
      return TokenType::get(getContext(), alignment, armSize.getFixedValue());
    }
  }
  auto size = dataLayout.getTypeSize(rcBoxType);
  return TokenType::get(getContext(), alignment, size.getFixedValue());
}

//===----------------------------------------------------------------------===//
// RcCreateOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult
ReussirRcCreateOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  return verifyRcCreateLikeSymbolUses(getOperation(), getRegion(),
                                      getValue().getType(), getVtableAttr(),
                                      symbolTable);
}

//===----------------------------------------------------------------------===//
// RcCreateCompoundOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcCreateCompoundOp::verify() {
  RecordType recordType = getRecordType();
  if (failed(verifyCompoundFields(getOperation(), recordType, getFields())))
    return mlir::failure();
  if (failed(verifySkippedFields(getOperation(), getFields().size())))
    return mlir::failure();
  if (failed(verifyHoleFields(getOperation(), getFields().getTypes(),
                              getHoles().getTypes())))
    return mlir::failure();
  return verifyRcCreateLikeOp(getOperation(), getRcPtr().getType(), recordType,
                              getToken(), getRegion());
}

//===----------------------------------------------------------------------===//
// RcCreateCompoundOp TokenAcceptor Interface
//===----------------------------------------------------------------------===//
TokenType ReussirRcCreateCompoundOp::getTokenType() {
  auto rcBoxType =
      RcBoxType::get(getContext(), getRcPtr().getType().getElementType(),
                     getRegion() != nullptr);
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  auto size = dataLayout.getTypeSize(rcBoxType);
  return TokenType::get(getContext(), alignment, size.getFixedValue());
}

//===----------------------------------------------------------------------===//
// RcCreateCompoundOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcCreateCompoundOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  return verifyRcCreateLikeSymbolUses(getOperation(), getRegion(),
                                      getRcPtr().getType().getElementType(),
                                      getVtableAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// RcCreateVariantOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcTaggedOp::verify() {
  RcType rcType = getRcPtr().getType();
  if (rcType.getCapability() != Capability::shared &&
      rcType.getCapability() != Capability::unspecified)
    return emitOpError("tagged immediates require a plain shared box");
  if (rcType.getAtomicKind() != AtomicKind::normal)
    return emitOpError("tagged immediates require a nonatomic box");
  auto variantType = llvm::dyn_cast<RecordType>(rcType.getElementType());
  if (!variantType || !variantType.isVariant() || !variantType.getComplete())
    return emitOpError("RC element type must be a complete variant record");
  size_t tag = getTag().getZExtValue();
  if (tag >= variantType.getMembers().size())
    return emitOpError("tag out of bounds");
  // The encoding stores `tag + 1` in the pointer's top byte.
  if (tag + 1 > 0xFF)
    return emitOpError("tag does not fit the top-byte encoding");
  auto arm = llvm::dyn_cast<RecordType>(variantType.getMembers()[tag]);
  if (!arm || !arm.isCompound() || !arm.getMembers().empty())
    return emitOpError("tagged immediates only encode nullary variants");
  return mlir::success();
}

mlir::LogicalResult ReussirRcCompareImmortalOp::verify() {
  RcType rcType = getRcPtr().getType();
  auto variantType = llvm::dyn_cast<RecordType>(rcType.getElementType());
  if (!variantType || !variantType.isVariant() || !variantType.getComplete())
    return emitOpError("RC element type must be a complete variant record");
  size_t tag = getTag().getZExtValue();
  if (tag >= variantType.getMembers().size())
    return emitOpError("tag out of bounds");
  auto arm = llvm::dyn_cast<RecordType>(variantType.getMembers()[tag]);
  if (!arm || !arm.isCompound() || !arm.getMembers().empty())
    return emitOpError("only nullary variant arms have immediates");
  if (!rcType.mayCarrySpecialPointerTag())
    return emitOpError("the box type cannot carry a special pointer tag");
  return mlir::success();
}

mlir::LogicalResult ReussirRcCreateVariantOp::verify() {
  RecordType variantType = getRecordType();
  if (!variantType)
    return emitOpError("RC element type must be a record type");
  if (!variantType.getComplete())
    return emitOpError("cannot assemble incomplete variant record");
  if (!variantType.isVariant())
    return emitOpError("RC element type must be a variant record");

  size_t tag = getTag().getZExtValue();
  if (tag >= variantType.getMembers().size())
    return emitOpError("tag out of bounds");

  mlir::Type targetVariantType = variantType.getMembers()[tag];
  bool targetVariantIsField = variantType.getMemberIsField()[tag];
  if (targetVariantIsField)
    return emitOpError("TODO: check this is nested in a region operation");

  bool hasValue = getValue() != nullptr;
  bool hasFields = !getFields().empty();
  if (hasValue == hasFields) {
    if (!hasValue && !hasFields && mlir::isa<RecordType>(targetVariantType)) {
      auto compoundType = llvm::cast<RecordType>(targetVariantType);
      if (compoundType.isCompound() && compoundType.getMembers().empty())
        hasFields = true;
    }
  }
  if (hasValue == hasFields)
    return emitOpError(
        "expected exactly one payload form: either a value or compound fields");

  if (hasValue) {
    mlir::Type projectedType = reussir::getProjectedType(
        targetVariantType, targetVariantIsField, Capability::flex);
    if (projectedType != getValue().getType())
      return emitOpError("value type must match projected type, ")
             << "value type: " << getValue().getType()
             << ", projected type: " << projectedType;
    if (getHoleFields())
      return emitOpError("holeFields require compound fields payload form");
    if (!getHoles().empty())
      return emitOpError("hole results require compound fields payload form");
  } else {
    auto compoundType = llvm::dyn_cast<RecordType>(targetVariantType);
    if (!compoundType || !compoundType.isCompound())
      return emitOpError("compound fields payload requires the selected "
                         "variant member to be a compound record");
    if (failed(verifyCompoundFields(getOperation(), compoundType, getFields())))
      return mlir::failure();
    if (failed(verifyHoleFields(getOperation(), getFields().getTypes(),
                                getHoles().getTypes())))
      return mlir::failure();
  }
  if (failed(verifySkippedFields(getOperation(), getFields().size())))
    return mlir::failure();

  return verifyRcCreateLikeOp(getOperation(), getRcPtr().getType(), variantType,
                              getToken(), getRegion());
}

//===----------------------------------------------------------------------===//
// RcCreateVariantOp TokenAcceptor Interface
//===----------------------------------------------------------------------===//
TokenType ReussirRcCreateVariantOp::getTokenType() {
  mlir::Type elementType = getRcPtr().getType().getElementType();
  auto rcBoxType =
      RcBoxType::get(getContext(), elementType, getRegion() != nullptr);
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  // Per-constructor box sizing (default; `fixed` variants opt out): a
  // fused-header
  // variant box IS its record, and this op constructs a statically known
  // arm — size the token for `header + arm[tag]` instead of the max-arm
  // width. Alignment (hence the payload offset and every field offset) is
  // unchanged. Regional boxes keep the uniform size: their bump-allocation
  // header/lifecycle is handled separately (phase 4 of the landing plan).
  if (auto recordType = llvm::dyn_cast<RecordType>(elementType);
      recordType && recordType.isVariant() && recordType.getComplete() &&
      rcBoxType.isHeaderFused() && !getRegion() && !recordType.getFixed()) {
    llvm::TypeSize armSize =
        recordType.getVariantArmAllocSize(dataLayout, getTag().getZExtValue());
    return TokenType::get(getContext(), alignment, armSize.getFixedValue());
  }
  auto size = dataLayout.getTypeSize(rcBoxType);
  return TokenType::get(getContext(), alignment, size.getFixedValue());
}

//===----------------------------------------------------------------------===//
// RcCreateVariantOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcCreateVariantOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  return verifyRcCreateLikeSymbolUses(getOperation(), getRegion(),
                                      getRcPtr().getType().getElementType(),
                                      getVtableAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// Reussir Borrow Operation
//===----------------------------------------------------------------------===//
// BorrowOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcBorrowOp::verify() {
  RcType rcType = getRcPtr().getType();
  RefType refType = getBorrowed().getType();

  // Allow unspecified capability or matching capability
  if (refType.getCapability() != reussir::Capability::unspecified &&
      refType.getCapability() != rcType.getCapability())
    return emitOpError("borrowed type capability must be unspecified or match "
                       "RC type capability, ")
           << "borrowed type capability: "
           << stringifyCapability(refType.getCapability())
           << ", RC type capability: "
           << stringifyCapability(rcType.getCapability());

  if (refType.getElementType() != rcType.getElementType())
    return emitOpError(
               "borrowed type element type must match RC element type, ")
           << "borrowed type element type: " << refType.getElementType()
           << ", RC element type: " << rcType.getElementType();

  if (refType.getAtomicKind() != rcType.getAtomicKind())
    return emitOpError(
               "borrowed type atomic kind must match RC type atomic kind, ")
           << "borrowed type atomic kind: "
           << stringifyAtomicKind(refType.getAtomicKind())
           << ", RC type atomic kind: "
           << stringifyAtomicKind(rcType.getAtomicKind());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcFreezeOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcFreezeOp::verify() {
  RcType inputRcType = getRcPtr().getType();
  RcType outputRcType = getFrozen().getType();

  // Check that input RC has flex capability
  if (inputRcType.getCapability() != reussir::Capability::flex)
    return emitOpError("input RC pointer must have flex capability, ")
           << "got: " << stringifyCapability(inputRcType.getCapability());

  // Check that output RC has rigid capability
  if (outputRcType.getCapability() != reussir::Capability::rigid)
    return emitOpError("output RC pointer must have rigid capability, ")
           << "got: " << stringifyCapability(outputRcType.getCapability());

  // Check that element types match
  if (inputRcType.getElementType() != outputRcType.getElementType())
    return emitOpError("input and output RC element types must match, ")
           << "input element type: " << inputRcType.getElementType()
           << ", output element type: " << outputRcType.getElementType();

  // Check that atomic kinds match
  if (inputRcType.getAtomicKind() != outputRcType.getAtomicKind())
    return emitOpError("input and output RC atomic kinds must match, ")
           << "input atomic kind: "
           << stringifyAtomicKind(inputRcType.getAtomicKind())
           << ", output atomic kind: "
           << stringifyAtomicKind(outputRcType.getAtomicKind());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcIsUniqueOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcIsUniqueOp::verify() {
  RcType rcType = getRcPtr().getType();

  // Check that RC pointer is not regional (i.e., has shared capability)
  if (rcType.isRegional())
    return emitOpError("isUnique can only be applied to non-regional RC "
                       "(shared capability), ")
           << "got: " << stringifyCapability(rcType.getCapability());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RcAssumeUniqueOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRcAssumeUniqueOp::verify() {
  RcType rcType = getRcPtr().getType();

  if (rcType.isRegional())
    return emitOpError("assumeUnique can only be applied to non-regional RC "
                       "(shared capability), ")
           << "got: " << stringifyCapability(rcType.getCapability());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Constructor Context Operations
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirCctxExtendOp::verify() {
  RcType eleTy = getCtx().getType().getElementType();
  if (getNewCtx().getType() != getCtx().getType())
    return emitOpError("extended context type must match the incoming "
                       "context type, got ")
           << getNewCtx().getType() << " from " << getCtx().getType();
  if (getRcPtr().getType() != eleTy)
    return emitOpError("plugged constructor type must match the context "
                       "element type, got ")
           << getRcPtr().getType() << " for " << eleTy;
  if (getHole().getType().getElementType() != eleTy)
    return emitOpError("refocused hole must hold the context element type, "
                       "got ")
           << getHole().getType().getElementType() << " for " << eleTy;
  return mlir::success();
}

mlir::LogicalResult ReussirCctxApplyOp::verify() {
  RcType eleTy = getCtx().getType().getElementType();
  if (getValue().getType() != eleTy)
    return emitOpError("applied value type must match the context element "
                       "type, got ")
           << getValue().getType() << " for " << eleTy;
  if (getResult().getType() != eleTy)
    return emitOpError("result type must match the context element type, "
                       "got ")
           << getResult().getType() << " for " << eleTy;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Record Operations
//===----------------------------------------------------------------------===//
// RecordCompoundOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordCompoundOp::verify() {
  auto compoundType = getCompound().getType();
  if (!compoundType.getComplete())
    return emitOpError("cannot assemble incomplete compound record");
  if (!compoundType.isCompound())
    return emitOpError("compound type must be a compound record");
  if (compoundType.getMembers().size() != getFields().size())
    return emitOpError("number of fields must match number of members");
  for (auto [field, member, memberCapability] :
       llvm::zip(getFields(), compoundType.getMembers(),
                 compoundType.getMemberIsField())) {
    // Since this is assemble phase, assume flex ref capability.
    mlir::Type projectedType =
        reussir::getProjectedType(member, memberCapability, Capability::flex);
    if (projectedType != field.getType())
      return emitOpError("field type must match projected member type, ")
             << "field type: " << field.getType()
             << ", projected member type: " << projectedType;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Record Variant Op
//===----------------------------------------------------------------------===//
// RecordVariantOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordVariantOp::verify() {
  auto variantType = getVariant().getType();
  if (!variantType.getComplete())
    return emitOpError("cannot assemble incomplete variant record");
  if (!variantType.isVariant())
    return emitOpError("variant type must be a variant record");
  size_t tag = getTag().getZExtValue();
  if (tag >= variantType.getMembers().size())
    return emitOpError("tag out of bounds");
  mlir::Type targetVariantType = variantType.getMembers()[tag];
  bool targetVariantIsIsField = variantType.getMemberIsField()[tag];
  mlir::Type projectedType = reussir::getProjectedType(
      targetVariantType, targetVariantIsIsField, Capability::flex);
  if (projectedType != getValue().getType())
    return emitOpError("value type must match projected type, ")
           << "value type: " << getValue().getType()
           << ", projected type: " << projectedType;
  if (targetVariantIsIsField)
    return emitOpError("TODO: check this is nested in a region operation");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RecordTagOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordTagOp::verify() {
  RefType variantType = getVariant().getType();

  // Check that the input is a reference to a record type
  mlir::Type elementType = variantType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
  if (!recordType)
    return emitOpError("input must be a reference to a record type, got: ")
           << elementType;

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot get tag of incomplete record");

  // Check that the record is a variant record
  if (!recordType.isVariant())
    return emitOpError("can only get tag of variant records");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Record Coerce Op
//===----------------------------------------------------------------------===//
// RecordCoerceOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordCoerceOp::verify() {
  RefType variantRefType = getVariant().getType();
  RefType coercedRefType = getCoerced().getType();

  // Check that the input reference is a reference to a record type
  mlir::Type variantElementType = variantRefType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(variantElementType);
  if (!recordType)
    return emitOpError("input must be a reference to a record type, got: ")
           << variantElementType;

  // Check that the record is a variant record
  if (!recordType.isVariant())
    return emitOpError("input must be a reference to a variant record");

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot coerce incomplete variant record");

  // Get the tag and validate it's within bounds
  size_t tag = getTag().getZExtValue();
  if (tag >= recordType.getMembers().size())
    return emitOpError("tag out of bounds: ")
           << tag << " >= " << recordType.getMembers().size();

  // Get the target variant element type at the specified tag position
  mlir::Type targetVariantElementType = getProjectedType(
      recordType.getMembers()[tag], recordType.getMemberIsField()[tag],
      variantRefType.getCapability());

  // Check that the output reference element type matches the target variant
  // element type
  mlir::Type coercedElementType = coercedRefType.getElementType();
  if (coercedElementType != targetVariantElementType)
    return emitOpError("output reference element type must match target "
                       "variant element type, ")
           << "expected: " << targetVariantElementType
           << ", got: " << coercedElementType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Array Operations
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirArrayViewOp::verify() {
  RefType refType = getRef().getType();
  ArrayType arrayType = llvm::dyn_cast<ArrayType>(refType.getElementType());
  if (!arrayType)
    return emitOpError(
        "array.view input must be a reference to a reussir.array");
  return verifyArrayViewType(getOperation(), getView().getType(), arrayType,
                             "array.view result");
}

mlir::LogicalResult ReussirArrayProjectOp::verify() {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(getView().getType());
  if (!memrefType)
    return emitOpError(
        "array.project input must be a statically shaped memref");
  if (!memrefType.hasStaticShape() || !memrefType.getLayout().isIdentity())
    return emitOpError("array.project input memref must have a static "
                       "identity-layout type");
  ArrayType arrayType = ArrayType::get(getContext(), memrefType.getShape(),
                                       memrefType.getElementType());
  if (arrayType.getRank() == 0)
    return emitOpError("array view must have at least one extent");

  if (arrayType.getRank() == 1) {
    RefType projectedType = llvm::dyn_cast<RefType>(getProjected().getType());
    if (!projectedType)
      return emitOpError("projecting the last array dimension must produce a "
                         "reference result");
    if (projectedType.getElementType() != arrayType.getElementType())
      return emitOpError("projected reference element type mismatch: expected ")
             << arrayType.getElementType() << ", got "
             << projectedType.getElementType();
    if (projectedType.getCapability() != Capability::field &&
        projectedType.getCapability() != Capability::unspecified)
      return emitOpError("projected reference capability must be field or "
                         "unspecified, got ")
             << stringifyCapability(projectedType.getCapability());
    return mlir::success();
  }

  auto projectedMemRefType =
      llvm::dyn_cast<mlir::MemRefType>(getProjected().getType());
  if (!projectedMemRefType)
    return emitOpError("projecting a non-final array dimension must produce "
                       "another statically shaped memref");
  if (projectedMemRefType != getArrayViewMemRefType(arrayType.dropFront()))
    return emitOpError("projected subview type mismatch: expected ")
           << getArrayViewMemRefType(arrayType.dropFront()) << ", got "
           << projectedMemRefType;
  return mlir::success();
}

mlir::LogicalResult ReussirArrayWithUniqueViewOp::verify() {
  RcType rcType = getArray().getType();
  ArrayType arrayType = llvm::dyn_cast<ArrayType>(rcType.getElementType());
  if (!arrayType)
    return emitOpError("input must be an RC array");

  mlir::Region &body = getBody();
  if (body.empty())
    return emitOpError("body region must not be empty");
  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1)
    return emitOpError("body region must take exactly one view argument");
  if (block.empty() || !llvm::isa<ReussirScfYieldOp>(block.getTerminator()))
    return emitOpError("body region must terminate with reussir.scf.yield");

  if (failed(verifyArrayViewType(getOperation(), block.getArgument(0).getType(),
                                 arrayType, "body argument view")))
    return mlir::failure();

  auto yieldOp = llvm::cast<ReussirScfYieldOp>(block.getTerminator());
  if (getResult() && yieldOp.getNumOperands() == 0 &&
      getResult().getType() != getArray().getType())
    return emitOpError("implicit array result is only valid when the result "
                       "type matches the input RC array type");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Reference Operations
//===----------------------------------------------------------------------===//
// ReferenceProjectOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefProjectOp::verify() {
  RefType refType = getRef().getType();
  RefType projectedType = getProjected().getType();

  // Check that the reference element type is a record type
  mlir::Type elementType = refType.getElementType();
  RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
  if (!recordType)
    return emitOpError("reference element type must be a record type, got: ")
           << elementType;

  // Check that the record is complete
  if (!recordType.getComplete())
    return emitOpError("cannot project into incomplete record");

  // Check that the index is within bounds
  size_t index = getIndex().getZExtValue();
  if (index >= recordType.getMembers().size())
    return emitOpError("index out of bounds: ")
           << index << " >= " << recordType.getMembers().size();

  // Get the member type and capability at the specified index
  mlir::Type memberType = recordType.getMembers()[index];
  bool memberIsField = recordType.getMemberIsField()[index];

  // Calculate the expected projected type based on the member type, member
  // capability, and reference capability
  mlir::Type expectedProjectedType = reussir::getProjectedType(
      memberType, memberIsField, refType.getCapability());

  // Check that the projected type matches the expected type
  if (expectedProjectedType != projectedType.getElementType())
    return emitOpError("projected type mismatch: expected ")
           << expectedProjectedType << ", got "
           << projectedType.getElementType();

  // Check that the projected reference has the same capability as the
  // original reference. Or if the refType is flex, then the projected type
  // can be field if target is field.
  bool isOfSameCapability =
      projectedType.getCapability() == refType.getCapability();
  bool projectFieldOutOfFlex =
      projectedType.getCapability() == Capability::field &&
      refType.getCapability() == Capability::flex && memberIsField;
  if (!isOfSameCapability && !projectFieldOutOfFlex)
    return emitOpError(
               "projected reference capability must match original "
               "reference capability or be a field projection: original "
               "capability: ")
           << stringifyCapability(refType.getCapability()) << ", got "
           << stringifyCapability(projectedType.getCapability());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefSpilledOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefSpilledOp::verify() {
  mlir::Type valueType = getValue().getType();
  RefType refType = getSpilled().getType();
  if (valueType != refType.getElementType())
    return emitOpError("value type must match spilled element type, ")
           << "value type: " << valueType
           << ", spilled element type: " << refType.getElementType();
  if (refType.getCapability() != reussir::Capability::unspecified)
    return emitOpError("spilled type capability must be unspecified, ")
           << "spilled type capability: "
           << stringifyCapability(refType.getCapability());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefToMemrefOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefToMemrefOp::verify() {
  RefType refType = getRef().getType();
  return verifyZeroRankMemRefType(getOperation(), getView().getType(),
                                  refType.getElementType(), "view result");
}

//===----------------------------------------------------------------------===//
// RefFromMemrefOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefFromMemrefOp::verify() {
  RefType refType = getRef().getType();
  return verifyZeroRankMemRefType(getOperation(), getView().getType(),
                                  refType.getElementType(), "view input");
}

//===----------------------------------------------------------------------===//
// RefLoadOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefLoadOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type valueType = getValue().getType();
  if (valueType != refType.getElementType())
    return emitOpError("value type must match reference element type, ")
           << "value type: " << valueType
           << ", reference element type: " << refType.getElementType();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefStoreOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefStoreOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type valueType = getValue().getType();

  // Check that the target reference has field capability
  if (refType.getCapability() != reussir::Capability::field)
    return emitOpError("target reference must have field capability, got: ")
           << stringifyCapability(refType.getCapability());

  // Check that the value type matches the reference element type
  if (valueType != refType.getElementType())
    return emitOpError("value type must match reference element type, ")
           << "value type: " << valueType
           << ", reference element type: " << refType.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefMemcpyOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefMemcpyOp::verify() {
  RefType srcType = getSrc().getType();
  RefType dstType = getDst().getType();

  // Check that element types are identical
  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError(
               "source and destination element types must be identical, ")
           << "source element type: " << srcType.getElementType()
           << ", destination element type: " << dstType.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Region Operations
//===----------------------------------------------------------------------===//
// RegionRunOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRegionRunOp::verify() {
  // - Check that the region has exactly one argument of !reussir.region type
  // - Check that the region can optionally yield a value
  if (getRegion().getNumArguments() != 1)
    return emitOpError("region must have exactly one argument");
  mlir::Type argType = getRegion().getArgumentTypes()[0];
  if (argType != reussir::RegionType::get(getContext()))
    return emitOpError("region argument must be of !reussir.region type");
  if (getResults().size() > 1)
    return emitOpError("region must have at most one result");
  if (mlir::Value result = getResult()) {
    RcType rcType = llvm::dyn_cast<RcType>(result.getType());
    if (rcType && rcType.getCapability() != reussir::Capability::rigid &&
        rcType.getCapability() != reussir::Capability::shared)
      return emitOpError("region result must be of rigid or shared RC type");
  }
  // Check that the region is not nested in the same function
  if (this->getOperation()->getParentOfType<ReussirRegionRunOp>() != nullptr)
    return emitOpError("region cannot be nested in the same function");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRegionYieldOp::verify() {
  // Check that it is consistent on whether yielding a value or not
  // If yielding a value, check that the value is of RC type, and it the
  // capability of the Rc type is flex, convert the return type to rigid
  // counterpart. Then verify that the value is of the same type as the region
  // argument.
  auto parentOp = this->getParentOp();
  if (getValue() != nullptr) {
    if (parentOp->getNumResults() != 1)
      return emitOpError("region must have exactly one result");
    mlir::Type valueType = getValue().getType();
    auto rcType = llvm::dyn_cast<RcType>(valueType);
    if (rcType && rcType.getCapability() == reussir::Capability::flex)
      valueType = RcType::get(getContext(), rcType.getElementType(),
                              reussir::Capability::rigid);
    if (valueType != parentOp->getResult(0).getType())
      return emitOpError("value type must match region result type, ")
             << "value type: " << valueType
             << ", region result type: " << parentOp->getResult(0).getType();
  } else if (parentOp->getNumResults() != 0)
    return emitOpError("region must have no result");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionRunOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirRegionRunOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.emplace_back(&getRegion());
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.emplace_back(getOperation(), getResults());
}

//===----------------------------------------------------------------------===//
// NullableDispatchOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirNullableDispatchOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the parent operation, branch into one of the
  // regions.
  if (point.isParent()) {
    regions.emplace_back(&getNonNullRegion());
    regions.emplace_back(&getNullRegion());
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.emplace_back(getOperation(), getResults());
}

//===----------------------------------------------------------------------===//
// RecordDispatchOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
void ReussirRecordDispatchOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the parent operation, branch into one of the
  // regions.
  if (point.isParent()) {
    for (mlir::Region &region : getRegions())
      regions.emplace_back(&region);
    return;
  }
  // Otherwise, the region branches back to the parent operation.
  regions.emplace_back(getOperation(), getResults());
}

//===----------------------------------------------------------------------===//
// Reussir Region VTable Op
//===----------------------------------------------------------------------===//
// RegionVTableOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRegionVTableOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  if (getDropAttr()) {
    auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
        getOperation(), getDropAttr());
    if (!funcOp)
      return emitOpError("drop function not found: ") << getDropAttr();

    // Check that the drop function has the correct signature:
    // single input parameter of RefType with unspecified capability, zero
    // outputs
    mlir::FunctionType funcType = funcOp.getFunctionType();

    // Must have exactly one input and zero outputs
    if (funcType.getNumInputs() != 1)
      return emitOpError(
                 "drop function must have exactly one input parameter, got: ")
             << funcType.getNumInputs();

    if (funcType.getNumResults() != 0)
      return emitOpError("drop function must have zero outputs, got: ")
             << funcType.getNumResults();

    // Input parameter must be RefType with unspecified capability
    mlir::Type inputType = funcType.getInput(0);
    RefType refType = llvm::dyn_cast<RefType>(inputType);
    if (!refType)
      return emitOpError("drop function input parameter must be RefType, got: ")
             << inputType;

    if (refType.getCapability() != reussir::Capability::unspecified)
      return emitOpError("drop function input parameter must have unspecified "
                         "capability, got: ")
             << stringifyCapability(refType.getCapability());

    // Check that the drop function input reference element type matches the
    // type attribute
    mlir::Type elementType = refType.getElementType();
    mlir::Type vtableType = getTypeAttr().getValue();
    if (elementType != vtableType)
      return emitOpError("drop function input reference element type must "
                         "match vtable type attribute, got: ")
             << elementType << " but expected: " << vtableType;
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Trampoline Op
//===----------------------------------------------------------------------===//
// TrampolineOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirTrampolineOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  if (symbolTable.lookupNearestSymbolFrom(getOperation(), getTargetAttr()))
    return mlir::success();
  return emitOpError("target function not found: ") << getTargetAttr();
}

//===----------------------------------------------------------------------===//
// Reussir Record Dispatch Op
//===----------------------------------------------------------------------===//
// RecordDispatchOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordDispatchOp::verify() {
  // Get the variant reference type and extract the record type
  RefType variantRefType = getVariant().getType();
  RecordType recordType =
      llvm::dyn_cast<RecordType>(variantRefType.getElementType());
  if (!recordType)
    return emitOpError("variant operand must be a reference to a record type");

  if (!recordType.isVariant())
    return emitOpError(
        "variant operand must be a reference to a variant record type");

  if (!recordType.getComplete())
    return emitOpError("variant record type must be complete");

  // Get the number of variant members
  auto members = recordType.getMembers();
  size_t numMembers = members.size();

  // Check that tagSets array size matches the number of regions
  auto tagSets = getTagSets();
  auto regions = getRegions();

  if (tagSets.size() != regions.size())
    return emitOpError("number of tag sets must match number of regions, ")
           << "tag sets: " << tagSets.size() << ", regions: " << regions.size();

  // Track which tags are covered
  llvm::SmallSet<int64_t, 8> coveredTags;

  // Verify each tag set and corresponding region
  for (size_t i = 0; i < tagSets.size(); ++i) {
    auto tagSetAttr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(tagSets[i]);
    if (!tagSetAttr)
      return emitOpError("tag set ") << i << " must be a DenseI64ArrayAttr";

    auto tagSet = tagSetAttr.asArrayRef();
    if (tagSet.empty())
      return emitOpError("tag set ") << i << " must have at least one value";

    // Check that all tags in this set are valid and not already covered
    for (int64_t tag : tagSet) {
      if (tag < 0 || static_cast<size_t>(tag) >= numMembers)
        return emitOpError("tag ")
               << tag << " in tag set " << i << " is out of range [0, "
               << numMembers << ")";
      if (coveredTags.contains(tag))
        return emitOpError("tag ")
               << tag << " in tag set " << i
               << " is already covered by a previous tag set";
      coveredTags.insert(tag);
    }

    // Verify region argument types based on tag set size
    mlir::Region &region = regions[i];
    if (region.empty())
      return emitOpError("region ") << i << " cannot be empty";

    mlir::Block &block = region.front();
    if (tagSet.size() == 1) {
      // Single tag: region should accept a reference to the target variant
      // element type
      if (block.getNumArguments() != 1)
        return emitOpError("region ")
               << i << " must have exactly one argument for single tag";

      int64_t tag = tagSet[0];
      mlir::Type expectedType =
          getProjectedType(members[tag], recordType.getMemberIsField()[tag],
                           variantRefType.getCapability());
      mlir::Type actualType = block.getArgument(0).getType();

      // The argument should be a reference to the member type
      RefType actualRefType = llvm::dyn_cast<RefType>(actualType);
      if (!actualRefType)
        return emitOpError("region ")
               << i << " argument must be a reference type";

      if (actualRefType.getElementType() != expectedType)
        return emitOpError("region ")
               << i << " argument type must match variant member type, "
               << "argument type: " << actualRefType.getElementType()
               << ", expected type: " << expectedType;
    } else if (block.getNumArguments() != 0)
      return emitOpError("region ")
             << i << " must have no arguments for multiple tags";
  }

  // Check that all possible tags are covered
  for (size_t i = 0; i < numMembers; ++i)
    if (!coveredTags.contains(i))
      return emitOpError("tag ") << i << " is not covered by any tag set";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RecordDispatchOp custom assembly format
//===----------------------------------------------------------------------===//
mlir::ParseResult ReussirRecordDispatchOp::parse(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand variantRefOperand;
  llvm::SMLoc variantRefOperandsLoc;
  RefType variantRefType;
  mlir::Type valueType;
  llvm::SmallVector<std::unique_ptr<mlir::Region>> regions;
  llvm::SmallVector<mlir::Attribute> tagSets;
  if (parser.parseLParen())
    return mlir::failure();

  variantRefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(variantRefOperand))
    return mlir::failure();
  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseCustomTypeWithFallback(variantRefType))
    return mlir::failure();

  if (parser.parseRParen())
    return mlir::failure();
  if (llvm::succeeded(parser.parseOptionalArrow()))
    if (llvm::failed(parser.parseType(valueType)))
      return mlir::failure();

  if (parser.parseLBrace())
    return mlir::failure();

  llvm::SmallVector<int64_t> tags;
  auto parseTag = [&]() -> mlir::ParseResult {
    llvm::APInt tag;
    if (parser.parseInteger(tag))
      return mlir::failure();
    if (tag.isNegative())
      return parser.emitError(parser.getCurrentLocation(),
                              "tag must be positive");
    tags.push_back(tag.getZExtValue());
    return mlir::success();
  };
  while (llvm::succeeded(parser.parseOptionalLSquare())) {
    if (llvm::failed(parser.parseCommaSeparatedList(parseTag)))
      return llvm::failure();
    if (llvm::failed(parser.parseRSquare()))
      return llvm::failure();
    if (llvm::failed(parser.parseArrow()))
      return mlir::failure();
    if (llvm::failed(parser.parseRegion(
            *regions.emplace_back(std::make_unique<mlir::Region>()))))
      return mlir::failure();
    tagSets.push_back(mlir::DenseI64ArrayAttr::get(parser.getContext(), tags));
    tags.clear();
  }
  if (parser.parseRBrace())
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();
  if (valueType)
    result.addTypes(valueType);
  result.addRegions(regions);
  result.addAttribute("tagSets",
                      mlir::ArrayAttr::get(parser.getContext(), tagSets));
  if (llvm::failed(parser.resolveOperands({variantRefOperand}, variantRefType,
                                          variantRefOperandsLoc,
                                          result.operands)))
    return mlir::failure();
  return mlir::success();
}

void ReussirRecordDispatchOp::print(mlir::OpAsmPrinter &p) {
  // Print the variant reference operand and type
  p << "(";
  p.printOperand(getVariant());
  p << ' ' << ":";
  p.printType(getVariant().getType());
  p << ")";

  // Print optional result type
  if (getValue()) {
    p << ' ' << "->" << ' ';
    p.printType(getValue().getType());
  }

  // Print the dispatch body
  p << "{";
  p.increaseIndent();
  for (auto [region, tagSetAttr] : llvm::zip(getRegions(), getTagSets())) {
    p.printNewline();
    auto tagSet = llvm::cast<mlir::DenseI64ArrayAttr>(tagSetAttr);
    p << '[';
    llvm::interleaveComma(tagSet.asArrayRef(), p,
                          [&](int64_t tag) { p << tag; });
    p << ']' << ' ' << "->" << ' ';
    p.printRegion(region);
  }
  p.decreaseIndent();
  p.printNewline();
  p << "}";
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"tagSets"});
}

//===----------------------------------------------------------------------===//
// Reussir Array WithUniqueView Op
//===----------------------------------------------------------------------===//
mlir::ParseResult
ReussirArrayWithUniqueViewOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SMLoc operandLoc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand arrayOperand;
  RcType arrayType;
  mlir::Type resultType;
  auto bodyRegion = std::make_unique<mlir::Region>();

  if (parser.parseLParen() || parser.parseOperand(arrayOperand) ||
      parser.parseColon() || parser.parseCustomTypeWithFallback(arrayType) ||
      parser.parseRParen())
    return mlir::failure();

  if (llvm::succeeded(parser.parseOptionalArrow()))
    if (parser.parseType(resultType))
      return mlir::failure();

  if (parser.parseRegion(*bodyRegion) ||
      parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  result.addRegion(std::move(bodyRegion));
  if (resultType)
    result.addTypes(resultType);
  if (parser.resolveOperands({arrayOperand}, arrayType, operandLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

void ReussirArrayWithUniqueViewOp::print(mlir::OpAsmPrinter &p) {
  p << "(";
  p.printOperand(getArray());
  p << " : ";
  p.printType(getArray().getType());
  p << ")";
  if (getResult()) {
    p << " -> ";
    p.printType(getResult().getType());
  }
  p << " ";
  p.printRegion(getBody());
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

//===----------------------------------------------------------------------===//
// Reussir Nullable Coerce Op
//===----------------------------------------------------------------------===//
// NullableCoerceOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirNullableCoerceOp::verify() {
  NullableType nullableType = getNullable().getType();
  mlir::Type coercedType = getNonnull().getType();

  // Check that the coerced type is the same as the PtrTy of the nullable
  // input
  mlir::Type expectedType = nullableType.getPtrTy();
  if (coercedType != expectedType)
    return emitOpError("coerced type must match nullable pointer type, ")
           << "expected: " << expectedType << ", got: " << coercedType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Nullable Dispatch Op
//===----------------------------------------------------------------------===//
// NullableDispatchOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirNullableDispatchOp::verify() {
  // Get the nullable type and extract the inner pointer type
  NullableType nullableType = getNullable().getType();
  mlir::Type innerType = nullableType.getPtrTy();

  // Verify nonNullRegion has exactly one argument matching the inner type
  mlir::Region &nonNullRegion = getNonNullRegion();
  if (nonNullRegion.empty())
    return emitOpError("nonnull region cannot be empty");

  mlir::Block &nonNullBlock = nonNullRegion.front();
  if (nonNullBlock.getNumArguments() != 1)
    return emitOpError("nonnull region must have exactly one argument");

  mlir::Type nonNullArgType = nonNullBlock.getArgument(0).getType();
  if (nonNullArgType != innerType)
    return emitOpError("nonnull region argument type must match nullable "
                       "inner type, ")
           << "argument type: " << nonNullArgType
           << ", expected type: " << innerType;

  // Verify nullRegion has no arguments
  mlir::Region &nullRegion = getNullRegion();
  if (nullRegion.empty())
    return emitOpError("null region cannot be empty");

  mlir::Block &nullBlock = nullRegion.front();
  if (nullBlock.getNumArguments() != 0)
    return emitOpError("null region must have no arguments");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Scf Yield Op
//===----------------------------------------------------------------------===//
// ScfYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirScfYieldOp::verify() {
  mlir::Type yieldedType = getValue() ? getValue().getType() : mlir::Type{};
  mlir::Type expectedType = mlir::Type{};
  bool allowImplicitArrayResult = false;
  if (auto nullableParent =
          getOperation()->getParentOfType<ReussirNullableDispatchOp>())
    expectedType = nullableParent.getValue()
                       ? nullableParent.getValue().getType()
                       : mlir::Type{};
  else if (auto recordParent =
               getOperation()->getParentOfType<ReussirRecordDispatchOp>())
    expectedType = recordParent.getValue() ? recordParent.getValue().getType()
                                           : mlir::Type{};
  else if (auto arrayParent =
               getOperation()
                   ->getParentOfType<ReussirArrayWithUniqueViewOp>()) {
    expectedType = arrayParent.getResult() ? arrayParent.getResult().getType()
                                           : mlir::Type{};
    allowImplicitArrayResult =
        expectedType && expectedType == arrayParent.getArray().getType();
  } else
    llvm_unreachable("unexpected parent operation");

  if (expectedType && !yieldedType && !allowImplicitArrayResult)
    return emitOpError(
        "parent operation expected a value, but nothing is yielded");
  if (yieldedType && !expectedType)
    return emitOpError(
        "parent operation did not expect a value, but one is yielded");

  if (yieldedType && expectedType && yieldedType != expectedType)
    return emitOpError("yielded type must match parent operation result type, ")
           << "yielded type: " << yieldedType
           << ", expected type: " << expectedType;

  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// Reussir Closure Create Op
//===-----------------------------------------------------------------------===//
// ClosureCreateOp custom assembly format
//===-----------------------------------------------------------------------===//
mlir::ParseResult ReussirClosureCreateOp::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  llvm::SMLoc operationLoc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand tokenOperand;
  mlir::FlatSymbolRefAttr vtableAttr [[maybe_unused]];
  std::unique_ptr<mlir::Region> bodyRegion = std::make_unique<mlir::Region>();
  TokenType tokenType;
  RcType closureType;
  enum class Keyword {
    vtable,
    body,
    token,
    unknown,
  };
  constexpr size_t NUM_KEYWORDS = static_cast<size_t>(Keyword::unknown);
  std::array<bool, NUM_KEYWORDS + 1> appeared{false, false, false, false};
  // Parse return type
  if (parser.parseArrow())
    return mlir::failure();
  if (parser.parseCustomTypeWithFallback(closureType))
    return mlir::failure();
  if (!llvm::isa<ClosureType>(closureType.getElementType()))
    return parser.emitError(operationLoc, "expected a Rc closure type");
  if (parser.parseLBrace())
    return mlir::failure();
  // Parse order insensitive fields (vtable, body, token)
  for (;;) {
    llvm::SMLoc keywordLoc = parser.getCurrentLocation();
    llvm::StringRef keyword;
    if (llvm::failed(parser.parseOptionalKeyword(&keyword)))
      break;
    auto dispatch = llvm::StringSwitch<Keyword>(keyword)
                        .Case("vtable", Keyword::vtable)
                        .Case("body", Keyword::body)
                        .Case("token", Keyword::token)
                        .Default(Keyword::unknown);
    if (appeared[static_cast<size_t>(dispatch)])
      return parser.emitError(keywordLoc,
                              "keyword " + keyword + " appeared twice");
    appeared[static_cast<size_t>(dispatch)] = true;
    switch (dispatch) {
    case Keyword::vtable: {
      if (parser.parseLParen())
        return mlir::failure();
      if (parser.parseCustomAttributeWithFallback(vtableAttr))
        return mlir::failure();
      if (parser.parseRParen())
        return mlir::failure();
      break;
    }
    case Keyword::body: {
      if (parser.parseRegion(*bodyRegion))
        return mlir::failure();
      break;
    }
    case Keyword::token: {
      if (parser.parseLParen())
        return mlir::failure();
      if (parser.parseOperand(tokenOperand))
        return mlir::failure();
      if (parser.parseColon())
        return mlir::failure();
      if (parser.parseCustomTypeWithFallback(tokenType))
        return mlir::failure();
      if (parser.parseRParen())
        return mlir::failure();
      break;
    }
    case Keyword::unknown:
      return parser.emitError(keywordLoc, "unknown keyword: " + keyword);
    }
  }

  if (parser.parseRBrace())
    return mlir::failure();

  // Token is now optional
  if (vtableAttr)
    result.addAttribute("vtable", vtableAttr);
  result.addRegion(std::move(bodyRegion));
  result.addTypes(closureType);

  // Only resolve token operands if token was present
  if (appeared[static_cast<size_t>(Keyword::token)]) {
    if (llvm::failed(parser.resolveOperands({tokenOperand}, tokenType,
                                            operationLoc, result.operands)))
      return mlir::failure();
  }

  return mlir::success();
}

void ReussirClosureCreateOp::print(mlir::OpAsmPrinter &p) {
  // Print return type
  p << " -> ";
  p.printType(getClosure().getType());
  p << " {";
  p.increaseIndent();

  // Print order insensitive fields: token, vtable, body
  // Print token if present
  if (getToken()) {
    p.printNewline();
    p << " token (";
    p.printOperand(getToken());
    p << " : ";
    p.printStrippedAttrOrType(getToken().getType());
    p << ")";
  }

  // Print vtable if present
  if (getVtableAttr()) {
    p.printNewline();
    p << " vtable (" << getVtableAttr() << ")";
  }

  // Print body region if present
  if (!getBody().empty()) {
    p.printNewline();
    p << " body ";
    p.printRegion(getBody());
  }

  p.decreaseIndent();
  p.printNewline();
  p << "}";
}

//===-----------------------------------------------------------------------===//
// ClosureCreateOp verification
//===-----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCreateOp::verify() {
  bool outlinedFlag = isOutlined();
  bool inlinedFlag = isInlined();
  ClosureType closureType =
      llvm::cast<ClosureType>(getClosure().getType().getElementType());
  if (!outlinedFlag && !inlinedFlag)
    return emitOpError("closure must be outlined or inlined");

  // Only verify token layout if token is present
  if (getToken()) {
    RcBoxType closureBoxType = getRcClosureBoxType();
    auto dataLayout = mlir::DataLayout::closest(this->getOperation());
    auto closureBoxSize = dataLayout.getTypeSize(closureBoxType);
    auto closureBoxAlignment = dataLayout.getTypeABIAlignment(closureBoxType);
    TokenType tokenType = getToken().getType();
    if (closureBoxSize != tokenType.getSize())
      return emitOpError("closure box size must match token size")
             << ", closure box size: " << closureBoxSize.getFixedValue()
             << ", token size: " << tokenType.getSize();
    if (closureBoxAlignment != tokenType.getAlign())
      return emitOpError("closure box alignment must match token alignment")
             << ", closure box alignment: " << closureBoxAlignment
             << ", token alignment: " << tokenType.getAlign();
  }

  // Check that region arguments match the closure input types
  if (inlinedFlag) {
    auto types = getBody().getArgumentTypes();
    if (types.size() != closureType.getInputTypes().size())
      return emitOpError("inlined closure body must have the same number of "
                         "arguments as the closure input types");
    if (llvm::any_of(llvm::zip(types, closureType.getInputTypes()),
                     [](auto &&argAndType) {
                       auto [argTy, type] = argAndType;
                       return argTy != type;
                     }))
      return emitOpError("inlined closure body arguments must match the "
                         "closure input types");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ClosureCreateOp TokenAcceptor Interface
//===----------------------------------------------------------------------===//
TokenType ReussirClosureCreateOp::getTokenType() {
  auto rcBoxType = getRcClosureBoxType();
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  auto alignment = dataLayout.getTypeABIAlignment(rcBoxType);
  auto size = dataLayout.getTypeSize(rcBoxType);
  return TokenType::get(getContext(), alignment, size.getFixedValue());
}

//===----------------------------------------------------------------------===//
// ClosureCreateOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCreateOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  if (getVtableAttr()) {
    auto vtableOp = symbolTable.lookupNearestSymbolFrom<ReussirClosureVtableOp>(
        getOperation(), getVtableAttr());
    if (!vtableOp)
      return emitOpError("virtual table not found: ") << getVtableAttr();
  }
  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// ClosureCreateOp helper methods
//===-----------------------------------------------------------------------===//
bool ReussirClosureCreateOp::isOutlined() {
  return getBody().empty() && getVtableAttr();
}

bool ReussirClosureCreateOp::isInlined() {
  return !getBody().empty() && !getVtableAttr();
}

ClosureBoxType ReussirClosureCreateOp::getClosureBoxType() {
  ClosureType closureType =
      llvm::cast<ClosureType>(getClosure().getType().getElementType());
  llvm::SmallVector<mlir::Type> payloadTypes;
  std::transform(
      closureType.getInputTypes().begin(), closureType.getInputTypes().end(),
      std::back_inserter(payloadTypes), [](mlir::Type type) {
        return reussir::getProjectedType(type, false, Capability::shared);
      });
  return ClosureBoxType::get(getContext(), payloadTypes);
}

RcBoxType ReussirClosureCreateOp::getRcClosureBoxType() {
  ClosureBoxType closureBoxType = getClosureBoxType();
  return RcBoxType::get(getContext(), closureBoxType);
}

//===----------------------------------------------------------------------===//
// Reussir Closure Vtable Op
//===----------------------------------------------------------------------===//
// ClosureVtableOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureVtableOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  // NYI for body
  auto funcOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      getOperation(), getFuncAttr());
  if (!funcOp)
    return emitOpError("function not found: ") << getFuncAttr();

  auto dropOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      getOperation(), getDropAttr());
  if (!dropOp)
    return emitOpError("drop function not found: ") << getDropAttr();

  auto cloneOp = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(
      getOperation(), getCloneAttr());
  if (!cloneOp)
    return emitOpError("clone function not found: ") << getCloneAttr();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Yield Op
//===----------------------------------------------------------------------===//
// ClosureYieldOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureYieldOp::verify() {
  // Get the parent closure operation
  auto parentOp = getOperation()->getParentOfType<ReussirClosureCreateOp>();
  if (!parentOp)
    return emitOpError(
        "closure yield must be inside a closure create operation");

  // Get the closure type to determine if it has a return value
  ClosureType closureType =
      llvm::cast<ClosureType>(parentOp.getClosure().getType().getElementType());
  mlir::Type expectedReturnType = closureType.getOutputType();

  // Check consistency between yield value and closure return type
  if (expectedReturnType) {
    // Closure has a return type, so yield must provide a value
    if (!getValue())
      return emitOpError("closure has return type ")
             << expectedReturnType << " but yield provides no value";

    // Check that the yielded value type matches the closure return type
    mlir::Type yieldedType = getValue().getType();
    if (yieldedType != expectedReturnType)
      return emitOpError("yielded type must match closure return type, ")
             << "yielded type: " << yieldedType
             << ", expected type: " << expectedReturnType;
  } else {
    // Closure has no return type, so yield must not provide a value
    if (getValue())
      return emitOpError("closure has no return type but yield provides "
                         "value of type ")
             << getValue().getType();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Apply Op
//===----------------------------------------------------------------------===//
// ClosureApplyOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureApplyOp::verify() {
  ClosureType closureType =
      llvm::cast<ClosureType>(getClosure().getType().getElementType());
  mlir::Type argType = getArg().getType();

  // Get the input types of the closure
  auto inputTypes = closureType.getInputTypes();

  // Check that the closure has at least one input type
  if (inputTypes.empty())
    return emitOpError("cannot apply to closure with no input types");

  // Check that the argument type matches the first input type
  mlir::Type expectedArgType = inputTypes[0];
  if (argType != expectedArgType)
    return emitOpError("argument type must match first closure input type, ")
           << "argument type: " << argType
           << ", expected type: " << expectedArgType;

  // Verify the result type
  ClosureType resultType =
      llvm::cast<ClosureType>(getApplied().getType().getElementType());

  // The result closure should have one less input type
  auto expectedInputTypes = inputTypes.drop_front(1);
  auto resultInputTypes = resultType.getInputTypes();

  if (resultInputTypes.size() != expectedInputTypes.size())
    return emitOpError("result closure must have one less input type, ")
           << "expected " << expectedInputTypes.size() << " input types, "
           << "but got " << resultInputTypes.size();

  // Check that the remaining input types match
  for (size_t i = 0; i < expectedInputTypes.size(); ++i) {
    if (resultInputTypes[i] != expectedInputTypes[i])
      return emitOpError("result closure input types must match remaining "
                         "input types, ")
             << "mismatch at index " << i << ": expected "
             << expectedInputTypes[i] << ", but got " << resultInputTypes[i];
  }

  // Check that the output types match
  mlir::Type closureOutputType = closureType.getOutputType();
  mlir::Type resultOutputType = resultType.getOutputType();

  if (closureOutputType != resultOutputType)
    return emitOpError("result closure output type must match original closure "
                       "output type, ")
           << "original output type: " << closureOutputType
           << ", result output type: " << resultOutputType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Eval Op
//===----------------------------------------------------------------------===//
// ClosureEvalOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureEvalOp::verify() {
  ClosureType closureType =
      llvm::cast<ClosureType>(getClosure().getType().getElementType());

  // The `with` pack must supply every remaining input exactly (the plain
  // form is the empty pack: a fully applied closure).
  auto inputTypes = closureType.getInputTypes();
  if (inputTypes.size() != getArgs().size())
    return emitOpError("closure has ")
           << inputTypes.size() << " remaining input types but "
           << getArgs().size() << " eval arguments are supplied";
  for (auto [index, pair] : llvm::enumerate(llvm::zip(getArgs(), inputTypes))) {
    auto [arg, inputType] = pair;
    if (arg.getType() != inputType)
      return emitOpError("eval argument ")
             << index << " has type " << arg.getType()
             << " but the closure expects " << inputType;
  }

  // Check that the result type matches the closure's output type
  mlir::Type closureOutputType = closureType.getOutputType();

  // Check if we have a result
  if (getNumResults() > 0) {
    mlir::Type resultType = getResult().getType();

    // If the closure has no output type, the result should be empty
    if (!closureOutputType)
      return emitOpError("closure has no output type but result is not empty");

    // If the closure has an output type, the result should match
    if (resultType != closureOutputType)
      return emitOpError("result type must match closure output type, ")
             << "result type: " << resultType
             << ", closure output type: " << closureOutputType;
  } else {
    // No result provided
    if (closureOutputType)
      return emitOpError("closure has output type ")
             << closureOutputType << " but result is empty";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Uniqify Op
//===----------------------------------------------------------------------===//
// ClosureUniqifyOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureUniqifyOp::verify() {
  // Get the input and output closure types
  mlir::Type inputClosureType = getClosure().getType();
  mlir::Type outputClosureType = getUniqified().getType();

  // Check that input and output types are the same
  if (inputClosureType != outputClosureType)
    return emitOpError("input and output closure types must be the same, ")
           << "input type: " << inputClosureType
           << ", output type: " << outputClosureType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Closure Clone Op
//===----------------------------------------------------------------------===//
// ClosureCloneOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCloneOp::verify() {
  // Get the input and output closure types
  mlir::Type inputClosureType = getClosure().getType();
  mlir::Type outputClosureType = getCloned().getType();

  // Check that input and output types are the same
  if (inputClosureType != outputClosureType)
    return emitOpError("input and output closure types must be the same, ")
           << "input type: " << inputClosureType
           << ", output type: " << outputClosureType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ClosureCursorOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureCursorOp::verify() {
  RefType cursorRefType = getCursor().getType();
  mlir::Type elementType = cursorRefType.getElementType();

  // Check that the element type is i8
  if (!elementType.isInteger(8))
    return emitOpError("cursor element type must be i8, ")
           << "got: " << elementType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ClosureInstantiateOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirClosureInstantiateOp::verify() {
  RcType rcType = getClosureBoxRc().getType();
  TokenType tokenType = getToken().getType();
  ClosureBoxType closureBoxType =
      llvm::dyn_cast<ClosureBoxType>(rcType.getElementType());
  if (!closureBoxType)
    return emitOpError("rc type must be a closure box type, got: ") << rcType;

  RcBoxType rcBoxType = rcType.getInnerBoxType();
  auto dataLayout = mlir::DataLayout::closest(getOperation());
  // Check that the token type is a valid token type
  if (tokenType.getAlign() != dataLayout.getTypeABIAlignment(rcBoxType) ||
      tokenType.getSize() != dataLayout.getTypeSize(rcBoxType))
    return emitOpError("token type must match rc box type, got: ")
           << tokenType << " and " << rcBoxType;

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Reference Drop Op
//===----------------------------------------------------------------------===//
// RefDropOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefDropOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type elementType = refType.getElementType();

  // Check if variant attribute is specified
  if (auto variantAttr = getVariant()) {
    // When variant is specified, the inner element must be a variant record
    // type
    RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
    if (!recordType)
      return emitOpError("when variant is specified, reference element type "
                         "must be a record type, got: ")
             << elementType;

    // Check that the record is a variant record
    if (!recordType.isVariant())
      return emitOpError("when variant is specified, reference element type "
                         "must be a variant record type");

    // Check that the record is complete
    if (!recordType.getComplete())
      return emitOpError("cannot drop incomplete variant record");

    // Check that the index is inbound
    size_t variantIndex = variantAttr->getZExtValue();
    size_t numVariants = recordType.getMembers().size();
    if (variantIndex >= numVariants)
      return emitOpError("variant index out of bounds: ")
             << variantIndex << " >= " << numVariants;
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Reussir Reference Acquire Op
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRefAcquireOp::verify() {
  RefType refType = getRef().getType();
  mlir::Type elementType = refType.getElementType();

  // Check if variant attribute is specified
  if (auto variantAttr = getVariant()) {
    // When variant is specified, the inner element must be a variant record
    // type
    RecordType recordType = llvm::dyn_cast<RecordType>(elementType);
    if (!recordType)
      return emitOpError("when variant is specified, reference element type "
                         "must be a record type, got: ")
             << elementType;

    if (!recordType.isVariant())
      return emitOpError("when variant is specified, reference element type "
                         "must be a variant record type");

    if (!recordType.getComplete())
      return emitOpError("cannot acquire incomplete variant record");

    size_t variantIndex = variantAttr->getZExtValue();
    size_t numVariants = recordType.getMembers().size();
    if (variantIndex >= numVariants)
      return emitOpError("variant index out of bounds: ")
             << variantIndex << " >= " << numVariants;
  }

  // Element type must be something emitOwnershipAcquisition can handle:
  // Rc, Record (compound or variant), or trivially copyable
  if (isTriviallyCopyable(elementType))
    return mlir::success();

  if (llvm::isa<RcType>(elementType))
    return mlir::success();

  if (auto recordType = llvm::dyn_cast<RecordType>(elementType)) {
    if (!recordType.getComplete())
      return emitOpError("cannot acquire incomplete record type");
    return mlir::success();
  }

  if (llvm::isa<ArrayType>(elementType))
    return mlir::success();

  return emitOpError("unsupported element type for acquisition: ")
         << elementType;
}

//===----------------------------------------------------------------------===//
// Reussir Poly FFI Op
//===----------------------------------------------------------------------===//
// ReussirPolyFFIOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirPolyFFIOp::verify() {
  auto moduleTexture = getModuleTextureAttr();
  auto compiledModule = getCompiledModuleAttr();
  auto substitutions = getSubstitutionsAttr();

  // Check for empty string in moduleTexture
  if (moduleTexture && moduleTexture.getValue().empty())
    return emitOpError("moduleTexture cannot be empty");

  // Check for empty array in compiledModule
  if (compiledModule && compiledModule.empty())
    return emitOpError("compiledModule cannot be empty");

  // Check for empty dictionary in substitutions
  if (substitutions && substitutions.empty())
    return emitOpError("substitutions cannot be empty");

  // Check that either moduleTexture or compiledModule is specified, but not
  // both (after checking for empty values)
  bool hasModuleTexture = moduleTexture != nullptr;
  bool hasCompiledModule = compiledModule != nullptr;

  if (!hasModuleTexture && !hasCompiledModule)
    return emitOpError(
        "either moduleTexture or compiledModule must be specified");

  if (hasModuleTexture && hasCompiledModule)
    return emitOpError("cannot specify both moduleTexture and compiledModule");

  // Check that when compiledModule is specified, substitutions cannot be
  // specified
  if (hasCompiledModule && substitutions)
    return emitOpError(
        "substitutions cannot be specified when compiledModule is used");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// emitOwnershipAcquisition
//===----------------------------------------------------------------------===//
static mlir::LogicalResult
emitArrayOwnershipAcquisition(mlir::Value view, mlir::OpBuilder &builder,
                              mlir::Location loc);

mlir::LogicalResult emitOwnershipAcquisition(mlir::Value value,
                                             mlir::OpBuilder &builder,
                                             mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type type = value.getType();

  return llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
      // For Rc types, emit an Inc operation
      .Case<RcType>([&](RcType) {
        ReussirRcIncOp::create(builder, loc, value);
        return mlir::success();
      })
      // For Ref types, check what they point to and handle accordingly
      .Case<RefType>([&](RefType refType) {
        mlir::Type elementType = refType.getElementType();

        if (isTriviallyCopyable(elementType))
          return mlir::success();

        // If reference points to an RC pointer, load it and recursively apply
        if (llvm::isa<RcType>(elementType)) {
          auto loadedValue =
              ReussirRefLoadOp::create(builder, loc, elementType, value);
          return emitOwnershipAcquisition(loadedValue, builder, loc);
        }

        if (auto arrayType = llvm::dyn_cast<ArrayType>(elementType)) {
          auto view =
              ReussirArrayViewOp::create(
                  builder, loc, getArrayViewMemRefType(arrayType), value)
                  .getView();
          return emitArrayOwnershipAcquisition(view, builder, loc);
        }

        // If reference points to a record, handle fields directly
        if (auto recordType = llvm::dyn_cast<RecordType>(elementType)) {
          // The value is already a reference, so we can use it directly

          if (recordType.getKind() == RecordKind::compound) {
            // For compound types, recursively apply to each field
            for (auto [i, actualMemberType, actualMemberCap] : llvm::enumerate(
                     recordType.getMembers(), recordType.getMemberIsField())) {
              auto projectedType = getProjectedType(
                  actualMemberType, actualMemberCap, refType.getCapability());
              if (isTriviallyCopyable(projectedType))
                continue;
              auto fieldRef = ReussirRefProjectOp::create(
                  builder, loc,
                  RefType::get(builder.getContext(), projectedType), value,
                  builder.getIndexAttr(i));

              if (emitOwnershipAcquisition(fieldRef, builder, loc).failed())
                return mlir::failure();
            }
          } else if (recordType.getKind() == RecordKind::variant) {
            // For variant types, emit a RecordDispatch operation
            // First prepare the tag sets
            llvm::SmallVector<mlir::Attribute> tagSetAttrs;
            for (auto i : llvm::seq<int64_t>(0, recordType.getMembers().size()))
              tagSetAttrs.push_back(builder.getDenseI64ArrayAttr({i}));

            auto tagSetsAttr = builder.getArrayAttr(tagSetAttrs);

            // Create the dispatch operation with the correct number of
            // regions
            auto dispatchOp = ReussirRecordDispatchOp::create(
                builder, loc, mlir::Type{}, value, tagSetsAttr,
                recordType.getMembers().size());

            // Create regions for each variant and apply ownership acquisition
            // in each region
            for (auto [i, actualVariantType, actualVariantCap] :
                 llvm::enumerate(recordType.getMembers(),
                                 recordType.getMemberIsField())) {

              auto projectedType = getProjectedType(
                  actualVariantType, actualVariantCap, refType.getCapability());

              // Create a block for this variant region
              RefType projectedRefTy =
                  RefType::get(builder.getContext(), projectedType);
              auto *block = builder.createBlock(
                  &dispatchOp.getRegions()[i],
                  dispatchOp.getRegions()[i].begin(), {projectedRefTy}, {loc});

              // Set insertion point to the block
              builder.setInsertionPointToStart(block);
              if (!isTriviallyCopyable(projectedType)) {
                if (emitOwnershipAcquisition(block->getArgument(0), builder,
                                             loc)
                        .failed())
                  return mlir::failure();
              }

              // Add a terminator
              ReussirScfYieldOp::create(builder, loc, nullptr);
            }
          }
        }

        // For other reference types, this is a no-op
        return mlir::success();
      })
      .Case<mlir::MemRefType>([&](mlir::MemRefType) {
        return emitArrayOwnershipAcquisition(value, builder, loc);
      })
      // For other types, return failure
      .Default([&](mlir::Type) { return mlir::failure(); });
}

static mlir::LogicalResult
emitArrayOwnershipAcquisition(mlir::Value view, mlir::OpBuilder &builder,
                              mlir::Location loc) {
  auto viewType = llvm::cast<mlir::MemRefType>(view.getType());
  ArrayType arrayType = ArrayType::get(
      builder.getContext(), viewType.getShape(), viewType.getElementType());
  auto *context = builder.getContext();
  for (int64_t index : llvm::seq<int64_t>(0, arrayType.getShape().front())) {
    auto idx = mlir::arith::ConstantIndexOp::create(builder, loc, index);
    if (arrayType.getRank() == 1) {
      RefType projectedType = RefType::get(context, arrayType.getElementType(),
                                           Capability::unspecified);
      auto elementRef = ReussirArrayProjectOp::create(
          builder, loc, projectedType, view, idx.getResult());
      if (emitOwnershipAcquisition(elementRef.getProjected(), builder, loc)
              .failed())
        return mlir::failure();
      continue;
    }

    auto projectedType = getArrayViewMemRefType(arrayType.dropFront());
    auto nestedView = ReussirArrayProjectOp::create(builder, loc, projectedType,
                                                    view, idx.getResult());
    if (emitArrayOwnershipAcquisition(nestedView.getProjected(), builder, loc)
            .failed())
      return mlir::failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// inheritSanitizerPassthrough
//===----------------------------------------------------------------------===//
void inheritSanitizerPassthrough(mlir::ModuleOp moduleOp,
                                 mlir::Operation *func) {
  auto attrs = moduleOp->getAttrOfType<mlir::ArrayAttr>(kSanitizeAttr);
  if (!attrs || attrs.empty())
    return;
  llvm::SmallVector<mlir::Attribute> merged;
  if (auto existing = func->getAttrOfType<mlir::ArrayAttr>("passthrough"))
    merged.append(existing.begin(), existing.end());
  for (mlir::Attribute attr : attrs)
    if (!llvm::is_contained(merged, attr))
      merged.push_back(attr);
  func->setAttr("passthrough",
                mlir::ArrayAttr::get(moduleOp.getContext(), merged));
}

//===----------------------------------------------------------------------===//
// createDtorIfNotExists
//===----------------------------------------------------------------------===//
mlir::func::FuncOp createDtorIfNotExists(mlir::ModuleOp moduleOp,
                                         RecordType type,
                                         mlir::OpBuilder &builder) {
  mlir::SymbolTable symTable(moduleOp);
  auto dtorName = type.getDtorName();
  if (!dtorName)
    llvm::report_fatal_error("only named record types have destructors");
  std::string funcName = dtorName.getValue().str();
  if (auto funcOp = symTable.lookup<mlir::func::FuncOp>(funcName))
    return funcOp;
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  RefType refType = builder.getType<RefType>(type);
  auto dtor =
      mlir::func::FuncOp::create(builder, builder.getUnknownLoc(), funcName,
                                 builder.getFunctionType({refType}, {}));
  dtor.setPrivate();
  dtor->setAttr("llvm.linkage", builder.getAttr<mlir::LLVM::LinkageAttr>(
                                    mlir::LLVM::linkage::Linkage::LinkonceODR));
  dtor->setAttr("passthrough",
                builder.getStrArrayAttr(
                    {"mustprogress", "nounwind", "willreturn", "nocallback"}));
  // Drop glue is where a stale box's memory is actually touched; it must be
  // instrumented like the code it was outlined from.
  inheritSanitizerPassthrough(moduleOp, dtor);
  dtor.setArgAttr(0, "llvm.noalias", builder.getUnitAttr());
  dtor.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  dtor.setArgAttr(0, "llvm.noundef", builder.getUnitAttr());
  mlir::Block *entryBlock = dtor.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto ref = entryBlock->getArgument(0);
  ReussirRefDropOp::create(builder, builder.getUnknownLoc(), ref, true,
                           nullptr);
  mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  return dtor;
}

//===----------------------------------------------------------------------===//
// emitOwnershipAcquisitionFuncIfNotExists
//===----------------------------------------------------------------------===//
mlir::func::FuncOp emitOwnershipAcquisitionFuncIfNotExists(
    mlir::ModuleOp moduleOp, RecordType type, mlir::OpBuilder &builder) {
  mlir::SymbolTable symTable(moduleOp);
  auto acquireName = type.getAcquireName();
  if (!acquireName)
    llvm::report_fatal_error(
        "only named record types have ownership acquisition functions");
  std::string funcName = acquireName.getValue().str();

  if (auto funcOp = symTable.lookup<mlir::func::FuncOp>(funcName))
    return funcOp;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());

  // Construct RefType from RecordType
  RefType refType = builder.getType<RefType>(type);

  auto funcOp =
      mlir::func::FuncOp::create(builder, builder.getUnknownLoc(), funcName,
                                 builder.getFunctionType({refType}, {}));
  funcOp.setPrivate();
  funcOp->setAttr("llvm.linkage",
                  builder.getAttr<mlir::LLVM::LinkageAttr>(
                      mlir::LLVM::linkage::Linkage::LinkonceODR));
  inheritSanitizerPassthrough(moduleOp, funcOp);

  mlir::Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto ref = entryBlock->getArgument(0);

  if (emitOwnershipAcquisition(ref, builder, builder.getUnknownLoc())
          .failed()) {
    llvm::report_fatal_error("failed to emit ownership acquisition");
  }

  mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  return funcOp;
}

//===----------------------------------------------------------------------===//
// StrLiteralOp SymbolUserOpInterface
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirStrLiteralOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto strGlobalOp = symbolTable.lookupNearestSymbolFrom<ReussirStrGlobalOp>(
      getOperation(), getSymNameAttr());
  if (!strGlobalOp)
    return emitOpError("referenced symbol is not a reussir.str.global: ")
           << getSymNameAttr();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StrLiteralOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirStrLiteralOp::verify() {
  StrType strType = getLiteral().getType();
  if (strType.getLifeScope() != LifeScope::global)
    return emitOpError("literal type must have global lifescope, got: ")
           << stringifyLifeScope(strType.getLifeScope());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StrCastOp verification
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirStrCastOp::verify() {
  StrType localStrType = getLocalStr().getType();

  if (localStrType.getLifeScope() != LifeScope::local)
    return emitOpError("output must be a local string literal, got: ")
           << stringifyLifeScope(localStrType.getLifeScope());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RegionRunOp RegionBranchOpInterface implementation
//===----------------------------------------------------------------------===//
mlir::LogicalResult ReussirRecordExtractOp::verify() {
  auto recordType = getRecord().getType();
  auto members = recordType.getMembers();
  size_t indexVal = getIndex().getZExtValue();
  if (members.size() < indexVal)
    return emitOpError("index out of bounds");
  auto memberType = members[indexVal];
  auto projectedType = getProjectedType(
      memberType, recordType.getMemberIsField()[indexVal], Capability::value);
  if (projectedType != getField().getType())
    return emitOpError("projected type mismatch");
  return mlir::success();
}

//===-----------------------------------------------------------------------===//
// Reussir Dialect Operations Registration
//===-----------------------------------------------------------------------===//
void ReussirDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Reussir/IR/ReussirOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// gatherCompiledModules
//===----------------------------------------------------------------------===//
std::unique_ptr<llvm::Module>
gatherCompiledModules(mlir::ModuleOp moduleOp, llvm::LLVMContext &context,
                      llvm::StringRef dataLayout) {
  // Collect all polyffi operations with compiledModule
  llvm::SmallVector<ReussirPolyFFIOp> opsWithCompiledModule;
  moduleOp.walk([&](ReussirPolyFFIOp op) {
    if (op.getCompiledModule())
      opsWithCompiledModule.push_back(op);
  });
  // Handle empty case - create an empty module if no compiledModule is
  // found
  if (opsWithCompiledModule.empty()) {
    auto module = std::make_unique<llvm::Module>("empty", context);
    module->setDataLayout(dataLayout);
    return module;
  }
  // Parse bitcode and link all modules together
  std::unique_ptr<llvm::Module> finalModule;
  for (auto op : opsWithCompiledModule) {
    // Extract bitcode from DenseElementsAttr
    auto compiledModuleOpt = op.getCompiledModule();
    if (!compiledModuleOpt)
      continue;
    mlir::DenseElementsAttr bitcodeAttr =
        llvm::dyn_cast<mlir::DenseElementsAttr>(*compiledModuleOpt);
    if (!bitcodeAttr) {
      llvm::errs() << "compiledModule is not a DenseElementsAttr\n";
      return nullptr;
    }
    // Get the raw bitcode data
    llvm::ArrayRef<char> bitcodeData = bitcodeAttr.getRawData();
    // Create a MemoryBuffer from the bitcode data
    std::unique_ptr<llvm::MemoryBuffer> memBuffer =
        llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(bitcodeData.data(), bitcodeData.size()),
            "compiledModule", /*RequiresNullTerminator=*/false);
    // Parse the bitcode
    llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrErr =
        llvm::parseBitcodeFile(memBuffer->getMemBufferRef(), context);
    if (!moduleOrErr) {
      llvm::errs() << "Failed to parse bitcode from polyffi op: "
                   << llvm::toString(moduleOrErr.takeError()) << "\n";
      return nullptr;
    }
    std::unique_ptr<llvm::Module> parsedModule = std::move(*moduleOrErr);
    parsedModule->setDataLayout(dataLayout);
    // Link into final module
    if (!finalModule) {
      // First module becomes the base
      finalModule = std::move(parsedModule);
    } else {
      // Link subsequent modules into the final module
      if (llvm::Linker::linkModules(*finalModule, std::move(parsedModule))) {
        llvm::errs() << "Failed to link LLVM modules\n";
        return nullptr;
      }
    }
  }
  // Erase all the polyffi operations collected in step 1
  for (auto op : opsWithCompiledModule)
    op.erase();
  return finalModule;
}
} // namespace reussir
