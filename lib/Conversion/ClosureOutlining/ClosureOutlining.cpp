//===-- ClosureOutlining.cpp ------------------------------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/ClosureOutlining.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirOps.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <sstream>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRCLOSUREOUTLININGPASS
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// ClosureOutliningPass
//===----------------------------------------------------------------------===//

namespace {

std::string b62encode(size_t value) {
  const char *alphabet =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  if (value == 0)
    return std::string(1, alphabet[0]);

  std::string result;
  while (value != 0) {
    uint64_t rem = value % 62;
    value = value / 62;
    result.push_back(alphabet[rem]);
  }
  std::reverse(result.begin(), result.end());
  return result;
}

class ClosureNameUniquifier {
  llvm::StringMap<size_t> nameOccurrences;

public:
  ClosureNameUniquifier() = default;
  std::string uniquify(ReussirClosureCreateOp op) {
    auto function = op->getParentOfType<mlir::func::FuncOp>();
    auto closureIdx = nameOccurrences[function.getSymName()]++;
    std::stringstream ss;
    llvm::StringRef funcNameWithoutPrefix = function.getSymName().ltrim("_R");
    std::string_view funcNameWithoutPrefixView = funcNameWithoutPrefix;
    ss << "Nv" << funcNameWithoutPrefixView << "s" << b62encode(closureIdx)
       << "_8$closure";
    return ss.str();
  }
};
struct ClosureOutliningPass
    : public impl::ReussirClosureOutliningPassBase<ClosureOutliningPass> {
  using Base::Base;

  /// Creates a function from the closure's inlined region and returns it.
  /// The function will have:
  /// - Arguments matching the closure's input types
  /// - Return type matching the closure's output type (if any)
  /// - Body cloned from the closure's region with yield replaced by return
  mlir::func::FuncOp createFunctionAndInlineRegion(ReussirClosureCreateOp op,
                                                   llvm::Twine name,
                                                   mlir::IRRewriter &rewriter);

  /// Create the closure drop function.
  mlir::func::FuncOp createClosureDropFunction(ReussirClosureCreateOp op,
                                               llvm::Twine name,
                                               mlir::IRRewriter &rewriter);

  /// Create the closure clone function
  mlir::func::FuncOp createClosureCloneFunction(ReussirClosureCreateOp op,
                                                llvm::Twine name,
                                                mlir::IRRewriter &rewriter);

  void runOnOperation() override;
};

void ClosureOutliningPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<ReussirClosureCreateOp> closureCreateOps;
  moduleOp.walk([&](ReussirClosureCreateOp op) {
    if (op.isInlined())
      closureCreateOps.push_back(op);
  });
  ClosureNameUniquifier nameUniquifier{};
  mlir::IRRewriter rewriter(moduleOp.getContext());
  for (auto op : closureCreateOps) {
    auto name = nameUniquifier.uniquify(op);
    // First, create a function, inline the region into the function and
    // change the yield op to return op during the translation.
    mlir::func::FuncOp evaluateFunction =
        createFunctionAndInlineRegion(op, name, rewriter);

    // Second, create closure's clone and drop functions
    mlir::func::FuncOp cloneFunction =
        createClosureCloneFunction(op, name, rewriter);
    mlir::func::FuncOp dropFunction =
        createClosureDropFunction(op, name, rewriter);

    // Third, create vtable operation
    rewriter.setInsertionPointToEnd(moduleOp.getBody());
    std::string vtableName = "_RNv" + name + "6vtable";

    // Create the vtable operation
    mlir::FlatSymbolRefAttr dropAttr = mlir::FlatSymbolRefAttr::get(
        rewriter.getContext(), dropFunction.getName());
    mlir::FlatSymbolRefAttr cloneAttr = mlir::FlatSymbolRefAttr::get(
        rewriter.getContext(), cloneFunction.getName());
    mlir::FlatSymbolRefAttr evaluateAttr = mlir::FlatSymbolRefAttr::get(
        rewriter.getContext(), evaluateFunction.getName());

    ReussirClosureVtableOp::create(rewriter, op.getLoc(),
                                            rewriter.getStringAttr(vtableName),
                                            evaluateAttr, dropAttr, cloneAttr);

    // Fourth, update the closure to use
    // outlined version (e.g. remove the
    // region and set vtable attribute)
    mlir::FlatSymbolRefAttr vtableAttr =
        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), vtableName);
    op.setVtableAttr(vtableAttr);
  }
}

mlir::func::FuncOp ClosureOutliningPass::createFunctionAndInlineRegion(
    ReussirClosureCreateOp op, llvm::Twine name, mlir::IRRewriter &rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  std::string invokeName = ("_RNv" + name + "8evaluate").str();
  mlir::ModuleOp moduleOp = getOperation();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  // Get the closure type to determine function signature
  ClosureType closureType =
      llvm::cast<ClosureType>(op.getClosure().getType().getElementType());
  mlir::Type outputType = closureType.getOutputType();
  ClosureBoxType closureBoxType = op.getClosureBoxType();
  RcType specializedRcType = RcType::get(rewriter.getContext(), closureBoxType);

  llvm::SmallVector<mlir::Type> resultTypes;
  if (outputType)
    resultTypes.push_back(outputType);
  mlir::FunctionType funcType =
      rewriter.getFunctionType(specializedRcType, resultTypes);

  // Create the function
  auto funcOp = mlir::func::FuncOp::create(rewriter, op.getLoc(), invokeName,
                                           funcType);
  funcOp.setPrivate();
  funcOp->setAttr("llvm.linkage",
                  mlir::LLVM::LinkageAttr::get(rewriter.getContext(),
                                               mlir::LLVM::Linkage::Internal));
  funcOp->setAttr("passthrough",
                  rewriter.getStrArrayAttr({"mustprogress", "nounwind",
                                            "willreturn", "nocallback"}));
  inheritSanitizerPassthrough(funcOp->getParentOfType<mlir::ModuleOp>(),
                              funcOp);
  funcOp.setArgAttr(0, "llvm.noalias", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.nonnull", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.noundef", rewriter.getUnitAttr());
  // Create entry block for the function
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Location loc = op.getLoc();

  // === PROLOGUE: Borrow the Rc and load all fields ===
  mlir::Value rcPtr = entryBlock->getArgument(0);

  // Borrow the rc to get a reference to the closure box
  RefType closureBoxRefType =
      rewriter.getType<RefType>(closureBoxType, Capability::unspecified);
  mlir::Value closureBoxRef =
      ReussirRcBorrowOp::create(rewriter, loc, closureBoxRefType, rcPtr);

  // Load all payload fields from the closure box
  auto payloadTypes = closureBoxType.getPayloadTypes();
  llvm::SmallVector<mlir::Value> blockArgs;
  for (size_t i = 0; i < payloadTypes.size(); ++i) {
    mlir::Type payloadType = payloadTypes[i];
    RefType payloadRefType =
        rewriter.getType<RefType>(payloadType, Capability::unspecified);

    // Get reference to this payload field
    mlir::Value payloadRef = ReussirClosureInspectPayloadOp::create(rewriter, 
        loc, payloadRefType, rewriter.getIndexAttr(i), closureBoxRef);

    // Load the payload value
    mlir::Value payloadValue =
        ReussirRefLoadOp::create(rewriter, loc, payloadType, payloadRef);
    blockArgs.push_back(payloadValue);
  }

  // Add the input arguments (from function args after rcPtr)
  for (size_t i = 1; i < entryBlock->getNumArguments(); ++i)
    blockArgs.push_back(entryBlock->getArgument(i));

  // Inline the closure block into the function entry block
  mlir::Block &closureBlock = op.getBody().front();
  rewriter.inlineBlockBefore(&closureBlock, entryBlock, entryBlock->end(),
                             blockArgs);

  // === EPILOGUE: Replace yield with fetch + conditional dec/free + return ===
  funcOp.walk([&](ReussirClosureYieldOp yieldOp) {
    rewriter.setInsertionPoint(yieldOp);
    mlir::Location yieldLoc = yieldOp.getLoc();

    // Fetch the current reference count on the rc pointer.
    auto fetch = ReussirRcFetchOp::create(rewriter, yieldLoc, rcPtr);

    // Only write the decremented count when the closure remains shared.
    auto one = mlir::arith::ConstantIndexOp::create(rewriter, yieldLoc, 1);
    auto isShared = mlir::arith::CmpIOp::create(rewriter, 
        yieldLoc, mlir::arith::CmpIPredicate::ugt, fetch.getRefCount(), one);

    auto ifOp =
        mlir::scf::IfOp::create(rewriter, yieldLoc, mlir::TypeRange{}, isShared,
                                         /*addThenRegion=*/true,
                                         /*addElseRegion=*/true);

    // Inside the then block: decrement the refcount and skip the free.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    auto decremented = mlir::arith::SubIOp::create(rewriter, 
        yieldLoc, fetch.getRefCount(), one);
    ReussirRcSetOp::create(rewriter, yieldLoc, rcPtr, decremented.getResult());
    mlir::scf::YieldOp::create(rewriter, yieldLoc);

    // Inside the else block: free the token on the last reference.
    rewriter.setInsertionPointToStart(ifOp.elseBlock());

    // Compute TokenType from RcBoxType using DataLayout
    RcBoxType rcBoxType = specializedRcType.getInnerBoxType();
    mlir::DataLayout dataLayout = mlir::DataLayout::closest(funcOp);
    size_t size = dataLayout.getTypeSize(rcBoxType).getFixedValue();
    size_t align = dataLayout.getTypeABIAlignment(rcBoxType);
    TokenType tokenType = TokenType::get(rewriter.getContext(), align, size);

    mlir::Value token =
        ReussirRcReinterpretOp::create(rewriter, yieldLoc, tokenType, rcPtr);
    ReussirTokenFreeOp::create(rewriter, yieldLoc, token);
    mlir::scf::YieldOp::create(rewriter, yieldLoc);

    // After the if, create the return
    rewriter.setInsertionPointAfter(ifOp);
    if (yieldOp.getValue())
      mlir::func::ReturnOp::create(rewriter, yieldLoc, yieldOp.getValue());
    else
      mlir::func::ReturnOp::create(rewriter, yieldLoc);

    rewriter.eraseOp(yieldOp);
  });

  return funcOp;
}

mlir::func::FuncOp ClosureOutliningPass::createClosureDropFunction(
    ReussirClosureCreateOp op, llvm::Twine name, mlir::IRRewriter &rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  std::string dropName = ("_RNv" + name + "4drop").str();
  mlir::ModuleOp moduleOp = getOperation();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  ClosureBoxType closureBoxType = op.getClosureBoxType();

  RcType specializedRcType =
      RcType::get(rewriter.getContext(), closureBoxType, Capability::shared);
  // transparently use RcType as refType of the Rc-wrapped closure box
  mlir::FunctionType funcType = rewriter.getFunctionType(
      llvm::ArrayRef<mlir::Type>{specializedRcType}, {});
  auto funcOp =
      mlir::func::FuncOp::create(rewriter, op.getLoc(), dropName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("llvm.linkage",
                  mlir::LLVM::LinkageAttr::get(rewriter.getContext(),
                                               mlir::LLVM::Linkage::Internal));
  funcOp->setAttr("passthrough",
                  rewriter.getStrArrayAttr({"mustprogress", "nounwind",
                                            "willreturn", "nocallback"}));
  inheritSanitizerPassthrough(funcOp->getParentOfType<mlir::ModuleOp>(),
                              funcOp);

  funcOp.setArgAttr(0, "llvm.noalias", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.nonnull", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.noundef", rewriter.getUnitAttr());
  // Create entry block for the function
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Value rcPtr = entryBlock->getArgument(0);
  mlir::Location loc = op.getLoc();

  // First, fetch the current reference count on the rc pointer.
  auto fetch = ReussirRcFetchOp::create(rewriter, loc, rcPtr);

  // Only write the decremented count when the closure remains shared.
  auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto isShared = mlir::arith::CmpIOp::create(rewriter, 
      loc, mlir::arith::CmpIPredicate::ugt, fetch.getRefCount(), one);

  auto ifOp = mlir::scf::IfOp::create(rewriter, loc, mlir::TypeRange{}, isShared,
                                               /*addThenRegion=*/true,
                                               /*addElseRegion=*/true);

  // Inside the then block: decrement the refcount and skip the drop.
  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  auto decremented =
      mlir::arith::SubIOp::create(rewriter, loc, fetch.getRefCount(), one);
  ReussirRcSetOp::create(rewriter, loc, rcPtr, decremented.getResult());
  mlir::scf::YieldOp::create(rewriter, loc);

  // Inside the else block: drop the closure box contents on the last
  // reference.
  rewriter.setInsertionPointToStart(ifOp.elseBlock());

  // Use rc.borrow to get a reference to the closure box
  RefType closureBoxRefType =
      rewriter.getType<RefType>(closureBoxType, Capability::unspecified);
  mlir::Value closureBoxRef =
      ReussirRcBorrowOp::create(rewriter, loc, closureBoxRefType, rcPtr);

  // Extract the cursor pointer from the closure box
  auto payloadTypes = closureBoxType.getPayloadTypes();
  RefType cursorRefType =
      rewriter.getType<RefType>(rewriter.getI8Type(), Capability::unspecified);
  mlir::Value cursorRef = ReussirClosureCursorOp::create(rewriter, 
      loc, cursorRefType, rewriter.getIndexAttr(0), closureBoxRef);

  // For each payload field, check if cursor is strictly greater than the field
  // pointer and if so, add a drop operation for the field
  for (size_t i = 0; i < payloadTypes.size(); ++i) {
    mlir::Type payloadType = payloadTypes[i];
    if (isTriviallyCopyable(payloadType))
      continue;
    RefType payloadRefType =
        rewriter.getType<RefType>(payloadType, Capability::unspecified);

    // Get reference to this payload field
    mlir::Value payloadRef = ReussirClosureInspectPayloadOp::create(rewriter, 
        loc, payloadRefType, rewriter.getIndexAttr(i), closureBoxRef);

    // Compare cursor with payload reference: if cursor > payload, the field
    // was initialized and needs to be dropped
    auto cursorGtPayload = ReussirRefCmpOp::create(rewriter, 
        loc, rewriter.getI1Type(), mlir::arith::CmpIPredicate::ugt, cursorRef,
        payloadRef);

    // Conditionally drop the field if it was initialized
    auto dropIfOp = mlir::scf::IfOp::create(rewriter, 
        loc, mlir::TypeRange{}, cursorGtPayload, /*addThenRegion=*/true,
        /*addElseRegion=*/false);
    rewriter.setInsertionPointToStart(dropIfOp.thenBlock());
    ReussirRefDropOp::create(rewriter, loc, payloadRef);
    mlir::scf::YieldOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(dropIfOp);
  }

  // Free the token after dropping all fields
  // Compute TokenType from RcBoxType using DataLayout
  RcBoxType rcBoxType = specializedRcType.getInnerBoxType();
  mlir::DataLayout dataLayout = mlir::DataLayout::closest(funcOp);
  size_t size = dataLayout.getTypeSize(rcBoxType).getFixedValue();
  size_t align = dataLayout.getTypeABIAlignment(rcBoxType);
  TokenType tokenType = TokenType::get(rewriter.getContext(), align, size);

  mlir::Value token =
      ReussirRcReinterpretOp::create(rewriter, loc, tokenType, rcPtr);
  ReussirTokenFreeOp::create(rewriter, loc, token);

  // Yield from the outer if
  mlir::scf::YieldOp::create(rewriter, loc);

  // Return from the function (after the if)
  rewriter.setInsertionPointAfter(ifOp);
  mlir::func::ReturnOp::create(rewriter, loc);

  return funcOp;
}

mlir::func::FuncOp ClosureOutliningPass::createClosureCloneFunction(
    ReussirClosureCreateOp op, llvm::Twine name, mlir::IRRewriter &rewriter) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  std::string cloneName = ("_RNv" + name + "5clone").str();
  mlir::ModuleOp moduleOp = getOperation();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  ClosureBoxType closureBoxType = op.getClosureBoxType();

  RcType specializedRcType =
      RcType::get(rewriter.getContext(), closureBoxType, Capability::shared);
  // Clone function takes Rc<ClosureBox> and returns Rc<ClosureBox>
  mlir::FunctionType funcType =
      rewriter.getFunctionType(llvm::ArrayRef<mlir::Type>{specializedRcType},
                               llvm::ArrayRef<mlir::Type>{specializedRcType});
  auto funcOp =
      mlir::func::FuncOp::create(rewriter, op.getLoc(), cloneName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("llvm.linkage",
                  mlir::LLVM::LinkageAttr::get(rewriter.getContext(),
                                               mlir::LLVM::Linkage::Internal));
  funcOp->setAttr("passthrough",
                  rewriter.getStrArrayAttr({"mustprogress", "nounwind",
                                            "willreturn", "nocallback"}));
  inheritSanitizerPassthrough(funcOp->getParentOfType<mlir::ModuleOp>(),
                              funcOp);

  funcOp.setArgAttr(0, "llvm.noalias", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.nonnull", rewriter.getUnitAttr());
  funcOp.setArgAttr(0, "llvm.noundef", rewriter.getUnitAttr());
  funcOp.setResultAttr(0, "llvm.noalias", rewriter.getUnitAttr());
  funcOp.setResultAttr(0, "llvm.nonnull", rewriter.getUnitAttr());
  funcOp.setResultAttr(0, "llvm.noundef", rewriter.getUnitAttr());
  // Create entry block for the function
  mlir::Block *entryBlock = funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);
  mlir::Value srcRcPtr = entryBlock->getArgument(0);
  mlir::Location loc = op.getLoc();

  // Use ClosureAllocaOp to create a place for assembling new closure
  RefType closureBoxRefType =
      rewriter.getType<RefType>(closureBoxType, Capability::unspecified);

  // Allocate token for RcBox<ClosureBox>
  RcBoxType rcBoxType = specializedRcType.getInnerBoxType();
  mlir::DataLayout dataLayout = mlir::DataLayout::closest(funcOp);
  size_t size = dataLayout.getTypeSize(rcBoxType).getFixedValue();
  size_t align = dataLayout.getTypeABIAlignment(rcBoxType);
  TokenType tokenType = TokenType::get(rewriter.getContext(), align, size);
  mlir::Value token = ReussirTokenAllocOp::create(rewriter, loc, tokenType);

  // Allocate assemble space on stack first
  mlir::Value dstRc = ReussirClosureInstantiateOp::create(rewriter, 
      loc, specializedRcType, token);

  // Borrow the destination closure box
  mlir::Value dstClosureBoxRef =
      ReussirRcBorrowOp::create(rewriter, loc, closureBoxRefType, dstRc);

  // Borrow the source closure box
  mlir::Value srcClosureBoxRef =
      ReussirRcBorrowOp::create(rewriter, loc, closureBoxRefType, srcRcPtr);

  // Extract the cursor pointer from the source closure box
  auto payloadTypes = closureBoxType.getPayloadTypes();
  RefType cursorRefType =
      rewriter.getType<RefType>(rewriter.getI8Type(), Capability::unspecified);
  mlir::Value srcCursorRef = ReussirClosureCursorOp::create(rewriter, 
      loc, cursorRefType, rewriter.getIndexAttr(0), srcClosureBoxRef);

  // Transfer the closure box contents, including header and applied arguments
  ReussirClosureTransferOp::create(rewriter, loc, srcClosureBoxRef,
                                            dstClosureBoxRef);

  // For each payload field, check if cursor is greater than the field
  // pointer and if so, copy the data and emit ownership acquisition
  for (size_t i = 0; i < payloadTypes.size(); ++i) {
    mlir::Type payloadType = payloadTypes[i];
    if (isTriviallyCopyable(payloadType))
      continue;
    RefType payloadRefType =
        rewriter.getType<RefType>(payloadType, Capability::unspecified);

    // Get reference to source payload field
    mlir::Value srcPayloadRef = ReussirClosureInspectPayloadOp::create(rewriter, 
        loc, payloadRefType, rewriter.getIndexAttr(i), srcClosureBoxRef);

    // Get reference to destination payload field
    mlir::Value dstPayloadRef = ReussirClosureInspectPayloadOp::create(rewriter, 
        loc, payloadRefType, rewriter.getIndexAttr(i), dstClosureBoxRef);

    // Compare cursor with payload reference: if cursor > payload, the field
    // was initialized and needs to be copied
    auto cursorGtPayload = ReussirRefCmpOp::create(rewriter, 
        loc, rewriter.getI1Type(), mlir::arith::CmpIPredicate::ugt,
        srcCursorRef, srcPayloadRef);

    // Conditionally copy the field if it was initialized
    auto copyIfOp = mlir::scf::IfOp::create(rewriter, 
        loc, mlir::TypeRange{}, cursorGtPayload, /*addThenRegion=*/true,
        /*addElseRegion=*/false);
    rewriter.setInsertionPointToStart(copyIfOp.thenBlock());

    // Emit ownership acquisition for non-trivially copyable types
    if (emitOwnershipAcquisition(dstPayloadRef, rewriter, loc).failed())
      funcOp.emitError("failed to emit ownership acquisition for payload");

    mlir::scf::YieldOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(copyIfOp);
  }

  // Return the destination Rc pointer
  mlir::func::ReturnOp::create(rewriter, loc, dstRc);

  return funcOp;
}

} // namespace

} // namespace reussir
