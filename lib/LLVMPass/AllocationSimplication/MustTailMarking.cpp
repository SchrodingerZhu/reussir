//===- MustTailMarking.cpp - Reussir self-tail-call marking pass ---------===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Reussir lowers each function as a single-exit CFG: every return-position
// value branches (possibly through several forwarding blocks and phis) into
// one common return block. A direct self recursive call in tail position
// therefore never sits directly in front of a `ret`, and once
// DeadArgumentElimination shrinks an aggregate return the forwarding chain
// gains extractvalue/insertvalue wrappers that neither TailCallElimination
// nor CodeGenPrepare's dup-ret recognize — the recursion keeps real stack
// frames and deep workloads overflow the stack (functional-queue's `churn`).
//
// This pass runs before the module optimization pipeline, while the
// translated IR still has the plain shape `%r = call @self(...); br ...phi...
// ret`. For every direct self call whose result only flows through empty
// forwarding blocks into the function's return, it duplicates the return
// into the call's block and marks the call `musttail`: from that point on
// TailCallElimination usually rewrites the recursion into a loop, and even
// if no IR pass does, instruction selection must emit a real tail call — the
// stack no longer grows with recursion depth. Marking is skipped whenever a
// call argument is derived from a caller alloca (`musttail` forbids the
// callee to access the caller's frame).
//
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/MustTailMarking.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace reussir::llvmpass {
namespace {

// Whether any call argument may point into the caller's own frame. A
// `musttail` callee must not access the caller's allocas; the caller's frame
// is gone when the callee runs.
bool anyArgumentDerivedFromAlloca(llvm::CallInst *call) {
  for (llvm::Value *arg : call->args())
    if (arg->getType()->isPointerTy() &&
        llvm::isa<llvm::AllocaInst>(llvm::getUnderlyingObject(arg)))
      return true;
  return false;
}

// Follows the self call's result from its block through unconditional
// branches into the common return block. Every hop must be an "empty"
// forwarding block — phis plus an unconditional branch, or phis plus the
// `ret` — and the tracked value must remain the returned value (either
// rebound through a phi or passed through by dominance). Returns true iff
// the walk ends at `ret <tracked value>` (or `ret void` for a void self
// call).
bool returnsCallResult(llvm::CallInst *call) {
  llvm::Value *tracked = call;
  llvm::BasicBlock *block = call->getParent();
  llvm::Instruction *terminator = block->getTerminator();
  // A bound well past any realistic forwarding chain; guards degenerate CFGs.
  constexpr unsigned kMaxHops = 32;
  for (unsigned hop = 0; hop < kMaxHops; ++hop) {
    if (auto *ret = llvm::dyn_cast<llvm::ReturnInst>(terminator))
      return call->getType()->isVoidTy() ? ret->getNumOperands() == 0
                                         : ret->getReturnValue() == tracked;
    auto *branch = llvm::dyn_cast<llvm::BranchInst>(terminator);
    if (!branch || branch->isConditional())
      return false;
    llvm::BasicBlock *successor = branch->getSuccessor(0);
    // The successor must compute nothing of its own: phis, then the
    // terminator. Rebind the tracked value through the phi that receives it;
    // a phi that merges an unrelated value is fine as long as the eventual
    // `ret` still returns the tracked value.
    for (llvm::PHINode &phi : successor->phis())
      if (phi.getIncomingValueForBlock(block) == tracked)
        tracked = &phi;
    if (&*successor->getFirstNonPHIOrDbg() != successor->getTerminator())
      return false;
    block = successor;
    terminator = successor->getTerminator();
  }
  return false;
}

// Rewrites `call; br` into `call; ret` (removing the call's block from its
// former successor's phis) and marks the call. Void, scalar, and pointer
// returns take `musttail` — the instruction-selection guarantee. An
// aggregate return only gets the plain `tail` marker: `musttail` on a
// return the target demotes to sret is a hard instruction-selection error
// ("failed to perform tail call elimination on a call site marked
// musttail"), and how many scalars still fit return registers is a
// target-specific limit an IR pass must not encode. The duplicated `ret`
// puts the call in the exact `call; ret` shape TailCallElimination (run
// right after this pass) folds into a loop for any first-class type.
void markSelfTailCall(llvm::CallInst *call) {
  llvm::BasicBlock *block = call->getParent();
  llvm::Instruction *terminator = block->getTerminator();
  if (auto *branch = llvm::dyn_cast<llvm::BranchInst>(terminator)) {
    branch->getSuccessor(0)->removePredecessor(block);
    terminator->eraseFromParent();
    llvm::Value *result = call->getType()->isVoidTy() ? nullptr : call;
    llvm::ReturnInst::Create(call->getContext(), result, block);
  }
  llvm::Type *type = call->getType();
  bool registerTrivial = type->isVoidTy() || type->isIntegerTy() ||
                         type->isPointerTy() || type->isFloatingPointTy();
  call->setTailCallKind(registerTrivial ? llvm::CallInst::TCK_MustTail
                                        : llvm::CallInst::TCK_Tail);
}

bool processFunction(llvm::Function &function) {
  if (function.isDeclaration() || function.isVarArg())
    return false;
  llvm::SmallVector<llvm::CallInst *> selfTailCalls;
  for (llvm::BasicBlock &block : function) {
    auto *call = llvm::dyn_cast_if_present<llvm::CallInst>(
        block.getTerminator()->getPrevNode());
    // `musttail` requires the call to immediately precede the return, the
    // matching self signature is given by construction, and the calling
    // conventions coincide on a self call.
    if (!call || call->getCalledFunction() != &function ||
        call->getCallingConv() != function.getCallingConv() ||
        anyArgumentDerivedFromAlloca(call) || !returnsCallResult(call))
      continue;
    selfTailCalls.push_back(call);
  }
  for (llvm::CallInst *call : selfTailCalls)
    markSelfTailCall(call);
  return !selfTailCalls.empty();
}

} // namespace

llvm::PreservedAnalyses
MustTailMarkingPass::run(llvm::Module &module, llvm::ModuleAnalysisManager &) {
  bool changed = false;
  for (llvm::Function &function : module)
    changed |= processFunction(function);
  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace reussir::llvmpass
