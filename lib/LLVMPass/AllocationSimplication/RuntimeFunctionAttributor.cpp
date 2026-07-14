//===- RuntimeFunctionAttributor.cpp - Reussir runtime attrib pass ------===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"

#include <optional>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

namespace reussir::llvmpass {
namespace {

bool setAllocateAttributes(llvm::Function &fn) {
  bool changed = false;
  llvm::LLVMContext &ctx = fn.getContext();
  constexpr auto allocKind = llvm::AllocFnKind::Alloc |
                             llvm::AllocFnKind::Uninitialized |
                             llvm::AllocFnKind::Aligned;

  if (!fn.hasFnAttribute(llvm::Attribute::AllocKind) ||
      fn.getFnAttribute(llvm::Attribute::AllocKind).getAllocKind() !=
          allocKind) {
    fn.addFnAttr(llvm::Attribute::getWithAllocKind(ctx, allocKind));
    changed = true;
  }

  if (!fn.hasFnAttribute(llvm::Attribute::AllocSize)) {
    fn.addFnAttr(llvm::Attribute::getWithAllocSizeArgs(ctx, 1, std::nullopt));
    changed = true;
  } else {
    auto allocSizeArgs =
        fn.getFnAttribute(llvm::Attribute::AllocSize).getAllocSizeArgs();
    if (allocSizeArgs.first != 0 || allocSizeArgs.second.has_value()) {
      fn.addFnAttr(llvm::Attribute::getWithAllocSizeArgs(ctx, 1, std::nullopt));
      changed = true;
    }
  }

  if (fn.getFnAttribute("alloc-family").getValueAsString() != "reussir") {
    fn.addFnAttr("alloc-family", "reussir");
    changed = true;
  }

  return changed;
}

// `__reussir_allocate_small(size)`: same allocation family and kind as the
// generic entry, but the size is the sole argument (allocsize index 0) and
// there is no alignment parameter — the natural-alignment contract is
// discharged by the compiler at every call site.
bool setAllocateSmallAttributes(llvm::Function &fn) {
  bool changed = false;
  llvm::LLVMContext &ctx = fn.getContext();
  constexpr auto allocKind =
      llvm::AllocFnKind::Alloc | llvm::AllocFnKind::Uninitialized;

  if (!fn.hasFnAttribute(llvm::Attribute::AllocKind) ||
      fn.getFnAttribute(llvm::Attribute::AllocKind).getAllocKind() !=
          allocKind) {
    fn.addFnAttr(llvm::Attribute::getWithAllocKind(ctx, allocKind));
    changed = true;
  }

  auto hasExpectedAllocSize = [&] {
    if (!fn.hasFnAttribute(llvm::Attribute::AllocSize))
      return false;
    auto args =
        fn.getFnAttribute(llvm::Attribute::AllocSize).getAllocSizeArgs();
    return args.first == 0 && !args.second.has_value();
  };
  if (!hasExpectedAllocSize()) {
    fn.addFnAttr(llvm::Attribute::getWithAllocSizeArgs(ctx, 0, std::nullopt));
    changed = true;
  }

  if (fn.getFnAttribute("alloc-family").getValueAsString() != "reussir") {
    fn.addFnAttr("alloc-family", "reussir");
    changed = true;
  }

  return changed;
}

bool setReallocateAttributes(llvm::Function &fn) {
  bool changed = false;
  llvm::LLVMContext &ctx = fn.getContext();
  constexpr auto reallocKind = llvm::AllocFnKind::Realloc |
                               llvm::AllocFnKind::Uninitialized |
                               llvm::AllocFnKind::Aligned;

  if (!fn.hasFnAttribute(llvm::Attribute::AllocKind) ||
      fn.getFnAttribute(llvm::Attribute::AllocKind).getAllocKind() !=
          reallocKind) {
    fn.addFnAttr(llvm::Attribute::getWithAllocKind(ctx, reallocKind));
    changed = true;
  }

  if (!fn.hasFnAttribute(llvm::Attribute::AllocSize)) {
    fn.addFnAttr(llvm::Attribute::getWithAllocSizeArgs(ctx, 4, std::nullopt));
    changed = true;
  } else {
    auto allocSizeArgs =
        fn.getFnAttribute(llvm::Attribute::AllocSize).getAllocSizeArgs();
    if (allocSizeArgs.first != 4 || allocSizeArgs.second.has_value()) {
      fn.addFnAttr(llvm::Attribute::getWithAllocSizeArgs(ctx, 4, std::nullopt));
      changed = true;
    }
  }

  if (fn.getFnAttribute("alloc-family").getValueAsString() != "reussir") {
    fn.addFnAttr("alloc-family", "reussir");
    changed = true;
  }

  return changed;
}

bool setDeallocateAttributes(llvm::Function &fn) {
  bool changed = false;
  llvm::LLVMContext &ctx = fn.getContext();
  constexpr auto freeKind = llvm::AllocFnKind::Free;

  if (!fn.hasFnAttribute(llvm::Attribute::AllocKind) ||
      fn.getFnAttribute(llvm::Attribute::AllocKind).getAllocKind() !=
          freeKind) {
    fn.addFnAttr(llvm::Attribute::getWithAllocKind(ctx, freeKind));
    changed = true;
  }

  if (fn.getFnAttribute("alloc-family").getValueAsString() != "reussir") {
    fn.addFnAttr("alloc-family", "reussir");
    changed = true;
  }

  if (fn.hasFnAttribute(llvm::Attribute::AllocSize)) {
    fn.removeFnAttr(llvm::Attribute::AllocSize);
    changed = true;
  }

  return changed;
}

} // namespace

llvm::PreservedAnalyses
RuntimeFunctionAttributorPass::run(llvm::Module &module,
                                   llvm::ModuleAnalysisManager &) {
  bool changed = false;

  if (auto *allocate = module.getFunction("__reussir_allocate"))
    changed |= setAllocateAttributes(*allocate);

  if (auto *allocateSmall = module.getFunction("__reussir_allocate_small"))
    changed |= setAllocateSmallAttributes(*allocateSmall);

  if (auto *deallocate = module.getFunction("__reussir_deallocate"))
    changed |= setDeallocateAttributes(*deallocate);

  if (auto *reallocate = module.getFunction("__reussir_reallocate"))
    changed |= setReallocateAttributes(*reallocate);

  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace reussir::llvmpass
