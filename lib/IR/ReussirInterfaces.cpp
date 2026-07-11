//===-- ReussirInterfaces.cpp - Reussir Interfaces -------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces for the Reussir dialect.
//
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirInterfaces.h"
#include "Reussir/Conversion/ClosureWpd.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Transforms/InliningUtils.h>
/// Include the generated interface definitions.
#include "Reussir/IR/ReussirInterfaces.cpp.inc"

namespace reussir {
namespace {
// Translation of the Reussir ops and attributes that survive the conversion
// pipeline into the LLVM dialect module because the LLVM dialect cannot
// express their artifacts (`!type` metadata and `llvm.type.test`'s
// metadata-string operand):
//
//  * `reussir.closure.wpd_test` becomes `llvm.type.test(%vtable, !"id")`
//    feeding `llvm.assume`;
//  * the `reussir.wpd.type_ids` attribute on a closure vtable
//    `llvm.mlir.global` becomes `!type {i64 0, !"id"}` entries (the vtable
//    has a single address point) plus translation-unit `!vcall_visibility`
//    on the translated global. The visibility is what makes
//    devirtualization legal without LTO: the lowering only emits WPD
//    artifacts on a closed world (see docs/design/closure-wpd.md), so every
//    possible callee of a tested call site is inside this module.
struct ReussirLLVMTranslation : public mlir::LLVMTranslationDialectInterface {
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  mlir::LogicalResult
  convertOperation(mlir::Operation *op, llvm::IRBuilderBase &builder,
                   mlir::LLVM::ModuleTranslation &state) const override {
    auto test = llvm::dyn_cast<ReussirClosureWpdTestOp>(op);
    if (!test)
      return op->emitError("cannot translate operation to LLVM IR");
    llvm::LLVMContext &ctx = builder.getContext();
    llvm::Value *vtable = state.lookupValue(test.getVtable());
    if (!vtable || !vtable->getType()->isPointerTy())
      return test.emitError("expected the vtable to lower to a pointer");
    llvm::Value *id = llvm::MetadataAsValue::get(
        ctx, llvm::MDString::get(ctx, test.getId()));
    llvm::CallInst *hit = builder.CreateIntrinsic(llvm::Intrinsic::type_test,
                                                  {}, {vtable, id});
    builder.CreateAssumption(hit);
    return mlir::success();
  }

  mlir::LogicalResult
  amendOperation(mlir::Operation *op,
                 llvm::ArrayRef<llvm::Instruction *> instructions,
                 mlir::NamedAttribute attribute,
                 mlir::LLVM::ModuleTranslation &state) const override {
    if (attribute.getName() != kClosureWpdTypeIdsAttr)
      return mlir::success(); // Not ours to amend (e.g. the module-level
                              // vtable map, kept for introspection).
    auto global = llvm::dyn_cast<mlir::LLVM::GlobalOp>(op);
    auto ids = llvm::dyn_cast<mlir::ArrayAttr>(attribute.getValue());
    if (!global || !ids)
      return op->emitError("expected '")
             << kClosureWpdTypeIdsAttr
             << "' to be an array of strings on an llvm.mlir.global";
    auto *vtable =
        llvm::dyn_cast_or_null<llvm::GlobalObject>(state.lookupGlobal(global));
    if (!vtable)
      return op->emitError("vtable global was not translated");
    llvm::LLVMContext &ctx = vtable->getContext();
    for (mlir::Attribute id : ids)
      vtable->addTypeMetadata(
          0, llvm::MDString::get(
                 ctx, llvm::cast<mlir::StringAttr>(id).getValue()));
    vtable->setVCallVisibilityMetadata(
        llvm::GlobalObject::VCallVisibilityTranslationUnit);
    return mlir::success();
  }
};

struct ReussirInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
};
} // namespace

void ReussirDialect::registerInterfaces() {
  addInterfaces<ReussirInlinerInterface>();
  addInterfaces<ReussirLLVMTranslation>();
}
} // namespace reussir
