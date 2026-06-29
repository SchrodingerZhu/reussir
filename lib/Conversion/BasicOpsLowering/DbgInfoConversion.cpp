
//===-- DbgInfoConversion.cpp - Reussir debug info conversion ---*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Reussir debug info attributes to LLVM DI.
//
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRDEBUGINFOCONVERSIONPASS
#include "Reussir/Conversion/Passes.h.inc"

namespace {

// Helper function to extract line and column from MLIR locations.
// Walks through FusedLoc to find the first FileLineColLoc with valid line info.
std::pair<unsigned, unsigned> extractLineCol(mlir::Location loc) {
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc))
    return {fileLoc.getLine(), fileLoc.getColumn()};
  if (auto fusedLoc = llvm::dyn_cast<mlir::FusedLoc>(loc)) {
    for (auto inner : fusedLoc.getLocations()) {
      auto [line, col] = extractLineCol(inner);
      if (line != 0)
        return {line, col};
    }
  }
  if (auto nameLoc = llvm::dyn_cast<mlir::NameLoc>(loc))
    return extractLineCol(nameLoc.getChildLoc());
  if (auto callSiteLoc = llvm::dyn_cast<mlir::CallSiteLoc>(loc))
    return extractLineCol(callSiteLoc.getCallee());
  return {0, 0};
}

mlir::Type getUnderlyingTypeFromDbgAttr(mlir::Attribute dbgAttr) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Type>(dbgAttr)
      .template Case<DBGFPTypeAttr>(
          [](DBGFPTypeAttr fptAttr) { return fptAttr.getInnerType(); })
      .template Case<DBGIntTypeAttr>(
          [](DBGIntTypeAttr intAttr) { return intAttr.getInnerType(); })
      .template Case<DBGBoxedTypeAttr>([](DBGBoxedTypeAttr boxed) {
        return getUnderlyingTypeFromDbgAttr(boxed.getDbgType());
      })
      .template Case<DBGRecordTypeAttr>(
          [](DBGRecordTypeAttr recAttr) { return recAttr.getUnderlyingType(); })
      .Default([](auto attr) { return mlir::Type{}; });
}

template <typename RetType = mlir::Attribute>
RetType translateDBGAttrToLLVM(mlir::ModuleOp moduleOp, mlir::Attribute dbgAttr,
                               mlir::LLVM::DIFileAttr diFile,
                               mlir::LLVM::DICompileUnitAttr diCU,
                               mlir::LLVM::LLVMFuncOp funcOp,
                               mlir::LLVM::DIScopeAttr funcScope,
                               mlir::Location loc) {
  mlir::DataLayout dataLayout{moduleOp};
  return llvm::dyn_cast_if_present<RetType>(
      llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(dbgAttr)
          .template Case<DBGFPTypeAttr>([&](DBGFPTypeAttr fptAttr) {
            auto sizeInBits =
                dataLayout.getTypeSizeInBits(fptAttr.getInnerType());
            return mlir::LLVM::DIBasicTypeAttr::get(
                moduleOp.getContext(), llvm::dwarf::DW_TAG_base_type,
                fptAttr.getDbgName(), sizeInBits, llvm::dwarf::DW_ATE_float);
          })
          .template Case<DBGIntTypeAttr>([&](DBGIntTypeAttr intAttr) {
            auto sizeInBits =
                dataLayout.getTypeSizeInBits(intAttr.getInnerType());
            return mlir::LLVM::DIBasicTypeAttr::get(
                moduleOp.getContext(), llvm::dwarf::DW_TAG_base_type,
                intAttr.getDbgName(), sizeInBits,
                intAttr.getIsSigned() ? llvm::dwarf::DW_ATE_signed
                                      : llvm::dwarf::DW_ATE_unsigned);
          })
          // TODO: we ignore boxed and variant for now
          .template Case<DBGRecordTypeAttr>([&](DBGRecordTypeAttr recAttr)
                                                -> mlir::Attribute {
            if (recAttr.getIsVariant())
              return nullptr;
            llvm::SmallVector<mlir::LLVM::DINodeAttr> members;
            size_t currentOffset = 0;
            for (auto element : recAttr.getMembers()) {
              auto memberAttr = llvm::dyn_cast<DBGRecordMemberAttr>(element);
              if (!memberAttr)
                return nullptr;
              auto memberTy = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                  moduleOp, memberAttr.getTypeAttr(), diFile, diCU, funcOp,
                  funcScope, loc);
              if (!memberTy)
                return nullptr;
              auto memberUnderlyingTy =
                  getUnderlyingTypeFromDbgAttr(memberAttr.getTypeAttr());
              if (!memberUnderlyingTy)
                return nullptr;
              auto sizeInBits =
                  dataLayout.getTypeSizeInBits(memberUnderlyingTy);
              auto alignInBytes =
                  dataLayout.getTypeABIAlignment(memberUnderlyingTy);
              currentOffset = llvm::alignTo(currentOffset, alignInBytes);
              // align current offset
              members.push_back(mlir::LLVM::DIDerivedTypeAttr::get(
                  moduleOp->getContext(), llvm::dwarf::DW_TAG_member,
                  memberAttr.getName(), memberTy, sizeInBits, alignInBytes * 8,
                  currentOffset * 8,
                  /*address space=*/std::nullopt, /*extraData=*/nullptr));
              currentOffset += sizeInBits / 8;
            }
            auto sizeInBits =
                dataLayout.getTypeSizeInBits(recAttr.getUnderlyingType());
            auto alignInBits =
                dataLayout.getTypeABIAlignment(recAttr.getUnderlyingType()) * 8;
#if LLVM_VERSION_MAJOR >= 22
            return mlir::LLVM::DICompositeTypeAttr::get(
                moduleOp.getContext(), llvm::dwarf::DW_TAG_class_type,
                recAttr.getDbgName(), /*file=*/diFile, /*line=*/0, diCU,
                /*baseType=*/nullptr, /*flags=*/mlir::LLVM::DIFlags::Zero,
                sizeInBits, alignInBits,
                /*dataLocation=*/nullptr, /*rank=*/nullptr,
                /*allocated=*/nullptr, /*associated=*/nullptr, members);
#else
            return mlir::LLVM::DICompositeTypeAttr::get(
                moduleOp.getContext(), llvm::dwarf::DW_TAG_class_type,
                recAttr.getDbgName(), /*file=*/diFile, /*line=*/0, diCU,
                /*baseType=*/nullptr, /*flags=*/mlir::LLVM::DIFlags::Zero,
                sizeInBits, alignInBits, members, nullptr, nullptr, nullptr,
                nullptr);
#endif
          })
          // A boxed value is presented as its payload composite; the pointer is
          // dereferenced (and the header skipped) by the `dbg.declare`'s
          // `DIExpression`, built where the variable is declared.
          .template Case<DBGBoxedTypeAttr>(
              [&](DBGBoxedTypeAttr boxed) -> mlir::Attribute {
                return translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                    moduleOp, boxed.getDbgType(), diFile, diCU, funcOp,
                    funcScope, loc);
              })
          .template Case<DBGSubprogramAttr>(
              [&](DBGSubprogramAttr spAttr) -> mlir::Attribute {
                auto linkageName = funcOp.getSymNameAttr();
                llvm::SmallVector<mlir::LLVM::DINodeAttr> argTypes;
                for (auto paramAttr : spAttr.getTypeParams()) {
                  auto paramTy = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                      moduleOp, paramAttr, diFile, diCU, funcOp, funcScope,
                      loc);
                  if (!paramTy)
                    return nullptr;
                  argTypes.push_back(paramTy);
                }
                // TODO: function type is not emitted now
                auto emptyRoutine = mlir::LLVM::DISubroutineTypeAttr::get(
                    moduleOp.getContext(), {});
                // Extract the line from the function's location. The fifth and
                // sixth arguments are `line` and `scopeLine` (both lines) — not
                // line/column.
                auto [line, col] = extractLineCol(loc);
                (void)col;
                auto res = mlir::LLVM::DISubprogramAttr::get(
                    moduleOp.getContext(),
                    mlir::DistinctAttr::create(
                        mlir::UnitAttr::get(moduleOp.getContext())),
                    false,
                    mlir::DistinctAttr::create(
                        mlir::UnitAttr::get(moduleOp.getContext())),
                    diCU, diFile, spAttr.getRawName(), linkageName, diFile,
                    line, /*scopeLine=*/line,
                    mlir::LLVM::DISubprogramFlags::Definition, emptyRoutine, {},
                    {});

                return res;
              })
          .template Case<DBGLocalVarAttr>(
              [&](DBGLocalVarAttr localVarAttr) -> mlir::Attribute {
                auto underlyingTy =
                    getUnderlyingTypeFromDbgAttr(localVarAttr.getDbgType());
                auto varType = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                    moduleOp, localVarAttr.getDbgType(), diFile, diCU, funcOp,
                    funcScope, loc);
                if (!varType)
                  return nullptr;
                auto scope = funcScope ? funcScope : diFile;
                auto alignInBits =
                    dataLayout.getTypeABIAlignment(underlyingTy) * 8;
                // Extract line/column from the operation's location
                // Note: 5th parameter is 'arg' (argument number), not column.
                // For local variables (not function parameters), arg should be
                // 0.
                auto [line, col] = extractLineCol(loc);
                (void)col; // Column is not used in DILocalVariable
                return mlir::LLVM::DILocalVariableAttr::get(
                    moduleOp.getContext(), scope, localVarAttr.getVarName(),
                    diFile, line, /*arg=*/0, alignInBits, varType,
                    mlir::LLVM::DIFlags::Zero);
              })
          .template Case<DBGFuncArgAttr>(
              [&](DBGFuncArgAttr funcArgAttr) -> mlir::Attribute {
                auto underlyingTy =
                    getUnderlyingTypeFromDbgAttr(funcArgAttr.getDbgType());
                auto varType = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                    moduleOp, funcArgAttr.getDbgType(), diFile, diCU, funcOp,
                    funcScope, loc);
                if (!varType)
                  return nullptr;
                auto scope = funcScope ? funcScope : diFile;
                auto alignInBits =
                    dataLayout.getTypeABIAlignment(underlyingTy) * 8;
                // For function arguments, use the 1-based arg index
                auto [line, col] = extractLineCol(loc);
                (void)col;
                return mlir::LLVM::DILocalVariableAttr::get(
                    moduleOp.getContext(), scope, funcArgAttr.getArgName(),
                    diFile, line, funcArgAttr.getArgIndex(), alignInBits,
                    varType, mlir::LLVM::DIFlags::Zero);
              })
          .Default([](auto attr) { return attr; }));
}

// Spill `value` to a fresh stack slot at the builder's insertion point and
// describe the slot with `dbg.declare`. A stack slot is a stable location: a
// `dbg.value` on an SSA value goes stale once its registers are reused, so the
// debugger would read garbage. `expr` transforms the slot address into the
// variable's location (empty for a plain variable; a deref + offset for a boxed
// one, where the slot holds an `rc` pointer to the payload).
void spillAndDeclare(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::MLIRContext *context, mlir::Value value,
                     mlir::LLVM::DILocalVariableAttr varInfo,
                     mlir::LLVM::DIExpressionAttr expr = {}) {
  auto one = mlir::LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                            builder.getI32IntegerAttr(1));
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  auto slot = mlir::LLVM::AllocaOp::create(builder, loc, ptrType,
                                           value.getType(), one);
  mlir::LLVM::StoreOp::create(builder, loc, value, slot);
  mlir::LLVM::DbgDeclareOp::create(builder, loc, slot, varInfo, expr);
}

// If `dbgType` describes a boxed (`[shared]`) value, return the `DIExpression`
// that reaches the payload from the slot holding the `rc` pointer: dereference
// the pointer, then skip the box header. Otherwise an empty expression.
mlir::LLVM::DIExpressionAttr boxedExpr(mlir::ModuleOp moduleOp,
                                       mlir::Attribute dbgType) {
  auto boxed = llvm::dyn_cast_if_present<DBGBoxedTypeAttr>(dbgType);
  if (!boxed)
    return {};
  auto payloadTy = getUnderlyingTypeFromDbgAttr(boxed.getDbgType());
  if (!payloadTy)
    return {};
  auto *context = moduleOp.getContext();
  mlir::DataLayout dataLayout{moduleOp};
  // The box header is a run of pointer-sized words: one ref-count word for a
  // shared box `{ count, payload }`, or three (`{ status, next, vtable }`) for a
  // regional box. The payload follows, aligned up to its own alignment.
  uint64_t headerWords = boxed.getRegional() ? 3 : 1;
  uint64_t indexSize =
      dataLayout.getTypeSize(mlir::IndexType::get(context));
  uint64_t offset = llvm::alignTo(headerWords * indexSize,
                                  dataLayout.getTypeABIAlignment(payloadTy));
  llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr, 2> elems = {
      mlir::LLVM::DIExpressionElemAttr::get(context, llvm::dwarf::DW_OP_deref,
                                            {}),
      mlir::LLVM::DIExpressionElemAttr::get(
          context, llvm::dwarf::DW_OP_plus_uconst, {offset}),
  };
  return mlir::LLVM::DIExpressionAttr::get(context, elems);
}
} // namespace

void lowerFusedDBGAttributeInLocations(mlir::ModuleOp moduleOp) {
  auto context = moduleOp.getContext();
  auto fileBasename = mlir::dyn_cast_if_present<mlir::StringAttr>(
      moduleOp->getAttr("reussir.dbg.file_basename"));
  auto fileDirectory = mlir::dyn_cast_if_present<mlir::StringAttr>(
      moduleOp->getAttr("reussir.dbg.file_directory"));
  if (!fileBasename || !fileDirectory)
    return;
  auto llvmDIFileAttr =
      mlir::LLVM::DIFileAttr::get(context, fileBasename, fileDirectory);
  auto dbgCompileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
      mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::DW_LANG_C_plus_plus_20, llvmDIFileAttr,
      mlir::StringAttr::get(context, "reussir"), true,
      mlir::LLVM::DIEmissionKind::Full);
  mlir::OpBuilder builder(moduleOp);
  moduleOp->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
    mlir::LocationAttr funcLoc = funcOp->getLoc();
    if (auto fused =
            llvm::dyn_cast_if_present<mlir::FusedLocWith<DBGSubprogramAttr>>(
                funcLoc)) {

      auto subprogram = translateDBGAttrToLLVM<mlir::LLVM::DISubprogramAttr>(
          moduleOp, fused.getMetadata(), llvmDIFileAttr, dbgCompileUnitAttr,
          funcOp, nullptr, funcLoc);
      if (subprogram) {
        auto updated =
            mlir::FusedLoc::get(context, fused.getLocations(), subprogram);
        funcOp->setLoc(updated);
      }

      // Process function argument debug info attribute
      // Note: Use funcOp->getLoc() which has the updated subprogram
      auto updatedFuncLoc = funcOp->getLoc();
      if (auto dbgFuncArgsAttr =
              funcOp->getAttrOfType<mlir::ArrayAttr>("reussir.dbg_func_args")) {
        for (auto [idx, argAttr] :
             llvm::enumerate(dbgFuncArgsAttr.getValue())) {
          if (auto funcArgAttr = mlir::dyn_cast<DBGFuncArgAttr>(argAttr)) {
            auto translated =
                translateDBGAttrToLLVM<mlir::LLVM::DILocalVariableAttr>(
                    moduleOp, funcArgAttr, llvmDIFileAttr, dbgCompileUnitAttr,
                    funcOp, subprogram, updatedFuncLoc);
            if (translated && idx < funcOp.getNumArguments()) {
              auto argValue = funcOp.getArgument(idx);
              auto &entryBlock = funcOp.getBody().front();
              builder.setInsertionPointToStart(&entryBlock);
              // Spill the argument to a stack slot and describe *that* with
              // `dbg.declare`. A `dbg.value` on the SSA argument goes stale once
              // its registers are reused (right after the prologue spills it),
              // so a debugger reads garbage; a stack slot is a stable location.
              //
              // The spill is emitted at a line-0 (prologue) location so it is
              // covered by the function prologue: a debugger setting a breakpoint
              // on the function skips to the first body statement, by which point
              // the argument has been stored and is inspectable.
              mlir::Location prologueLoc =
                  mlir::FileLineColLoc::get(context, fileBasename.getValue(),
                                            /*line=*/0, /*column=*/0);
              spillAndDeclare(builder, prologueLoc, context, argValue,
                              translated,
                              boxedExpr(moduleOp, funcArgAttr.getDbgType()));
            }
          }
        }
        // Remove the attribute after processing
        funcOp->removeAttr("reussir.dbg_func_args");
      }
      // Local variables: codegen tags the op defining a `let` binding with a
      // `DBGLocalVar` on its fused location. Conversion copies that location onto
      // every op it expands the definition into, so to describe the variable just
      // once we pick the op whose result *escapes* the tagged group — i.e. is
      // used by an op that does not carry the same metadata; that is the value
      // the rest of the program sees. We spill it and `dbg.declare` it (declaring
      // each variable once), then strip the Reussir metadata off every tagged op
      // so they translate cleanly.
      //
      // Declaration is deferred to after the walk so the escape check sees the
      // original, un-stripped locations.
      llvm::SmallVector<std::tuple<mlir::Operation *, mlir::Location,
                                   mlir::LLVM::DILocalVariableAttr,
                                   mlir::LLVM::DIExpressionAttr>>
          toDeclare;
      llvm::SmallVector<std::pair<mlir::Operation *, mlir::Location>> toStrip;
      llvm::DenseSet<mlir::Attribute> declared;
      funcOp->walk([&](mlir::Operation *op) {
        auto innerFused =
            llvm::dyn_cast_if_present<mlir::FusedLoc>(op->getLoc());
        if (!innerFused)
          return;
        mlir::Attribute meta = innerFused.getMetadata();
        auto localVar = translateDBGAttrToLLVM<mlir::LLVM::DILocalVariableAttr>(
            moduleOp, meta, llvmDIFileAttr, dbgCompileUnitAttr, funcOp,
            subprogram, op->getLoc());
        if (!localVar)
          return;
        mlir::Location varLoc = innerFused.getLocations().empty()
                                    ? mlir::Location(innerFused)
                                    : innerFused.getLocations().front();
        toStrip.push_back({op, varLoc});
        if (op->getNumResults() != 1 || !declared.insert(localVar).second)
          return;
        bool escapes = false;
        for (mlir::Operation *user : op->getResult(0).getUsers()) {
          auto userFused =
              llvm::dyn_cast_if_present<mlir::FusedLoc>(user->getLoc());
          if (!userFused || userFused.getMetadata() != meta) {
            escapes = true;
            break;
          }
        }
        if (escapes) {
          auto localVarAttr = llvm::dyn_cast<DBGLocalVarAttr>(meta);
          auto expr = localVarAttr ? boxedExpr(moduleOp, localVarAttr.getDbgType())
                                   : mlir::LLVM::DIExpressionAttr{};
          toDeclare.push_back({op, varLoc, localVar, expr});
        } else {
          declared.erase(localVar);
        }
      });
      for (auto &[op, varLoc, localVar, expr] : toDeclare) {
        builder.setInsertionPointAfter(op);
        spillAndDeclare(builder, varLoc, context, op->getResult(0), localVar,
                        expr);
      }
      for (auto &[op, varLoc] : toStrip)
        op->setLoc(varLoc);
    }
  });
}

namespace {
// Runs the fused-debug-info → LLVM DI conversion as a standalone pass. It must be
// scheduled after `convert-to-llvm`, when functions are `llvm.func` (carrying the
// fused subprogram location and `reussir.dbg_func_args` the conversion preserves):
// a function's debug info only survives LLVM-IR translation once its `llvm.func`
// has a `DISubprogram`.
struct DebugInfoConversionPass
    : public impl::ReussirDebugInfoConversionPassBase<DebugInfoConversionPass> {
  using Base::Base;
  void runOnOperation() override {
    lowerFusedDBGAttributeInLocations(getOperation());
  }
};
} // namespace
} // namespace reussir
