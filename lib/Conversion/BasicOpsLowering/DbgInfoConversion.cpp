//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements lowering of Reussir debug info attributes to LLVM DI.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/BasicOpsLowering.h"
#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/Path.h>
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

// Whether `boxed` wraps a fused-header variant record (`{i32 count-slot,
// i32 tag, payload}` — see TypeConverter). Such a record's debug composite
// describes the TAG-FIRST VIEW (base at the tag, 4 bytes into the record):
// the discriminant must sit at offset 0 of the composite or lldb's variant
// parser misreads it (see the variant emission below). Regional boxes carry
// their three-word header in front of the record; shared boxes of a
// fused-header record ARE the record.
bool boxedFusedVariant(DBGBoxedTypeAttr boxed) {
  auto recordTy = llvm::dyn_cast_if_present<RecordType>(
      getUnderlyingTypeFromDbgAttr(boxed.getDbgType()));
  return recordTy && recordTy.hasFusedHeader();
}

// Byte offset from the box pointer to the data the payload's debug composite
// describes: the tag-first view for a fused-header variant, the payload
// itself otherwise, past a regional box's three-word `{status, next, vtable}`
// header (a shared non-fused box has a one-word refcount header; a shared
// fused box has none — the record overlays it).
uint64_t boxedViewOffset(const mlir::DataLayout &dataLayout,
                         mlir::MLIRContext *ctx, DBGBoxedTypeAttr boxed,
                         mlir::Type payloadTy) {
  uint64_t indexSize = dataLayout.getTypeSize(mlir::IndexType::get(ctx));
  uint64_t payloadAlign = dataLayout.getTypeABIAlignment(payloadTy);
  if (boxedFusedVariant(boxed)) {
    uint64_t recordStart =
        boxed.getRegional() ? llvm::alignTo(3 * indexSize, payloadAlign) : 0;
    return recordStart + 4;
  }
  uint64_t headerWords = boxed.getRegional() ? 3 : 1;
  return llvm::alignTo(headerWords * indexSize, payloadAlign);
}

// State for emitting recursive record debug types. A record's debug attr is
// re-entered on a recursive field (the frontend hands us a memberless,
// same-named placeholder to break its own recursion); MLIR attributes are
// immutable and cannot reference themselves directly, so the cycle is expressed
// with the `DIRecursiveTypeAttrInterface`: the outer composite is tagged with a
// `recId` and the recursive reference becomes a `getRecSelf(recId)` placeholder
// that LLVM stitches back to the outer composite. `ancestors` maps a record
// name to the `recId` of the occurrence currently being built; `used` records
// which of those were actually referenced, so only genuinely recursive
// composites carry a `recId`.
struct RecursionState {
  llvm::DenseMap<mlir::StringAttr, mlir::DistinctAttr> ancestors;
  llvm::DenseSet<mlir::DistinctAttr> used;
};

template <typename RetType = mlir::Attribute>
RetType translateDBGAttrToLLVM(mlir::ModuleOp moduleOp, mlir::Attribute dbgAttr,
                               mlir::LLVM::DIFileAttr diFile,
                               mlir::LLVM::DICompileUnitAttr diCU,
                               mlir::LLVM::LLVMFuncOp funcOp,
                               mlir::LLVM::DIScopeAttr funcScope,
                               mlir::Location loc,
                               RecursionState *recState = nullptr) {
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
            // `bool` gets the boolean encoding so debuggers print
            // `true`/`false` instead of a character literal.
            unsigned encoding = intAttr.getIsSigned()
                                    ? llvm::dwarf::DW_ATE_signed
                                    : llvm::dwarf::DW_ATE_unsigned;
            if (intAttr.getDbgName() == "bool")
              encoding = llvm::dwarf::DW_ATE_boolean;
            return mlir::LLVM::DIBasicTypeAttr::get(
                moduleOp.getContext(), llvm::dwarf::DW_TAG_base_type,
                intAttr.getDbgName(), sizeInBits, encoding);
          })
          .template Case<DBGRecordTypeAttr>([&](DBGRecordTypeAttr recAttr)
                                                -> mlir::Attribute {
            auto *ctx = moduleOp.getContext();
            auto name = recAttr.getDbgName();

            // A record reached recursively reappears with the same name (the
            // frontend's forward-declared placeholder). If this name is already
            // being built, emit a `rec-self` placeholder tied to that
            // occurrence's `recId` instead of expanding again — LLVM resolves
            // it back to the enclosing composite, giving the recursive field a
            // real pointee type the debugger can descend.
            RecursionState localState;
            RecursionState *state = recState ? recState : &localState;
            if (auto it = state->ancestors.find(name);
                it != state->ancestors.end()) {
              state->used.insert(it->second);
              return mlir::Attribute(
                  mlir::LLVM::DICompositeTypeAttr::getRecSelf(it->second));
            }
            auto recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(ctx));
            state->ancestors.insert({name, recId});

            auto makeComposite =
                [&](unsigned tag, mlir::StringAttr name, uint64_t sizeInBits,
                    uint64_t alignInBits,
                    llvm::ArrayRef<mlir::LLVM::DINodeAttr> members)
                -> mlir::LLVM::DICompositeTypeAttr {
              return mlir::LLVM::DICompositeTypeAttr::get(
                  ctx, tag, name, /*file=*/diFile, /*line=*/0, diCU,
                  /*baseType=*/nullptr, /*flags=*/mlir::LLVM::DIFlags::Zero,
                  sizeInBits, alignInBits, /*dataLocation=*/nullptr,
                  /*rank=*/nullptr, /*allocated=*/nullptr,
                  /*associated=*/nullptr, members);
            };

            // The debug type, size, and alignment (bits) for one member. A
            // boxed (rc) member is a *pointer* to the payload composite — not
            // the composite inlined: the `DW_OP_deref` expression that presents
            // a boxed *variable* as its payload applies only to that variable's
            // slot, never to a field. Emitting it as a pointer is also what
            // keeps a recursive record (a field whose box points back to the
            // record) from sending the debugger into an unbounded layout
            // recursion.
            auto memberType = [&](mlir::Attribute typeAttr)
                -> std::optional<
                    std::tuple<mlir::LLVM::DITypeAttr, uint64_t, uint64_t>> {
              auto diType = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                  moduleOp, typeAttr, diFile, diCU, funcOp, funcScope, loc,
                  state);
              if (!diType)
                return std::nullopt;
              if (auto boxed = llvm::dyn_cast<DBGBoxedTypeAttr>(typeAttr)) {
                auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
                uint64_t ptrBits = dataLayout.getTypeSizeInBits(ptrTy);
                uint64_t ptrAlignBits =
                    dataLayout.getTypeABIAlignment(ptrTy) * 8;
                // The rc pointer aims at the box *header*, not the payload
                // view, and a member cannot carry the `DW_OP_deref`/skip
                // expression a variable would. So the pointee models the box
                // `{ header, value }` with `value` — the payload composite —
                // at the view offset: past the refcount word (one shared,
                // three regional), or at the tag (record + 4) for a
                // fused-header variant whose composite is the tag-first view.
                auto payloadUnderlying = getUnderlyingTypeFromDbgAttr(typeAttr);
                if (!payloadUnderlying)
                  return std::nullopt;
                uint64_t payloadBits =
                    dataLayout.getTypeSizeInBits(payloadUnderlying);
                uint64_t payloadAlignBytes =
                    dataLayout.getTypeABIAlignment(payloadUnderlying);
                uint64_t valueOffBytes =
                    boxedViewOffset(dataLayout, ctx, boxed, payloadUnderlying);
                uint64_t valueBits =
                    boxedFusedVariant(boxed) ? payloadBits - 32 : payloadBits;
                auto valueMember = mlir::LLVM::DIDerivedTypeAttr::get(
                    ctx, llvm::dwarf::DW_TAG_member,
                    mlir::StringAttr::get(ctx, "value"), diType, valueBits,
                    payloadAlignBytes * 8, valueOffBytes * 8,
                    /*address space=*/std::nullopt, /*extraData=*/nullptr);
                // Named `<record>$box` so the lldb formatters' `^_R` type
                // match engages on the wrapper (and its pointers) and can
                // collapse the box to its payload's active case.
                mlir::StringAttr boxName = mlir::StringAttr::get(ctx, "");
                if (auto payloadRec =
                        llvm::dyn_cast_if_present<DBGRecordTypeAttr>(
                            boxed.getDbgType()))
                  boxName = mlir::StringAttr::get(
                      ctx, payloadRec.getDbgName().getValue() + "$box");
                auto pointee =
                    makeComposite(llvm::dwarf::DW_TAG_structure_type, boxName,
                                  valueOffBytes * 8 + valueBits,
                                  payloadAlignBytes * 8, {valueMember});
                auto pointer = mlir::LLVM::DIDerivedTypeAttr::get(
                    ctx, llvm::dwarf::DW_TAG_pointer_type,
                    /*name=*/mlir::StringAttr{}, pointee, ptrBits, ptrAlignBits,
                    /*offsetInBits=*/0, /*address space=*/std::nullopt,
                    /*extraData=*/nullptr);
                return std::make_tuple(
                    mlir::cast<mlir::LLVM::DITypeAttr>(pointer), ptrBits,
                    ptrAlignBits);
              }
              auto underlying = getUnderlyingTypeFromDbgAttr(typeAttr);
              if (!underlying)
                return std::nullopt;
              return std::make_tuple(
                  diType, dataLayout.getTypeSizeInBits(underlying),
                  dataLayout.getTypeABIAlignment(underlying) * 8);
            };

            auto sizeInBits =
                dataLayout.getTypeSizeInBits(recAttr.getUnderlyingType());
            auto alignInBits =
                dataLayout.getTypeABIAlignment(recAttr.getUnderlyingType()) * 8;

            // Physical field offsets (bits) for `count` fields with the given
            // sizes/alignments, starting at `baseBits`. Compound records are
            // *packed* — members laid out in descending storage-alignment
            // order (TypeConverter/getPackedOrder), not declaration order —
            // so the offsets must be accumulated along the packed order and
            // reported back per declared field. Falls back to declaration
            // order when the underlying record is unavailable (e.g. a layout
            // rebuilt from textual MIR).
            // The base is added AFTER packing: a fused-header variant's case
            // fields are view-relative (view base = the tag, 4 bytes into an
            // 8-aligned record), so aligning `base + offset` per field would
            // snap them past their real positions.
            auto physicalOffsets =
                [&](RecordType recTy, llvm::ArrayRef<uint64_t> fieldSizeBits,
                    llvm::ArrayRef<uint64_t> fieldAlignBits,
                    uint64_t baseBits) -> llvm::SmallVector<uint64_t> {
              llvm::SmallVector<uint32_t> order(fieldSizeBits.size());
              for (uint32_t i = 0; i < order.size(); ++i)
                order[i] = i;
              if (recTy && recTy.isCompound() &&
                  recTy.getMembers().size() == fieldSizeBits.size())
                order = recTy.getPackedOrder(dataLayout);
              llvm::SmallVector<uint64_t> offsets(fieldSizeBits.size());
              uint64_t current = 0;
              for (uint32_t logical : order) {
                current = llvm::alignTo(current, fieldAlignBits[logical]);
                offsets[logical] = baseBits + current;
                current += fieldSizeBits[logical];
              }
              return offsets;
            };

            if (recAttr.getIsVariant()) {
              // A variant lays out as a tag plus a payload area large enough
              // for any case. MLIR's `DICompositeTypeAttr` cannot express a
              // discriminated `DW_TAG_variant_part`, so the cases are modelled
              // as an overlapping union carried in a `payload` member next to
              // the `tag`; `fixupVariantDebugInfo` rewrites that into the real
              // variant_part after translation.
              //
              // The encoding matches rustc's — the shape debuggers are
              // actually tested against: the DISCRIMINANT SITS AT OFFSET 0 of
              // the described composite, each case is a struct spanning the
              // whole composite with its fields at composite-relative
              // offsets, and every case member is at offset 0. Anything else
              // is valid DWARF that lldb misreads: its variant parser drops
              // the discriminant's own member location (the value is even
              // misapplied as a bitfield width), so a tag anywhere but the
              // composite's start decodes garbage.
              //
              // A value variant already starts at its tag. A fused-header
              // variant (`{i32 count, i32 tag, payload}`) does not — so its
              // composite describes the TAG-FIRST VIEW of the box, base at
              // the tag (box + 4): the variable-side `DIExpression` and the
              // member-side box wrapper (see boxedExpr / memberType) both
              // present the box from that view, and gdb reads the same
              // standard DWARF.
              auto recordTy = llvm::dyn_cast_if_present<RecordType>(
                  recAttr.getUnderlyingType());

              // Pass 1: translate every case's fields; the payload area's
              // alignment (needed for its offset) is the max field alignment.
              struct CaseField {
                mlir::StringAttr name;
                mlir::LLVM::DITypeAttr type;
                uint64_t sizeBits;
                uint64_t alignBits;
              };
              struct CaseInfo {
                mlir::StringAttr name;
                RecordType record; // the arm's record type; may be null
                llvm::SmallVector<CaseField> fields;
              };
              llvm::SmallVector<CaseInfo> caseInfos;
              uint64_t payloadAlignBytes = 1;
              for (auto element : recAttr.getMembers()) {
                auto memberAttr = llvm::dyn_cast<DBGRecordMemberAttr>(element);
                if (!memberAttr)
                  return nullptr;
                CaseInfo info{memberAttr.getName(), {}, {}};
                if (auto caseRec = llvm::dyn_cast<DBGRecordTypeAttr>(
                        memberAttr.getTypeAttr())) {
                  info.record = llvm::dyn_cast_if_present<RecordType>(
                      caseRec.getUnderlyingType());
                  for (auto fieldElem : caseRec.getMembers()) {
                    auto fieldAttr =
                        llvm::dyn_cast<DBGRecordMemberAttr>(fieldElem);
                    if (!fieldAttr)
                      return nullptr;
                    auto fieldInfo = memberType(fieldAttr.getTypeAttr());
                    if (!fieldInfo)
                      return nullptr;
                    auto [fieldTy, fieldBits, fieldAlignBits] = *fieldInfo;
                    payloadAlignBytes =
                        std::max(payloadAlignBytes, fieldAlignBits / 8);
                    info.fields.push_back({fieldAttr.getName(), fieldTy,
                                           fieldBits, fieldAlignBits});
                  }
                } else {
                  // Not a record payload: model it as the case's single field.
                  auto fieldInfo = memberType(memberAttr.getTypeAttr());
                  if (!fieldInfo)
                    return nullptr;
                  auto [fieldTy, fieldBits, fieldAlignBits] = *fieldInfo;
                  payloadAlignBytes =
                      std::max(payloadAlignBytes, fieldAlignBits / 8);
                  info.fields.push_back({mlir::StringAttr::get(ctx, "f0"),
                                         fieldTy, fieldBits, fieldAlignBits});
                }
                caseInfos.push_back(std::move(info));
              }

              // Mirror the converted layout exactly (see TypeConverter): a
              // fused-header variant is `{i32 count-slot, i32 tag, payload}`
              // — its view starts at the tag, 4 bytes in — and a value
              // variant `{minimal-width tag, payload}` (tag at 0, payload at
              // its natural alignment past the tag). Every offset below is
              // view-relative.
              mlir::Type tagIntTy = recordTy
                                        ? mlir::Type(recordTy.getTagType())
                                        : mlir::Type(mlir::IndexType::get(ctx));
              uint64_t tagSizeBits = dataLayout.getTypeSizeInBits(tagIntTy);
              uint64_t tagSizeBytes = dataLayout.getTypeSize(tagIntTy);
              bool fusedHeader = recordTy && recordTy.hasFusedHeader();
              uint64_t viewShiftBytes = fusedHeader ? 4 : 0;
              uint64_t payloadOffsetBytes =
                  llvm::alignTo(fusedHeader ? 8 : tagSizeBytes,
                                payloadAlignBytes) -
                  viewShiftBytes;
              uint64_t viewSizeBits = sizeInBits - viewShiftBytes * 8;
              auto tagTy = mlir::LLVM::DIBasicTypeAttr::get(
                  ctx, llvm::dwarf::DW_TAG_base_type,
                  mlir::StringAttr::get(ctx, "<tag>"), tagSizeBits,
                  llvm::dwarf::DW_ATE_unsigned);
              auto tagMember = mlir::LLVM::DIDerivedTypeAttr::get(
                  ctx, llvm::dwarf::DW_TAG_member,
                  mlir::StringAttr::get(ctx, "tag"), tagTy, tagSizeBits,
                  tagSizeBytes * 8, /*offsetInBits=*/0,
                  /*address space=*/std::nullopt, /*extraData=*/nullptr);

              // Pass 2: build each view-size case struct with view-relative
              // field offsets (packed order within the arm), and the
              // overlapping union of the cases.
              llvm::SmallVector<mlir::LLVM::DINodeAttr> cases;
              for (auto &info : caseInfos) {
                llvm::SmallVector<uint64_t> fieldSizes, fieldAligns;
                for (auto &field : info.fields) {
                  fieldSizes.push_back(field.sizeBits);
                  fieldAligns.push_back(field.alignBits);
                }
                auto offsets =
                    physicalOffsets(info.record, fieldSizes, fieldAligns,
                                    payloadOffsetBytes * 8);
                llvm::SmallVector<mlir::LLVM::DINodeAttr> fields;
                for (auto [field, offset] : llvm::zip(info.fields, offsets))
                  fields.push_back(mlir::LLVM::DIDerivedTypeAttr::get(
                      ctx, llvm::dwarf::DW_TAG_member, field.name, field.type,
                      field.sizeBits, field.alignBits, offset,
                      /*address space=*/std::nullopt, /*extraData=*/nullptr));
                auto caseComposite =
                    makeComposite(llvm::dwarf::DW_TAG_structure_type, info.name,
                                  viewSizeBits, alignInBits, fields);
                cases.push_back(mlir::LLVM::DIDerivedTypeAttr::get(
                    ctx, llvm::dwarf::DW_TAG_member, info.name, caseComposite,
                    viewSizeBits, alignInBits, /*offsetInBits=*/0,
                    /*address space=*/std::nullopt, /*extraData=*/nullptr));
              }
              auto payloadUnion =
                  makeComposite(llvm::dwarf::DW_TAG_union_type,
                                mlir::StringAttr::get(ctx, ""), viewSizeBits,
                                payloadAlignBytes * 8, cases);
              auto payloadMember = mlir::LLVM::DIDerivedTypeAttr::get(
                  ctx, llvm::dwarf::DW_TAG_member,
                  mlir::StringAttr::get(ctx, "payload"), payloadUnion,
                  viewSizeBits, payloadAlignBytes * 8,
                  /*offsetInBits=*/0, /*address space=*/std::nullopt,
                  /*extraData=*/nullptr);
              llvm::SmallVector<mlir::LLVM::DINodeAttr> members = {
                  tagMember, payloadMember};
              auto composite =
                  makeComposite(llvm::dwarf::DW_TAG_structure_type, name,
                                viewSizeBits, alignInBits, members);
              state->ancestors.erase(name);
              if (!state->used.contains(recId))
                return composite;
              return mlir::cast<mlir::LLVM::DICompositeTypeAttr>(
                  composite.withRecId(recId));
            }

            // A compound record: translate the fields, then place them at
            // their *packed* physical offsets (declaration order only breaks
            // ties — see physicalOffsets above).
            auto recordTy = llvm::dyn_cast_if_present<RecordType>(
                recAttr.getUnderlyingType());
            struct FieldInfo {
              mlir::StringAttr name;
              mlir::LLVM::DITypeAttr type;
              uint64_t sizeBits;
              uint64_t alignBits;
            };
            llvm::SmallVector<FieldInfo> fields;
            for (auto element : recAttr.getMembers()) {
              auto memberAttr = llvm::dyn_cast<DBGRecordMemberAttr>(element);
              if (!memberAttr)
                return nullptr;
              auto info = memberType(memberAttr.getTypeAttr());
              if (!info)
                return nullptr;
              auto [memberTy, memberSizeBits, memberAlignBits] = *info;
              fields.push_back({memberAttr.getName(), memberTy, memberSizeBits,
                                memberAlignBits});
            }
            llvm::SmallVector<uint64_t> fieldSizes, fieldAligns;
            for (auto &field : fields) {
              fieldSizes.push_back(field.sizeBits);
              fieldAligns.push_back(field.alignBits);
            }
            auto offsets =
                physicalOffsets(recordTy, fieldSizes, fieldAligns, /*base*/ 0);
            llvm::SmallVector<mlir::LLVM::DINodeAttr> members;
            for (auto [field, offset] : llvm::zip(fields, offsets))
              members.push_back(mlir::LLVM::DIDerivedTypeAttr::get(
                  moduleOp->getContext(), llvm::dwarf::DW_TAG_member,
                  field.name, field.type, field.sizeBits, field.alignBits,
                  offset,
                  /*address space=*/std::nullopt, /*extraData=*/nullptr));
            auto composite = makeComposite(llvm::dwarf::DW_TAG_class_type, name,
                                           sizeInBits, alignInBits, members);
            state->ancestors.erase(name);
            if (!state->used.contains(recId))
              return composite;
            return mlir::cast<mlir::LLVM::DICompositeTypeAttr>(
                composite.withRecId(recId));
          })
          // A boxed value is presented as its payload composite; the pointer is
          // dereferenced (and the header skipped) by the `dbg.declare`'s
          // `DIExpression`, built where the variable is declared.
          .template Case<DBGBoxedTypeAttr>(
              [&](DBGBoxedTypeAttr boxed) -> mlir::Attribute {
                return translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                    moduleOp, boxed.getDbgType(), diFile, diCU, funcOp,
                    funcScope, loc, recState);
              })
          .template Case<DBGSubprogramAttr>([&](DBGSubprogramAttr spAttr)
                                                -> mlir::Attribute {
            auto linkageName = funcOp.getSymNameAttr();
            llvm::SmallVector<mlir::LLVM::DINodeAttr> argTypes;
            for (auto paramAttr : spAttr.getTypeParams()) {
              auto paramTy = translateDBGAttrToLLVM<mlir::LLVM::DITypeAttr>(
                  moduleOp, paramAttr, diFile, diCU, funcOp, funcScope, loc);
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
                diCU, diFile, spAttr.getRawName(), linkageName, diFile, line,
                /*scopeLine=*/line, mlir::LLVM::DISubprogramFlags::Definition,
                emptyRoutine, {}, {});

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
  auto slot =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrType, value.getType(), one);
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
  // Dereference the rc pointer, then skip to the data the payload composite
  // describes: past the box header (one ref-count word shared, three words
  // regional), or to the tag-first view (record + 4) for a fused-header
  // variant. See boxedViewOffset.
  uint64_t offset = boxedViewOffset(dataLayout, context, boxed, payloadTy);
  llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr, 2> elems = {
      mlir::LLVM::DIExpressionElemAttr::get(context, llvm::dwarf::DW_OP_deref,
                                            {})};
  if (offset != 0)
    elems.push_back(mlir::LLVM::DIExpressionElemAttr::get(
        context, llvm::dwarf::DW_OP_plus_uconst, {offset}));
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
  // Reussir is a Rust-like language (sum types, `_R…` mangling). Tagging the
  // compile unit as Rust is both more accurate and what makes debuggers expose
  // the `DW_TAG_variant_part` an enum lowers to (see `fixupVariantDebugInfo`):
  // under a C/C++ language they ignore variant parts entirely.
  auto dbgCompileUnitAttr = mlir::LLVM::DICompileUnitAttr::get(
      mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
      llvm::dwarf::DW_LANG_Rust, llvmDIFileAttr,
      mlir::StringAttr::get(context, "reussir"), true,
      mlir::LLVM::DIEmissionKind::Full);
  mlir::OpBuilder builder(moduleOp);
  moduleOp->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
    mlir::LocationAttr funcLoc = funcOp->getLoc();
    if (auto fused =
            llvm::dyn_cast_if_present<mlir::FusedLocWith<DBGSubprogramAttr>>(
                funcLoc)) {
      // The subprogram's own file: a multi-file translation unit stamps each
      // function's `FileLineColLoc` with its declaration file, which may not
      // be the compile unit's (the module attribute names the crate root).
      // Deriving the `DIFile` from the location keeps `decl_file` correct;
      // the module-level file remains the fallback for a function whose
      // location carries no file (e.g. re-ingested IR with missing sources).
      mlir::LLVM::DIFileAttr funcFileAttr = llvmDIFileAttr;
      for (mlir::Location loc : fused.getLocations()) {
        if (auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
          llvm::StringRef file = flc.getFilename().getValue();
          if (!file.empty())
            funcFileAttr = mlir::LLVM::DIFileAttr::get(
                context, llvm::sys::path::filename(file),
                llvm::sys::path::parent_path(file));
          break;
        }
      }

      auto subprogram = translateDBGAttrToLLVM<mlir::LLVM::DISubprogramAttr>(
          moduleOp, fused.getMetadata(), funcFileAttr, dbgCompileUnitAttr,
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
        // `dbg_func_args` describes the SOURCE parameters; the lowered
        // function may take leading implicit arguments in front of them — a
        // `regional` function receives the region it allocates into as its
        // first argument. Bind each debug attr to the corresponding trailing
        // argument, or a regional fn's first source parameter gets described
        // as the region pointer.
        size_t implicitLeadingArgs =
            funcOp.getNumArguments() >= dbgFuncArgsAttr.size()
                ? funcOp.getNumArguments() - dbgFuncArgsAttr.size()
                : 0;
        for (auto [idx, argAttr] :
             llvm::enumerate(dbgFuncArgsAttr.getValue())) {
          if (auto funcArgAttr = mlir::dyn_cast<DBGFuncArgAttr>(argAttr)) {
            auto translated =
                translateDBGAttrToLLVM<mlir::LLVM::DILocalVariableAttr>(
                    moduleOp, funcArgAttr, funcFileAttr, dbgCompileUnitAttr,
                    funcOp, subprogram, updatedFuncLoc);
            if (translated &&
                idx + implicitLeadingArgs < funcOp.getNumArguments()) {
              auto argValue = funcOp.getArgument(idx + implicitLeadingArgs);
              auto &entryBlock = funcOp.getBody().front();
              builder.setInsertionPointToStart(&entryBlock);
              // Spill the argument to a stack slot and describe *that* with
              // `dbg.declare`. A `dbg.value` on the SSA argument goes stale
              // once its registers are reused (right after the prologue spills
              // it), so a debugger reads garbage; a stack slot is a stable
              // location.
              //
              // The spill is emitted at a line-0 (prologue) location so it is
              // covered by the function prologue: a debugger setting a
              // breakpoint on the function skips to the first body statement,
              // by which point the argument has been stored and is inspectable.
              mlir::Location prologueLoc = mlir::FileLineColLoc::get(
                  context, funcFileAttr.getName().getValue(),
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
      // `DBGLocalVar` on its fused location. Conversion copies that location
      // onto every op it expands the definition into, so to describe the
      // variable just once we pick the op whose result *escapes* the tagged
      // group — i.e. is used by an op that does not carry the same metadata;
      // that is the value the rest of the program sees. We spill it and
      // `dbg.declare` it (declaring each variable once), then strip the Reussir
      // metadata off every tagged op so they translate cleanly.
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
            moduleOp, meta, funcFileAttr, dbgCompileUnitAttr, funcOp,
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
          auto expr = localVarAttr
                          ? boxedExpr(moduleOp, localVarAttr.getDbgType())
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
// Runs the fused-debug-info → LLVM DI conversion as a standalone pass. It must
// be scheduled after `convert-to-llvm`, when functions are `llvm.func`
// (carrying the fused subprogram location and `reussir.dbg_func_args` the
// conversion preserves): a function's debug info only survives LLVM-IR
// translation once its `llvm.func` has a `DISubprogram`.
struct DebugInfoConversionPass
    : public impl::ReussirDebugInfoConversionPassBase<DebugInfoConversionPass> {
  using Base::Base;
  void runOnOperation() override {
    lowerFusedDBGAttributeInLocations(getOperation());
  }
};
} // namespace
} // namespace reussir
