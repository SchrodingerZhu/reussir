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
/// MLIR's `DICompositeTypeAttr` has no discriminator field, so the debug-info
/// conversion can only describe an enum as a `{ tag, payload-union }` struct: a
/// debugger then shows every case overlapping and leaves the user to read the
/// tag. This post-translation pass rewrites that struct into a real DWARF
/// `DW_TAG_variant_part` — a discriminant member plus one `DW_TAG_variant` per
/// case carrying its `DW_AT_discr_value` — which gdb uses to display only the
/// live case.
///
/// It runs on the translated `llvm::Module` because the `discriminator` operand
/// (and the per-case discriminant constant) exist only at the LLVM-DI level,
/// not in MLIR's attribute. The composite is rewritten *in place*: every
/// Reussir composite is a `distinct` node, so mutating its `elements` operand
/// preserves node identity — a recursive payload that points back at the enum,
/// and every variable whose type is the enum, see the variant_part with no
/// further work.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/Passes.h"

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Module.h>

#include <optional>

using namespace llvm;

namespace reussir {
namespace {

// The shape the debug-info conversion emits for an enum: a struct whose two
// members are `tag` (the discriminant) and `payload` (a union of the per-case
// payload structs).
struct VariantShape {
  DIDerivedType *tag;
  DIDerivedType *payload;
  DICompositeType *cases; // the union; its elements are the per-case members
};

std::optional<VariantShape> matchVariant(DICompositeType *composite) {
  if (composite->getTag() != dwarf::DW_TAG_structure_type)
    return std::nullopt;
  // `distinct` is required to mutate the `elements` operand below; every
  // composite the conversion emits is distinct, but guard regardless.
  if (!composite->isDistinct())
    return std::nullopt;
  DINodeArray elements = composite->getElements();
  if (elements.size() != 2)
    return std::nullopt;
  auto *tag = dyn_cast_or_null<DIDerivedType>(elements[0]);
  auto *payload = dyn_cast_or_null<DIDerivedType>(elements[1]);
  if (!tag || !payload || tag->getName() != "tag" ||
      payload->getName() != "payload")
    return std::nullopt;
  auto *cases = dyn_cast_or_null<DICompositeType>(payload->getBaseType());
  if (!cases || cases->getTag() != dwarf::DW_TAG_union_type)
    return std::nullopt;
  return VariantShape{tag, payload, cases};
}

// Rewrite `composite`'s body into a `DW_TAG_variant_part` in place.
void rewriteVariant(LLVMContext &ctx, DICompositeType *composite,
                    const VariantShape &v) {
  DIFile *file = composite->getFile();
  uint64_t payloadOffset = v.payload->getOffsetInBits();

  // The discriminant is the tag member, marked artificial
  // (compiler-synthesised, not a source field). Scoped to the enum itself,
  // matching what rustc emits. It keeps the tag's own offset: 0 for a value
  // variant, 4 for a fused-header one (past the overlaid refcount slot).
  auto *discriminator = DIDerivedType::get(
      ctx, dwarf::DW_TAG_member, /*Name=*/StringRef(), file, /*Line=*/0,
      /*Scope=*/composite, /*BaseType=*/v.tag->getBaseType(),
      v.tag->getSizeInBits(), v.tag->getAlignInBits(), v.tag->getOffsetInBits(),
      /*DWARFAddressSpace=*/std::nullopt, /*PtrAuthData=*/std::nullopt,
      DINode::FlagArtificial);

  // One `DW_TAG_member` per case, tagged with its discriminant value in
  // `extraData` (the case index, which is the tag the constructor stores). Each
  // case payload sits at the offset the union occupied.
  auto *i64Ty = Type::getInt64Ty(ctx);
  SmallVector<Metadata *> variants;
  unsigned index = 0;
  for (DINode *element : v.cases->getElements()) {
    auto *caseMember = dyn_cast<DIDerivedType>(element);
    if (!caseMember) {
      ++index;
      continue;
    }
    auto *discrValue = ConstantAsMetadata::get(ConstantInt::get(i64Ty, index));
    variants.push_back(DIDerivedType::get(
        ctx, dwarf::DW_TAG_member, caseMember->getName(), file, /*Line=*/0,
        /*Scope=*/composite, caseMember->getBaseType(),
        caseMember->getSizeInBits(), caseMember->getAlignInBits(),
        payloadOffset, /*DWARFAddressSpace=*/std::nullopt,
        /*PtrAuthData=*/std::nullopt, DINode::FlagZero,
        /*ExtraData=*/discrValue));
    ++index;
  }

  auto *variantPart = DICompositeType::get(
      ctx, dwarf::DW_TAG_variant_part, /*Name=*/StringRef(), file, /*Line=*/0,
      /*Scope=*/composite, /*BaseType=*/nullptr, composite->getSizeInBits(),
      composite->getAlignInBits(), /*OffsetInBits=*/0, DINode::FlagZero,
      /*Elements=*/DINodeArray(MDTuple::get(ctx, variants)), /*RuntimeLang=*/0,
      /*EnumKind=*/std::nullopt, /*VTableHolder=*/nullptr,
      /*TemplateParams=*/nullptr, /*Identifier=*/StringRef(),
      /*Discriminator=*/discriminator);

  // Mutating the distinct node's `elements` keeps its identity, so all existing
  // references (recursive payloads, variable types) now resolve to the
  // variant_part.
  composite->replaceElements(DINodeArray(MDTuple::get(ctx, {variantPart})));
}

} // namespace

void fixupVariantDebugInfo(llvm::Module &module) {
  DebugInfoFinder finder;
  finder.processModule(module);

  // Collect matches first; `rewriteVariant` mutates the nodes the finder holds.
  SmallVector<std::pair<DICompositeType *, VariantShape>> work;
  for (DIType *type : finder.types())
    if (auto *composite = dyn_cast<DICompositeType>(type))
      if (auto shape = matchVariant(composite))
        work.emplace_back(composite, *shape);

  for (auto &[composite, shape] : work)
    rewriteVariant(module.getContext(), composite, shape);
}

} // namespace reussir
