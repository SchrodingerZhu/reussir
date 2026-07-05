//===-- TypeConverter.cpp - Reussir type converter impl ---------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/TypeConverter.h"
#include "Reussir/IR/ReussirTypes.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DataLayout.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <memory>

namespace reussir {
namespace {
mlir::LowerToLLVMOptions buildLowerOptions(mlir::ModuleOp op) {
  llvm::StringRef dataLayoutString;
  auto dataLayoutAttr = op->template getAttrOfType<mlir::StringAttr>(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (dataLayoutAttr)
    dataLayoutString = dataLayoutAttr.getValue();

  auto options = mlir::LowerToLLVMOptions(op.getContext());
  auto llvmDL = llvm::DataLayout(dataLayoutString);
  // FIXME: Should translateDataLayout in the MLIR layer be doing this?
  if (llvmDL.getPointerSizeInBits(0) == 32)
    options.overrideIndexBitwidth(32);

  options.dataLayout = llvmDL;
  return options;
}

class PopGuard {
  llvm::SmallVectorImpl<mlir::Type> *callStack;

public:
  PopGuard() : callStack(nullptr) {}

  void install(llvm::SmallVectorImpl<mlir::Type> &callStack) {
    this->callStack = &callStack;
  }
  ~PopGuard() {
    if (callStack)
      callStack->pop_back();
  }
};

thread_local llvm::DenseMap<const mlir::LLVMTypeConverter *,
                            llvm::SmallVector<mlir::Type, 8>>
    conversionCallStacks;

std::shared_ptr<mlir::DataLayout>
buildMLIRDataLayout(const mlir::LLVMTypeConverter &converter) {
  mlir::MLIRContext *context = &converter.getContext();
  context->getOrLoadDialect<mlir::DLTIDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  auto layoutModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  std::string dataLayoutString =
      converter.getDataLayout().getStringRepresentation();
  if (!dataLayoutString.empty()) {
    layoutModule->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                          mlir::StringAttr::get(context, dataLayoutString));
    layoutModule->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                          mlir::translateDataLayout(converter.getDataLayout(),
                                                    context));
  }
  return std::make_shared<mlir::DataLayout>(layoutModule);
}

std::optional<llvm::LogicalResult>
convertRecordType(mlir::LLVMTypeConverter &converter,
                  const mlir::DataLayout &dataLayout, RecordType type,
                  llvm::SmallVectorImpl<mlir::Type> &results) {
  PopGuard popGuard;
  mlir::StringAttr name = type.getName();
  mlir::LLVM::LLVMStructType structType;
  auto &callStack = conversionCallStacks[&converter];

  if (name) {
    structType =
        mlir::LLVM::LLVMStructType::getIdentified(type.getContext(), name);
    if (llvm::is_contained(callStack, structType)) {
      results.push_back(structType);
      return mlir::success();
    }
    callStack.push_back(structType);
    popGuard.install(callStack);
  }

  llvm::SmallVector<mlir::Type> members;
  if (type.getKind() == reussir::RecordKind::variant) {
    // A fused-header variant is `{i32 count slot, i32 tag, payload}` — the
    // rc box overlays its refcount on field 0. Value variants carry just
    // their minimal-width tag; either way the payload lands at its natural
    // alignment boundary after the header.
    if (type.hasFusedHeader())
      members.push_back(mlir::IntegerType::get(type.getContext(), 32));
    members.push_back(type.getTagType());
    auto [size, _unused, representative] =
        type.getElementRegionLayoutInfo(dataLayout);
    if (representative) {
      members.push_back(converter.convertType(representative));
      auto representativeSize = dataLayout.getTypeSize(representative);
      if (representativeSize < size)
        members.push_back(mlir::LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(type.getContext(), 8),
            size.getFixedValue() - representativeSize.getFixedValue()));
    }
  } else {
    size_t expectedTotalSize = dataLayout.getTypeSize(type);
    size_t currentSize = 0;
    // Struct fields are emitted in packed physical order (descending storage
    // alignment, stable on declaration order); lowerings translate logical
    // member indices through getPhysicalMemberIndex.
    for (uint32_t logical : type.getPackedOrder(dataLayout)) {
      mlir::Type member = type.getMembers()[logical];
      bool capability = type.getMemberIsField()[logical];
      mlir::Type projectedType =
          getProjectedType(member, capability, Capability::unspecified);
      auto align = dataLayout.getTypeABIAlignment(projectedType);
      if (currentSize % align != 0) {
        size_t lastMemberNeedToPad = align - (currentSize % align);
        mlir::Type lastMemberType = members.back();
        size_t lastMemberSize = dataLayout.getTypeSize(lastMemberType);
        llvm::SmallVector<mlir::Type> liftCandidates = {
            mlir::IntegerType::get(type.getContext(), 64),
            mlir::IntegerType::get(type.getContext(), 32),
            mlir::IntegerType::get(type.getContext(), 16),
            mlir::IntegerType::get(type.getContext(), 8),
        };
        bool lift = false;
        for (auto liftCandidate : liftCandidates) {
          auto liftCandidateSize = dataLayout.getTypeSize(liftCandidate);
          if (lastMemberSize < liftCandidateSize &&
              lastMemberSize + lastMemberNeedToPad == liftCandidateSize) {
            members.back() = liftCandidate;
            lift = true;
            break;
          }
        }
        if (!lift) {
          members.back() = mlir::LLVM::LLVMStructType::getLiteral(
              type.getContext(),
              {lastMemberType, mlir::LLVM::LLVMArrayType::get(
                                   mlir::IntegerType::get(type.getContext(), 8),
                                   lastMemberNeedToPad)});
        }
        currentSize += lastMemberNeedToPad;
      }
      members.push_back(converter.convertType(projectedType));
      currentSize += dataLayout.getTypeSize(projectedType);
    }
    if (currentSize < expectedTotalSize)
      members.push_back(mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(type.getContext(), 8),
          expectedTotalSize - currentSize));
  }

  if (!name)
    structType = mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), members);
  if (name && failed(structType.setBody(members, false)))
    return mlir::failure();

  results.push_back(structType);
  return mlir::success();
}
} // namespace

mlir::LowerToLLVMOptions getReussirToLLVMOptions(mlir::ModuleOp op) {
  return buildLowerOptions(op);
}

void populateReussirToLLVMTypeConversions(mlir::LLVMTypeConverter &converter) {
  auto dataLayout = buildMLIRDataLayout(converter);
  converter.addConversion(
      [&converter, dataLayout](RecordType type,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
        return convertRecordType(converter, *dataLayout, type, results);
      });

  converter.addConversion([](RefType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](HoleType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  // A constructor context is a {root, hole} pointer pair; a null hole is the
  // empty context.
  converter.addConversion([](CctxType type) {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  {ptrTy, ptrTy});
  });
  converter.addConversion([](RegionType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](RcType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](ViewType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](TokenType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](RawPtrType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });

  converter.addConversion([](ClosureType type) {
    llvm::SmallVector<mlir::Type> members;
    members.push_back(mlir::LLVM::LLVMPointerType::get(type.getContext()));
    members.push_back(mlir::LLVM::LLVMPointerType::get(type.getContext()));
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), members);
  });

  converter.addConversion([&converter](NullableType type) {
    return converter.convertType(type.getPtrTy());
  });

  converter.addConversion([&converter](RcBoxType type) -> mlir::Type {
    // A box of a fused-header variant IS the variant record: its refcount
    // lives in the record's leading count slot.
    if (auto recordTy = llvm::dyn_cast<RecordType>(type.getElementType());
        recordTy && recordTy.hasFusedHeader() && !type.isRegional())
      return converter.convertType(recordTy);
    llvm::SmallVector<mlir::Type> members;
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    if (type.isRegional()) {
      members.push_back(ptrTy);
      members.push_back(ptrTy);
      members.push_back(ptrTy);
    } else
      members.push_back(mlir::IntegerType::get(type.getContext(), 32));
    members.push_back(converter.convertType(type.getElementType()));
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), members);
  });

  converter.addConversion([&converter](ClosureBoxType type) {
    llvm::SmallVector<mlir::Type> members;
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    members.push_back(ptrTy);
    members.push_back(ptrTy);
    for (auto payloadType : type.getPayloadTypes())
      members.push_back(converter.convertType(payloadType));
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), members);
  });

  converter.addConversion([&converter](StrType type) {
    auto indexTy = converter.getIndexType();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  {ptrTy, indexTy});
  });

  converter.addConversion([&converter](ArrayType type) {
    mlir::Type lowered = converter.convertType(type.getElementType());
    for (int64_t extent : llvm::reverse(type.getShape()))
      lowered = mlir::LLVM::LLVMArrayType::get(lowered, extent);
    return lowered;
  });
}
} // namespace reussir
