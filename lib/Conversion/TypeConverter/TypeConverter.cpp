
#include "Reussir/Conversion/TypeConverter.h"
#include "Reussir/IR/ReussirTypes.h"
#include <mlir/IR/BuiltinTypes.h>

namespace reussir {
namespace {
mlir::LowerToLLVMOptions getLowerOptions(mlir::ModuleOp op) {
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
}; // namespace

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp op)
    : mlir::LLVMTypeConverter(op.getContext(), getLowerOptions(op)),
      dataLayout(op) {
  // Record types
  addConversion(
      [this](RecordType type, llvm::SmallVectorImpl<mlir::Type> &results) {
        return convertRecordType(type, results);
      });

  // Pointer-like types: RefType, RegionType, RcType, TokenType
  addConversion([this](RefType type) {
    return mlir::LLVM::LLVMPointerType::get(&getContext());
  });
  addConversion([this](RegionType type) {
    return mlir::LLVM::LLVMPointerType::get(&getContext());
  });
  addConversion([this](RcType type) {
    return mlir::LLVM::LLVMPointerType::get(&getContext());
  });
  addConversion([this](TokenType type) {
    return mlir::LLVM::LLVMPointerType::get(&getContext());
  });
  addConversion([this](RawPtrType type) {
    return mlir::LLVM::LLVMPointerType::get(&getContext());
  });

  // Closure types
  addConversion([this](ClosureType type) {
    // Convert to LLVM struct: { void* vtable, void* arg_start, void* arg_cursor
    // }
    llvm::SmallVector<mlir::Type> members;
    members.push_back(
        mlir::LLVM::LLVMPointerType::get(&getContext())); // vtable
    members.push_back(
        mlir::LLVM::LLVMPointerType::get(&getContext())); // arg_start
    members.push_back(
        mlir::LLVM::LLVMPointerType::get(&getContext())); // arg_cursor
    return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members);
  });

  // Nullable types
  addConversion(
      [this](NullableType type) { return convertType(type.getPtrTy()); });

  // RcBox types
  addConversion([this](RcBoxType type) {
    llvm::SmallVector<mlir::Type> members;
    if (type.isRegional()) {
      members.push_back(mlir::LLVM::LLVMPointerType::get(&getContext()));
      members.push_back(mlir::LLVM::LLVMPointerType::get(&getContext()));
      members.push_back(mlir::LLVM::LLVMPointerType::get(&getContext()));
    } else
      members.push_back(getIndexType());
    members.push_back(convertType(type.getElementType()));
    return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members);
  });
}

std::optional<llvm::LogicalResult> LLVMTypeConverter::convertRecordType(
    RecordType type, llvm::SmallVectorImpl<mlir::Type> &results) {
  PopGuard popGuard;
  mlir::StringAttr name = type.getName();
  mlir::LLVM::LLVMStructType structType;

  if (name) {
    structType = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    auto &callStack = getCurrentThreadRecursiveStack();
    if (llvm::is_contained(callStack, structType)) {
      results.push_back(structType);
      return mlir::success();
    }
    callStack.push_back(structType);
    popGuard.install(callStack);
  }

  llvm::SmallVector<mlir::Type> members;
  if (type.getKind() == reussir::RecordKind::variant) {
    // For variant records, we need to include the tag type as the first member
    members.push_back(getIndexType());
    auto [size, _y, representative] =
        type.getElementRegionLayoutInfo(getDataLayout());
    // member can be all empty
    if (representative) {
      members.push_back(convertType(representative));
      auto representativeSize = dataLayout.getTypeSize(representative);
      // Pad the representative type to the size of the record
      if (representativeSize < size)
        members.push_back(mlir::LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(&getContext(), 8),
            size.getFixedValue() - representativeSize.getFixedValue()));
    }
  } else {
    for (auto [member, capability] :
         llvm::zip(type.getMembers(), type.getMemberCapabilities())) {
      mlir::Type projectedType =
          getProjectedType(member, capability, Capability::unspecified);
      members.push_back(convertType(projectedType));
    }
  }
  if (!name)
    structType = mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members);
  if (name && failed(structType.setBody(members, false)))
    return mlir::failure();

  results.push_back(structType);
  return mlir::success();
}
} // namespace reussir
