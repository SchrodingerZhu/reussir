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
/// This file implements the CompilePolymorphicFFI pass which compiles
/// polymorphic FFI template into concrete LLVM bitcode and embeds them into the
/// module.
///
//===----------------------------------------------------------------------===//

#include "Reussir/Conversion/Passes.h"
#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/RustCompiler.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Target/LLVMIR/Import.h>

namespace reussir {

#define GEN_PASS_DEF_REUSSIRCOMPILEPOLYMORPHICFFIPASS
#include "Reussir/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// CompilePolymorphicFFI Helper Functions
//===----------------------------------------------------------------------===//

static void formatInto(mlir::ModuleOp moduleOp, llvm::raw_ostream &os,
                       mlir::OpBuilder &builder, mlir::TypeAttr type,
                       llvm::Twine name, bool isTopLevel = true) {
  llvm::TypeSwitch<mlir::Type>(type.getValue())
      .Case<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(
          [&](mlir::Type ty) { os << ty; })
      .Case<RcType>([&](RcType rcType) {
        if (auto eleTy =
                mlir::dyn_cast<FFIObjectType>(rcType.getElementType())) {
          os << "\ntype " << name << " = " << eleTy.getFfiName().getValue()
             << ";\n";
          return;
        }
        if (rcType.getCapability() != Capability::shared &&
            rcType.getCapability() != Capability::rigid)
          llvm::report_fatal_error("rc with other capabilities is not "
                                   "supported in FFI generation");
        llvm::StringRef typePrefix =
            (rcType.getCapability() == Capability::shared)
                ? "::reussir_rt::rc::Rc"
                : "::reussir_rt::region::rusty::RigidRc";
        llvm::Twine innerName = name + "Pointee";
        formatInto(moduleOp, os, builder,
                   mlir::TypeAttr::get(rcType.getElementType()), innerName,
                   false);
        os << "\ntype " << name << " = " << typePrefix << "<" << innerName
           << ">;\n";
      })
      .Case<RecordType>([&](RecordType ty) {
        mlir::DataLayout dataLayout(moduleOp);
        mlir::SymbolTable symbolTable(moduleOp);
        bool withoutCleanup = !isTopLevel || isTriviallyCopyable(ty);
        if (withoutCleanup) {
          os << "\n#[derive(Copy, Clone)]\n#[repr(C, align("
             << dataLayout.getTypeABIAlignment(ty) << "))]\n";
          os << "struct " << name << "([u8; " << dataLayout.getTypeSize(ty)
             << "]);\n";
        } else {
          os << "#[repr(C, align(" << dataLayout.getTypeABIAlignment(ty)
             << "))]\n";
          os << "struct " << name << "([u8; " << dataLayout.getTypeSize(ty)
             << "]);\n";
          os << "unsafe extern \"C\" {\n";
          {
            os << "#[link_name = \"" << ty.getDtorName().getValue() << "\"]\n"
               << "unsafe fn " << "drop_in_place" << "(ptr: *mut " << name
               << ");\n";
            os << "#[link_name = \"" << ty.getAcquireName().getValue()
               << "\"]\n"
               << "unsafe fn " << "acquire_in_place" << "(ptr: *mut " << name
               << ");\n";
          }
          os << "}\n";
          os << "impl Drop for " << name << " {\n";
          os << "    fn drop(&mut self) {\n";
          os << "        unsafe { drop_in_place(self) };\n";
          os << "    }\n";
          os << "}\n";
          // clang-format off
            os << "impl Clone for " << name << " {\n";
            os << "    fn clone(&self) -> Self {\n";
            os << "        let mut raw_data = ::std::mem::MaybeUninit::uninit();\n";
            os << "        unsafe { ::std::ptr::copy_nonoverlapping(self, raw_data.as_mut_ptr(), 1) };\n";
            os << "        unsafe { acquire_in_place(raw_data.as_mut_ptr()) };\n";
            os << "        unsafe { raw_data.assume_init() }\n";
            os << "    }\n";
            os << "}\n";
          // clang-format on
          createDtorIfNotExists(moduleOp, ty, builder);
          emitOwnershipAcquisitionFuncIfNotExists(moduleOp, ty, builder);
        }
      })
      .Default([](mlir::Type) {
        llvm::report_fatal_error("unsupported type in FFI generation");
      });
}

static std::string monomorphize(mlir::ModuleOp moduleOp, ReussirPolyFFIOp op) {
  mlir::OpBuilder builder(moduleOp);
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  size_t cursor = 0;
  size_t index = 0;
  constexpr llvm::StringRef SUBSTART = "[:";
  constexpr llvm::StringRef SUBEND = ":]";
  const llvm::StringRef text = *op.getModuleTexture();
  bool inSubstitution = false;
  while (index < text.size()) {
    if (!inSubstitution) {
      if (text.substr(index, 2) != SUBSTART) {
        index++;
        continue;
      } else {
        inSubstitution = true;
        os << text.substr(cursor, index - cursor);
        index += 2;
        cursor = index;
      }
    } else {
      if (text.substr(index, 2) != SUBEND) {
        index++;
        continue;
      } else {
        inSubstitution = false;
        llvm::StringRef key = text.substr(cursor, index - cursor);
        mlir::Attribute attr = op.getSubstitutions()
                                   ? op.getSubstitutions()->get(key)
                                   : mlir::Attribute();
        if (!attr) {
          os << SUBSTART << key << SUBEND;
        } else {
          llvm::TypeSwitch<mlir::Attribute>(attr)
              .Case<mlir::StringAttr>(
                  [&](mlir::StringAttr str) { os << str.getValue(); })
              .Case<mlir::IntegerAttr>(
                  [&](mlir::IntegerAttr value) { os << value.getValue(); })
              .Case([&](mlir::TypeAttr attr) {
                formatInto(moduleOp, os, builder, attr, key, true);
              })
              .Default([](mlir::Attribute) {
                llvm::report_fatal_error(
                    "unsupported attribute in substitution");
              });
        }
        index += 2;
        cursor = index;
      }
    }
  }
  os << text.substr(cursor);
  return buffer;
}

} // namespace

//===----------------------------------------------------------------------===//
// CompilePolymorphicFFI Standalone Function
//===----------------------------------------------------------------------===//
mlir::LogicalResult
compilePolymorphicFFI(mlir::ModuleOp moduleOp, bool optimized,
                      llvm::StringRef rustPath,
                      llvm::ArrayRef<llvm::StringRef> libDirs) {
  llvm::LLVMContext context;
  llvm::SmallVector<ReussirPolyFFIOp> uncompiledOps;
  moduleOp.walk([&](ReussirPolyFFIOp op) {
    if (!op.getCompiledModule())
      uncompiledOps.push_back(op);
  });

  llvm::SmallVector<llvm::StringRef> additionalArgs;
  if (optimized)
    additionalArgs.push_back("-O");

  for (ReussirPolyFFIOp op : uncompiledOps) {
    std::string monomorphized = monomorphize(moduleOp, op);
    std::unique_ptr<llvm::MemoryBuffer> bitcode = compileRustSourceToBitcode(
        context, monomorphized, additionalArgs, rustPath, libDirs);
    if (!bitcode)
      return mlir::failure();
    llvm::ArrayRef<char> buffer(bitcode->getBufferStart(),
                                bitcode->getBufferSize());
    mlir::DenseElementsAttr blob = mlir::DenseElementsAttr::getFromRawBuffer(
        mlir::RankedTensorType::get(
            {static_cast<int64_t>(bitcode->getBufferSize())},
            mlir::IntegerType::get(moduleOp->getContext(), 8)),
        buffer);
    op.setCompiledModuleAttr(blob);
    op.removeModuleTextureAttr();
    op.removeSubstitutionsAttr();
  }

  return mlir::success();
}

namespace {

//===----------------------------------------------------------------------===//
// CompilePolymorphicFFI Pass
//===----------------------------------------------------------------------===//

class ReussirCompilePolymorphicFFIPass
    : public impl::ReussirCompilePolymorphicFFIPassBase<
          ReussirCompilePolymorphicFFIPass> {
public:
  using impl::ReussirCompilePolymorphicFFIPassBase<
      ReussirCompilePolymorphicFFIPass>::ReussirCompilePolymorphicFFIPassBase;

  void runOnOperation() override {
    llvm::SmallVector<llvm::StringRef> dirs(libDirs.begin(), libDirs.end());
    if (failed(
            compilePolymorphicFFI(getOperation(), optimized, rustPath, dirs)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace reussir
