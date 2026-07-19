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
/// This file implements the Reussir translation driver.
///
//===----------------------------------------------------------------------===//

#include "Reussir/IR/ReussirDialect.h"
#include "Reussir/IR/ReussirOps.h"
#include <llvm/Linker/Linker.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllTranslations.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::TranslateFromMLIRRegistration registration(
      "reussir-to-llvmir", "Translate MLIR to LLVMIR",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        mlir::ModuleOp moduleOp = mlir::cast<mlir::ModuleOp>(op);
        std::unique_ptr<llvm::Module> polyffiModule =
            reussir::gatherCompiledModules(moduleOp, llvmContext, "");
        if (!polyffiModule)
          return mlir::failure();
        auto llvmModule = translateModuleToLLVMIR(moduleOp, llvmContext);
        if (!llvmModule)
          return mlir::failure();
        polyffiModule->setDataLayout(llvmModule->getDataLayout());
        if (llvm::Linker::linkModules(*llvmModule, std::move(polyffiModule)))
          return mlir::failure();
        llvmModule->removeDebugIntrinsicDeclarations();
        llvmModule->print(output, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        // The Reussir dialect carries its own LLVM translation interface
        // (registered by the dialect itself), so inserting it suffices for
        // the ops/attributes that survive conversion into the LLVM module.
        registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect,
                        reussir::ReussirDialect>();
        mlir::registerAllToLLVMIRTranslations(registry);
      });
  return failed(
      mlir::mlirTranslateMain(argc, argv, "Reussir translation driver\n"));
}
