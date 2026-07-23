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
/// This file implements a minimal `opt`-style driver over LLVM IR that
/// registers the Reussir LLVM passes with the new pass manager, so lit tests
/// can exercise them by name (alone or inside standard pipelines) without
/// relying on an external `opt` binary carrying our passes.
///
//===----------------------------------------------------------------------===//

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>

#include "Reussir/LLVMPass/AllocationSimplication.h"
#include "Reussir/LLVMPass/LinearRecurrence.h"
#include "Reussir/LLVMPass/RuntimeFunctionAttributor.h"

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename("o",
                                          llvm::cl::desc("Output filename"),
                                          llvm::cl::value_desc("filename"),
                                          llvm::cl::init("-"));
llvm::cl::opt<std::string>
    passPipeline("passes",
                 llvm::cl::desc("Textual pass pipeline (same syntax as opt)"),
                 llvm::cl::init(""));
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Reussir LLVM pass driver\n");

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(inputFilename, diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }

  llvm::PassBuilder passBuilder;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  passBuilder.registerModuleAnalyses(mam);
  passBuilder.registerCGSCCAnalyses(cgam);
  passBuilder.registerFunctionAnalyses(fam);
  passBuilder.registerLoopAnalyses(lam);
  passBuilder.crossRegisterProxies(lam, fam, cgam, mam);

  passBuilder.registerPipelineParsingCallback(
      [](llvm::StringRef name, llvm::ModulePassManager &mpm,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (name == "reussir-recursion-linearization") {
          mpm.addPass(reussir::llvmpass::RecursionLinearizationPass());
          return true;
        }
        if (name == "reussir-allocation-simplication") {
          mpm.addPass(reussir::llvmpass::AllocationSimplicationPass());
          return true;
        }
        if (name == "reussir-runtime-function-attributor") {
          mpm.addPass(reussir::llvmpass::RuntimeFunctionAttributorPass());
          return true;
        }
        return false;
      });
  passBuilder.registerPipelineParsingCallback(
      [](llvm::StringRef name, llvm::FunctionPassManager &fpm,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (name == "reussir-linear-recurrence-matexp") {
          fpm.addPass(reussir::llvmpass::LinearRecurrenceMatExpPass());
          return true;
        }
        return false;
      });

  llvm::ModulePassManager mpm;
  if (llvm::Error error =
          passBuilder.parsePassPipeline(mpm, passPipeline)) {
    llvm::errs() << argv[0] << ": " << llvm::toString(std::move(error))
                 << "\n";
    return 1;
  }
  mpm.run(*module, mam);

  if (llvm::verifyModule(*module, &llvm::errs())) {
    llvm::errs() << argv[0] << ": module verification failed\n";
    return 1;
  }

  std::error_code errorCode;
  llvm::ToolOutputFile output(outputFilename, errorCode,
                              llvm::sys::fs::OF_Text);
  if (errorCode) {
    llvm::errs() << argv[0] << ": " << errorCode.message() << "\n";
    return 1;
  }
  module->print(output.os(), nullptr);
  output.keep();
  return 0;
}
