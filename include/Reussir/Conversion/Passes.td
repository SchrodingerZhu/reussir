//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the conversion passes for the Reussir dialect.
//
//===----------------------------------------------------------------------===//
#ifndef REUSSIR_CONVERSION_PASSES_TD
#define REUSSIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

class ReussirLoweringPass<string mnemonic, string target = "">
  : Pass<"reussir-lowering-" # mnemonic, target>;


//===----------------------------------------------------------------------===//
// BasicOpsLowering
//===----------------------------------------------------------------------===//
def ReussirBasicOpsLoweringPass : 
  ReussirLoweringPass<"basic-ops", "::mlir::ModuleOp"> {
  let summary = "lower straight-forward operations to MLIR basic operations";
  let description = [{
    `reussir-lowering-basic-ops` lowers straight-forward operations to MLIR basic
    operations.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect", 
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect", 
                           "::mlir::math::MathDialect", 
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::ub::UBDialect",
                           "::reussir::ReussirDialect"];
}

#endif // REUSSIR_CONVERSION_PASSES_TD
