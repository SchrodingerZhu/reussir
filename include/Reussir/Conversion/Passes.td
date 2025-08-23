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
    : Pass<"reussir-lowering-"#mnemonic, target>;

//===----------------------------------------------------------------------===//
// BasicOpsLowering
//===----------------------------------------------------------------------===//
def ReussirBasicOpsLoweringPass
    : ReussirLoweringPass<"basic-ops", "::mlir::ModuleOp"> {
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

//===----------------------------------------------------------------------===//
// SCFOpsLowering
//===----------------------------------------------------------------------===//
def ReussirSCFOpsLoweringPass
    : ReussirLoweringPass<"scf-ops"> {
  let summary = "lower Reussir SCF operations to MLIR SCF operations";
  let description = [{
    `reussir-lowering-scf-ops` lowers Reussir SCF operations to MLIR SCF
    operations.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::reussir::ReussirDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::ub::UBDialect"];
}

//===----------------------------------------------------------------------===//
// DropExpansion
//===----------------------------------------------------------------------===//
def ReussirDropExpansionPass
    : Pass<"reussir-drop-expansion"> {
  let summary = "expand Reussir drop operations";
  let description = [{
    `reussir-drop-expansion` expands Reussir drop operations.

    0. it elminates no-op drop operations.
    1. for outlined drop, it creates a new function for the drop operation.
       For type `T`, in normal version,the function name is `core::reussir::drop_in_place<T>`;
       as for flex version, the function name is `core::reussir::drop_in_place_regional<T>`.
    2. for normal drop, it inserts destruction operations for inner elements.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect"];
}
#endif // REUSSIR_CONVERSION_PASSES_TD
