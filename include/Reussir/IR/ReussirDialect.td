//===-- ReussirDialect.td - Reussir dialect definition -----*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reussir dialect. This is to be used as a common
// tablegen header that provides common dialect references.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIRDIALECT_TD
#define REUSSIR_IR_REUSSIRDIALECT_TD

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ReussirDialect : Dialect {
  let name = "reussir";
  let cppNamespace = "::reussir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 1;
  let dependentDialects = [
      // arithmetic operations
      "mlir::arith::ArithDialect",
      // structured control flow operations
      "mlir::scf::SCFDialect",
      // math operations
      "mlir::math::MathDialect",
      // TBAA Tag types
      "mlir::LLVM::LLVMDialect"];
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Types.
    void registerTypes();
    // Register the builtin Operations.
    void registerOperations();
  public:
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;
 
    mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const override;
    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &p) const override;
  }];
}

#endif // REUSSIR_IR_REUSSIRDIALECT_TD
