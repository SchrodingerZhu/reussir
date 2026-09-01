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
/// This file defines the Reussir dialect. This is to be used as a common
/// tablegen header that provides common dialect references.
///
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
  let dependentDialects = [
      // arithmetic operations
      "mlir::arith::ArithDialect",
      // structured control flow operations
      "mlir::scf::SCFDialect",
      // math operations
      "mlir::math::MathDialect",
      // TBAA Tag types
      "mlir::LLVM::LLVMDialect",
      // UB Dialect
      "mlir::ub::UBDialect",
      // Lock-guarded cells: their data layout is the layout of the
      // corresponding `!sync.*` primitive, so any DataLayout query on such a
      // cell constructs a sync type — including paths that run before any
      // pass can declare a pass-level dependency, e.g. the parser-time
      // `cell.create` verifier sizing a token-attached mutex cell box.
      "mlir::sync::SyncDialect",
  ];
  let extraClassDeclaration = [{
  private:
    // Register the builtin Attributes.
    void registerAttributes();
    // Register the builtin Types.
    void registerTypes();
    // Register the builtin Operations.
    void registerOperations();
    // Register interfaces.
    void registerInterfaces();
    // Whether compound record members are laid out in packed physical order
    // (descending storage alignment) rather than declaration order. On by
    // default. This is a whole-compilation layout contract, held on the
    // context-loaded dialect so `RecordType::getPackedOrder` can read it, and
    // set once by the lowering driver before any layout is computed.
    bool packRecordMembers = true;
  public:
    void setPackRecordMembers(bool enable) { packRecordMembers = enable; }
    bool getPackRecordMembers() const { return packRecordMembers; }

    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

    mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const override;
    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &p) const override;
  }];
}

#endif // REUSSIR_IR_REUSSIRDIALECT_TD
