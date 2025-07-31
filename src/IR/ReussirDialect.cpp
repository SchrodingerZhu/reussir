#include "Reussir/IR/ReussirDialect.h"

#include "Reussir/IR/ReussirDialect.cpp.inc"

namespace reussir {

void ReussirDialect::initialize() {
  // Register the builtin Attributes.
  registerAttributes();
  // Register the builtin Types.
  registerTypes();
}

mlir::Type ReussirDialect::parseType(mlir::DialectAsmParser &parser) const {
  return {};
}
void ReussirDialect::printType(mlir::Type ty,
                               mlir::DialectAsmPrinter &p) const {}

mlir::Attribute ReussirDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                               mlir::Type type) const {
  return {};
}

void ReussirDialect::printAttribute(mlir::Attribute attr,
                                    mlir::DialectAsmPrinter &p) const {}

void ReussirDialect::registerAttributes() {}
// Register the builtin Types.
void ReussirDialect::registerTypes() {}

} // namespace reussir
