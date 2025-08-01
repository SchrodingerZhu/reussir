#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Reussir/IR/ReussirDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<reussir::ReussirDialect>();
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  return failed(mlir::MlirOptMain(
      argc, argv, "Reussir analysis and optimization driver\n", registry));
}
