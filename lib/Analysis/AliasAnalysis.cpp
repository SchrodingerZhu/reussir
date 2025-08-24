#include "Reussir/Analysis/AliasAnalysis.h"

using namespace mlir;

namespace reussir {
namespace {
/// A simple AliasAnalysis implementation for Reussir that conservatively
/// returns MayAlias/ModRef for all queries. This is intended to be registered
/// into an instance of mlir::AliasAnalysis via addAnalysisImplementation.
struct ReussirAliasAnalysisImpl {
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);
};

AliasResult ReussirAliasAnalysisImpl::alias(Value lhs, Value rhs) {
  (void)lhs;
  (void)rhs;
  return AliasResult::MayAlias;
}

ModRefResult ReussirAliasAnalysisImpl::getModRef(Operation *op,
                                                 Value location) {
  (void)op;
  (void)location;
  return ModRefResult::getModAndRef();
}
} // namespace

void registerAliasAnalysisImplementations(AliasAnalysis &aa) {
  aa.addAnalysisImplementation(ReussirAliasAnalysisImpl{});
}
} // namespace reussir
