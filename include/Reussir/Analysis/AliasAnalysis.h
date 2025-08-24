#ifndef REUSSIR_ANALYSIS_ALIASANALYSIS_H
#define REUSSIR_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace reussir {

/// Register Reussir alias analysis implementations on the given aggregate.
void registerAliasAnalysisImplementations(mlir::AliasAnalysis &aliasAnalysis);

} // namespace reussir

#endif // REUSSIR_ANALYSIS_ALIASANALYSIS_H
