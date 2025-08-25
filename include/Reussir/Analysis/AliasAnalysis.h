#ifndef REUSSIR_ANALYSIS_ALIASANALYSIS_H
#define REUSSIR_ANALYSIS_ALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace reussir {

/// Register Reussir alias analysis implementations on the given aggregate.
///
/// Reussir alias analysis is a must-alias analysis (pointer equivalence
/// analysis) that tracks pointer relationships through Reussir's type system.
/// It uses a union-find data structure to maintain equivalence classes of
/// aliased pointers for rapid lookup.
///
/// Key rules:
/// 1. Identity: Same value always aliases itself (MustAlias)
/// 2. Type compatibility: Only values of the same type can alias.
///    The type compatibility is decided by the type ID in MLIR type system, not
///    the exact type equivalence.
/// 3. Reference projection: References projected from the same index of aliased
///    parents must alias
/// 4. RC borrowing: References borrowed from the same RC pointer must alias
/// 5. Nullable coercion: Values coerced from the same nullable must alias
/// 6. Record coercion: Values coerced from the same record variant must alias
/// 7. Load operations: RC/Nullable values loaded from the same reference must
/// alias
///
/// The analysis conservatively returns MayAlias when no must-alias relationship
/// can be established.
///
void registerAliasAnalysisImplementations(mlir::AliasAnalysis &aliasAnalysis);

} // namespace reussir

#endif // REUSSIR_ANALYSIS_ALIASANALYSIS_H
