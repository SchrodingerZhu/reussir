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
/// C API entry points for the Reussir dialect. These let Rust bindings built on
/// the MLIR C API (mlir-sys / melior) register and load the out-of-tree Reussir
/// dialect alongside the upstream dialects it depends on.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_DIALECTS_H
#define REUSSIR_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the dialect handle for the Reussir dialect. The handle can be
// inserted into a dialect registry or registered/loaded into a context.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Reussir, reussir);

// Registers the Reussir dialect together with every upstream dialect, extension
// and LLVM/builtin translation it relies on into the given context, then loads
// all available dialects. This mirrors the dialect set the C++ backend builds
// its compilation context from, so a freshly created context is ready to parse
// and lower Reussir IR.
void reussirRegisterAllDialects(MlirContext context);

// Populates the given dialect registry with the Reussir dialect, its dependent
// dialects, and the extensions and translations the backend pipeline relies on.
// Building a context from this registry (rather than appending afterward)
// ensures dialect extensions (e.g. the arith/func ConvertToLLVM interfaces) are
// applied as dialects load.
void reussirPopulateRegistry(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_DIALECTS_H
