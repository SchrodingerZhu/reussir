//===-- MlirCApiShim.cpp - libMLIR-C.so anchor ----------------*- C++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// Anchor translation unit for the fabricated libMLIR-C shared library. The
// distro MLIR 22 packages ship the C API only as static archives
// (libMLIRCAPI*.a) and never as the libMLIR-C.so that mlir-sys expects in
// shared-link mode. The CAPI CMake target whole-archives those archives into a
// shared object against the system libMLIR.so; this file just gives the target
// a source so CMake will build it.
//
//===----------------------------------------------------------------------===//

namespace reussir::capi {
// Exported so the shared object is never considered empty by aggressive
// linkers; carries no semantics.
extern "C" const char *reussirMlirCApiShimVersion() { return "22"; }
} // namespace reussir::capi
