//===-- Jit.h - Reussir JIT codegen C API ----------------------*- C -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// LLVM-side codegen helpers for the Rust ORC JIT (reussir-jit). These are the
// pieces that cannot be expressed through the LLVM-C ORC API the Rust engine is
// built on: the Reussir custom LLVM optimization pipeline (which runs C++
// passes) and TPDE object compilation. They operate on opaque `LLVMModuleRef`
// values so the Rust side can keep driving the ORC session through llvm-sys.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_JIT_H
#define REUSSIR_C_JIT_H

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimization level, mirroring the backend's `ReussirOptOption` C enum.
typedef enum ReussirJitOptLevel {
  ReussirJitOptNone = 0,
  ReussirJitOptDefault = 1,
  ReussirJitOptAggressive = 2,
  ReussirJitOptSize = 3,
  ReussirJitOptTpde = 4,
} ReussirJitOptLevel;

// Runs the Reussir LLVM optimization pipeline on `module` in place: the
// RuntimeFunctionAttributor pass, the default per-module pipeline at the
// requested level, then the AllocationSimplification pass. A no-op for the
// `None` and `Tpde` levels (matching the C++ backend).
void reussirRunBackendLLVMPipeline(LLVMModuleRef module, ReussirJitOptLevel opt);

// Compiles `module` to an ELF object file using TPDE, after setting the given
// data layout and target triple on it. Returns NULL if TPDE support was not
// compiled in or compilation fails; otherwise the caller owns the returned
// memory buffer (e.g. hands it to `LLVMOrcLLJITAddObjectFile`).
LLVMMemoryBufferRef reussirTpdeCompileToObject(LLVMModuleRef module,
                                               const char *dataLayout,
                                               const char *triple);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_JIT_H
