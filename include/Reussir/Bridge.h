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
/// This file implements the bridge between rust frontend and C++ backend.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_BRIDGE_H
#define REUSSIR_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// C-style enums for FFI compatibility
typedef enum ReussirOutputTarget {
  REUSSIR_OUTPUT_LLVMIR = 0,
  REUSSIR_OUTPUT_ASM = 1,
  REUSSIR_OUTPUT_OBJECT = 2
} ReussirOutputTarget;

typedef enum ReussirOptOption {
  REUSSIR_OPT_NONE = 0,
  REUSSIR_OPT_DEFAULT = 1,
  REUSSIR_OPT_AGGRESSIVE = 2,
  REUSSIR_OPT_SIZE = 3,
  REUSSIR_OPT_TPDE = 4
} ReussirOptOption;

typedef enum ReussirLogLevel {
  REUSSIR_LOG_ERROR = 0,
  REUSSIR_LOG_WARNING = 1,
  REUSSIR_LOG_INFO = 2,
  REUSSIR_LOG_DEBUG = 3,
  REUSSIR_LOG_TRACE = 4
} ReussirLogLevel;

typedef enum ReussirCodeModel {
  REUSSIR_CODE_MODEL_TINY = 0,
  REUSSIR_CODE_MODEL_SMALL = 1,
  REUSSIR_CODE_MODEL_KERNEL = 2,
  REUSSIR_CODE_MODEL_MEDIUM = 3,
  REUSSIR_CODE_MODEL_LARGE = 4,
  REUSSIR_CODE_MODEL_DEFAULT = 5
} ReussirCodeModel;

typedef enum ReussirRelocationModel {
  REUSSIR_RELOC_MODEL_STATIC = 0,
  REUSSIR_RELOC_MODEL_PIC = 1,
  REUSSIR_RELOC_MODEL_DYNAMIC = 2,
  REUSSIR_RELOC_MODEL_ROPI = 3,
  REUSSIR_RELOC_MODEL_RWPI = 4,
  REUSSIR_RELOC_MODEL_ROPI_RWPI = 5,
  REUSSIR_RELOC_MODEL_DEFAULT = 6
} ReussirRelocationModel;

// Install LLVM segfault handler and pretty stack trace support.
// Safe to call multiple times; subsequent calls are no-ops.
void reussir_bridge_setup(void);

// Query if TPDE support is compiled in
int reussir_bridge_has_tpde(void);

// Get default target triple (caller must free with free())
char *reussir_bridge_get_default_target_triple(void);

// Get default target CPU (caller must free with free())
char *reussir_bridge_get_default_target_cpu(void);

// Get the features string
char *reussir_bridge_get_default_target_features(void);

// Compile for a specific target
void reussir_bridge_compile_for_target(
    const char *mlir_module, const char *source_name, const char *output_file,
    ReussirOutputTarget target, ReussirOptOption opt, ReussirLogLevel log_level,
    const char *target_triple, const char *target_cpu,
    const char *target_features, ReussirCodeModel code_model,
    ReussirRelocationModel reloc_model, int reuse_token_across_call,
    int enable_invariant_analysis);

// An opaque stable pointer for AST.
typedef void *ASTStablePtr;
// A callback function that returns the MLIR IR of the AST.
// This callback took away the ownership of the ASTStablePtr.
// The callback shall use reussir_bridge_alloc_byte_buffer to allocate the
// memory.
typedef const char *(*ASTCallbackFn)(ASTStablePtr);
// A callback function that frees the ASTStablePtr. This is used when the AST
// unit is never materialized.
typedef void (*ASTFreeFn)(ASTStablePtr);
// An opaque handle for the JIT engine.
typedef void *ReussirJIT;

// Create a new JIT engine.
ReussirJIT reussir_bridge_jit_create(ASTCallbackFn ast_callback_fn,
                                     ASTFreeFn ast_free_fn,
                                     ReussirOptOption opt,
                                     ReussirLogLevel level);
// Destroy the JIT engine.
void reussir_bridge_jit_destroy(ReussirJIT jit);
// Add a module that should be loaded lazily.
bool reussir_bridge_jit_add_lazy_module(ReussirJIT jit, ASTStablePtr ast,
                                        char *symbol_names[],
                                        uint8_t symbol_flags[],
                                        size_t symbol_count);
// Add a module that should be loaded immediately. This function does not clean
// up texture.
bool reussir_bridge_jit_add_module(ReussirJIT jit, const char *texture);
// Lookup a symbol in the JIT engine.
void *reussir_bridge_jit_lookup_symbol(ReussirJIT jit, const char *symbol_name,
                                       bool mangled);

static inline char *reussir_bridge_alloc_byte_buffer(size_t size) {
  return (char *)malloc(size);
}

typedef void *ReussirLogger;

ReussirLogger reussir_bridge_create_stdout_logger(ReussirLogLevel level,
                                                  char *name);
void reussir_bridge_destroy_logger(ReussirLogger logger);
void reussir_bridge_log_with_level(ReussirLogger logger, ReussirLogLevel level,
                                   const char *message);
typedef struct {
  uint64_t values[4];
} ReussirStringHash;

void reussir_bridge_hash_bytes(const uint8_t *str, size_t len,
                               ReussirStringHash *out);

// String result from JIT execution
typedef struct {
  const char *ptr;
  uint64_t len;
} ReussirStrResult;

// Call JIT functions that use an explicit leading return pointer.
// The function pointer should be obtained from reussir_bridge_jit_lookup_symbol
void reussir_bridge_call_f32_func(void *func_ptr, float *result);
void reussir_bridge_call_f64_func(void *func_ptr, double *result);
void reussir_bridge_call_str_func(void *func_ptr, ReussirStrResult *result);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_BRIDGE_H
