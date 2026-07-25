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
/// Artifact packaging for the AOT driver (`rrc`): LTO bitcode emission and
/// archive writing. Neither is reachable through the LLVM-C API — writing a
/// ThinLTO module needs the summary index, and `llvm::writeArchive` (what
/// `llvm-ar` itself calls, symbol table over native *and* bitcode members
/// alike) is C++ only.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_ARTIFACT_H
#define REUSSIR_C_ARTIFACT_H

#include <stddef.h>

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

// How a compilation is packaged for link-time optimization. This selects the
// LLVM pipeline a module runs (`reussirRunBackendLLVMPipeline`) as well as
// what lands in a static library: native objects for `None`, bitcode for the
// two LTO modes.
typedef enum ReussirLtoMode {
  // No LTO: the per-module pipeline runs in full and native code is emitted.
  ReussirLtoNone = 0,
  // ThinLTO: the pre-link pipeline runs, and the bitcode carries the module
  // summary index the linker's ThinLTO plugin needs to import across modules.
  ReussirLtoThin = 1,
  // Full ("fat") LTO: the pre-link pipeline runs and plain bitcode is
  // emitted; whole-program optimization happens at link time.
  ReussirLtoFat = 2,
} ReussirLtoMode;

// Writes `module` to `path` as LLVM bitcode for the given LTO mode, which
// must not be `ReussirLtoNone`. `ReussirLtoThin` computes and attaches the
// module summary index (and a module hash, which the ThinLTO cache keys on);
// `ReussirLtoFat` writes plain bitcode.
//
// Returns NULL on success, or an error message the caller frees with
// `LLVMDisposeMessage`.
char *reussirWriteLtoBitcode(LLVMModuleRef module, const char *path,
                             ReussirLtoMode mode);

// Writes an archive at `path` holding `count` member files, in order, with a
// symbol table indexing each member. Members may be native objects or LLVM
// bitcode: the symbol table is built by reading each member as a symbolic
// file, so a bitcode archive resolves through a linker's LTO plugin exactly
// as `llvm-ar` output does.
//
// `triple` selects the archive flavor (GNU / Darwin / COFF / AIX big) so a
// cross-compiled library is written in the target's format; pass NULL for the
// host default. The archive is written deterministically (zeroed timestamps
// and ids), so identical inputs give identical bytes.
//
// Returns NULL on success, or an error message the caller frees with
// `LLVMDisposeMessage`.
char *reussirWriteArchive(const char *path, const char *const *members,
                          size_t count, const char *triple);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_ARTIFACT_H
