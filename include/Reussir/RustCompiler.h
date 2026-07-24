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
/// This file declares helpers for locating and invoking the Rust compiler.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_RUSTCOMPILER_H
#define REUSSIR_RUSTCOMPILER_H
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <memory>

namespace reussir {
/// Locate the `rustc` used to compile FFI textures. A non-empty `preferred`
/// path (an explicit `--polyffi-rust-path` threaded down from the driver) is
/// returned verbatim; otherwise the `REUSSIR_RUSTC` environment variable and
/// then a list of conventional locations are probed. Empty when nothing is
/// found.
llvm::StringRef findRustCompiler(llvm::StringRef preferred = {});
/// Locate the directory holding the Rust packages (`libreussir_rt` and
/// friends) FFI textures link against (`rustc -L`). A non-empty `preferred`
/// directory (an explicit `--polyffi-libdir`) is returned verbatim; otherwise
/// the `REUSSIR_RUSTC_DEPS` environment variable and then conventional
/// locations are probed. Empty when nothing is found.
llvm::StringRef findRustCompilerDeps(llvm::StringRef preferred = {});
std::unique_ptr<llvm::Module>
compileRustSource(llvm::LLVMContext &context, llvm::StringRef sourceCode,
                  llvm::ArrayRef<llvm::StringRef> additionalArgs = {},
                  llvm::StringRef rustcPath = {},
                  llvm::StringRef rustcDepsDir = {});
std::unique_ptr<llvm::MemoryBuffer>
compileRustSourceToBitcode(llvm::LLVMContext &context,
                           llvm::StringRef sourceCode,
                           llvm::ArrayRef<llvm::StringRef> additionalArgs = {},
                           llvm::StringRef rustcPath = {},
                           llvm::StringRef rustcDepsDir = {});
} // namespace reussir
#endif // REUSSIR_RUSTCOMPILER_H
