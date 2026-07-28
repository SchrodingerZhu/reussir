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
/// This file implements helpers for locating and invoking the Rust compiler.
///
//===----------------------------------------------------------------------===//

#include "Reussir/RustCompiler.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <system_error>

#ifdef _WIN32
#define EXEC_SUFFIX ".exe"
#else
#define EXEC_SUFFIX ""
#endif

#ifdef _WIN32
#define LIB_NAME "reussir_rt.dll"
#elif defined(__APPLE__)
#define LIB_NAME "libreussir_rt.dylib"
#else
#define LIB_NAME "libreussir_rt.so"
#endif

namespace reussir {
namespace {
constexpr std::array<llvm::StringRef, 9> RUSTC_HINTS = {
    "rustc" EXEC_SUFFIX,
    "build/bin/rustc" EXEC_SUFFIX,
    "bin/rustc" EXEC_SUFFIX,
    "../bin/rustc" EXEC_SUFFIX,
    "../../bin/rustc" EXEC_SUFFIX,
    "../../../bin/rustc" EXEC_SUFFIX,
    "/usr/bin/rustc" EXEC_SUFFIX,
    "/usr/local/bin/rustc" EXEC_SUFFIX,
    "/opt/reussir/bin/rustc" EXEC_SUFFIX,
};
constexpr std::array<llvm::StringRef, 14> RUSTC_DEPS_HINTS = {
    "lib/" LIB_NAME,
    "../lib/" LIB_NAME,
    "../../lib/" LIB_NAME,
    "../../../lib/" LIB_NAME,
    "bin/" LIB_NAME,
    "../bin/" LIB_NAME,
    "../../bin/" LIB_NAME,
    "../../../bin/" LIB_NAME,
    "/usr/lib/" LIB_NAME,
    "/usr/local/lib/" LIB_NAME,
    "/opt/reussir/lib/" LIB_NAME,
};
} // namespace

llvm::StringRef findRustCompiler(llvm::StringRef preferred) {
  // an explicit path from the driver wins over the environment and the probe
  if (!preferred.empty())
    return preferred;
  // first check if REUSSIR_RUSTC is set
  if (const char *env_p = std::getenv("REUSSIR_RUSTC"))
    return env_p;
  // locate rustc in known paths
  for (const auto &path : RUSTC_HINTS) {
    if (llvm::sys::fs::exists(path))
      return path;
  }

  return "";
}

llvm::SmallVector<std::string>
findRustCompilerDeps(llvm::ArrayRef<llvm::StringRef> preferred) {
  // explicit directories from the driver win over the environment and the
  // probe
  if (!preferred.empty()) {
    llvm::SmallVector<std::string> dirs;
    for (llvm::StringRef dir : preferred)
      if (!dir.empty())
        dirs.push_back(dir.str());
    if (!dirs.empty())
      return dirs;
  }
  // first check if REUSSIR_RUSTC_DEPS is set; it may list several directories
  // separated by the platform's environment path separator
  if (const char *env_p = std::getenv("REUSSIR_RUSTC_DEPS")) {
    llvm::SmallVector<std::string> dirs;
    llvm::SmallVector<llvm::StringRef> parts;
    llvm::StringRef(env_p).split(parts, llvm::sys::EnvPathSeparator,
                                 /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef dir : parts)
      dirs.push_back(dir.str());
    if (!dirs.empty())
      return dirs;
  }
  for (const auto &path : RUSTC_DEPS_HINTS) {
    if (llvm::sys::fs::exists(path))
      return {llvm::sys::path::parent_path(path).str()};
  }
  return {};
}

std::unique_ptr<llvm::MemoryBuffer> compileRustSourceToBitcode(
    llvm::LLVMContext &context, llvm::StringRef sourceCode,
    llvm::ArrayRef<llvm::StringRef> additionalArgs, llvm::StringRef rustcPath,
    llvm::ArrayRef<llvm::StringRef> rustcDepsDirs) {
  rustcPath = findRustCompiler(rustcPath);
  llvm::SmallVector<std::string> rustcDepsPaths =
      findRustCompilerDeps(rustcDepsDirs);
  if (rustcPath.empty() || rustcDepsPaths.empty()) {
    llvm::SmallString<16> cwd;
    auto code = llvm::sys::fs::current_path(cwd);
    if (code) {
      cwd = "<unknown>";
    }
    llvm::errs() << "Could not find rustc or its dependencies, current "
                    "working directory: "
                 << cwd << "\n";
    return nullptr;
  }
  // Create a temporary file for the source code
  llvm::SmallString<32> srcFilePath;
  llvm::SmallString<32> resultBitcodeFilePath;
  std::error_code srcFileCode = llvm::sys::fs::createUniqueFile(
      "reussir_rust_module_%%%%%%.rs", srcFilePath);
  std::error_code resultBitcodeFileCode = llvm::sys::fs::createUniqueFile(
      "reussir_rust_module_%%%%%%.bc", resultBitcodeFilePath);
  if (srcFileCode || resultBitcodeFileCode) {
    llvm::errs() << "Could not create temporary files for Rust compilation\n";
    return nullptr;
  }
  {
    int fd = -1;
    std::error_code code = llvm::sys::fs::openFileForWrite(srcFilePath, fd);
    if (code) {
      llvm::errs() << "Could not open temporary file for writing: "
                   << code.message() << "\n";
      return nullptr;
    }
    llvm::raw_fd_ostream srcStream(fd, /*shouldClose=*/true);
    srcStream << sourceCode;
    srcStream.flush();
  }
  // Prepare rustc command
  llvm::SmallVector<llvm::StringRef, 24> args = {
      "rustc",          "-A",           "warnings",
      srcFilePath,      "--crate-type", "cdylib",
      "--emit=llvm-bc", "-o",           resultBitcodeFilePath};
  for (const std::string &depsPath : rustcDepsPaths) {
    args.push_back("-L");
    args.push_back(depsPath);
  }
  for (auto arg : additionalArgs)
    args.push_back(arg);
  // Execute rustc. `REUSSIR_PHASE_LOG` (set by a verbose rrc) narrates the
  // spawn to stderr so a wedged toolchain process is attributable from a
  // captured log: a `begin` line with no `done` names this rustc.
  bool phaseLog = std::getenv("REUSSIR_PHASE_LOG") != nullptr;
  if (phaseLog)
    llvm::errs() << "[polyffi] begin texture rustc: " << rustcPath << "\n";
  auto spawnStart = std::chrono::steady_clock::now();
  int code = llvm::sys::ExecuteAndWait(rustcPath, args);
  if (phaseLog)
    llvm::errs() << "[polyffi] texture rustc done ("
                 << std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - spawnStart)
                        .count()
                 << " ms, exit " << code << ")\n";
  if (code != 0) {
    llvm::errs() << "Rust compilation failed with exit code " << code << "\n";
    llvm::errs() << "Full command: " << rustcPath << " ";
    llvm::interleave(args, llvm::errs(), " ");
    llvm::errs() << "\n";
    return nullptr;
  }
  if (auto err = llvm::sys::fs::remove(srcFilePath))
    llvm::errs() << "Failed to discard source file\n";
  // Load the bitcode file into a buffer
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  {
    auto bufferOrErr = llvm::MemoryBuffer::getFile(resultBitcodeFilePath);
    if (!bufferOrErr) {
      llvm::errs() << "Failed to read bitcode file: "
                   << bufferOrErr.getError().message() << "\n";
      return {};
    }
#ifdef _WIN32
    buffer = llvm::MemoryBuffer::getMemBufferCopy((*bufferOrErr)->getBuffer());
#else
    buffer = std::move(*bufferOrErr);
#endif
  }
  if (auto err = llvm::sys::fs::remove(resultBitcodeFilePath))
    llvm::errs() << "Failed to discard bitcode file\n";
  return buffer;
}

std::unique_ptr<llvm::Module>
compileRustSource(llvm::LLVMContext &context, llvm::StringRef sourceCode,
                  llvm::ArrayRef<llvm::StringRef> additionalArgs,
                  llvm::StringRef rustcPath,
                  llvm::ArrayRef<llvm::StringRef> rustcDepsDirs) {
  std::unique_ptr<llvm::MemoryBuffer> bitcode = compileRustSourceToBitcode(
      context, sourceCode, additionalArgs, rustcPath, rustcDepsDirs);
  if (!bitcode)
    return nullptr;
  auto moduleOrErr =
      llvm::parseBitcodeFile(bitcode->getMemBufferRef(), context);
  if (!moduleOrErr) {
    llvm::errs() << "Failed to parse bitcode file: "
                 << llvm::toString(moduleOrErr.takeError()) << "\n";
    return nullptr;
  }
  return std::move(*moduleOrErr);
}
} // namespace reussir
