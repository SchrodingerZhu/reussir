//===-- Bridge.h - Reussir backend bridge -----------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements the bridge between rust frontend and C++ backend.
//
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_BRIDGE_H
#define REUSSIR_BRIDGE_H

#include <string_view>

namespace reussir {
enum class OutputTarget { LLVMIR, ASM, Object };
enum class OptOption { None, Default, Aggressive, Size };
enum class LogLevel { Error, Warning, Info, Debug, Trace };
struct CompileOptions {
  OutputTarget target;
  OptOption opt;
  LogLevel logLevel;
  void (*backendLog)(std::string_view, LogLevel level);
};

// currently, we only support compiling for native machine target
void compileForNativeMachine(std::string_view mlirTextureModule,
                             std::string_view sourceName,
                             std::string_view outputFile,
                             CompileOptions options);
} // namespace reussir

#endif // REUSSIR_BRIDGE_H
