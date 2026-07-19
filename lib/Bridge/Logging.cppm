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
/// This file implements the Reussir logging bridge module.
///
//===----------------------------------------------------------------------===//

module;
#include "Reussir/Bridge.h"
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
export module Reussir.Logging;

namespace reussir {
namespace {
ReussirLogger createStdOutLogger(ReussirLogLevel level, char *name) {
  auto logger = spdlog::stdout_color_mt(name);
  switch (level) {
  case REUSSIR_LOG_ERROR:
    logger->set_level(spdlog::level::err);
    break;
  case REUSSIR_LOG_WARNING:
    logger->set_level(spdlog::level::warn);
    break;
  case REUSSIR_LOG_INFO:
    logger->set_level(spdlog::level::info);
    break;
  case REUSSIR_LOG_DEBUG:
    logger->set_level(spdlog::level::debug);
    break;
  case REUSSIR_LOG_TRACE:
    logger->set_level(spdlog::level::trace);
    break;
  }
  auto result =
      static_cast<ReussirLogger>(new std::shared_ptr(std::move(logger)));
  return result;
}
void destroyLogger(ReussirLogger logger) {
  auto loggerPtr = static_cast<std::shared_ptr<spdlog::logger> *>(logger);
  delete loggerPtr;
}
void logWithLevel(ReussirLogger logger, ReussirLogLevel level,
                  const char *message) {
  auto loggerPtr = static_cast<std::shared_ptr<spdlog::logger> *>(logger);
  switch (level) {
  case REUSSIR_LOG_ERROR:
    (*loggerPtr)->error(message);
    break;
  case REUSSIR_LOG_WARNING:
    (*loggerPtr)->warn(message);
    break;
  case REUSSIR_LOG_INFO:
    (*loggerPtr)->info(message);
    break;
  case REUSSIR_LOG_DEBUG:
    (*loggerPtr)->debug(message);
    break;
  case REUSSIR_LOG_TRACE:
    (*loggerPtr)->trace(message);
  }
}
} // namespace
} // namespace reussir

extern "C" {
ReussirLogger reussir_bridge_create_stdout_logger(ReussirLogLevel level,
                                                  char *name) {
  return reussir::createStdOutLogger(level, name);
}
void reussir_bridge_destroy_logger(ReussirLogger logger) {
  reussir::destroyLogger(logger);
}
void reussir_bridge_log_with_level(ReussirLogger logger, ReussirLogLevel level,
                                   const char *message) {
  reussir::logWithLevel(logger, level, message);
}
}
