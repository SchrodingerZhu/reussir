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
/// This header file provides the ClosureOutlining pass declaration.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifndef REUSSIR_CONVERSION_CLOSUREOUTLINING_H
#define REUSSIR_CONVERSION_CLOSUREOUTLINING_H

#include <mlir/Pass/Pass.h>

namespace reussir {

#define GEN_PASS_DECL_REUSSIRCLOSUREOUTLININGPASS
#include "Reussir/Conversion/Passes.h.inc"

} // namespace reussir

#endif // REUSSIR_CONVERSION_CLOSUREOUTLINING_H
