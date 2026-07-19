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
/// This header file provides lowering patterns for attaching vtables to
/// regions.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_CONVERSION_ATTACHREGIONVTABLES_H
#define REUSSIR_CONVERSION_ATTACHREGIONVTABLES_H

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace reussir {

// Backward-compat header to ease transition; include the new one.
#include "Reussir/Conversion/RegionPatterns.h"

} // namespace reussir

#endif // REUSSIR_CONVERSION_ATTACHREGIONVTABLES_H
