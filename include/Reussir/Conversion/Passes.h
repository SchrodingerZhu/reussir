//===-- Passes.h - Reussir conversion passes --------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides the definitions for conversion passes used in the
// Reussir dialect.
//
//===----------------------------------------------------------------------===//
#pragma once
#ifndef REUSSIR_CONVERSION_PASSES_H
#define REUSSIR_CONVERSION_PASSES_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>

namespace llvm {
class Module;
} // namespace llvm

namespace reussir {
#define GEN_PASS_DECL
#include "Reussir/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// CompilePolymorphicFFI Standalone Function
//===----------------------------------------------------------------------===//
//
// Compiles all uncompiled polymorphic FFI operations in the module by
// monomorphizing their templates and compiling them to LLVM bitcode.
//
//===----------------------------------------------------------------------===//
mlir::LogicalResult compilePolymorphicFFI(mlir::ModuleOp moduleOp,
                                          bool optimized = false);

//===----------------------------------------------------------------------===//
// Variant debug-info fixup (post-translation)
//===----------------------------------------------------------------------===//
//
// Rewrites the `{ tag, payload-union }` debug type the debug-info conversion
// emits for each enum into a real DWARF `DW_TAG_variant_part` (discriminant
// plus per-case `DW_AT_discr_value`), so a debugger shows only the active case.
// Runs on the translated `llvm::Module` because MLIR's `DICompositeTypeAttr`
// has no discriminator field — the operands only exist at the LLVM-DI level. A
// no-op for modules without debug info.
//
//===----------------------------------------------------------------------===//
void fixupVariantDebugInfo(llvm::Module &module);

void registerReussirBasicOpsLoweringInterface(mlir::DialectRegistry &registry);

} // namespace reussir

#endif // REUSSIR_CONVERSION_PASSES_H
