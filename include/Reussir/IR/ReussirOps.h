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
/// This header file provides the definitions for operations used in the Reussir
/// dialect.
///
//===----------------------------------------------------------------------===//

#pragma once
#ifndef REUSSIR_IR_REUSSIROPS_H
#define REUSSIR_IR_REUSSIROPS_H

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/IR/Module.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirInterfaces.h"
#include "Reussir/IR/ReussirTypes.h"

#define GET_OP_CLASSES
#include "Reussir/IR/ReussirOps.h.inc"

namespace reussir {

//===----------------------------------------------------------------------===//
// inheritSanitizerPassthrough
//===----------------------------------------------------------------------===//
//
// When a build targets sanitizers, codegen stamps the module with this
// attribute — an array of the LLVM sanitizer function-attribute strings
// (e.g. "sanitize_address", "sanitize_thread") — and annotates every function
// it emits with the same strings via `passthrough` (mirrored in
// crates/reussir-codegen/src/lower/mod.rs). LLVM's sanitizer passes only
// instrument plain memory accesses in functions carrying the attribute, so
// every pass that CREATES a function after codegen (outlined closure bodies,
// outlined drop/acquire glue, trampolines) must call this helper on it, or
// the new function's accesses would silently escape instrumentation.
//
//===----------------------------------------------------------------------===//
inline constexpr llvm::StringRef kSanitizeAttr = "reussir.sanitize";

// Appends the module's sanitizer attribute strings (if any) to `func`'s
// `passthrough` attribute. `func` may be a `func.func` or an
// `llvm.func` — both carry `passthrough` onto the translated LLVM function.
void inheritSanitizerPassthrough(mlir::ModuleOp moduleOp,
                                 mlir::Operation *func);

//===----------------------------------------------------------------------===//
// emitOwnershipAcquisition
//===----------------------------------------------------------------------===//
//
// This function emits the ownership acquisition for the given value. The input
// value can either be a reference or a rc pointer. If other type is provided,
// the function returns failure.
// - When RC value is passed in, the function emits a RcInc operation.
// - When Reference value is passed in, the function checks the following:
//   + if the reference points to a rc pointer, then the function loads it and
//     recursively apply ownership acquisition.
//   + if the reference points to a record, then the function spills it to get a
//     reference and recursively apply ownership acquisition. For variant types,
//     the function emits a RecordDispatch operation and continue the process
//     for each variant by recursively calling emitOwnershipAcquisition in
//     corresponding regions.
//   + otherwise, the function is a no-op.
//
//===----------------------------------------------------------------------===//
mlir::LogicalResult emitOwnershipAcquisition(mlir::Value value,
                                             mlir::OpBuilder &builder,
                                             mlir::Location loc);

/// Traverses a statically shaped Reussir array view and invokes `emitElement`
/// with a reference to each element.
mlir::LogicalResult emitArrayElementTraversal(
    mlir::Value view, mlir::OpBuilder &builder, mlir::Location loc,
    llvm::function_ref<mlir::LogicalResult(mlir::OpBuilder &, mlir::Location,
                                           mlir::Value)>
        emitElement);

//===----------------------------------------------------------------------===//
// createDtorIfNotExists
//===----------------------------------------------------------------------===//
//
// Creates a destructor function for the given record type if it doesn't already
// exist. The destructor takes a reference to the record type and performs the
// drop operation. Returns the existing destructor if one is already present.
//
//===----------------------------------------------------------------------===//
// `kind` is the atomic context the glue is dropped from: the glue's argument
// reference carries it, and the expanded body derives member links from it
// (whole-subtree rule), so each kind gets its own outlined symbol.
mlir::func::FuncOp
createDtorIfNotExists(mlir::ModuleOp moduleOp, RecordType type,
                      mlir::OpBuilder &builder,
                      AtomicKind kind = AtomicKind::normal);

//===----------------------------------------------------------------------===//
// emitOwnershipAcquisitionFuncIfNotExists
//===----------------------------------------------------------------------===//
//
// Creates a function that performs ownership acquisition for the given
// record type if it doesn't already exist. The function takes a reference to
// the type and performs the acquisition operation. Returns the existing
// function if one is already present. The RecordType must be a named type.
//
//===----------------------------------------------------------------------===//
mlir::func::FuncOp emitOwnershipAcquisitionFuncIfNotExists(
    mlir::ModuleOp moduleOp, RecordType type, mlir::OpBuilder &builder,
    AtomicKind kind = AtomicKind::normal);

//===----------------------------------------------------------------------===//

// gatherCompiledModules

//===----------------------------------------------------------------------===//

//

// Gathers all the compiled modules from the polymorphic FFI operations.

//

//===----------------------------------------------------------------------===//

// Every gathered module is stamped with `dataLayout` and, when it is
// non-empty, `targetTriple` — the destination module's own spelling of the
// machine, which rustc-produced bitcode need not share (see the comment at
// the definition).
std::unique_ptr<llvm::Module>
gatherCompiledModules(mlir::ModuleOp moduleOp, llvm::LLVMContext &context,
                      llvm::StringRef dataLayout,
                      llvm::StringRef targetTriple = {});

constexpr llvm::StringRef REUSSIR_EXPANDED_ENSURE_ATTR =
    "reussir.expanded_ensure";
} // namespace reussir

#endif // REUSSIR_IR_REUSSIROPS_H
