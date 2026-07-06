//===-- ReussirTypes.h - Reussir dialect types ------------------*- c++ -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This header file provides the definitions for types used in the Reussir
// dialect.
//
//===----------------------------------------------------------------------===//
#pragma once
#include <mlir/IR/Builders.h>
#ifndef REUSSIR_IR_REUSSIRTYPES_H
#define REUSSIR_IR_REUSSIRTYPES_H

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <variant>

#include "Reussir/IR/ReussirAttrs.h"
#include "Reussir/IR/ReussirEnumAttrs.h"
#include "Reussir/IR/ReussirTypeDetails.h"

namespace reussir {
std::optional<std::tuple<llvm::TypeSize, llvm::Align, mlir::Type>>
deriveCompoundSizeAndAlignment(mlir::MLIRContext *context,
                               llvm::ArrayRef<mlir::Type> members,
                               llvm::ArrayRef<bool> memberIsField,
                               const mlir::DataLayout &dataLayout,
                               bool memBoxInternal = false);
bool isNonNullPointerType(mlir::Type type);
bool isTriviallyCopyable(mlir::Type type);
mlir::Type getProjectedType(mlir::Type type, bool fieldCap, Capability refCap);
mlir::Type memberStorageType(mlir::MLIRContext *context, mlir::Type rawMember,
                             bool isField, bool memBoxInternal = false);

namespace scanner {
// Encoding shared with reussir-rt's region scanner interpreter
// (crates/reussir-rt/src/region/scanner.rs): 0 = end, -1/-3/-4 = variant tag
// read of 1/2/4 bytes, -2 = field, > 0 = advance, <= -5 = jump.
inline int32_t end() {
  return 0; // Assuming 0 represents the 'end' instruction
}
inline int32_t variant(uint64_t tagByteWidth) {
  switch (tagByteWidth) {
  case 1:
    return -1;
  case 2:
    return -3;
  case 4:
    return -4;
  default:
    llvm_unreachable("unsupported variant tag width");
  }
}
inline int32_t field() { return -2; }
inline int32_t advance(uint32_t bytes) {
  assert(bytes <= INT32_MAX && "advance bytes must fit in int32");
  return static_cast<int32_t>(bytes);
}
inline int32_t skip(size_t count) {
  assert(count <= INT32_MAX - 5 && "skip count must fit in int32");
  return static_cast<int32_t>(-5 - count);
}
struct End {};
struct Variant {
  uint64_t tagByteWidth;
};
struct Field {};
struct Advance {
  uint32_t bytes;
};
struct Skip {
  size_t count;
};
using Instr = std::variant<End, Variant, Field, Advance, Skip>;
inline Instr decode(int32_t code) {
  if (code == 0)
    return End{};
  else if (code == -1)
    return Variant{1};
  else if (code == -2)
    return Field{};
  else if (code == -3)
    return Variant{2};
  else if (code == -4)
    return Variant{4};
  else if (code <= -5)
    return Skip{static_cast<size_t>(-5 - code)};
  else
    return Advance{static_cast<uint32_t>(code)};
}
struct EmitState {
  size_t cursorPosition = 0;
  size_t scannedBytes = 0;
};
} // namespace scanner
} // namespace reussir

#define GET_TYPEDEF_CLASSES
#include "Reussir/IR/ReussirOpsTypes.h.inc"

#endif // REUSSIR_IR_REUSSIRTYPES_H
