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
/// This file implements the Reussir attribute C API.
///
//===----------------------------------------------------------------------===//

#include "Reussir-c/Attrs.h"

// Complete mlir::DialectVersion before MLIR C API headers reach
// BytecodeWriter.h; MSVC's STL rejects destroying unique_ptr<T> when T is only
// forward declared.
#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>

#include "Reussir/IR/ReussirAttrs.h"

#include "llvm/ADT/SmallVector.h"

using namespace reussir;

namespace {
// Collects an MlirAttribute array into a vector of unwrapped MLIR attributes.
llvm::SmallVector<mlir::Attribute> unwrapAttrs(intptr_t n,
                                               MlirAttribute const *attrs) {
  llvm::SmallVector<mlir::Attribute> result;
  result.reserve(n);
  for (intptr_t i = 0; i < n; ++i)
    result.push_back(unwrap(attrs[i]));
  return result;
}

mlir::StringAttr str(MlirAttribute attr) {
  return llvm::cast<mlir::StringAttr>(unwrap(attr));
}
} // namespace

MlirAttribute reussirDBGIntTypeAttrGet(MlirContext context, MlirType innerType,
                                       bool isSigned, MlirAttribute dbgName) {
  return wrap(DBGIntTypeAttr::get(unwrap(context), unwrap(innerType), isSigned,
                                  str(dbgName)));
}

MlirAttribute reussirDBGFPTypeAttrGet(MlirContext context, MlirType innerType,
                                      MlirAttribute dbgName) {
  return wrap(
      DBGFPTypeAttr::get(unwrap(context), unwrap(innerType), str(dbgName)));
}

MlirAttribute reussirDBGRecordMemberAttrGet(MlirContext context,
                                            MlirAttribute name,
                                            MlirAttribute typeAttr) {
  return wrap(
      DBGRecordMemberAttr::get(unwrap(context), str(name), unwrap(typeAttr)));
}

MlirAttribute
reussirDBGRecordTypeAttrGet(MlirContext context, intptr_t nMembers,
                            MlirAttribute const *members, bool isVariant,
                            MlirType underlyingType, MlirAttribute dbgName) {
  auto memberArray =
      mlir::ArrayAttr::get(unwrap(context), unwrapAttrs(nMembers, members));
  return wrap(DBGRecordTypeAttr::get(unwrap(context), memberArray, isVariant,
                                     unwrap(underlyingType), str(dbgName)));
}

MlirAttribute reussirDBGSubprogramAttrGet(MlirContext context,
                                          MlirAttribute rawName,
                                          intptr_t nTypeParams,
                                          MlirAttribute const *typeParams) {
  auto params = mlir::ArrayAttr::get(unwrap(context),
                                     unwrapAttrs(nTypeParams, typeParams));
  return wrap(DBGSubprogramAttr::get(unwrap(context), str(rawName), params));
}

MlirAttribute reussirDBGBoxedTypeAttrGet(MlirContext context,
                                         MlirAttribute dbgType, bool regional) {
  return wrap(
      DBGBoxedTypeAttr::get(unwrap(context), unwrap(dbgType), regional));
}

MlirAttribute reussirDBGLocalVarAttrGet(MlirContext context,
                                        MlirAttribute dbgType,
                                        MlirAttribute varName) {
  return wrap(
      DBGLocalVarAttr::get(unwrap(context), unwrap(dbgType), str(varName)));
}

MlirAttribute reussirDBGFuncArgAttrGet(MlirContext context,
                                       MlirAttribute dbgType,
                                       MlirAttribute argName,
                                       unsigned argIndex) {
  return wrap(DBGFuncArgAttr::get(unwrap(context), unwrap(dbgType),
                                  str(argName), argIndex));
}

void reussirOperationSetLocation(MlirOperation op, MlirLocation location) {
  unwrap(op)->setLoc(unwrap(location));
}
