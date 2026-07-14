//===-- Attrs.h - Reussir dialect attribute C API --------------*- C -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// C API constructors for the Reussir debug-info attributes. These attributes
// describe a value's debug type / variable so the debug-info conversion pass
// can emit LLVM DI; they use custom storage that the generic MLIR C API cannot
// build, so the code generator constructs them through these entry points.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_C_ATTRS_H
#define REUSSIR_C_ATTRS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Debug-info attributes
//===----------------------------------------------------------------------===//

// A debug integer type: `innerType` is the lowered MLIR integer, `dbgName` a
// `StringAttr` naming it in DWARF.
MlirAttribute reussirDBGIntTypeAttrGet(MlirContext context, MlirType innerType,
                                       bool isSigned, MlirAttribute dbgName);

// A debug floating-point type.
MlirAttribute reussirDBGFPTypeAttrGet(MlirContext context, MlirType innerType,
                                      MlirAttribute dbgName);

// One member of a debug record type: a field `name` (`StringAttr`) and its
// debug type attribute (a `DBG*Type` attribute).
MlirAttribute reussirDBGRecordMemberAttrGet(MlirContext context,
                                            MlirAttribute name,
                                            MlirAttribute typeAttr);

// A debug record (struct / variant) type. `members` is an array of
// `reussirDBGRecordMemberAttrGet` results; `underlyingType` is the lowered MLIR
// record used for size/offset computation.
MlirAttribute
reussirDBGRecordTypeAttrGet(MlirContext context, intptr_t nMembers,
                            MlirAttribute const *members, bool isVariant,
                            MlirType underlyingType, MlirAttribute dbgName);

// A debug subprogram (function) attribute: `rawName` is the source function
// name
// (`StringAttr`); `typeParams` an array of parameter debug-type attributes.
MlirAttribute reussirDBGSubprogramAttrGet(MlirContext context,
                                          MlirAttribute rawName,
                                          intptr_t nTypeParams,
                                          MlirAttribute const *typeParams);

// A debug type for a reference-counted (boxed) value, wrapping the payload's
// debug type. `regional` selects the box header to skip past: a shared box has
// a single ref-count word, a regional box a three-word header.
MlirAttribute reussirDBGBoxedTypeAttrGet(MlirContext context,
                                         MlirAttribute dbgType, bool regional);

// A debug local-variable attribute: its debug type and source name.
MlirAttribute reussirDBGLocalVarAttrGet(MlirContext context,
                                        MlirAttribute dbgType,
                                        MlirAttribute varName);

// A debug function-argument attribute: its debug type, source name, and 1-based
// argument index.
MlirAttribute reussirDBGFuncArgAttrGet(MlirContext context,
                                       MlirAttribute dbgType,
                                       MlirAttribute argName,
                                       unsigned argIndex);

// Set an operation's location. Used by the code generator to attach a fused
// debug-info location (carrying a `DBGLocalVar`) to the op that defines a local
// variable; `mlir-sys` does not bind `mlirOperationSetLocation`.
void reussirOperationSetLocation(MlirOperation op, MlirLocation location);

#ifdef __cplusplus
}
#endif

#endif // REUSSIR_C_ATTRS_H
