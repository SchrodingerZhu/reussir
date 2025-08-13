//===-- ReussirAttrs.td - Reussir dialect attributes -------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reussir dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIRATTRS_TD
#define REUSSIR_IR_REUSSIRATTRS_TD

include "Reussir/IR/ReussirDialect.td"
include "mlir/IR/EnumAttr.td"

class ReussirAttr<string name> : AttrDef<ReussirDialect, name>;

//===----------------------------------------------------------------------===//
// Atomicity Attributes
//===----------------------------------------------------------------------===//
def AK_Normal : I32EnumAttrCase<"normal", 0>;
def AK_Atomic : I32EnumAttrCase<"atomic", 1>;
def AtomicKind
    : I32EnumAttr<"AtomicKind", "atomic kind", [AK_Normal, AK_Atomic]> {
  let cppNamespace = "::reussir";
}

//===----------------------------------------------------------------------===//
// Capability Attributes
//===----------------------------------------------------------------------===//
def CAP_Default : I32EnumAttrCase<"unspecified", 0>;
def CAP_Value : I32EnumAttrCase<"value", 1>;
def CAP_Shared : I32EnumAttrCase<"shared", 2>;
def CAP_Flex : I32EnumAttrCase<"flex", 3>;
def CAP_Rigid : I32EnumAttrCase<"rigid", 4>;
def CAP_Field : I32EnumAttrCase<"field", 5>;

def Capability : I32EnumAttr<"Capability", "capability",
                             [CAP_Default, CAP_Value, CAP_Shared, CAP_Flex,
                              CAP_Rigid, CAP_Field]> {
  let cppNamespace = "::reussir";
}

//===----------------------------------------------------------------------===//
// Record Kind Attributes
//===----------------------------------------------------------------------===//
def RK_Struct : I32EnumAttrCase<"compound", 0>;
def RK_Enum : I32EnumAttrCase<"variant", 1>;

def RecordKind
    : I32EnumAttr<"RecordKind", "record kind", [RK_Struct, RK_Enum]> {
  let cppNamespace = "::reussir";
}
#endif // REUSSIR_IR_REUSSIRATTRS_TD
