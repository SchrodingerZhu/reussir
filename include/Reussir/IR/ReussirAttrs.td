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

def AK_Normal : I32EnumAttrCase<"Normal", 0>;
def AK_Atomic : I32EnumAttrCase<"Atomic", 1>;
def AtomicKind : I32EnumAttr<
    "AtomicKind",
    "atomic kind",
    [AK_Normal, AK_Atomic]> {
  let cppNamespace = "::reussir";
}

def CAP_Default : I32EnumAttrCase<"Default", 0>;
def CAP_Value : I32EnumAttrCase<"Value", 1>;
def CAP_Shared : I32EnumAttrCase<"Shared", 2>;
def CAP_Flex : I32EnumAttrCase<"Flex", 3>;
def CAP_Rigid : I32EnumAttrCase<"Rigid", 4>;

def Capability : I32EnumAttr<
    "Capability",
    "capability",
    [CAP_Default, CAP_Value, CAP_Shared, CAP_Flex, CAP_Rigid]> {
  let cppNamespace = "::reussir";
}

#endif // REUSSIR_IR_REUSSIRATTRS_TD
