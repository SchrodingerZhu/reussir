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

class ReussirAttr<string name, string attrMnemonic, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Attribute">
    : AttrDef<ReussirDialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

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
// Cell storage kind
//===----------------------------------------------------------------------===//
// A single enum deliberately models these as alternatives. In particular,
// `exclusive` and `atomic` are storage strategies, not independent flags.
def CK_Plain : I32EnumAttrCase<"plain", 0>;
def CK_Exclusive : I32EnumAttrCase<"exclusive", 1>;
def CK_Atomic : I32EnumAttrCase<"atomic", 2>;
def CellKind
    : I32EnumAttr<"CellKind", "cell storage kind",
                  [CK_Plain, CK_Exclusive, CK_Atomic]> {
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
def CAP_Regional : I32EnumAttrCase<"regional", 6>;

def Capability : I32EnumAttr<"Capability", "capability",
                             [CAP_Default, CAP_Value, CAP_Shared, CAP_Flex,
                              CAP_Rigid, CAP_Field, CAP_Regional]> {
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

//===----------------------------------------------------------------------===//
// LifeScope Attributes
//===----------------------------------------------------------------------===//
def LS_Global : I32EnumAttrCase<"global", 0>;
def LS_Local : I32EnumAttrCase<"local", 1>;

def LifeScope : I32EnumAttr<"LifeScope", "life scope", [LS_Global, LS_Local]> {
  let cppNamespace = "::reussir";
}

//===----------------------------------------------------------------------===//
// Trampoline Attributes
//===----------------------------------------------------------------------===//
def TD_Import : I32EnumAttrCase<"Import", 0, "import">;
def TD_Export : I32EnumAttrCase<"Export", 1, "export">;

def TrampolineDirection
    : I32EnumAttr<"TrampolineDirection",
                  "trampoline direction", [TD_Import, TD_Export]> {
  let cppNamespace = "::reussir";
}

//===----------------------------------------------------------------------===//
// Debug Attributes
//===----------------------------------------------------------------------===//
// Debug Types
//===----------------------------------------------------------------------===//
def Reussir_DBGIntType : ReussirAttr<"DBGIntType", "dbg_inttype"> {
  let description = "Debug integer type attribute";
  let parameters = (ins "::mlir::Type":$innerType, "bool":$isSigned,
      "::mlir::StringAttr":$dbgName);
  let assemblyFormat = [{
    `<` `signed` `:`  $isSigned `,` $innerType `,` `name` `:` $dbgName `>`
  }];
}
def Reussir_DBGFPType : ReussirAttr<"DBGFPType", "dbg_fptype"> {
  let description = "Debug floating point type attribute";
  let parameters = (ins "::mlir::Type":$innerType,
      "::mlir::StringAttr":$dbgName);
  let assemblyFormat = [{
    `<` $innerType `,` `name` `:` $dbgName `>`
  }];
}
def Reussir_DBGRecordMember
    : ReussirAttr<"DBGRecordMember", "dbg_record_member"> {
  let description = "Debug record type attribute";
  let parameters = (ins "::mlir::StringAttr":$name,
      "::mlir::Attribute":$typeAttr);
  let assemblyFormat = [{
    `<` `name` `:` $name `,` 
        `type` `:` $typeAttr
    `>`
  }];
}
def Reussir_DBGRecordType : ReussirAttr<"DBGRecordType", "dbg_recordtype"> {
  let description = "Debug record type attribute";
  let parameters = (ins "::mlir::ArrayAttr":$members, "bool":$isVariant,
      "::mlir::Type":$underlyingType, "::mlir::StringAttr":$dbgName);
  let assemblyFormat = [{
    `<` 
        `members` `:` $members `,`
        `is_variant` `:` $isVariant `,`
        `underlying_type` `:` $underlyingType `,`
        `dbg_name` `:` $dbgName
    `>`
  }];
}
def Reussir_DBGSubprogram : ReussirAttr<"DBGSubprogram", "dbg_subprogram"> {
  let description = "Debug subprogram attribute";
  let parameters = (ins "::mlir::StringAttr":$rawName,
      "::mlir::ArrayAttr":$typeParams

  );
  let assemblyFormat = [{
    `<`
        `raw_name` `:` $rawName `,`
        `type_params` `:` $typeParams
    `>`
  }];
}
// TODO: we need to handle boxed type and generate DI expressions later on
// For now, let's focus on basic types and unboxed record types
def Reussir_DBGLocalVar : ReussirAttr<"DBGLocalVar", "dbg_localvar"> {
  let description = "Debug local variable attribute";
  let parameters = (ins "::mlir::Attribute":$dbgType,
      "::mlir::StringAttr":$varName);
  let assemblyFormat = [{
    `<`
        `type` `:` $dbgType `,`
        `name` `:` $varName
    `>`
  }];
}
// A reference-counted (`[shared]`/`[regional]`) value: the variable is an `rc`
// pointer to a box `{ header, payload }`. `dbgType` describes the payload; the
// debug-info pass presents the variable as the payload, reached by
// dereferencing the pointer and skipping the box header (a `DW_OP_deref,
// DW_OP_plus_uconst` expression on the `dbg.declare`). The two box flavours
// have different header sizes — a shared box has a single ref-count word, a
// regional box a three-word `{ status, next, vtable }` header — so `regional`
// selects which to skip past. Once the `rc` is lowered to an opaque pointer the
// box type is unrecoverable, hence the flag is carried here.
def Reussir_DBGBoxedType : ReussirAttr<"DBGBoxedType", "dbg_boxedtype"> {
  let description = "Debug type for a reference-counted (boxed) value";
  let parameters = (ins "::mlir::Attribute":$dbgType, "bool":$regional);
  let assemblyFormat = [{ `<` $dbgType `,` `regional` `:` $regional `>` }];
}
def Reussir_DBGFuncArg : ReussirAttr<"DBGFuncArg", "dbg_func_arg"> {
  let description = "Debug function argument attribute";
  let parameters = (ins "::mlir::Attribute":$dbgType,
      "::mlir::StringAttr":$argName, "unsigned":$argIndex);
  let assemblyFormat = [{
    `<`
        `type` `:` $dbgType `,`
        `name` `:` $argName `,`
        `index` `:` $argIndex
    `>`
  }];
}
#endif // REUSSIR_IR_REUSSIRATTRS_TD
