//===-- ReussirTypes.td - Reussir dialect types -------------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reussir dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIRTYPES_TD
#define REUSSIR_IR_REUSSIRTYPES_TD

include "Reussir/IR/ReussirDialect.td"
include "Reussir/IR/ReussirAttrs.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"


class ReussirType<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<ReussirDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def ReussirRawPtrType : ReussirType<"RawPtr", "raw_ptr"> {
  let summary = "Like LLVM pointer type";

  let description = [{
    A raw pointer type that does not have any of the constraints or semantics.
    This is sometimes needed for raw operations and FFI calls.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let assemblyFormat = "`<` $eleTy `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}

def ReussirRecordType : ReussirType<"Record", "record", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  MutableType,
]> {
  let summary = "A record type that is either a composite type or an enum";

  let description = [{
    A record type that can be either a struct or an enum. It is used to represent
    complex data structures in Reussir. Records are uniquely indentified by their
    names.

    To support self-referential records, the `members` field is an optional. An incomplete record can be later completed once its members are known.
  }];

  let parameters = (ins 
    OptionalArrayRefParameter<"mlir::Type">:$members,
    OptionalArrayRefParameter<"reussir::Capability">:$memberCapabilities,
    OptionalParameter<"mlir::StringAttr">:$name,
    "bool":$complete,
    "reussir::RecordKind":$kind,
    "reussir::Capability":$defaultCapability
  );

  let storageClass = "RecordTypeStorage";
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Create an identified and complete record type.
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "llvm::ArrayRef<reussir::Capability>":$memberCapabilities,
      "mlir::StringAttr":$name,
      "reussir::RecordKind":$kind,
      "reussir::Capability":$defaultCapability
    ), [{
      return $_get($_ctxt, members, memberCapabilities, name, /*complete=*/true,
                       kind, defaultCapability);
    }]>,

    // Create an identified and incomplete record type.
    TypeBuilder<(ins
      "mlir::StringAttr":$name,
      "reussir::RecordKind":$kind
    ), [{
      return $_get($_ctxt, 
        /* members= */ llvm::ArrayRef<Type>{}, 
        /* memberCapabilities= */ llvm::ArrayRef<reussir::Capability>{},
        name,
        /* complete= */ false, 
        kind,
        /* defaultCapability= */ reussir::Capability::unspecified);
    }]>,

    // Create an anonymous record type (always complete).
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "llvm::ArrayRef<reussir::Capability>":$memberCapabilities,
      "reussir::RecordKind":$kind,
      "reussir::Capability":$defaultCapability
    ), [{
      return $_get($_ctxt, members, memberCapabilities, mlir::StringAttr{}, /*complete=*/true,
                      kind, defaultCapability);
    }]>];

    let extraClassDeclaration = [{
      using Base::verifyInvariants;
      bool isCompound() const { return getKind() == RecordKind::compound; };
      bool isVariant() const { return getKind() == RecordKind::variant; };
      void complete(llvm::ArrayRef<mlir::Type> members, 
                    llvm::ArrayRef<reussir::Capability> memberCapabilities,
                    reussir::Capability defaultCapability);
    }];
}

#endif // REUSSIR_IR_REUSSIRTYPES_TD
