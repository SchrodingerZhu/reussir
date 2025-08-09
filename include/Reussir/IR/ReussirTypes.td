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

//===----------------------------------------------------------------------===//
// Pointer-like Type
//===----------------------------------------------------------------------===//

def ReussirNonNullPointerType : Type<CPred<"::reussir::isNonNullPointerType($_self)">,
                          "Reussir non-null pointer type">;

//===----------------------------------------------------------------------===//
// Reussir Raw Pointer Type
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Reussir Record Type
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Reussir Token Type
//===----------------------------------------------------------------------===//
def ReussirTokenType : ReussirType<"Token", "token",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Memory Token";
  let description = [{
    `reussir.token` represents a block of raw memory.
  }];
  let parameters = (ins
    "size_t":$align,
    "size_t":$size
  );
  let assemblyFormat = [{
    `<` `align` `:` $align `,`  `size` `:` $size `>`
  }];
  let genVerifyDecl = 1;
}

///===----------------------------------------------------------------------===//
// Reussir Region Type
//===----------------------------------------------------------------------===//
def ReussirRegionType : ReussirType<"Region", "region",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Region Type";
  let description = [{
    `reussir.region` represents a arena inside which flexible references can be
    reassigned. The region type itself serves as the context that keeps track of
    all allocations. Currently, it is just a pointer to the following struct:
    ```c++
    struct Region {
      ObjHeader* obj;
    };
    ```
  }];
}
///===----------------------------------------------------------------------===//
// Reussir Rc Type
//===----------------------------------------------------------------------===//
def ReussirRcType : ReussirType<"Rc", "rc",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Reference Counted Type";
  let description = [{
    `reussir.rc` represents a reference counted object. It is used to manage the
    lifetime of objects in Reussir. The type itself serves as the context that
    keeps track of all allocations.
  }];
  let parameters = (ins
    "mlir::Type":$eleTy,
    "reussir::Capability":$capability,
    "reussir::AtomicKind":$atomicKind
  );
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
                 "reussir::AtomicKind":$atomicKind), [{
      return $_get($_ctxt, eleTy, capability, atomicKind);
    }]>,
    TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability), [{
      return $_get($_ctxt, eleTy, capability, ::reussir::AtomicKind::normal);
    }]>,
    TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, ::reussir::Capability::unspecified, 
        ::reussir::AtomicKind::normal);
    }]>
  ];
  let extraClassDeclaration = [{
    using Base::verifyInvariants;
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}
//===----------------------------------------------------------------------===//
// Reussir Nullable Type
//===----------------------------------------------------------------------===//
def ReussirNullableType : ReussirType<"Nullable", "nullable",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Nullable Type";
  let description = [{
    Pointers in reussir are nonnull by default. This type is used to
    represent that a underlying pointer can be null.
  }];
  let parameters = (ins ReussirNonNullPointerType:$ptrTy);
  let assemblyFormat = "`<` $ptrTy `>`";
}
//===----------------------------------------------------------------------===//
// Reussir Reference Type
//===----------------------------------------------------------------------===//
def ReussirRefType : ReussirType<"Ref", "ref", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  MutableType
]> {
  let summary = "Reussir Reference Type";
  let description = [{
    A reference type that is used to represent a reference to an object in Reussir.
    It is used to manage the lifetime of objects in Reussir.
  }];
  
  let parameters = (ins 
    "mlir::Type":$eleTy,
    "reussir::Capability":$capability,
    "reussir::AtomicKind":$atomicKind
  );
  
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  
  let builders = [
    TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
                 "reussir::AtomicKind":$atomicKind), [{
      return $_get($_ctxt, eleTy, capability, atomicKind);
    }]>,
    TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability), [{
      return $_get($_ctxt, eleTy, capability, ::reussir::AtomicKind::normal);
    }]>,
    TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, ::reussir::Capability::unspecified,
        ::reussir::AtomicKind::normal);
    }]>
  ];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}

///===----------------------------------------------------------------------===//
// Reussir RcBox Type
//===----------------------------------------------------------------------===//
def ReussirRcBoxType : ReussirType<"RcBox", "rc_box",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Reference Counted Box Type";
  let description = [{
    `reussir.rc_box` represents a heap-allocated box that contains a value managed
    by reference counting. The box contains a reference count and the value itself.
    The layout of the box for shared Rc is as follows:
    ```
    RcBox<T> {
        size_t status;
        T data;
    };
    ```
    For flex/rigid varaints, status is also of the same size but it is actually 
    a tagged pointer. Please refer to https://dl.acm.org/doi/abs/10.1145/3652024.3665507
    for more details:
    ```
    RcBox<T> {
        union {
          size_t status;
          RcBox* root;
        } status;
        RcBox* next;
        void* vtable;
        T data;
    };
    struct VTable {
      void*  drop;
      size_t size;
      size_t alignment;
      ssize_t scan_count;
      size_t scan_offset[0];
    }
    ```
  ]}];
  let parameters = (ins
    "mlir::Type":$eleTy,
    "bool":$regional
  );
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "mlir::Type":$eleTy, "bool":$regional), [{
      return $_get($_ctxt, eleTy, regional);
    }]>,
    TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, /*regional=*/false);
    }]>
  ];
  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    bool isRegional() const { return getRegional(); }
  }];
}
  
  

///===----------------------------------------------------------------------===//
// Special Type Constraints
//===----------------------------------------------------------------------===//
def ReussirNullableTokenType : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointer) { return ",
         SubstLeaves<"$_self", "pointer", AnyTypeOf<[ReussirTokenType]>.predicate>,
         "; }(::llvm::cast<::reussir::NullableType>($_self).getPtrTy())">,
  /* descr */ "Nullable Token Type",
  /* baseType */ "::reussir::NullableType"
>;

def ReussirCompoundType : Type<CPred<[{
  ::llvm::isa<::reussir::RecordType>($_self) 
    && ::llvm::cast<::reussir::RecordType>($_self).isCompound()}]>,
                          "Reussir compound type">;

def ReussirVariantType : Type<CPred<[{
  ::llvm::isa<::reussir::RecordType>($_self) 
    && ::llvm::cast<::reussir::RecordType>($_self).isVariant()}]>,
                          "Reussir variant type">;

#endif // REUSSIR_IR_REUSSIRTYPES_TD
