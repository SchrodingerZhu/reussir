//===-- ReussirTypes.td - Reussir dialect types -------------*- tablegen
//-*-===//
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
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

class ReussirType<string name, string typeMnemonic, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Type">
    : TypeDef<ReussirDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Pointer-like Type
//===----------------------------------------------------------------------===//

def ReussirNonNullPointerType
    : Type<CPred<"::reussir::isNonNullPointerType($_self)">,
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

  let builders = [TypeBuilderWithInferredContext<
                      (ins "mlir::Type":$elementType), [{
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
def ReussirRecordType
    : ReussirType<"Record", "record",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
                   MutableType,
]> {
  let summary = "A record type that is either a composite type or an enum";

  let description = [{
    A record type that can be either a struct or an enum. It is used to represent
    complex data structures in Reussir. Records are uniquely identified by their
    names.

    To support self-referential records, the `members` field is an optional. 
    An incomplete record can be later completed once its members are known.

    Notice that rc and reference types are not allowed as members of a record.
    Instead, use the `shared` capability for rc managed fields; `rigid` for
    immutable rc managed fields can escape its region; `value` for inplace
    storage. `flex` is not allowed as a member type capability. If the capability
    is unspecified, it is set to the type's default capability.

    For variant records, the default capability is `value` if not specified.
  }];

  let parameters = (ins OptionalArrayRefParameter<"mlir::Type">:$members,
      OptionalArrayRefParameter<"bool">:$memberIsField,
      OptionalParameter<"mlir::StringAttr">:$name, "bool":$complete,
      "reussir::RecordKind":$kind, "reussir::Capability":$defaultCapability);

  let storageClass = "RecordTypeStorage";
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
      // Create an identified and complete record type.
      TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$members,
                      "llvm::ArrayRef<bool>":$memberIsField,
                      "mlir::StringAttr":$name, "reussir::RecordKind":$kind,
                      "reussir::Capability":$defaultCapability),
                  [{
      return $_get($_ctxt, members, memberIsField, name, /*complete=*/true,
                       kind, defaultCapability);
    }]>,

      // Create an identified and incomplete record type.
      TypeBuilder<(ins "mlir::StringAttr":$name, "reussir::RecordKind":$kind),
                  [{
      return $_get($_ctxt, 
        /* members= */ llvm::ArrayRef<Type>{}, 
        /* memberIsField= */ llvm::ArrayRef<bool>{},
        name,
        /* complete= */ false, 
        kind,
        /* defaultCapability= */ reussir::Capability::unspecified);
    }]>,

      // Create an anonymous record type (always complete).
      TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$members,
                      "llvm::ArrayRef<bool>":$memberIsField,
                      "reussir::RecordKind":$kind,
                      "reussir::Capability":$defaultCapability),
                  [{
      return $_get($_ctxt, members, memberIsField, mlir::StringAttr{}, /*complete=*/true,
                      kind, defaultCapability);
    }]>];

  let extraClassDeclaration = [{
      using Base::verifyInvariants;
      bool isCompound() const { return getKind() == RecordKind::compound; };
      bool isVariant() const { return getKind() == RecordKind::variant; };
      void complete(llvm::ArrayRef<mlir::Type> members, 
                    llvm::ArrayRef<bool> memberIsField,
                    reussir::Capability defaultCapability);
      struct LayoutInfo {
        llvm::TypeSize size;
        llvm::Align alignment;
        mlir::Type memberWithLargestAlignment;
      };
      LayoutInfo getElementRegionLayoutInfo(const mlir::DataLayout &dataLayout) const;
      ::mlir::FlatSymbolRefAttr getDtorName() const;
      ::mlir::FlatSymbolRefAttr getAcquireName() const;
      size_t emitScannerInstructions(
        llvm::SmallVectorImpl<int32_t> &buffer,
        const mlir::DataLayout &dataLayout, const scanner::EmitState &EmitState) const;
      bool hasNoRegionalFields() const;
    }];
}

//===----------------------------------------------------------------------===//
// Reussir Token Type
//===----------------------------------------------------------------------===//
def ReussirTokenType
    : ReussirType<
          "Token",
          "token", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Memory Token";
  let description = [{
    `reussir.token` represents a block of raw memory.
  }];
  let parameters = (ins "size_t":$align, "size_t":$size);
  let assemblyFormat = [{
    `<` `align` `:` $align `,`  `size` `:` $size `>`
  }];
  let genVerifyDecl = 1;
}

///===----------------------------------------------------------------------===//
// Reussir Region Type
//===----------------------------------------------------------------------===//
def ReussirRegionType
    : ReussirType<
          "Region",
          "region", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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
def ReussirRcType
    : ReussirType<
          "Rc", "rc",
          [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
           MemRefElementTypeInterface]> {
  let summary = "Reussir Reference Counted Type";
  let description = [{
    `reussir.rc` represents a reference counted object. It is used to manage the
    lifetime of objects in Reussir. The type itself serves as the context that
    keeps track of all allocations.
  }];
  let parameters = (ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
      "reussir::AtomicKind":$atomicKind);
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let builders =
      [TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
                       "reussir::AtomicKind":$atomicKind),
                   [{
      return $_get($_ctxt, eleTy, capability, atomicKind);
    }]>,
       TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability),
                   [{
      return $_get($_ctxt, eleTy, capability, ::reussir::AtomicKind::normal);
    }]>,
       TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, ::reussir::Capability::unspecified, 
        ::reussir::AtomicKind::normal);
    }]>];
  let extraClassDeclaration = [{
    using Base::verifyInvariants;
    mlir::Type getElementType() const { return getEleTy(); }
    bool isRegional() const {
      return getCapability() == Capability::flex ||
             getCapability() == Capability::rigid;
    }
    RcBoxType getInnerBoxType() const;
    /// Whether values of this type may be special-pointer-tag immediates
    /// (see `reussir.rc.tagged`): a shared-capability box of a complete
    /// variant record with at least one nullary arm whose `tag + 1` fits the
    /// top byte, and a nonatomic refcount. Decrementing refcount stores on
    /// such values must be steered when the scheme is enabled.
    bool mayCarrySpecialPointerTag() const;
  }];
}
//===----------------------------------------------------------------------===//
// Reussir Nullable Type
//===----------------------------------------------------------------------===//
def ReussirNullableType
    : ReussirType<
          "Nullable",
          "nullable", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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
def ReussirRefType
    : ReussirType<"Ref", "ref",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
                   MutableType]> {
  let summary = "Reussir Reference Type";
  let description = [{
    A reference type that is used to represent a reference to an object in Reussir.
    It is used to manage the lifetime of objects in Reussir.
  }];

  let parameters = (ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
      "reussir::AtomicKind":$atomicKind);

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders =
      [TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability,
                       "reussir::AtomicKind":$atomicKind),
                   [{
      return $_get($_ctxt, eleTy, capability, atomicKind);
    }]>,
       TypeBuilder<(ins "mlir::Type":$eleTy, "reussir::Capability":$capability),
                   [{
      return $_get($_ctxt, eleTy, capability, ::reussir::AtomicKind::normal);
    }]>,
       TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, ::reussir::Capability::unspecified,
        ::reussir::AtomicKind::normal);
    }]>];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Hole Type
//===----------------------------------------------------------------------===//
def ReussirHoleType
    : ReussirType<
          "Hole",
          "hole", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir mutable hole type";
  let description = [{
    `reussir.hole` represents a write-once destination used by TRMC-style
    rewrites. Unlike `reussir.ref`, it is not a general-purpose aliasable
    mutable reference.
  }];

  let parameters = (ins "mlir::Type":$eleTy);
  let assemblyFormat = "`<` $eleTy `>`";

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}

///===----------------------------------------------------------------------===//
// Reussir RcBox Type
//===----------------------------------------------------------------------===//
def ReussirRcBoxType
    : ReussirType<
          "RcBox",
          "rc_box", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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
    For flex/rigid variants, status is also of the same size but it is actually 
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
  let parameters = (ins "mlir::Type":$eleTy, "bool":$regional);
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [TypeBuilder<(ins "mlir::Type":$eleTy, "bool":$regional), [{
      return $_get($_ctxt, eleTy, regional);
    }]>,
                  TypeBuilder<(ins "mlir::Type":$eleTy), [{
      return $_get($_ctxt, eleTy, /*regional=*/false);
    }]>];
  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    bool isRegional() const { return getRegional(); }
    size_t getElementIndex() const {
      return isRegional() ? 3 : 1;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Closure Type
//===----------------------------------------------------------------------===//
def ReussirClosureType
    : ReussirType<"Closure", "closure"> {
  let summary = "Reussir Closure Type";
  let description = [{
    `reussir.closure` represents a closure. 
    A closure must be within a shared Rc type. The RC pointer points to a 
    RcBox wrapping a ClosureBox. The ClosureBox contains the payload values
    captured by the closure.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputTypes,
    OptionalParameter<"::mlir::Type">:$outputType
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static constexpr size_t VTABLE_DROP_INDEX = 0;
    static constexpr size_t VTABLE_CLONE_INDEX = 1;
    static constexpr size_t VTABLE_EVALUATE_INDEX = 2;
  }];
}

def ReussirClosureBoxType
    : ReussirType<"ClosureBox", "closure_box", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Closure Box Type";
  let description = [{
    `reussir.closure` represents a closure. It roughly corresponds to the 
    following C++ struct:
    ```c++
    struct ClosureBox {
      void* vtable;
      void* arg_cursor;
      PayloadTypes... payload;
    }
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$payloadTypes
  );
  let assemblyFormat = "`<` $payloadTypes `>`";
  let extraClassDeclaration = [{
    static constexpr size_t VTABLE_INDEX = 0;
    static constexpr size_t ARG_CURSOR_INDEX = 1;
  }];
}

def ReussirArrayType
    : ReussirType<
          "Array", "array",
          [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
           ShapedTypeInterface]> {
  let summary = "Reussir fixed-size inline array type";
  let description = [{
    `reussir.array` represents a statically shaped inline array payload.
    This is the payload type wrapped by `!reussir.rc<...>` for managed arrays.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
      "mlir::Type":$eleTy);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    bool hasRank() const { return true; }
    ArrayType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                        mlir::Type elementType) const;
    size_t getRank() const { return getShape().size(); }
    ArrayType dropFront() const;
  }];
}

def ReussirViewType
    : ReussirType<
          "View", "view",
          [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir array view type";
  let description = [{
    `reussir.view` represents a temporary region-scoped view of an array payload.
    Mutable views are intended to support in-place update and future memref-style
    integrations.
  }];
  let parameters = (ins "bool":$isMutable, "reussir::ArrayType":$arrayTy);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    bool isMutable() const { return getIsMutable(); }
    ArrayType getArrayType() const { return getArrayTy(); }
    mlir::Type getElementType() const { return getArrayTy().getElementType(); }
  }];
}

def ReussirFFIObjectType
    : ReussirType<"FFIObject", "ffi_object", []> {
  let summary = "Reussir FFIObject Object Type";
  let description = [{
    `reussir.ffi_object` is a special opaque type that is always
    behind a RC pointer. Its dec operation, however, is handled by an FFI call.
  }];
  let parameters = (ins
    "::mlir::StringAttr":$ffiName,
    "::mlir::FlatSymbolRefAttr":$cleanupHook
  );
  let assemblyFormat = "`<` $ffiName `,` $cleanupHook `>`";
}

def ReussirStrType
  : ReussirType<"Str", "str", []> {
  let summary = "Reussir String Reference Type";
  let description = [{
    `reussir.str` is for str reference. Unlike rust, it only has two variants:
    - *global*: can be materialized
    - *local*: should not be materialized
  }];
  let parameters = (ins
    "::reussir::LifeScope":$lifeScope
  );
  let assemblyFormat = "`<` $lifeScope `>`";
}

///===---------------------------------------------------------------------===//
// Special Type Constraints
//===----------------------------------------------------------------------===//

def RcClosureType : Type<CPred<[{
  ::llvm::isa<::reussir::RcType>($_self) && 
  ::llvm::isa<::reussir::ClosureType>(
    ::llvm::cast<::reussir::RcType>($_self).getElementType()) &&
  !::llvm::cast<::reussir::RcType>($_self).isRegional()
  }]>, "Reussir rc closure type", "::reussir::RcType">;

def ReussirNullableTokenType : Type<CPred<[{
  ::llvm::isa<::reussir::NullableType>($_self) && 
  ::llvm::isa<::reussir::TokenType>(
    ::llvm::cast<::reussir::NullableType>($_self).getPtrTy())
  }]>, "Reussir nullable token type", "::reussir::NullableType">;

def ReussirCompoundType
    : Type<CPred<[{
  ::llvm::cast<::reussir::RecordType>($_self).isCompound()}]>,
           "Reussir compound type", "::reussir::RecordType">;

def ReussirVariantType : Type<CPred<[{
  ::llvm::cast<::reussir::RecordType>($_self).isVariant()}]>,
                              "Reussir variant type", "::reussir::RecordType">;

def ReussirRcClosureType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ClosureType>(
      ::llvm::cast<::reussir::RcType>($_self).getElementType())
    }]>, "Reussir rc closure type", "::reussir::RcType">;

def ReussirClosureBoxRefType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ClosureBoxType>(
      ::llvm::cast<::reussir::RefType>($_self).getElementType())
    }]>, "Reussir closure box reference type", "::reussir::RefType">;

def ReussirArrayPayloadType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ArrayType>($_self)
    }]>, "Reussir array payload type", "::reussir::ArrayType">;

def ReussirArrayViewType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ViewType>($_self)
    }]>, "Reussir array view type", "::reussir::ViewType">;

def ReussirRcArrayType
    : Type<CPred<[{
    ::llvm::isa<::reussir::RcType>($_self) &&
    ::llvm::isa<::reussir::ArrayType>(
      ::llvm::cast<::reussir::RcType>($_self).getElementType())
    }]>, "Reussir rc array type", "::reussir::RcType">;

#endif // REUSSIR_IR_REUSSIRTYPES_TD
