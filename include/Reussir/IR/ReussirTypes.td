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
/// This file defines the Reussir dialect types.
///
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
      "reussir::RecordKind":$kind, "reussir::Capability":$defaultCapability,
      "bool":$fixed);

  let storageClass = "RecordTypeStorage";
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
      // Create an identified and complete record type. `fixed` pins uniform
      // max-arm box sizing (variant-only; pass false for compounds and for
      // variant records that use the default per-constructor sizing).
      TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$members,
                      "llvm::ArrayRef<bool>":$memberIsField,
                      "mlir::StringAttr":$name, "reussir::RecordKind":$kind,
                      "reussir::Capability":$defaultCapability, "bool":$fixed),
                  [{
      return $_get($_ctxt, members, memberIsField, name, /*complete=*/true,
                       kind, defaultCapability, fixed);
    }]>,

      // Create an identified and incomplete record type. `fixed` is unknown
      // until completion, so it starts false and is set by `complete`.
      TypeBuilder<(ins "mlir::StringAttr":$name, "reussir::RecordKind":$kind),
                  [{
      return $_get($_ctxt,
        /* members= */ llvm::ArrayRef<Type>{},
        /* memberIsField= */ llvm::ArrayRef<bool>{},
        name,
        /* complete= */ false,
        kind,
        /* defaultCapability= */ reussir::Capability::unspecified,
        /* fixed= */ false);
    }]>,

      // Create an anonymous record type (always complete).
      TypeBuilder<(ins "llvm::ArrayRef<mlir::Type>":$members,
                      "llvm::ArrayRef<bool>":$memberIsField,
                      "reussir::RecordKind":$kind,
                      "reussir::Capability":$defaultCapability, "bool":$fixed),
                  [{
      return $_get($_ctxt, members, memberIsField, mlir::StringAttr{}, /*complete=*/true,
                      kind, defaultCapability, fixed);
    }]>];

  let extraClassDeclaration = [{
      using Base::verifyInvariants;
      bool isCompound() const { return getKind() == RecordKind::compound; };
      bool isVariant() const { return getKind() == RecordKind::variant; };
      // `getFixed()` (declared from the `fixed` parameter, defined in
      // ReussirTypes.cpp): whether this variant pins uniform max-arm box
      // sizing (`#[repr(fixed)]`). False (the default) sizes each boxed cell
      // for its constructed arm (`header + arm[k]`). Variant-only; always
      // false for compounds.
      void complete(llvm::ArrayRef<mlir::Type> members,
                    llvm::ArrayRef<bool> memberIsField,
                    reussir::Capability defaultCapability, bool fixed);
      struct LayoutInfo {
        llvm::TypeSize size;
        llvm::Align alignment;
        mlir::Type memberWithLargestAlignment;
      };
      LayoutInfo getElementRegionLayoutInfo(const mlir::DataLayout &dataLayout) const;
      // Per-constructor box sizing: the byte size a fused-header
      // variant box needs when it is known to hold arm `tag` — the fused
      // header plus that arm's storage, in place of the uniform max-arm
      // size `getTypeSizeInBits/8` reports. The payload offset (and thus
      // every field offset) is derived from the *variant-wide* alignment,
      // identical to the uniform layout: per-arm sizing only drops the
      // trailing padding up to the max arm. Callers gate on the variant not
      // being `#[repr(fixed)]` (i.e. `!getFixed()`); only valid on complete
      // fused-header variants.
      llvm::TypeSize getVariantArmAllocSize(const mlir::DataLayout &dataLayout,
                                            size_t tag) const;
      // Whether arm `tag` is *nullary*: an empty compound. On a taggable box
      // type (`RcType::mayCarrySpecialPointerTag`, which also guarantees at
      // the type level that every nullary arm fits `rc.tagged`'s tag slot),
      // constructions of exactly these arms are rewritten to unboxed
      // immediates that never allocate — so under the special-pointer-tag
      // scheme "nullary arm of a taggable type" means "never produces a
      // token". Single source of truth shared by the rewrite pattern and
      // the token-type computation.
      bool isNullaryArm(size_t tag) const;
      // A shared/regional variant record carries a fused 8-byte box header:
      // `{4-byte count slot, i32 tag, payload}`. When rc-boxed, the box IS
      // the record — the refcount overlays the leading slot, so the rc
      // pointer equals the record pointer. [value] variants (never a box
      // element; shared records embed as pointers) keep the compact
      // `{minimal-width tag, payload}` layout instead.
      bool hasFusedHeader() const;
      // The discriminant type of a variant record: i32 for fused-header
      // (shared/regional) variants, otherwise the narrowest integer
      // covering the arm count (i8/i16/i32).
      mlir::IntegerType getTagType() const;
      // Physical member order of a compound record: member indices sorted by
      // descending storage alignment (stable on declaration order), so no
      // padding ever appears between members. Identity for variants.
      llvm::SmallVector<uint32_t> getPackedOrder(const mlir::DataLayout &dataLayout) const;
      // Position of logical member `index` in the packed physical order —
      // i.e. the GEP field index of that member in the converted LLVM struct.
      uint32_t getPhysicalMemberIndex(const mlir::DataLayout &dataLayout, uint32_t index) const;
      // Outlined drop/acquire glue symbols. The glue's body derives member
      // links from its argument reference's atomic kind (the whole-subtree
      // rule), so each kind gets its own symbol and the atomic variant names
      // a distinct intrinsic path.
      ::mlir::FlatSymbolRefAttr
      getDtorName(AtomicKind kind = AtomicKind::normal) const;
      ::mlir::FlatSymbolRefAttr
      getAcquireName(AtomicKind kind = AtomicKind::normal) const;
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
    `reussir.token` represents a block of raw memory. Its `size` is normally a
    compile-time constant (`token<align: 8, size: 24>`). Under per-constructor
    box sizing an *unpinned* variant decrement cannot know its arm
    statically, so its token has a **dynamic** size, written
    `token<align: 16, size: ?>`: the size is carried at runtime. A dynamic
    token lowers to a fat `{ptr, size}` pair (align stays static, from the
    type) instead of a bare pointer, so its free and realloc pass the exact
    runtime size — sound on any allocator, mimalloc or talc alike.
  }];
  let parameters = (ins "size_t":$align, "size_t":$size);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Sentinel `size` marking a runtime-carried (dynamic) token size. Chosen
    // as the max representable size so it can never collide with a real
    // allocation size and so type equality/hashing treat all dynamic tokens
    // of a given alignment as one type.
    static constexpr size_t kDynamicSize = ~size_t{0};
    bool isDynamicSize() const { return getSize() == kDynamicSize; }
    // A dynamic token of the given alignment.
    static TokenType getDynamic(::mlir::MLIRContext *ctx, size_t align) {
      return get(ctx, align, kDynamicSize);
    }
  }];
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
    : ReussirType<"Rc", "rc",
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
// Reussir Cell Type
//===----------------------------------------------------------------------===//
def ReussirCellType
    : ReussirType<"Cell", "cell",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir mutable cell payload type";
  let description = [{
    `reussir.cell` is the one-element mutable payload of a shared reference
    counted cell. A cell value is always managed through
    `!reussir.rc<!reussir.cell<T>>`; when a cell occurs as a record member it
    is represented by that shared RC pointer, like closures and arrays.

    The element is stored inline in the cell box and may itself be an RC type.

    A plain cell (`!reussir.cell<T>`) supports only whole-element access
    (`reussir.cell.get`/`reussir.cell.set`). An exclusive cell
    (`!reussir.cell<T exclusive>`) additionally supports in-place
    read-modify-write (`reussir.cell.rmw`): its storage carries a trailing
    `i1` in-use flag recording whether the element is currently moved out
    into an rmw body, and every access is guarded against that state. The
    flag is addressed as slot `[1]` of the cell reference and can be
    observed with `reussir.cell.in_use`.

    An atomic cell (`!reussir.cell<T atomic>`) is an RC-contained atomic
    scalar, analogous to an inline `std::atomic<T>` inside the RC box. Its
    element must be a byte-addressable signless integer or floating-point
    primitive. Atomic cells use atomic get/set operations and support both
    the direct and region forms of `reussir.cell.rmw`. Atomic and exclusive
    are distinct, mutually exclusive storage kinds.

    A `mutex`, `flatlock`, or `rwlock` cell (`!reussir.cell<T mutex>`,
    `!reussir.cell<T flatlock>`, `!reussir.cell<T rwlock>`) wraps a payload `T`
    in a synchronization primitive from the `sync` dialect: a mutex, a
    flat-combining lock, and a reader-writer lock respectively. Access to the
    payload is mediated by a lock-holding critical-section region, which views
    the payload through a zero-ranked memref — so `T` must be a valid memref
    element type (an RC pointer qualifies; records and nullables do not).
    Mutex whole-element get/set operations lower to such regions, and every
    lock-guarded cell is constructable through `cell.create` (backed by the
    corresponding `sync` init); the remaining lock operations become available
    as their lowerings are implemented.

    An atomic, mutex, flatlock, or rwlock cell only exists to be shared across
    threads, so it is always managed by an atomic shared RC pointer
    (`!reussir.rc<!reussir.cell<T atomic> atomic>`): the box refcount and the
    payload slot are atomic together. Plain and exclusive cells may sit in
    either box flavor.
  }];
  let parameters = (ins "mlir::Type":$eleTy, "reussir::CellKind":$kind);
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "mlir::Type":$eleTy,
                     CArg<"reussir::CellKind",
                          "reussir::CellKind::plain">:$kind), [{
      return $_get($_ctxt, eleTy, kind);
    }]>,
    // Compatibility builder for the source-language/backend API, which still
    // models only plain versus exclusive cells. Atomic cells are currently an
    // MLIR-level facility and use the CellKind builder directly.
    TypeBuilder<(ins "mlir::Type":$eleTy, "bool":$exclusive), [{
      return $_get($_ctxt, eleTy,
                   exclusive ? reussir::CellKind::exclusive
                             : reussir::CellKind::plain);
    }]>
  ];
  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    bool getExclusive() const { return getKind() == CellKind::exclusive; }
    bool getAtomic() const { return getKind() == CellKind::atomic; }
    bool getMutex() const { return getKind() == CellKind::mutex; }
    bool getFlatlock() const { return getKind() == CellKind::flatlock; }
    bool getRwlock() const { return getKind() == CellKind::rwlock; }
    /// Whether this is a lock-guarded cell (mutex/flatlock/rwlock), whose
    /// payload is protected by a `sync` dialect synchronization primitive.
    bool getLockGuarded() const {
      return getKind() == CellKind::mutex || getKind() == CellKind::flatlock ||
             getKind() == CellKind::rwlock;
    }
    /// Whether this cell must be managed by an atomic shared RC box because it
    /// is meant to be shared across threads: atomic scalars and lock-guarded
    /// cells share that requirement.
    bool requiresAtomicSharedBox() const {
      return getAtomic() || getLockGuarded();
    }
  }];
}
//===----------------------------------------------------------------------===//
// Reussir Reference Type
//===----------------------------------------------------------------------===//
def ReussirRefType
    : ReussirType<"Ref", "ref",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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

//===----------------------------------------------------------------------===//
// Reussir Constructor Context Type
//===----------------------------------------------------------------------===//
def ReussirCctxType : ReussirType<"Cctx", "cctx"> {
  let summary = "Reussir first-class constructor context";
  let description = [{
    `reussir.cctx` is a first-class constructor context (Koka-style `ctx`): a
    tree of already-allocated constructors with exactly one unfilled hole. It
    is represented as a `{root, hole}` pointer pair — the root of the partial
    value and the address of the hole — with a null hole denoting the empty
    (identity) context.

    Contexts are linear values: each context is consumed exactly once, either
    by `reussir.cctx.extend` (plugging a new constructor into the hole and
    refocusing on one of the new constructor's holes) or by
    `reussir.cctx.apply` (plugging a finished value into the hole and yielding
    the completed root). The TRMC pass threads a context through
    tail-recursion-modulo-constructor chains so every constructor arm becomes
    a genuine tail call.
  }];

  let parameters = (ins "mlir::Type":$eleTy);
  let assemblyFormat = "`<` $eleTy `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    RcType getElementType() const { return llvm::cast<RcType>(getEleTy()); }
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
    // Whether the box IS its element: a non-regional box of a fused-header
    // variant overlays the refcount on the element's leading count slot, so
    // box pointer == element pointer.
    bool isHeaderFused() const;
    // Canonical physical header fields, shared by data-layout computation and
    // LLVM type conversion.
    llvm::SmallVector<mlir::Type> getHeaderTypes() const;
    size_t getElementIndex() const { return getHeaderTypes().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Closure Type
//===----------------------------------------------------------------------===//
def ReussirClosureType : ReussirType<"Closure", "closure"> {
  let summary = "Reussir Closure Type";
  let description = [{
    `reussir.closure` represents a closure. 
    A closure must be within a shared Rc type. The RC pointer points to a 
    RcBox wrapping a ClosureBox. The ClosureBox contains the payload values
    captured by the closure.
  }];
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$inputTypes,
      OptionalParameter<"::mlir::Type">:$outputType);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    static constexpr size_t VTABLE_DROP_INDEX = 0;
    static constexpr size_t VTABLE_CLONE_INDEX = 1;
    static constexpr size_t VTABLE_EVALUATE_INDEX = 2;
  }];
}

def ReussirClosureBoxType
    : ReussirType<"ClosureBox", "closure_box",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$payloadTypes);
  let assemblyFormat = "`<` $payloadTypes `>`";
  let extraClassDeclaration = [{
    static constexpr size_t VTABLE_INDEX = 0;
    static constexpr size_t ARG_CURSOR_INDEX = 1;
    static constexpr size_t PAYLOAD_TAIL_INDEX = 2;
    static constexpr size_t PAYLOAD_INDEX_WITHOUT_PADDING = 0;
    static constexpr size_t PAYLOAD_INDEX_WITH_PADDING = 1;

    // Canonical physical header fields, shared by data-layout computation and
    // LLVM type conversion.
    llvm::SmallVector<mlir::Type> getHeaderTypes() const;
    uint64_t getPayloadPadding(const ::mlir::DataLayout &dataLayout) const;
    size_t getPayloadIndex(const ::mlir::DataLayout &dataLayout) const;
  }];
}

def ReussirArrayType
    : ReussirType<"Array", "array",
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
          "View",
          "view", [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
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

def ReussirFFIObjectType : ReussirType<"FFIObject", "ffi_object",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir FFIObject Object Type";
  let description = [{
    `reussir.ffi_object` is a special opaque type that is always
    behind a RC pointer. Its dec operation, however, is handled by an FFI call.

    Layout queries see only what the FFI contract guarantees: the `u32`
    refcount the foreign `Rc` wrapper pins at offset 0. The payload behind
    the header stays opaque — debug info describing a boxed variable is the
    client, and it renders the box as that visible header.
  }];
  let parameters = (ins "::mlir::StringAttr":$ffiName,
      "::mlir::FlatSymbolRefAttr":$cleanupHook);
  let assemblyFormat = "`<` $ffiName `,` $cleanupHook `>`";
}

def ReussirStrType : ReussirType<"Str", "str",
                  [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir String Reference Type";
  let description = [{
    `reussir.str` is for str reference. Unlike rust, it only has two variants:
    - *global*: can be materialized
    - *local*: should not be materialized
  }];
  let parameters = (ins "::reussir::LifeScope":$lifeScope);
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
  }]>,
                         "Reussir rc closure type", "::reussir::RcType">;

def ReussirNullableTokenType
    : Type<CPred<[{
  ::llvm::isa<::reussir::NullableType>($_self) && 
  ::llvm::isa<::reussir::TokenType>(
    ::llvm::cast<::reussir::NullableType>($_self).getPtrTy())
  }]>,
           "Reussir nullable token type", "::reussir::NullableType">;

def ReussirCompoundType
    : Type<CPred<[{
  ::llvm::cast<::reussir::RecordType>($_self).isCompound()}]>,
           "Reussir compound type", "::reussir::RecordType">;

def ReussirVariantType : Type<CPred<[{
  ::llvm::cast<::reussir::RecordType>($_self).isVariant()}]>,
                              "Reussir variant type", "::reussir::RecordType">;

def ReussirRcClosureType : Type<CPred<[{
    ::llvm::isa<::reussir::ClosureType>(
      ::llvm::cast<::reussir::RcType>($_self).getElementType())
    }]>,
                                "Reussir rc closure type", "::reussir::RcType">;

def ReussirClosureBoxRefType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ClosureBoxType>(
      ::llvm::cast<::reussir::RefType>($_self).getElementType())
    }]>,
           "Reussir closure box reference type", "::reussir::RefType">;

def ReussirArrayPayloadType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ArrayType>($_self)
    }]>,
           "Reussir array payload type", "::reussir::ArrayType">;

def ReussirArrayViewType
    : Type<CPred<[{
    ::llvm::isa<::reussir::ViewType>($_self)
    }]>,
           "Reussir array view type", "::reussir::ViewType">;

def ReussirRcArrayType : Type<CPred<[{
    ::llvm::isa<::reussir::RcType>($_self) &&
    ::llvm::isa<::reussir::ArrayType>(
      ::llvm::cast<::reussir::RcType>($_self).getElementType())
    }]>,
                              "Reussir rc array type", "::reussir::RcType">;

def ReussirRcCellType : Type<CPred<[{
    ::llvm::isa<::reussir::RcType>($_self) &&
    ::llvm::isa<::reussir::CellType>(
      ::llvm::cast<::reussir::RcType>($_self).getElementType())
    }]>,
                             "Reussir rc cell type", "::reussir::RcType">;

#endif // REUSSIR_IR_REUSSIRTYPES_TD
