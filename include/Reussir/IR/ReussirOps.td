//===-- ReussirOps.td - Reussir dialect operations ---------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reussir dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIROPS_TD
#define REUSSIR_IR_REUSSIROPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "Reussir/IR/ReussirDialect.td"
include "Reussir/IR/ReussirAttrs.td"
include "Reussir/IR/ReussirTypes.td"
include "Reussir/IR/ReussirInterfaces.td"

class ReussirOp<string mnemonic, list<Trait> traits>
    : Op<ReussirDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Generic Operations
//===----------------------------------------------------------------------===//

def ReussirPanicOp : ReussirOp<"panic", []> {
  let summary = "Panic the program";
  let description = [{
    `reussir.panic` panics the program
  }];

  let arguments = (ins Arg<StrAttr, "message">:$message);

  let assemblyFormat = [{
    $message attr-dict
  }];
}

def ReussirExpectOp : ReussirOp<"expect", [Pure]> {
  let summary = "Mark a condition as expected";
  let description = [{
    `reussir.expect` hints that a boolean condition is expected to evaluate to
    true on the hot path.
  }];

  let arguments = (ins I1:$condition, I1Attr:$expected);
  let results = (outs I1:$likely);

  let assemblyFormat = [{
    `(` $condition `:` type($condition) `,` $expected `)` `:` type($likely) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Constructor Context Operations
//===----------------------------------------------------------------------===//
class ReussirCctxOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"cctx."#mnemonic, traits>;

def ReussirCctxEmptyOp : ReussirCctxOp<"empty", [Pure]> {
  let summary = "Create the empty constructor context";
  let description = [{
    `reussir.cctx.empty` produces the identity context: applying it to a value
    yields the value itself, and extending it makes the plugged constructor
    the context's root.
  }];

  let results = (outs Res<ReussirCctxType, "empty context">:$ctx);
  let assemblyFormat = [{
    `:` qualified(type($ctx)) attr-dict
  }];
}

def ReussirCctxExtendOp : ReussirCctxOp<"extend", []> {
  let summary = "Plug a constructor into a context and refocus on its hole";
  let description = [{
    `reussir.cctx.extend` plugs `rcPtr` — a constructor freshly created with a
    hole (see the `holeFields` attribute on the fused `rc.create_*` ops) —
    into the context's hole, and yields the context refocused on `hole`, one
    of the plugged constructor's own holes. If the incoming context is empty,
    the plugged constructor becomes the new context's root.

    Both the incoming context and the hole are consumed: contexts are linear,
    and the refocused hole must be filled exactly once through the resulting
    context.
  }];

  let arguments = (ins
      Arg<ReussirCctxType, "context to extend">:$ctx,
      Arg<ReussirRcType, "constructor plugged into the hole">:$rcPtr,
      Arg<ReussirHoleType, "the plugged constructor's hole to refocus on">:$hole);
  let results = (outs Res<ReussirCctxType, "extended context">:$newCtx);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $ctx `:` qualified(type($ctx)) `)`
    `plug` `(` $rcPtr `:` qualified(type($rcPtr)) `)`
    `hole` `(` $hole `:` qualified(type($hole)) `)`
    `:` qualified(type($newCtx)) attr-dict
  }];
}

def ReussirCctxApplyOp : ReussirCctxOp<"apply", []> {
  let summary = "Complete a constructor context with a final value";
  let description = [{
    `reussir.cctx.apply` plugs `value` into the context's hole and yields the
    context's root — the now-complete value. Applying the empty context
    yields `value` itself. The context is consumed.
  }];

  let arguments = (ins
      Arg<ReussirCctxType, "context to complete">:$ctx,
      Arg<ReussirRcType, "value filling the final hole">:$value);
  let results = (outs Res<ReussirRcType, "completed value">:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $ctx `:` qualified(type($ctx)) `)`
    `(` $value `:` qualified(type($value)) `)`
    `:` qualified(type($result)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Token Operation
//===----------------------------------------------------------------------===//

class ReussirTokenOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"token."#mnemonic, traits>;

def ReussirTokenAllocOp : ReussirTokenOp<"alloc", []> {
  let summary = "Allocate memory token";
  let description = [{
    `reussir.token.alloc` allocates a memory token.
    ```mlir
    reussir.token.alloc : !reussir.token<align: 8, size: 16>
    ```
  }];

  let results =
      (outs Res<ReussirTokenType,
                "Allocated Memory Token", [MemAlloc<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `:` type($token) attr-dict
  }];
}

def ReussirTokenFreeOp : ReussirTokenOp<"free", []> {
  let summary = "Free memory token";
  let description = [{
    `reussir.token.free` deallocates a memory token.
    ```mlir
    reussir.token.free (%x : !reussir.token<align: 8, size: 16>)
    ```
  }];

  let arguments =
      (ins Res<AnyTypeOf<[ReussirTokenType, ReussirNullableTokenType]>,
               "Free Memory Token", [MemFree<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict
  }];
}

def ReussirTokenReinterpretOp : ReussirTokenOp<"reinterpret", []> {
  let summary = "Reinterpret memory token";
  let description = [{
    `reussir.token.reinterpret` reinterprets a token into a reference.
    ```mlir
    %reinterpreted = reussir.token.reinterpret 
      (%token : !reussir.token<align: 8, size: 8>) : !reussir.ref<i64>
    ```
    The layout of the token and the referenced type must match.
  }];

  let results =
      (outs Res<ReussirRefType, "reinterpreted reference">:$reinterpreted);
  let arguments = (ins Arg<ReussirTokenType, "Token to reinterpret">:$token);
  let assemblyFormat = [{
    `(` $token `:` type($token) `)` `:` qualified(type($reinterpreted)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirTokenEnsureOp : ReussirTokenOp<"ensure", []> {
  let summary = "Ensure a nonnull token";
  let description = [{
    `reussir.token.ensure` converts a possibly nullable token to a token by 
    checking the nullability and allocate memory on need.
    ```mlir
    %result = reussir.token.ensure
      (%nullableToken : !reussir.nullable.token<align: 8, size: 8>) 
      : !reussir.token<align: 8, size: 8>
    ```
  }];
  let results = (outs Res<ReussirTokenType,
                          "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments =
      (ins Arg<ReussirNullableTokenType,
               "input token", [MemFree<DefaultResource>]>:$nullableToken);
  let assemblyFormat = [{
    `(` $nullableToken `:` type($nullableToken) `)` `:` type($result) attr-dict
  }];
}

def ReussirTokenLaunderOp : ReussirTokenOp<"launder", []> {
  let summary = "Launder a token";
  let description = [{
    `reussir.token.launder` launders a token for reuse.
    ```mlir
    %result = reussir.token.launder
      (%token : !reussir.token<align: 8, size: 8>) 
      : !reussir.token<align: 8, size: 8>
    ```
  }];
  let results = (outs Res<ReussirTokenType,
                          "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments =
      (ins Arg<ReussirTokenType,
               "input token", [MemFree<DefaultResource>]>:$token);
  let assemblyFormat = [{
    `(` $token `:` type($token) `)` `:` type($result) attr-dict
  }];
}

def ReussirTokenReallocOp : ReussirTokenOp<"realloc", []> {
  let summary = "Reallocate a token";
  let description = [{
    `reussir.token.realloc` converts a possibly nullable token to a token by checking
    the nullability and reallocate the memory on need. During the process, if the original
    layout and the new layout are different, the memory will be reallocated, such reallocation
    may happen in place depending on the allocator and the new size.

    The block's contents are NOT preserved across the resize: the operand token is dead
    storage about to be overwritten (token reuse), so the runtime relocates with a plain
    free + alloc instead of a copying realloc — the runtime entry points are declared
    `allockind("realloc,uninitialized,...")` accordingly.

    ```mlir
    %realloced = reussir.token.realloc
      (%token : !reussir.token<align: 8, size: 8>) 
      : !reussir.token<align: 8, size: 16>
    ```
  }];

  let arguments =
      (ins Res<AnyTypeOf<[ReussirTokenType, ReussirNullableTokenType]>,
               "Target Token to Realloc", [MemFree<DefaultResource>]>:$token);

  let results = (outs Res<
      ReussirTokenType,
      "Reallocated Memory Token", [MemAlloc<DefaultResource>]>:$realloced);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` `:` type($realloced) attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// Reussir Nullable Operations
//===----------------------------------------------------------------------===//
class ReussirNullableOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"nullable."#mnemonic, traits>;

def ReussirNullableCheckOp : ReussirNullableOp<"check", []> {
  let summary = "Check if a nullable token is null";
  let description = [{
    `reussir.nullable.check` checks if a nullable token is null.
    ```mlir
    %flag = reussir.nullable.check
      (%nullable : !reussir.nullable<token<align: 8, size: 8>>) : i1
    ```
  }];

  let results = (outs Res<I1, "is null">:$flag);
  let arguments = (ins Arg<ReussirNullableType, "nullable pointer">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` `:` type($flag) attr-dict
  }];
}

def ReussirNullableCreateOp : ReussirNullableOp<"create", []> {
  let summary = "Create a nullable token";
  let description = [{
    `reussir.nullable.create` creates a nullable token.
    It can accept a inner pointer, which means it is acutally non-nullable,
    or it can be a null pointer if there is no input argument.
    ```mlir
    %nullable = 
      reussir.nullable.create : !reussir.nullable<!reussir.rc<index>>
    ```
  }];

  let results = (outs Res<ReussirNullableType, "nullable pointer">:$nullable);
  let arguments = (ins Optional<ReussirNonNullPointerType>:$ptr);
  let assemblyFormat = [{
    ( `(` $ptr^ `:` type($ptr) `)` )? `:` type($nullable) attr-dict
  }];
}

// this is a high-level scf operation, 
// should be expanded before basic ops lowering pass
def ReussirNullableDispatchOp : ReussirNullableOp<"dispatch", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Dispatch a nullable token";
  let description = [{
    `reussir.nullable.dispatch` is a high-level structured control flow operation.
    It takes a nullable pointer and two regions where the first region accepts a 
    non-nullable pointer and second region has no region argument. Both regions
    must be terminated by `reussir.scf.yield` operation.

    ```mlir
    %flag = reussir.nullable.dispatch 
      (%nullable : !reussir.nullable<!reussir.rc<i64>>) -> i1 {
      nonnull -> {
        ^bb0(%nonnull_ptr : !reussir.rc<i64>):
          %true = arith.constant 1 : i1
          reussir.scf.yield %true : i1
      }
      null -> {
        ^bb0:
          %false = arith.constant 0 : i1
          reussir.scf.yield %false : i1
      }
    }
    ```
  }];
  
  let arguments = (ins Arg<ReussirNullableType, "nullable pointer">:$nullable);
  let results = (outs Res<Optional<AnyType>, "dispatched value">:$value);
  let regions = (region SizedRegion<1>:$nonNullRegion,
                 SizedRegion<1>:$nullRegion);

  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` ( `->` type($value)^)? `{` `\n`
      `nonnull` `->` $nonNullRegion `\n`
      `null` `->` $nullRegion
    `}` attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirNullableCoerceOp : ReussirNullableOp<"coerce", []> {
  let summary = "Coerce a nullable token to a non-nullable token";
  let description = [{
    `reussir.nullable.coerce` coerces a nullable token to a non-nullable token.
    This operation is a placeholder to make sure type is consistent during 
    intermediate lowering steps. Higher-level IR generation should typically use
    the `reussir.nullable.dispatch` operation to handle nullable tokens.

    ```mlir
    %nonnull = reussir.nullable.coerce 
      (%nullable : !reussir.nullable<!reussir.token<align: 8, size: 8>>) 
      : !reussir.token<align: 8, size: 8>
    ```
  }];

  let results = (outs Res<ReussirNonNullPointerType, "non-null pointer">:$nonnull);
  let arguments =
      (ins Arg<ReussirNullableType, "nullable pointer">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` `:` type($nonnull) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir RC Operations
//===----------------------------------------------------------------------===//
class ReussirRcOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"rc."#mnemonic, traits>;

def ReussirRcIncOp : ReussirRcOp<"inc", []> {
  let summary = "Increment reference count";
  let description = [{
    `reussir.rc.inc` increments the reference count of a RC pointer.
    ```mlir
    reussir.rc.inc (%rc_ptr : !reussir.rc<i64>)
    ```
  }];

  let arguments = (ins Arg<ReussirRcType, "Reference to increment">:$rcPtr);

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRcDecOp : ReussirRcOp<"dec", [
  DeclareOpInterfaceMethods<TokenProducerInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Decrement reference count";
  let description = [{
    `reussir.rc.dec` decrements the reference count of a RC pointer.
    If the reference count reaches zero, the memory will not be freed immediately.
    Rather, a nullable token is returned.
    ```mlir
    %nullable = reussir.rc.dec (%rc_ptr : !reussir.rc<i64>) -> 
      !reussir.nullable<!reussir.token<align: 8, size: 8>>
    ```

    For rigid RC pointer, this operation is basically a function call to
    __reussir_release_rigid_object. There is no token shall be returned.
  }];

  let arguments = (
    ins Arg<ReussirRcType, "Reference to decrement">:$rcPtr,
    OptionalAttr<IndexAttr>:$destructureTag,
    OptionalAttr<DenseI64ArrayAttr>:$boundMembers
  );
  let results =
      (outs Res<Optional<ReussirNullableTokenType>, "Nullable token">:$nullableToken);

  let extraClassDeclaration = [{
    /// Whether this decrement *destructures* its box: a pattern match on the
    /// variant arm `destructureTag` consumed the box, taking ownership of the
    /// `boundMembers` (their retain/release pairs were fused away). It
    /// expands shallowly instead of through the transitive drop glue: on the
    /// unique path the bound members transfer with the arm (only unbound rc
    /// members are released) and the box becomes a token; on the shared path
    /// the bound members are retained and the count drops. Introduced by
    /// `reussir-rc-dispatch-fusion`; cancelled against a preceding increment
    /// (bound retains rematerialize) by `reussir-inc-dec-cancellation`.
    bool isDestructuring() { return getDestructureTagAttr() != nullptr; }
  }];

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` 
    (`:` type($nullableToken)^)? 
    attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRcFetchOp : ReussirRcOp<"fetch", []> {
  let summary = "Fetch the reference count of a RC pointer";
  let description = [{
    `reussir.rc.fetch` fetches the current reference count from a RC pointer.

    This serves as a low-level operation to implement `reussir.rc.dec` and
    related RC management transforms. It is not recommended to use this
    operation directly.
  }];

  let arguments = (ins Arg<ReussirRcType, "Reference to inspect">:$rcPtr);
  let results = (outs Res<Index, "Reference count">:$refCount);

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` type($refCount) attr-dict
  }];
}

def ReussirRcSetOp : ReussirRcOp<"set", []> {
  let summary = "Set the reference count of a RC pointer";
  let description = [{
    `reussir.rc.set` writes a reference count to a RC pointer.

    This serves as a low-level operation to implement `reussir.rc.dec` and
    related RC management transforms. It is not recommended to use this
    operation directly.
  }];

  let arguments = (
    ins
      Arg<ReussirRcType, "Reference to update">:$rcPtr,
      Arg<Index, "New reference count">:$refCount
  );

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `,`
    $refCount `:` type($refCount) `)` attr-dict
  }];
}

def ReussirRcCreateOp : ReussirRcOp<"create", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<TokenAcceptorInterface>
]> {
  let summary = "Create a RC pointer";
  let description = [{
    `reussir.rc.create` creates a RC pointer from a value.
    The memory resource can be provided by an optional token. It additionally accepts
    a region argument.

    RC pointer created solely from token has shared capability, while RC pointer 
    created with region has flex capability.

    When a token is provided, its layout must meet the underlying Rc box layout.

    The created RC pointer will have reference count 1.

    RC with default capability:
    ```mlir
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 16>
    %val = arith.constant 42 : i64
    %rc_ptr = reussir.rc.create value(%val : i64) 
      token(%token : !reussir.token<align: 8, size: 16>)
      : !reussir.rc<i64>
    ```

    RC with flex capability:
    ```mlir
    func.func @region_funcion(%region: !reussir.region) {
      %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
      %val = arith.constant 42 : i64
      %rc_ptr = reussir.rc.create 
        value (%val : i64) 
        token (%token : !reussir.token<align: 8, size: 32>) 
        region (%region : !reussir.region)
        : !reussir.rc<i64, flex>
      func.return
    }
    ```
  }];

  let arguments = (ins 
      Arg<AnyType, "value">:$value, 
      Optional<ReussirTokenType>:$token,
      Optional<ReussirRegionType>:$region,
      OptionalAttr<FlatSymbolRefAttr>:$vtable,
      UnitAttr:$skipRc
  );
  let results = (outs Res<ReussirRcType, "Created RC pointer">:$rcPtr);

  let assemblyFormat = [{
    `value` `(` $value `:` type($value) `)` 
    oilist ( 
      `token` `(` $token `:` type($token) `)`
    | `region` `(` $region `:` type($region) `)` 
    | `vtable` `(` $vtable `)`
    | `skip_rc` $skipRc
    )
    `:` qualified(type($rcPtr)) attr-dict
  }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool needsVTable() {
      auto recordType = llvm::dyn_cast<RecordType>(getValue().getType());
      // A region-allocated object always needs a vtable: the region reclaims
      // every object it owns and reads the vtable for its layout (and any
      // drop/scan) to do so, even when the payload is trivially copyable.
      return getRegion() && recordType;
    }
  }];
}

def ReussirRcCreateCompoundOp : ReussirRcOp<"create_compound", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<TokenAcceptorInterface>
]> {
  let summary = "Create a RC pointer from a compound record payload";
  let description = [{
    `reussir.rc.create_compound` fuses `reussir.record.compound` and
    `reussir.rc.create` into a single operation. It stores the record fields
    directly into the RC box payload.
  }];

  let arguments = (ins
      Variadic<AnyType>:$fields,
      Optional<ReussirTokenType>:$token,
      Optional<ReussirRegionType>:$region,
      OptionalAttr<FlatSymbolRefAttr>:$vtable,
      UnitAttr:$skipRc,
      OptionalAttr<DenseI64ArrayAttr>:$skipFields,
      OptionalAttr<DenseI64ArrayAttr>:$holeFields
  );
  let results = (outs
      Res<ReussirRcType, "Created RC pointer">:$rcPtr,
      Variadic<ReussirHoleType>:$holes);

  let assemblyFormat = [{
    (`(` $fields^ `:` type($fields) `)`)? 
    oilist (
      `token` `(` $token `:` type($token) `)`
    | `region` `(` $region `:` type($region) `)`
    | `vtable` `(` $vtable `)`
    | `skip_rc` $skipRc
    )
    `:` qualified(type($rcPtr)) (`,` qualified(type($holes))^)? attr-dict
  }];

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    RecordType getRecordType() {
      return llvm::dyn_cast<RecordType>(getRcPtr().getType().getElementType());
    }

    bool needsVTable() {
      auto recordType = getRecordType();
      // A region-allocated object always needs a vtable: the region reclaims
      // every object it owns and reads the vtable for its layout (and any
      // drop/scan) to do so, even when the payload is trivially copyable.
      return getRegion() && recordType;
    }
  }];
}

def ReussirRcCreateVariantOp : ReussirRcOp<"create_variant", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<TokenAcceptorInterface>
]> {
  let summary = "Create a RC pointer from a variant record payload";
  let description = [{
    `reussir.rc.create_variant` fuses `reussir.record.variant` and
    `reussir.rc.create` into a single operation. When the selected variant
    payload is itself a compound record, the compound fields can be carried
    directly to represent a fully fused `variant(compound(...))` materialization.
  }];

  let arguments = (ins
      Arg<IndexAttr, "tag">:$tag,
      Optional<AnyType>:$value,
      Variadic<AnyType>:$fields,
      Optional<ReussirTokenType>:$token,
      Optional<ReussirRegionType>:$region,
      OptionalAttr<FlatSymbolRefAttr>:$vtable,
      UnitAttr:$skipRc,
      OptionalAttr<DenseI64ArrayAttr>:$skipFields,
      OptionalAttr<DenseI64ArrayAttr>:$holeFields
  );
  let results = (outs
      Res<ReussirRcType, "Created RC pointer">:$rcPtr,
      Variadic<ReussirHoleType>:$holes);

  let hasVerifier = 1;
  let extraClassDeclaration = [{
    RecordType getRecordType() {
      return llvm::dyn_cast<RecordType>(getRcPtr().getType().getElementType());
    }

    bool needsVTable() {
      auto recordType = getRecordType();
      // A region-allocated object always needs a vtable: the region reclaims
      // every object it owns and reads the vtable for its layout (and any
      // drop/scan) to do so, even when the payload is trivially copyable.
      return getRegion() && recordType;
    }
  }];
}

def ReussirRcTaggedOp : ReussirRcOp<"tagged", [Pure]> {
  let summary = "Materialize a nullary variant as a pointer immediate";
  let description = [{
    `reussir.rc.tagged` produces an rc pointer encoding a *nullary* variant
    (an arm whose payload is an empty compound) as an unboxed immediate. No
    allocation takes place — no token is ever attached — and no observable
    reference counting happens: the value points at a per-tag dummy box (a
    static global shaped like a real box header `{refcount, tag}`), so
    refcount/uniqueness/tag reads through it land in well-formed memory and
    observe a never-unique shared header — the hot lowerings need no
    immediate-checks at all; only the few guarded refcount *stores* must
    avoid writing the dummy box.

    The op itself is encoding-agnostic; the module's
    `reussir.special_ptr_tag` attribute selects how the immediate is
    encoded at LLVM lowering:
      * `immortal` (any target): the immediate is the dummy box address,
        plain; guards recognize immediates by the dummy refcount's
        immortal magnitude.
      * `tbi` (aarch64, 64-bit): additionally ORs `(tag + 1) << 56` into
        the pointer's top byte (hardware top-byte-ignore masks it on data
        accesses), so the tag also decodes from the pointer without a
        memory access; guards test the top byte.

    Under both encodings the op reserves one byte for the tag, `0` meaning
    "untagged, a real box pointer". The bound exists for the *guards*: under
    `tbi` a zero top byte is the classifier for "real box, stores may
    proceed", so every immediate must carry a nonzero top byte — an
    unencodable arm must stay a real box (see the scheme invariant at
    `kSpecialPtrTagAttr`). The verifier enforces `tag + 1 <= 0xFF` under
    both encodings so the same IR stays valid under either. The range is a
    *type-level* eligibility concern (`RcType::mayCarrySpecialPointerTag`
    refuses types whose nullary arms exceed the slot); per arm, "nullary on
    a taggable type" alone decides immediate-vs-boxed, and hence
    token-vs-no-token.
    ```mlir
    %leaf = reussir.rc.tagged tag(1) : !reussir.rc<!reussir.record<...>>
    ```
  }];

  let arguments = (ins Arg<IndexAttr, "tag">:$tag);
  let results = (outs Res<ReussirRcType, "tagged immediate">:$rcPtr);
  let assemblyFormat = [{
    `tag` `(` $tag `)` `:` qualified(type($rcPtr)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRcReinterpretOp : ReussirRcOp<"reinterpret", []> {
  let summary = "Reinterpret a Rc pointer into a token";
  let description = [{
    `reussir.rc.reinterpret` reinterprets a Rc pointer into a token.

    The token layout must meet the underlying Rc box layout.
  }];

  let results =
      (outs Res<ReussirTokenType, "reinterpreted token">:$reinterpreted);
  let arguments = (ins Arg<ReussirRcType, "Rc pointer to reinterpret">:$rcPtr);
  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` qualified(type($reinterpreted)) attr-dict
  }];
  let hasVerifier = 1;
}

// The following is an intermediate placeholder for regional rc freezing
def ReussirRcFreezeOp : ReussirRcOp<"freeze", []> {
  let summary = "Freeze a Rc pointer into a reference";
  let description = [{
    `reussir.rc.freeze` freezes a Rc pointer into a reference.
  }];

  let results = (outs Res<ReussirRcType, "frozen rc">:$frozen);
  let arguments = (ins Arg<ReussirRcType, "Rc pointer to freeze">:$rcPtr);
  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` qualified(type($frozen)) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir RC Borrow Operation
//===----------------------------------------------------------------------===//
def ReussirRcBorrowOp : ReussirRcOp<"borrow", []> {
  let summary = "Borrow a RC pointer";
  let description = [{
    `reussir.rc.borrow` borrows a RC pointer, which gives you an access reference 
    to the underlying value. The shared/flex/rigid capability from Rc will be inherited
    by the borrowed reference. Additionally, the result reference can be of unspecified
    capability. The borrow will not change the reference count of the RC pointer.
    ```mlir
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 8>
    %rc_ptr = reussir.rc.create 
      value(%val : i64) 
      token(%token : !reussir.token<align: 8, size: 8>) : !reussir.rc<i64>
    %ref = reussir.rc.borrow 
      (%rc_ptr : !reussir.rc<i64>) : !reussir.ref<i64 shared>
    ```
  }];

  let arguments = (ins Arg<ReussirRcType, "RC pointer to borrow">:$rcPtr);
  let results = (outs Res<ReussirRefType, "Borrowed reference">:$borrowed);

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` qualified(type($borrowed)) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Rc IsUnique Operation
//===----------------------------------------------------------------------===//
def ReussirRcIsUniqueOp : ReussirRcOp<"is_unique", []> {
  let summary = "Check if a RC pointer is unique";
  let description = [{
    `reussir.rc.is_unique` checks if a RC pointer is unique.
  }];
  let arguments = (ins Arg<ReussirRcType, "RC pointer to check">:$rcPtr);
  let results = (outs Res<I1, "Is unique">:$isUnique);
  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` type($isUnique) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRcAssumeUniqueOp : ReussirRcOp<"assume_unique", []> {
  let summary = "Assume that a RC pointer is unique";
  let description = [{
    `reussir.rc.assume_unique` asserts that a RC pointer is unique and is
    intended to lower to `llvm.assume`.
  }];
  let arguments = (ins Arg<ReussirRcType, "RC pointer to assume unique">:$rcPtr);
  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir Record Operations
//===----------------------------------------------------------------------===//
class ReussirRecordOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"record."#mnemonic, traits>;

def ReussirRecordCompoundOp : ReussirRecordOp<"compound", []> {
  let summary = "Create a compound record";
  let description = [{
    `reussir.record.compound` creates a compound record.
  }];

  let arguments = (ins Variadic<AnyType>:$fields);
  let results = (outs Res<ReussirCompoundType, "Created record">:$compound);

  let assemblyFormat = [{
    (`(` $fields^ `:` type($fields) `)`)? `:` qualified(type($compound)) attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRecordExtractOp : ReussirRecordOp<"extract", []> {
  let summary = "Extract a field from a record";
  let description = [{
    `reussir.record.extract` extracts a field from a record.
  }];

  let arguments = (ins Arg<ReussirRecordType, "Record to extract from">:$record,
      Arg<IndexAttr, "field index">:$index);
  let results = (outs Res<AnyType, "Extracted field">:$field);

  let assemblyFormat = [{
    `(` $record `:` qualified(type($record)) `)` `[` $index `]` `:` qualified(type($field)) attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRecordVariantOp : ReussirRecordOp<"variant", []> {
  let summary = "Create a variant record";
  let description = [{
    `reussir.record.variant` creates a variant record.
  }];

  let arguments = (ins Arg<IndexAttr, "tag">:$tag,
      Arg<AnyType, "value">:$value);
  let results = (outs Res<ReussirVariantType, "Created record">:$variant);

  let assemblyFormat = [{
   `[` $tag `]` `(` $value `:` type($value) `)` 
   `:` qualified(type($variant)) attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRecordTagOp : ReussirRecordOp<"tag", []> {
  let summary = "Get the tag of a variant record";
  let description = [{
    `reussir.record.tag` gets the tag of a variant record.

    The input must be a reference to a variant record.
  }];
  
  let arguments = (ins Arg<ReussirRefType, "variant record">:$variant);
  let results = (outs Res<Index, "tag">:$tag);

  let assemblyFormat = [{
    `(` $variant `:` qualified(type($variant)) `)` `:` type($tag) attr-dict
  }];

  let hasVerifier = 1;
}


// This is a high-level scf operation, 
// should be expanded before basic ops lowering pass
def ReussirRecordDispatchOp : ReussirRecordOp<"dispatch", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Dispatch a record";
  let description = [{
    `reussir.record.dispatch` is a high-level structured control flow operation.

    It provides a way to dispatch based on the tag of a variant record.
    The input must be a reference to a variant record.

    It can produce a optional value. If the result presents, all child regions 
    must yield a value of the same type.

    Each branch takes the form `[tag0, tag1, ...] -> region`. If it is a single tag, then
    the region should accept a reference to target variant element type. 
    If it is a set of tags, the region must not have any argument.

    If the dispatch operation has separated regions for each tag, this operation
    will be directly lowered to a scf.index_switch operation. Otherwise, it will
    be expanded into 2 consecutive scf.index_switch operations, where the first
    one dispatches tags to region indices and the second one dispatches the 
    region indices to the actual regions.
  }];

  let arguments = (ins Arg<ReussirRefType, "variant record">:$variant, 
      ArrayAttr:$tagSets);
  let results = (outs Res<Optional<AnyType>, "output value">:$value);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ReussirRecordCoerceOp : ReussirRecordOp<"coerce", []> {
  let summary = "Coerce a variant reference to a non-variant reference";
  let description = [{
    `reussir.record.coerce` coerces a variant reference to a non-variant reference.
    This operation is a placeholder to make sure type is consistent during
    intermediate lowering steps. Higher-level IR generation should typically use
    the `reussir.record.dispatch` operation to handle variant references.

    The operation itself is basically a GEP[0, 1] operation on the input pointer.
  }];

  let results = (outs Res<ReussirRefType, "coerced pointer">:$coerced);
  let arguments =
      (ins 
      IndexAttr:$tag,
      Arg<ReussirRefType, "variant reference">:$variant);
  let assemblyFormat = [{
    `[` $tag `]` 
    `(` $variant `:` qualified(type($variant)) `)` 
    `:` qualified(type($coerced)) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir array operations
//===----------------------------------------------------------------------===//
class ReussirArrayOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"array."#mnemonic, traits>;

def ReussirArrayProjectOp : ReussirArrayOp<"project", []> {
  let summary = "Project a fixed-size array view";
  let description = [{
    `reussir.array.project` projects the leading dimension of a statically
    shaped memref produced by `reussir.array.view`.
    When the remaining element is scalar, the result is a field reference.
    Otherwise, the result is another statically shaped memref with one fewer
    dimension.
  }];
  let arguments = (ins
    Arg<AnyType, "array view to project">:$view,
    Arg<Index, "dynamic index">:$index
  );
  let results = (outs Res<AnyType, "projected lane or subview">:$projected);
  let assemblyFormat = [{
    `(` $view `:` qualified(type($view)) `)` `[` $index `:` type($index) `]`
    `:` qualified(type($projected)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirArrayViewOp : ReussirArrayOp<"view", []> {
  let summary = "Materialize an array view from an array reference";
  let description = [{
    `reussir.array.view` turns a reference to an inline array payload into a
    statically shaped memref or tensor suitable for downstream bufferization
    and lowering.
  }];
  let arguments = (ins Arg<ReussirRefType, "array reference">:$ref);
  let results = (outs Res<AnyType, "materialized view">:$view);
  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` `:` qualified(type($view)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirArrayWithUniqueViewOp : ReussirArrayOp<"with_unique_view", []> {
  let summary = "Ensure a unique RC array, expose a mutable view, and run a region";
  let description = [{
    `reussir.array.with_unique_view` makes sure the input RC array is unique,
    clones it when needed, then executes the body region with a mutable view of
    the array payload as a statically shaped memref or tensor.

    The body region must take exactly one matching view argument and terminate
    with `reussir.scf.yield`.
  }];
  let arguments = (ins Arg<ReussirRcArrayType, "array to uniqify">:$array);
  let regions = (region MaxSizedRegion<1>:$body);
  let results = (outs Res<Optional<AnyType>, "body result">:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir reference operations
//===----------------------------------------------------------------------===//
class ReussirReferenceOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"ref."#mnemonic, traits>;

def ReussirRefProjectOp : ReussirReferenceOp<"project", []> {
  let summary = "Project a reference";
  let description = [{
    `reussir.ref.project` projects a reference. This is effectively a GEP 
    operation without the complication of self indexing at the first level.
    Similar to borrowing, the projected reference will inherit the capability 
    from the original reference.

    ```mlir
    !struct = !reussir.record<compound "foo" {i64, i64}>
    func.func @project_func(%ref : !reussir.ref<!struct>) {
      %field = reussir.ref.project (%ref : !reussir.ref<!struct>) [0] : !reussir.ref<i64>
      %field2 = reussir.ref.project (%ref : !reussir.ref<!struct>) [1] : !reussir.ref<i64>
      func.return
    }
    ```

    Notice that the projected type must match, which may be affected by the capability.
    For example, for field capability member under a rigid reference, the projected type
    is a nullable rigid rc pointer.

    ```mlir
    !struct = !reussir.record<compound "foo" {i64, field i64}>
    !nullable_rc = !reussir.nullable<!reussir.rc<i64 rigid>>
    %rc_field = reussir.ref.project (%ref : !reussir.ref<!struct rigid>) [1] 
      : !reussir.ref<!nullable_rc rigid>
    ```
  }];

  let arguments = (ins Arg<ReussirRefType, "reference to project">:$ref,
      Arg<IndexAttr, "index">:$index);
  let results = (outs Res<ReussirRefType, "projected reference">:$projected);

  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` `[` $index `]` `:` qualified(type($projected)) attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRefSpilledOp : ReussirReferenceOp<"spilled", []> {
  let summary = "Spill a value to stack allocation";
  let description = [{
    `reussir.ref.spilled` spills a value to a stack allocation, which can be
    used to pass values between regions or functions. This also gives user
    the power to project inner fields from a compound value. 
    
    The result reference must have unspecified capability.
  }];
  let arguments = (ins Arg<AnyType, "value to spill">:$value);
  let results =
      (outs Res<ReussirRefType, "spilled reference",
                [MemAlloc<AutomaticAllocationScopeResource>]>:$spilled);
  let assemblyFormat = [{
    `(` $value `:` type($value) `)` `:` qualified(type($spilled)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRefToMemrefOp : ReussirReferenceOp<"to_memref", []> {
  let summary = "Materialize a zero-rank memref view from a reference";
  let description = [{
    `reussir.ref.to_memref` turns a reference into a zero-rank memref view over
    the same payload. This is intended as a bridge to dialects that operate on
    memrefs, such as synchronization primitives that expect `memref<T>`.
  }];

  let arguments = (ins Arg<ReussirRefType, "reference to bridge">:$ref);
  let results = (outs Res<AnyType, "zero-rank memref view">:$view);
  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` `:` qualified(type($view)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRefFromMemrefOp : ReussirReferenceOp<"from_memref", []> {
  let summary = "Recover a reference from a zero-rank memref view";
  let description = [{
    `reussir.ref.from_memref` bridges a zero-rank memref view back to a
    reference over the same payload. Together with `reussir.ref.to_memref`,
    this provides a round-trip bridge between Reussir references and
    `memref<T>` values.
  }];

  let arguments = (ins Arg<AnyType, "zero-rank memref view">:$view);
  let results = (outs Res<ReussirRefType, "bridged reference">:$ref);
  let assemblyFormat = [{
    `(` $view `:` qualified(type($view)) `)` `:` qualified(type($ref)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRefLoadOp : ReussirReferenceOp<"load", []> {
  let summary = "Load a reference";
  let description = [{
    `reussir.ref.load` loads a reference.
  }];

  let arguments = (ins
    Arg<ReussirRefType, "reference to load">:$ref,
    UnitAttr:$invariant_group
  );
  let results = (outs Res<AnyType, "loaded value">:$value);

  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` (`invariant_group` $invariant_group^)?
    `:` type($value) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRefStoreOp : ReussirReferenceOp<"store", []> {
  let summary = "Store a value to a reference";
  let description = [{
    `reussir.ref.store` stores a value to a reference. Target reference must have
    field capability.
  }];
  let arguments = (ins Arg<ReussirRefType, "reference to store">:$ref,
      Arg<AnyType, "value to store">:$value);
  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` `(` $value `:` type($value) `)` attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRefDropOp : ReussirReferenceOp<"drop", []> {
  let summary = "Drop a reference";
  let description = [{
    `reussir.ref.drop` destructs the element behind the reference in place.
    
    For outlined version, the optimizer will not attempt to inline the drop operation
    for sub-elements.

    If the reference if of flex capability, the drop operation will skip the destruction
    of elements with field capability.

    When variant is specified, the inner element must be a variant record type and
    the index must be inbound.
  }];
  let arguments = (
    ins Arg<ReussirRefType, "reference to drop">:$ref,
    UnitAttr:$inlined,
    OptionalAttr<IndexAttr>:$variant 
  );

  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` 
    oilist (
      `inlined` $inlined  | 
      `variant` `[` $variant `]`
    )
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$ref),
    [{
      return build($_builder, $_state, ref, false, nullptr);
    }]>
  ];

  let hasVerifier = 1;
}

def ReussirRefAcquireOp : ReussirReferenceOp<"acquire", []> {
  let summary = "Acquire ownership of a reference";
  let description = [{
    `reussir.ref.acquire` acquires ownership of the element behind the reference
    in place. This increments reference counts for all RC pointers reachable
    from the reference.

    This is the dual of `reussir.ref.drop`. Where drop destructs, acquire
    increments ownership.

    For outlined version, the optimizer will not attempt to inline the acquire
    operation for sub-elements.

    When variant is specified, the inner element must be a variant record type and
    the index must be inbound.
  }];
  let arguments = (
    ins Arg<ReussirRefType, "reference to acquire">:$ref,
    UnitAttr:$inlined,
    OptionalAttr<IndexAttr>:$variant
  );

  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)`
    oilist (
      `inlined` $inlined  |
      `variant` `[` $variant `]`
    )
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$ref),
    [{
      return build($_builder, $_state, ref, false, nullptr);
    }]>
  ];

  let hasVerifier = 1;
}

def ReussirRefDiffOp : ReussirReferenceOp<"diff", []> {
  let summary = "Compute difference between two reference pointers";
  let description = [{
    `reussir.ref.diff` computes the difference (target - base) between two refernece pointers.
  }];
  let arguments = (ins 
    Arg<ReussirRefType, "base reference">:$base,
    Arg<ReussirRefType, "target reference">:$target
  );
  let results = (outs
    Res<Index, "difference">:$difference
  );
  let assemblyFormat = [{
    $base `,` $target `:` functional-type(operands, results) attr-dict
  }];
}

def ReussirRefCmpOp : ReussirReferenceOp<"cmp", []> {
  let summary = "Compare the address numeric value of two references";
  let description = [{
    `reussir.ref.cmp` computes the difference (target - base) between two refernece pointers.
  }];
  let arguments = (ins 
    Arith_CmpIPredicateAttr:$predicate,
    Arg<ReussirRefType, "left-hand side">:$lhs,
    Arg<ReussirRefType, "right-hand side">:$rhs
  );
  let results = (outs
    Res<I1, "compare result">:$flag
  );
  let assemblyFormat = [{
    $predicate $lhs `,` $rhs `:` functional-type(operands, results) attr-dict
  }];
}

def ReussirRefMemcpyOp : ReussirReferenceOp<"memcpy", []> {
  let summary = "Non-overlapping memory copy between two references";
  let description = [{
    `reussir.ref.memcpy` performs a non-overlapping memory copy from the source
    reference to the destination reference. The element types of both references
    must be identical.

    This is a low-level operation that does not perform any reference counting
    or ownership transfer. It simply copies the bytes from source to destination.
  }];
  let arguments = (ins 
    Arg<ReussirRefType, "source reference">:$src,
    Arg<ReussirRefType, "destination reference">:$dst
  );
  let assemblyFormat = [{
    $src `to` $dst `:` type($src) `to` type($dst) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir Region Operations
//===----------------------------------------------------------------------===//
class ReussirRegionOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"region."#mnemonic, traits>;

def ReussirRegionRunOp
    : ReussirRegionOp<
          "run", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Execute a region";
  let description = [{
    `reussir.region.run` executes a region. The inner region must accepts one 
    argument, which is of !reussir.region type. The inner region can optionally
    yield a flex value which will be freezed to rigid counterpart after the 
    region execution.

    `reussir.region.run` operation cannot be nested in the same function.
  }];
  let results = (outs Res<Optional<AnyType>, "result of the region">:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    (`->` type($result)^)? ($body^)? attr-dict 
  }];
  let hasVerifier = 1;
}

def ReussirRegionYieldOp
    : ReussirRegionOp<
          "yield", [Terminator,
                    ParentOneOf<["::reussir::ReussirRegionRunOp"]>,
                    ]> {
  let summary = "Yield a value from a region";
  let description = [{
    `reussir.region.yield` yields a value from a region.
  }];
  let arguments = (ins Arg<Optional<AnyType>, "value to yield">:$value);
  let assemblyFormat = [{
    ( $value^ `:` type($value) )? attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirRegionVTableOp 
    : ReussirRegionOp<"vtable", [
        Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Create a vtable for a region";
  let description = [{
    `reussir.region.vtable` creates a vtable for objects managed via flex/rigid
    reference. The vtable contains the drop functions and the layout of the object.
    ```rust
    #[repr(C)]
    pub struct VTable {
      pub drop: Option<unsafe extern "C" fn(*mut u8)>,
      pub scan_count: usize,
      pub scan_offsets: *const usize,
      pub size: usize,
      pub alignment: usize,
    }
    ```
    The size and alignment includes the header of the object.
    ```rust
    #[repr(C)]
    struct Header {
        status: PackedStatus,
        next: *mut Self,
        vtable: *mut VTable,
    }
    ```
    Similarly, all offsets are relative to the header start (not the object start).
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    OptionalAttr<FlatSymbolRefAttr>:$drop
  );
  let assemblyFormat = [{
    $sym_name `{`
      `type` `(` $type `)`
      ( `drop` `(` $drop^ `)` )?
    `}` attr-dict
  }];
}

// The following are intermediate placeholder operations for high-level constructs
// that should be lowered before code generation.
def ReussirRegionCreateOp
    : ReussirRegionOp<"create", []> {
  let summary = "Create a region";
  let description = [{
    `reussir.region.create` creates a region. This is just a alloca operation
    for a pointer type.
  }];
  let results = (outs Res<ReussirRegionType, "created region">:$region);
  let assemblyFormat = [{
    `:` qualified(type($region)) attr-dict
  }];
}

def ReussirRegionCleanupOp
    : ReussirRegionOp<"cleanup", []> {
  let summary = "Cleanup a region";
  let description = [{
    `reussir.region.cleanup` cleans up a region. This is a placeholder for
    runtime function call to cleanup the region.
  }];
  let arguments = (ins Arg<ReussirRegionType, "region to cleanup">:$region);
  let assemblyFormat = [{
    `(` $region `:` type($region) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Scf Placeholder Operations
//===----------------------------------------------------------------------===//

class ReussirScfOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"scf."#mnemonic, traits>;

def ReussirScfYieldOp : ReussirScfOp<"yield", [
  ReturnLike, Terminator,
  ParentOneOf<[
    "::reussir::ReussirNullableDispatchOp", 
    "::reussir::ReussirRecordDispatchOp",
    "::reussir::ReussirArrayWithUniqueViewOp"
  ]>]> {
  let summary = "Yield a value from a scf operation";
  let description = [{
    `reussir.scf.yield` terminates a high-level structured control flow operation.
  }];
  let arguments = (ins Arg<Optional<AnyType>, "value to yield">:$value);
  let assemblyFormat = [{
    ( $value^ `:` type($value) )? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir Closure Operations
//===----------------------------------------------------------------------===//

class ReussirClosureOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"closure."#mnemonic, traits>;


def ReussirClosureCreateOp : ReussirClosureOp<"create", [
  IsolatedFromAbove, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<TokenAcceptorInterface>
]> {
  let summary = "Create a closure";
  let description = [{
    `reussir.closure.create` creates a closure. 
    A closure is always wrapped in an rc pointer.

    The operation takes two form, either inlined or outlined. An optional token
    can be provided in both cases. When provided, the token shall match the layout
    of a RcBox with ClosureHeader and Payload, where the payload is the space for
    all arguments. This allows a closure to be partially applied.

    For the inlined form, the body region is required. The body region must 
    accept a list of arguments, each of which matches one of the closure 
    argument in order. If the closure has a return value, the body region must yield
    a single value, which is the result of the closure. No outlinedVtable shall be supplied.

    ```mlir
    !closure = reussir.closure.create
      token (%token : !reussir.token<...>) {
        ^bb(%arg0 : i64, %arg1 : i64):
          %result = arith.addi %arg0, %arg1 : i64
          reussir.region.yield %result : i64
      }
    ```

    For the outlined form, outlinedVtable is required. No region is allowed.
    Target vtable shall present as a vtable operation within the same module and
    the vtable shall match the signature of the closure.
    
    A closure looks like the following:
    ```c++
    struct Closure {
      size_t refcnt;
      void* vtable;
      size_t cursor;
      // trailing payload
    }
    ```
    When applying a new argument of type T. We first align the cursor to T's alignment
    , store the argument at the (ptr + aligned cursor) address, and then
    bump the cursor by T's size.

    Before each application, we check if the closure has a single reference. If so, 
    we do the operation inplace. Otherwise, we create a new closure using the clone
    function inside the vtable.

    Each evaluation of the closure consumes one reference.

    The clone operation takes both the closure pointer, the cursor, and the new storage. It consists
    of several if statements:

    ```c++
    if (cursor > offsetof(Closure, payload[0])) {
      // clone payload[0]
    } 

    if (cursor > offsetof(Closure, payload[1])) {
      // clone payload[1]
    } 

    if (cursor > offsetof(Closure, payload[2])) {
      // clone payload[2]
    } 

    // ...
    ```

    Similarly, the drop operation consists of if protected drop functions.
  }];
  let arguments = (ins
    Optional<ReussirTokenType>:$token,
    OptionalAttr<FlatSymbolRefAttr>:$vtable
  );
  let regions = (region MaxSizedRegion<1>:$body);
  let results = (outs Res<RcClosureType, "created closure">:$closure);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // empty region while function vtable presents
    bool isOutlined();
    // non-empty region while vtable does not present
    bool isInlined();
    // get the payload box for the closure
    ::reussir::ClosureBoxType getClosureBoxType();
    ::reussir::RcBoxType getRcClosureBoxType();
  }];

  let hasVerifier = 1;
}

def ReussirClosureYieldOp : ReussirClosureOp<"yield", [
  ReturnLike, Terminator,
  ParentOneOf<[
    "::reussir::ReussirClosureCreateOp"
  ]>]> {
  let summary = "Yield from a closure";
  let description = [{
    `reussir.closure.yield` yields from a closure.

    If the closure has a return value, the body region must yield
    a single value, which is the result of the closure. If the closure has no return
    value, the body region must not yield any value.
  }];
  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = [{
    ( $value^ `:` type($value) )? attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureVtableOp : ReussirClosureOp<"vtable", [
  Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create a vtable";
  let description = [{
    `reussir.closure.vtable` creates a vtable for closures. The VTable is in charge of
    record the function pointer and corresponding drop and clone functions.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$func,
    FlatSymbolRefAttr:$drop,
    FlatSymbolRefAttr:$clone
  );

  let assemblyFormat = [{
    $sym_name `{`
      `func` `(` $func `)`
      `drop` `(` $drop `)` 
      `clone` `(` $clone `)` 
    `}` attr-dict
  }];
}

def ReussirClosureApplyOp : ReussirClosureOp<"apply", []> {
  let summary = "Apply a closure";
  let description = [{
    `reussir.closure.apply` applies an argument to a closure.
    Notice that apply operation does not evaluate the closure.
    It is only to supply arguments to the closure.

    The closure must have input arguments in its list and the supplied value type 
    must match the type of the argument.

    The output closure is a new closure with one less argument.

    ```mlir
    %closure = reussir.closure.apply (%arg : i64) to (%closure : !reussir.closure<(i64) -> i64>)
       : !reussir.closure<() -> i64>
    ```
  }];
  let hasVerifier = 1;
  let arguments = (ins
    Arg<AnyType, "argument to apply">:$arg,
    Arg<RcClosureType, "closure to apply">:$closure
  );
  let results = (outs Res<RcClosureType, "applied closure">:$applied);
  let assemblyFormat = [{
    `(` $arg `:` type($arg) `)` `to` `(` $closure `:` qualified(type($closure)) `)`
    `:` qualified(type($applied)) attr-dict
  }];
}

def ReussirClosureEvalOp : ReussirClosureOp<"eval", []> {
  let summary = "Evaluate a closure";
  let description = [{
    `reussir.closure.eval` evaluates a closure.

    This operation takes a fully applied closure and evaluates it.
    The output is the result of the closure if exists.
  }];
  let arguments = (ins Arg<RcClosureType, "closure to evaluate">:$closure);
  let results = (outs Res<Optional<AnyType>, "result of the closure">:$result);
  let assemblyFormat = [{
    `(` $closure `:` qualified(type($closure)) `)` (`:` qualified(type($result))^)? attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureUniqifyOp : ReussirClosureOp<"uniqify", []> {
  let summary = "Make a unique closure from a closure";
  let description = [{
    `reussir.closure.uniqify` makes sure a unique closure from a closure.

    This is a SCF operation that will expand into rc.is_unique operation guarded
    scf if-else statement. If the original value is unique, the operation will
    call the clone operation.
  }];
  let arguments = (ins Arg<RcClosureType, "closure to specialize">:$closure);
  let results = (outs Res<RcClosureType, "specialized closure">:$uniqified);
  let assemblyFormat = [{
    `(` $closure `:` qualified(type($closure)) `)` `:` qualified(type($uniqified)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureCloneOp : ReussirClosureOp<"clone", []> {
  let summary = "Clone a closure";
  let description = [{
    `reussir.closure.clone` clones a closure.

    Since we typically have no way to understand the full closure, the memory layout
    and other configuration related operations are handled inside the vtable clone operation.
    This call just lowers into a function call to the clone operation via vtable.
  }];
  let arguments = (ins Arg<RcClosureType, "closure to clone">:$closure);
  let results = (outs Res<RcClosureType, "cloned closure">:$cloned);
  let assemblyFormat = [{
    `(` $closure `:` qualified(type($closure)) `)` `:` qualified(type($cloned)) 
    attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureWpdTestOp : ReussirClosureOp<"wpd_test", []> {
  let summary = "Assert a whole-program-devirtualization type id of a vtable";
  let description = [{
    `reussir.closure.wpd_test` asserts that `vtable` (a lowered pointer to a
    closure vtable global) carries the whole-program-devirtualization type id
    `id` — see `ClosureWpd.h` and docs/design/closure-wpd.md for the id
    scheme and the closed-world conditions that make the assertion sound.

    The op is emitted by the closure conversion patterns (when the basic-ops
    lowering runs with `closure-wpd`) next to each indirect vtable slot load,
    and survives conversion untouched: the LLVM dialect can express neither
    `!type` metadata nor the metadata-string operand of `llvm.type.test`, so
    the Reussir dialect's LLVM translation interface lowers this op directly
    to `llvm.type.test(%vtable, !"id")` + `llvm.assume` during
    `translateModuleToLLVMIR`.
  }];
  let arguments = (ins
    Arg<AnyType, "lowered pointer to the vtable">:$vtable,
    StrAttr:$id
  );
  let assemblyFormat = [{
    `(` $vtable `:` type($vtable) `)` `id` `(` $id `)` attr-dict
  }];
}

def ReussirClosureInspectPayloadOp : ReussirClosureOp<"inspect_payload", []> {
  let summary = "Inspect closure payload";
  let description = [{
    `reussir.closure.inspect_payload` serves as an low-level intermediate operation
    that achieves the conversion from ClosureBox reference to subfield reference.
  }];
  let arguments = (ins 
    IndexAttr:$payloadIndex,
    Arg<ReussirClosureBoxRefType, "reference to closure box">:$closureBoxRef);
  let results = (outs Res<ReussirRefType, "subfield reference">:$payloadRef);
  let assemblyFormat = [{
    `[` $payloadIndex `]` `(` $closureBoxRef `)` `:` 
    functional-type(operands, results) attr-dict
  }];
}

def ReussirClosureCursorOp : ReussirClosureOp<"cursor", []> {
  let summary = "Extract cursor pointer as a reference to i8";
  let description = [{
    `reussir.closure.cursor` operation extracts the cursor pointer from the
    closure box pointer as an i8 reference.
  }];
  let arguments = (ins 
    IndexAttr:$payloadIndex,
    Arg<ReussirClosureBoxRefType, "reference to closure box">:$closureBoxRef);
  let results = (outs Res<ReussirRefType, "i8 reference">:$cursor);
  let assemblyFormat = [{
   `(` $closureBoxRef `:` qualified(type($closureBoxRef)) `)` `:`
    qualified(type($cursor)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureInstantiateOp : ReussirClosureOp<"instantiate", []> {
  let summary = "Instantiate a token into an Rc wrapped closure box";
  let description = [{
    `reussir.closure.instantiate` operation instantiate a token into a special
    RC-wrapped payload box and setting the refcnt to 1.
  }];
  let arguments = (ins
    Arg<ReussirTokenType, "source token">:$token
  );
  let results = (outs Res<ReussirRcType, "pointer to the payload box">:$closureBoxRc);
  let assemblyFormat = [{
    `(` $token `:` qualified(type($token)) `)` `:` qualified(type($closureBoxRc)) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirClosureTransferOp : ReussirClosureOp<"transfer", []> {
  let summary = "Transfer bytes between closure boxes";
  let description = [{
    `reussir.closure.transfer` copy bytes between two closure boxes.
  }];
  let arguments = (ins 
    Arg<ReussirClosureBoxRefType, "source reference">:$src,
    Arg<ReussirClosureBoxRefType, "destination reference">:$dst
  );
  let assemblyFormat = [{
    $src `to` $dst `:` type($src) `to` type($dst) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reussir Poly FFI Operations
//===----------------------------------------------------------------------===//

def ReussirPolyFFIOp : ReussirOp<"polyffi", []> {
  let summary = "Polymorphic FFI instantiation";
  let description = [{
    `reussir.polyffi` instantiates the FFI format of the type.

    The operation takes a module texture string and a dictionary of parameters.
    The dictionary entries can specify types to be instantiated.

    For example, a dictionary entry can be "[:type:]" : SomeTypeAttr,
    meaning that instantiate the FFI format of the type.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$moduleTexture,
    OptionalAttr<ElementsAttr>:$compiledModule,
    OptionalAttr<DictionaryAttr>:$substitutions
  );

  let assemblyFormat = [{
    oilist (
      `texture` `(` $moduleTexture `)`
    | `compiled` `(` $compiledModule `)`
    | `substitutions` `(` $substitutions `)`
    ) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reussir Str Operations
//===----------------------------------------------------------------------===//

class ReussirStrOp<string mnemonic, list<Trait> traits>
    : ReussirOp<"str."#mnemonic, traits>;

def ReussirStrGlobalOp : ReussirStrOp<"global", [Symbol]> {
  let summary = "Create a global string literal";
  let description = [{
    `reussir.str.global` operation create a string literal in global storage.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$payload);
  let assemblyFormat = [{
    $sym_name `=` $payload attr-dict
  }];
}

def ReussirStrLiteralOp : ReussirStrOp<"literal", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create a global string literal";
  let description = [{
    `reussir.str.literal` creates a reference to a global string literal.
  }];
  let arguments = (ins FlatSymbolRefAttr:$sym_name);
  let results = (outs Res<ReussirStrType, "literal type">:$literal);
  let assemblyFormat = [{
    $sym_name `:` type($literal) attr-dict
  }];
  let hasVerifier = 1;
}

def ReussirStrLenOp : ReussirStrOp<"len", []> {
  let summary = "Get the length of the string";
  let description = [{
    `reussir.str.len` returns the length of the string.
  }];
  let arguments = (ins Arg<ReussirStrType, "String to get length from">:$str);
  let results = (outs Res<Index, "Length of the string">:$len);
  let assemblyFormat = [{
    `(` $str `:` type($str) `)` `:` type($len) attr-dict
  }];
}

def ReussirStrByteAtOp : ReussirStrOp<"byte_at", []> {
  let summary = "Get the character at a specific index";
  let description = [{
    `reussir.str.byte_at` returns the character at a byte offset. Return 0 if the index is out of bounds.
  }];
  let arguments = (ins Arg<ReussirStrType, "String to get character from">:$str, 
    Arg<Index, "Index of character to get">:$index);
  let results = (outs Res<I8, "Character at index">:$charVal);
  let assemblyFormat = [{
    `(` $str `:` type($str) `)` 
    `[` $index `:` type($index) `]` 
    `:` type($charVal) attr-dict
  }];
}

def ReussirStrUnsafeByteAtOp : ReussirStrOp<"unsafe_byte_at", []> {
  let summary = "Get the character at a specific index without bounds check";
  let description = [{
    `reussir.str.unsafe_byte_at` returns the character at a byte offset. 
    Undefined behavior if the index is out of bounds.
  }];
  let arguments = (ins Arg<ReussirStrType, "String to get character from">:$str, 
    Arg<Index, "Index of character to get">:$index);
  let results = (outs Res<I8, "Character at index">:$charVal);
  let assemblyFormat = [{
    `(` $str `:` type($str) `)` 
    `[` $index `:` type($index) `]` 
    `:` type($charVal) attr-dict
  }];
}

def ReussirStrCastOp : ReussirStrOp<"cast", []> {
   let summary = "Cast a global string literal to a local string literal";
   let description = [{
    `reussir.str.cast` casts a global string literal to a local string literal.
   }];
   let arguments = (ins Arg<ReussirStrType, "Global string literal">:$globalStr);
   let results = (outs Res<ReussirStrType, "Local string literal">:$localStr);
   let assemblyFormat = [{
    `(` $globalStr `:` type($globalStr) `)` `:` type($localStr) attr-dict
   }];
   let hasVerifier = 1;
}

def ReussirStrSelectOp : ReussirStrOp<"select", []> {
  let summary = "Select the first matched pattern of the string";
  let description = [{
    `reussir.str.select` selects the first matched pattern of the string.
  }];
  let arguments = (ins Arg<ReussirStrType, "String to select">:$str, 
    Arg<ArrayAttr, "Patterns to match">:$patterns);
  let results = (outs Res<Index, "Selected pattern">:$selected, 
    Res<I1, "Matched found">:$found);
  let assemblyFormat = [{
    `(` $str `)` $patterns `:` functional-type(operands, results) attr-dict
  }];
}

def ReussirStrStartWithOp : ReussirStrOp<"startswith", []> {
   let summary = "Check if the string starts with a prefix";
   let description = [{
    `reussir.str.startswith` checks if the string starts with a prefix.
   }];
   let arguments = (ins Arg<ReussirStrType, "String to check">:$str, Arg<StrAttr, "Prefix to check">:$prefix);
   let results = (outs Res<I1, "Whether the string starts with the prefix">:$startsWith);
   let assemblyFormat = [{
    `(` $str `:` type($str) `)` 
    $prefix 
    `:` type($startsWith) attr-dict
   }];
}

def ReussirStrUnsafeStartWithOp : ReussirStrOp<"unsafe_startswith", []> {
   let summary = "Check if the string starts with a prefix without bounds check";
   let description = [{
    `reussir.str.unsafe_startswith` checks if the string starts with a prefix.
    Undefined behavior if the prefix is longer than the string.
   }];
   let arguments = (ins Arg<ReussirStrType, "String to check">:$str, Arg<StrAttr, "Prefix to check">:$prefix);
   let results = (outs Res<I1, "Whether the string starts with the prefix">:$startsWith);
   let assemblyFormat = [{
    `(` $str `:` type($str) `)` 
    $prefix 
    `:` type($startsWith) attr-dict
   }];
}

def ReussirStrSliceOp : ReussirStrOp<"slice", []> {
  let summary = "Slice the string from a byte offset";
  let description = [{
    `reussir.str.slice` returns a substring starting from the given offset.
    If offset is greater than length, returns an empty string.
  }];
  let arguments = (ins Arg<ReussirStrType, "String to slice">:$str, Arg<Index, "Byte offset">:$offset);
  let results = (outs Res<ReussirStrType, "Sliced string">:$result);
  let assemblyFormat = [{
    `(` $str `:` type($str) `)` `[` $offset `]` `:` type($result) attr-dict
  }];
}

def ReussirTrampolineOp : ReussirOp<"trampoline", [
  Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Trampoline for FFI calls";
  let description = [{
    `reussir.trampoline` describes an FFI boundary shim. The `direction`
    attribute distinguishes between:

    - `export`: expose a Reussir-internal function through a C-facing boundary.
    - `import`: provide a Reussir symbol that forwards to a C-facing target.

    The backend classifies a trampoline as trivial when it accepts fewer than
    four integral or pointer arguments and returns at most one integral or
    pointer result. Trivial trampolines forward directly with best-effort
    minimal overhead.

    Nontrivial trampolines use a backend-defined explicit boundary:

    - nontrivial returns use a leading return pointer parameter;
    - nontrivial argument lists are packed into a literal LLVM struct passed by
      pointer in source order.

    Currently only the `"C"` ABI is supported.
  }];
  let arguments = (ins 
    DefaultValuedAttr<TrampolineDirection,
                      "::reussir::TrampolineDirection::Export">:$direction,
    Arg<FlatSymbolRefAttr, "Wrapped target function">:$target,
    Arg<StrAttr, "ABI Name">:$abi_name,
    Arg<SymbolNameAttr, "Trampoline symbol name">:$sym_name
  );
  let assemblyFormat = [{
    $direction $abi_name $sym_name `=` $target attr-dict
  }];
}
#endif // REUSSIR_IR_REUSSIROPS_TD
