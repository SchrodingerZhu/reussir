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

include "mlir/IR/BuiltinAttributes.td"
include "Reussir/IR/ReussirDialect.td"
include "Reussir/IR/ReussirAttrs.td"
include "Reussir/IR/ReussirTypes.td"

class ReussirOp<string mnemonic, list<Trait> traits>
    : Op<ReussirDialect, mnemonic, traits>;

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

def ReussirTokenReallocOp : ReussirTokenOp<"realloc", []> {
  let summary = "Reallocate a token";
  let description = [{
    `reussir.token.realloc` converts a possibly nullable token to a token by checking
    the nullability and reallocate the memory on need. During the process, if the original
    layout and the new layout are different, the memory will be reallocated, such reallocation
    may happen in place depending on the allocator and the new size.

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
    `(` $nullable `:` type($nullable) `)` ( `->` type($value)^)? `{` 
      `nonnull` `->` $nonNullRegion `\n`
      `null` `->` $nullRegion `\n`
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

def ReussirRcDecOp : ReussirRcOp<"dec", []> {
  let summary = "Decrement reference count";
  let description = [{
    `reussir.rc.dec` decrements the reference count of a RC pointer.
    If the reference count reaches zero, the memory will be freed.
    ```mlir
    %nullable = reussir.rc.dec (%rc_ptr : !reussir.rc<i64>) -> 
      !reussir.nullable<!reussir.token<align: 8, size: 8>>
    ```
  }];

  let arguments = (ins Arg<ReussirRcType, "Reference to decrement">:$rcPtr);
  let results =
      (outs Res<ReussirNullableTokenType, "Nullable tokeno">:$nullableToken);

  let assemblyFormat = [{
    `(` $rcPtr `:` qualified(type($rcPtr)) `)` `:` type($nullableToken) attr-dict
  }];

  let hasVerifier = 1;
}

def ReussirRcCreateOp : ReussirRcOp<"create", []> {
  let summary = "Create a RC pointer";
  let description = [{
    `reussir.rc.create` creates a RC pointer from a value.
    The memory resource must be provided by a token. It addtionally accepts
    a region argument.

    RC pointer created solely from token has shared capability, while RC pointer 
    created with region has flex capability.

    The token layout must meet the underlying Rc box layout.

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

  let arguments = (ins Arg<AnyType, "value">:$value, ReussirTokenType:$token,
      Optional<ReussirRegionType>:$region);
  let results = (outs Res<ReussirRcType, "Created RC pointer">:$rcPtr);

  let assemblyFormat = [{
    `value` `(` $value `:` type($value) `)` 
    `token` `(` $token `:` type($token) `)`
    ( `region` `(` $region^ `:` type($region) `)` )? 
    `:` qualified(type($rcPtr)) attr-dict
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
    by the borrowed reference. The borrow will not change the reference count of 
    the RC pointer.
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
    `(` $fields `:` type($fields) `)` `:` qualified(type($compound)) attr-dict
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

def ReussirRefLoadOp : ReussirReferenceOp<"load", []> {
  let summary = "Load a reference";
  let description = [{
    `reussir.ref.load` loads a reference.
  }];

  let arguments = (ins Arg<ReussirRefType, "reference to load">:$ref);
  let results = (outs Res<AnyType, "loaded value">:$value);

  let assemblyFormat = [{
    `(` $ref `:` qualified(type($ref)) `)` `:` type($value) attr-dict
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
  }];
  let arguments = (
    ins Arg<ReussirRefType, "reference to drop">:$ref,
    UnitAttr:$outlined
  );

  let assemblyFormat = [{
    (`outlined` $outlined^)?
    `(` $ref `:` qualified(type($ref)) `)` 
    attr-dict
  }];
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
          "yield", [ReturnLike, Terminator,
                    ParentOneOf<["::reussir::ReussirRegionRunOp"]>]> {
  let summary = "Yield a value from a region";
  let description = [{
    `reussir.region.yield` yields a value from a region.
  }];
  let arguments = (ins Arg<Optional<ReussirRcType>, "value to yield">:$value);
  let assemblyFormat = [{
    ( $value^ `:` type($value) )? attr-dict
  }];
  let hasVerifier = 1;
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
    "::reussir::ReussirRecordDispatchOp"
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
  IsolatedFromAbove, DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Create a closure";
  let description = [{
    `reussir.closure.create` creates a closure. 
    A closure is always wrapped in an rc pointer.

    The operation takes two form, either inlined or outlined. Token is required
    in both cases. The token shall match the layout of the a RcBox with 
    ClosureHeader and Payload, where the payload is the space for all arguments.
    This allows a closure to be partially applied.

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
    (TODO: add vtable operation and verification).
  }];
  let arguments = (ins
    ReussirTokenType:$token,
    OptionalAttr<FlatSymbolRefAttr>:$vtable
  );
  let regions = (region MaxSizedRegion<1>:$body);
  let results = (outs Res<ReussirClosureType, "created closure">:$closure);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // empty region while function vtable presents
    bool isOutlined();
    // non-empty region while vtable does not present
    bool isInlined();
    // get the payload box for the closure
    ::reussir::ClosureBoxType getClosureBoxType();

    // Return the target function is the closure is trivially forwarding.
    // Return null otherwise.
    // A closure is trivially forwarding if the only operation it does is to
    // call a function with exactly the same argument order and return the result.
    ::mlir::FlatSymbolRefAttr getTrivialForwardingTarget();
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

    Drop and clone functions are be null if closure payload is trivially copyable.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$func,
    OptionalAttr<FlatSymbolRefAttr>:$drop,
    OptionalAttr<FlatSymbolRefAttr>:$clone,
    TypeAttrOf<ReussirClosureType>:$closure
  );

  let assemblyFormat = [{
    $sym_name `{`
      `func` `(` $func `)`
      `closure` `(` $closure `)`
      oilist
      ( `drop` `(` $drop `)` 
      | `clone` `(` $clone `)` 
      ) 
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
    Arg<ReussirClosureType, "closure to apply">:$closure
  );
  let results = (outs Res<ReussirClosureType, "applied closure">:$applied);
  let assemblyFormat = [{
    `(` $arg `:` type($arg) `)` `to` `(` $closure `:` type($closure) `)`
    `:` type($applied) attr-dict
  }];
}

def ReussirClosureEvalOp : ReussirClosureOp<"eval", []> {
  let summary = "Evaluate a closure";
  let description = [{
    `reussir.closure.eval` evaluates a closure.

    This operation takes a fully applied closure and evaluates it.
    The output is the result of the closure if exists.
  }];
  let arguments = (ins Arg<ReussirClosureType, "closure to evaluate">:$closure);
  let results = (outs Res<Optional<AnyType>, "result of the closure">:$result);
  let assemblyFormat = [{
    `(` $closure `:` type($closure) `)` (`:` type($result)^)? attr-dict
  }];
  let hasVerifier = 1;
}

#endif // REUSSIR_IR_REUSSIROPS_TD
