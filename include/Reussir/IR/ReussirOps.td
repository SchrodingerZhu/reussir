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
  : ReussirOp<"token." # mnemonic, traits>;

def ReussirTokenAllocOp : ReussirTokenOp<"alloc", []> {
  let summary = "Allocate memory token";
  let description = [{
    `reussir.token.alloc` allocates a memory token.
    ```mlir
    reussir.token.alloc : !reussir.token<align: 8, size: 16>
    ```
  }];

  let results = (outs Res<
    ReussirTokenType,
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

  let arguments = (ins Res<
    AnyTypeOf<[ReussirTokenType, ReussirNullableTokenType]>,
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

  let results = (outs 
    Res<ReussirRefType, "reinterpreted reference">:$reinterpreted);
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
  let results = (outs 
    Res<ReussirTokenType, "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments = (ins 
    Arg<ReussirNullableTokenType, "input token", [MemFree<DefaultResource>]>:$nullableToken);
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

  let arguments = (ins Res<
    AnyTypeOf<[ReussirTokenType, ReussirNullableTokenType]>,
    "Target Token to Realloc", [MemFree<DefaultResource>]>:$token
  );

  let results = (outs
    Res<ReussirTokenType, "Reallocated Memory Token", 
      [MemAlloc<DefaultResource>]>:$realloced);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` `:` type($realloced) attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// Reussir Nullable Operations
//===----------------------------------------------------------------------===//
class ReussirNullableOp<string mnemonic, list<Trait> traits>
  : ReussirOp<"nullable." # mnemonic, traits>;

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
//===----------------------------------------------------------------------===//
// Reussir RC Operations
//===----------------------------------------------------------------------===//
class ReussirRcOp<string mnemonic, list<Trait> traits>
  : ReussirOp<"rc." # mnemonic, traits>;

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
  let results = (outs Res<ReussirNullableTokenType, "Nullable tokeno">:$nullableToken);

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
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 8>
    %val = arith.constant 42 : i64
    %rc_ptr = reussir.rc.create value(%val : i64) 
      token(%token : !reussir.token<align: 8, size: 8>)
      : !reussir.rc<i64>
    ```

    RC with flex capability:
    ```mlir
    func.func @region_funcion(%region: !reussir.region) {
      %token = reussir.token.alloc : !reussir.token<align: 8, size: 8>
      %val = arith.constant 42 : i64
      %rc_ptr = reussir.rc.create 
        value (%val : i64) 
        token (%token : !reussir.token<align: 8, size: 8>) 
        region (%region : !reussir.region)
        : !reussir.rc<i64, flex>
      func.return
    }
    ```
  }];

  let arguments = (ins Arg<AnyType, "value">:$value,
                    ReussirTokenType:$token,
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
// Reussir Spill Operation
//===----------------------------------------------------------------------===//
def ReussirSpillOp : ReussirOp<"spill", []> {
  let summary = "Spill a value to stack allocation";
  let description = [{
    `reussir.spill` spills a value to a stack allocation, which can be
    used to pass values between regions or functions. This also gives user
    the power to project inner fields from a compound value. 
    
    The result reference must have unspecified capability.
  }];
  let arguments = (ins Arg<AnyType, "value to spill">:$value);
  let results = (outs Res<ReussirRefType, "spilled reference">:$spilled);
  let assemblyFormat = [{
    `(` $value `:` type($value) `)` `:` qualified(type($spilled)) attr-dict
  }];
  let hasVerifier = 1;
}
#endif // REUSSIR_IR_REUSSIROPS_TD
