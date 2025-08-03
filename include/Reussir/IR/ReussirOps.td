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
#endif // REUSSIR_IR_REUSSIROPS_TD
