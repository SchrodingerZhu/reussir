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
/// This file defines the interfaces for the Reussir dialect.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_IR_REUSSIRINTERFACES_TD
#define REUSSIR_IR_REUSSIRINTERFACES_TD

include "mlir/IR/OpBase.td"
include "Reussir/IR/ReussirTypes.td"

//===----------------------------------------------------------------------===//
// TokenAcceptor Interface
//===----------------------------------------------------------------------===//

def TokenAcceptorInterface : OpInterface<"TokenAcceptor"> {
  let description = [{
    Interface for operations that can accept and manage tokens.
    Operations implementing this interface support token assignment, retrieval,
    and removal operations.
  }];

  let cppNamespace = "::reussir";

  let methods =
      [InterfaceMethod<
           /*desc=*/"Get the expected token type for this operation.",
           /*retTy=*/"::reussir::TokenType",
           /*methodName=*/"getTokenType",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">,
       InterfaceMethod<
           /*desc=*/"Check whether the operation has been assigned a token.",
           /*retTy=*/"bool",
           /*methodName=*/"hasToken",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        return $_op.getToken() != nullptr;
      }]>,
       InterfaceMethod<
           /*desc=*/"Assign a token value to this operation.",
           /*retTy=*/"void",
           /*methodName=*/"assignToken",
           /*args=*/(ins "::mlir::Value":$token),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        $_op.getTokenMutable().assign(token);
      }]>,
       InterfaceMethod<
           /*desc=*/"Remove the token from this operation and return it.",
           /*retTy=*/"::mlir::Value",
           /*methodName=*/"removeToken",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/[{
        ::mlir::Value token = $_op.getToken();
        $_op.getTokenMutable().clear();
        return token;
      }]>,
       InterfaceMethod<
           /*desc=*/"Get the current token value (may be null).",
           /*retTy=*/"::mlir::Value",
           /*methodName=*/"getToken",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">,
       InterfaceMethod<
           /*desc=*/"Get mutable access to the token operand.",
           /*retTy=*/"::mlir::MutableOperandRange",
           /*methodName=*/"getTokenMutable",
           /*args=*/(ins),
           /*methodBody=*/"",
           /*defaultImplementation=*/"">];
}

//===----------------------------------------------------------------------===//
// TokenProducer Interface
//===----------------------------------------------------------------------===//

def TokenProducerInterface : OpInterface<"TokenProducer"> {
  let description = [{
    Interface for operations that can generate tokens.
    Operations implementing this interface support token production, including
    nullable token production, and provide methods to query and create the
    produced token values.
  }];

  let cppNamespace = "::reussir";

  let methods = [InterfaceMethod<
                     /*desc=*/"Check if the operation should produce a token.",
                     /*retTy=*/"bool",
                     /*methodName=*/"shouldProduceToken",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/"Check if the operation's token production can "
                              "be nullable.",
                     /*retTy=*/"bool",
                     /*methodName=*/"isNullable",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/"Get the token type if shouldProduceToken, "
                              "otherwise return null.",
                     /*retTy=*/"::reussir::TokenType",
                     /*methodName=*/"getTokenType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/"Get the actual produced type (e.g., "
                              "Nullable<TokenType> if nullable).",
                     /*retTy=*/"::mlir::Type",
                     /*methodName=*/"getProducedType",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/[{
        if (!$_op.shouldProduceToken())
          return nullptr;
        ::reussir::TokenType tokenType = $_op.getTokenType();
        if ($_op.isNullable())
          return ::reussir::NullableType::get($_op.getContext(), tokenType);
        return tokenType;
      }]>,
                 InterfaceMethod<
                     /*desc=*/"Get the value being produced.",
                     /*retTy=*/"::mlir::Value",
                     /*methodName=*/"getProducedValue",
                     /*args=*/(ins),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">,
                 InterfaceMethod<
                     /*desc=*/"Create and return a new produced value if not "
                              "yet produced.",
                     /*retTy=*/"::mlir::LogicalResult",
                     /*methodName=*/"replaceWithProduced",
                     /*args=*/(ins "::mlir::PatternRewriter &":$builder),
                     /*methodBody=*/"",
                     /*defaultImplementation=*/"">];
}

#endif // REUSSIR_IR_REUSSIRINTERFACES_TD
