{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | JSON serialization for the surface syntax.

This module provides 'ToJSON'/'FromJSON' instances for every type that makes
up the surface syntax (the direct output of the parser). Together with the
serialization modules in @reussir-core@ this allows any step of the frontend
pipeline to be suspended to a JSON document and recovered later, which is the
foundation for testing the frontend against other implementations (e.g. the
planned Rust frontend).

Encoding conventions (shared by all serialization modules):

* Sum types use aeson's default tagged-object encoding:
  @{"tag": "Ctor", "contents": ...}@, with record constructors inlining
  their fields next to the @"tag"@ key.
* Sum types whose constructors are all nullary are encoded as plain strings.
* 'Identifier' is encoded as a plain JSON string.
* Strict boxed vectors are encoded as JSON arrays (orphans provided here
  because aeson does not yet ship instances for "Data.Vector.Strict").
-}
module Reussir.Parser.Serialization () where

import Data.Aeson (
    FromJSON (..),
    FromJSONKey (..),
    FromJSONKeyFunction (..),
    ToJSON (..),
    ToJSONKey (..),
    defaultOptions,
 )
import Data.Aeson.TH (deriveJSON)
import Data.Aeson.Types (toJSONKeyText)

import Data.Vector.Strict qualified as VS

import Reussir.Parser.Types.Capability (Capability)
import Reussir.Parser.Types.Expr (
    Access,
    BinaryOp,
    Constant,
    CtorCall,
    Expr,
    FuncCall,
    LambdaExpr,
    Pattern,
    PatternCtorArg,
    PatternKind,
    UnaryOp,
 )
import Reussir.Parser.Types.Lexer (Identifier (..), Path, WithSpan)
import Reussir.Parser.Types.Stmt (
    Function,
    Record,
    RecordFields,
    RecordKind,
    Stmt,
    Visibility,
 )
import Reussir.Parser.Types.Type (FloatingPointType, IntegralType, Type)

instance ToJSON Identifier where
    toJSON = toJSON . unIdentifier
    toEncoding = toEncoding . unIdentifier

instance FromJSON Identifier where
    parseJSON = fmap Identifier . parseJSON

instance ToJSONKey Identifier where
    toJSONKey = toJSONKeyText unIdentifier

instance FromJSONKey Identifier where
    fromJSONKey = FromJSONKeyText Identifier

instance (ToJSON a) => ToJSON (VS.Vector a) where
    toJSON = toJSON . VS.toList
    toEncoding = toEncoding . VS.toList

instance (FromJSON a) => FromJSON (VS.Vector a) where
    parseJSON = fmap VS.fromList . parseJSON

$(deriveJSON defaultOptions ''Path)
$(deriveJSON defaultOptions ''WithSpan)
$(deriveJSON defaultOptions ''Capability)
$(deriveJSON defaultOptions ''IntegralType)
$(deriveJSON defaultOptions ''FloatingPointType)
$(deriveJSON defaultOptions ''Type)

-- The expression types are mutually recursive, so they are derived in a
-- single declaration group.
$( concat
    <$> traverse
        (deriveJSON defaultOptions)
        [ ''Constant
        , ''BinaryOp
        , ''UnaryOp
        , ''Access
        , ''PatternCtorArg
        , ''PatternKind
        , ''Pattern
        , ''CtorCall
        , ''FuncCall
        , ''LambdaExpr
        , ''Expr
        ]
 )

$( concat
    <$> traverse
        (deriveJSON defaultOptions)
        [ ''Visibility
        , ''RecordKind
        , ''RecordFields
        , ''Record
        , ''Function
        , ''Stmt
        ]
 )
