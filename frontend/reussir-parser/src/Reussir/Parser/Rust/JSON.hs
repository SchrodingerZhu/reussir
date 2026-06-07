{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Reussir.Parser.Rust.JSON (
    RustParseResponse (..),
    decodeExprResponseText,
    decodeProgramResponseText,
    decodeStmtResponseText,
    decodeTypeResponseText,
) where

import Control.Applicative ((<|>))
import Data.Aeson
import Data.Aeson.Types (Parser)
import Data.Int (Int16, Int64)
import Data.Scientific (Scientific)

import Data.Aeson.Key qualified as Key
import Data.Aeson.KeyMap qualified as KeyMap
import Data.Text qualified as T
import Data.Text.Encoding qualified as TE
import Data.Vector qualified as V
import Data.Vector.Strict qualified as SV

import Reussir.Parser.Prog (Prog)
import Reussir.Parser.Types.Capability (Capability (..))
import Reussir.Parser.Types.Expr
import Reussir.Parser.Types.Lexer (Identifier (..), Path (..), WithSpan (..))
import Reussir.Parser.Types.Stmt
import Reussir.Parser.Types.Type

import Reussir.Parser.Types.Expr qualified as Expr
import Reussir.Parser.Types.Stmt qualified as Stmt

-- | JSON response envelope returned by the Rust syntax FFI.
data RustParseResponse a
    = RustParseOk a
    | RustParseError T.Text
    deriving (Eq, Show)

decodeProgramResponseText :: T.Text -> Either String (RustParseResponse Prog)
decodeProgramResponseText = decodeResponseText

decodeStmtResponseText :: T.Text -> Either String (RustParseResponse Stmt)
decodeStmtResponseText = decodeResponseText

decodeExprResponseText :: T.Text -> Either String (RustParseResponse Expr)
decodeExprResponseText = decodeResponseText

decodeTypeResponseText :: T.Text -> Either String (RustParseResponse Type)
decodeTypeResponseText = decodeResponseText

decodeResponseText ::
    (FromJSON a) => T.Text -> Either String (RustParseResponse a)
decodeResponseText = eitherDecodeStrict' . TE.encodeUtf8

instance (FromJSON a) => FromJSON (RustParseResponse a) where
    parseJSON = withObject "RustParseResponse" $ \obj -> do
        ok <- obj .: "ok"
        if ok
            then RustParseOk <$> obj .: "value"
            else RustParseError <$> obj .: "diagnostic"

instance (FromJSON a) => FromJSON (SV.Vector a) where
    parseJSON value = SV.fromList <$> parseJSON value

instance FromJSON Identifier where
    parseJSON = withText "Identifier" (pure . Identifier)

instance FromJSON Path where
    parseJSON = withObject "Path" $ \obj ->
        Path
            <$> obj .: "basename"
            <*> obj .: "segments"

instance (FromJSON a) => FromJSON (WithSpan a) where
    parseJSON = withObject "WithSpan" $ \obj -> do
        value <- obj .: "value"
        spanObj <- obj .: "span"
        start <- spanObj .: "start"
        end <- spanObj .: "end"
        pure $ WithSpan value (fromIntegral @Int64 start) (fromIntegral @Int64 end)

instance FromJSON Capability where
    parseJSON = withText "Capability" $ \case
        "shared" -> pure Shared
        "value" -> pure Value
        "flex" -> pure Flex
        "rigid" -> pure Rigid
        "field" -> pure Field
        "regional" -> pure Regional
        other -> fail $ "unknown capability: " <> T.unpack other

instance FromJSON IntegralType where
    parseJSON = parseTagged "IntegralType" $ \case
        ("signed", value) -> Signed . fromIntegral @Int16 <$> parseJSON value
        ("unsigned", value) -> Unsigned . fromIntegral @Int16 <$> parseJSON value
        (tag, _) -> fail $ "unknown integral type: " <> T.unpack tag

instance FromJSON FloatingPointType where
    parseJSON value = parseUnit <|> parseTagged "FloatingPointType" parseTaggedFP value
      where
        parseUnit =
            withText
                "FloatingPointType"
                ( \case
                    "b_float16" -> pure BFloat16
                    "bfloat16" -> pure BFloat16
                    "float8" -> pure Float8
                    other -> fail $ "unknown floating point type: " <> T.unpack other
                )
                value
        parseTaggedFP = \case
            ("ieee", bits) -> IEEEFloat . fromIntegral @Int16 <$> parseJSON bits
            (tag, _) -> fail $ "unknown floating point type: " <> T.unpack tag

instance FromJSON Type where
    parseJSON value = parseUnit value <|> parseTagged "Type" parseTaggedType value
      where
        parseUnit = withText "Type" $ \case
            "bool" -> pure TypeBool
            "str" -> pure TypeStr
            "unit" -> pure TypeUnit
            "bottom" -> pure TypeBottom
            other -> fail $ "unknown type: " <> T.unpack other
        parseTaggedType = \case
            ("expr", v) ->
                withObject "TypeExpr" (\obj -> TypeExpr <$> obj .: "path" <*> obj .: "args") v
            ("integral", v) -> TypeIntegral <$> parseJSON v
            ("float", v) -> TypeFP <$> parseJSON v
            ("arrow", v) ->
                withObject "TypeArrow" (\obj -> TypeArrow <$> obj .: "args" <*> obj .: "ret") v
            ("spanned", v) -> TypeSpanned <$> parseJSON v
            (tag, _) -> fail $ "unknown type tag: " <> T.unpack tag

instance FromJSON Constant where
    parseJSON = parseTagged "Constant" $ \case
        ("int", value) -> ConstInt <$> parseJSON value
        ("double", value) -> ConstDouble <$> parseScientificText value
        ("string", value) -> ConstString <$> parseJSON value
        ("bool", value) -> ConstBool <$> parseJSON value
        (tag, _) -> fail $ "unknown constant: " <> T.unpack tag

instance FromJSON BinaryOp where
    parseJSON = withText "BinaryOp" $ \case
        "add" -> pure Add
        "sub" -> pure Sub
        "mul" -> pure Mul
        "div" -> pure Div
        "mod" -> pure Mod
        "lt" -> pure Lt
        "gt" -> pure Gt
        "lte" -> pure Lte
        "gte" -> pure Gte
        "equ" -> pure Equ
        "neq" -> pure Neq
        "and" -> pure And
        "or" -> pure Or
        other -> fail $ "unknown binary operator: " <> T.unpack other

instance FromJSON UnaryOp where
    parseJSON = withText "UnaryOp" $ \case
        "negate" -> pure Negate
        "not" -> pure Not
        other -> fail $ "unknown unary operator: " <> T.unpack other

instance FromJSON Access where
    parseJSON = parseTagged "Access" $ \case
        ("named", value) -> Expr.Named <$> parseJSON value
        ("unnamed", value) -> Expr.Unnamed <$> parseJSON value
        (tag, _) -> fail $ "unknown access: " <> T.unpack tag

instance FromJSON Pattern where
    parseJSON = withObject "Pattern" $ \obj ->
        Pattern <$> obj .: "kind" <*> obj .: "guard"

instance FromJSON PatternKind where
    parseJSON value = parseUnit value <|> parseTagged "PatternKind" parseTaggedPattern value
      where
        parseUnit = withText "PatternKind" $ \case
            "wildcard" -> pure WildcardPat
            other -> fail $ "unknown pattern kind: " <> T.unpack other
        parseTaggedPattern = \case
            ("bind", v) -> BindPat <$> parseJSON v
            ("ctor", v) ->
                withObject
                    "CtorPattern"
                    ( \obj ->
                        CtorPat
                            <$> obj .: "path"
                            <*> (V.fromList <$> obj .: "args")
                            <*> obj .: "has_ellipsis"
                            <*> obj .: "is_named"
                    )
                    v
            ("const", v) -> ConstPat <$> parseJSON v
            (tag, _) -> fail $ "unknown pattern kind: " <> T.unpack tag

instance FromJSON PatternCtorArg where
    parseJSON = withObject "PatternCtorArg" $ \obj ->
        PatternCtorArg <$> obj .: "field" <*> obj .: "kind"

instance FromJSON CtorCall where
    parseJSON = withObject "CtorCall" $ \obj ->
        CtorCall <$> obj .: "name" <*> obj .: "ty_args" <*> obj .: "args"

instance FromJSON FuncCall where
    parseJSON = withObject "FuncCall" $ \obj ->
        FuncCall <$> obj .: "name" <*> obj .: "ty_args" <*> obj .: "args"

instance FromJSON LambdaExpr where
    parseJSON = withObject "LambdaExpr" $ \obj ->
        LambdaExpr <$> obj .: "args" <*> obj .: "body" <*> obj .: "ret_ty"

instance FromJSON Expr where
    parseJSON = parseTagged "Expr" $ \case
        ("const", v) -> ConstExpr <$> parseJSON v
        ("bin_op", v) ->
            withObject
                "BinOpExpr"
                (\obj -> BinOpExpr <$> obj .: "op" <*> obj .: "lhs" <*> obj .: "rhs")
                v
        ("unary_op", v) ->
            withObject
                "UnaryOpExpr"
                (\obj -> UnaryOpExpr <$> obj .: "op" <*> obj .: "expr")
                v
        ("if", v) ->
            withObject
                "If"
                (\obj -> If <$> obj .: "cond" <*> obj .: "then_expr" <*> obj .: "else_expr")
                v
        ("cast", v) -> withObject "Cast" (\obj -> Cast <$> obj .: "ty" <*> obj .: "expr") v
        ("let", v) ->
            withObject
                "Let"
                (\obj -> Let <$> obj .: "name" <*> obj .: "ty" <*> obj .: "value")
                v
        ("seq", v) -> ExprSeq <$> parseJSON v
        ("lambda", v) -> Lambda <$> parseJSON v
        ("match", v) ->
            withObject
                "Match"
                (\obj -> Match <$> obj .: "scrutinee" <*> (SV.fromList <$> obj .: "cases"))
                v
        ("var", v) -> Var <$> parseJSON v
        ("func_call", v) -> FuncCallExpr <$> parseJSON v
        ("regional", v) -> RegionalExpr <$> parseJSON v
        ("ctor_call", v) -> CtorCallExpr <$> parseJSON v
        ("call", v) ->
            withObject "CallExpr" (\obj -> CallExpr <$> obj .: "callee" <*> obj .: "args") v
        ("access_chain", v) ->
            withObject
                "AccessChain"
                (\obj -> AccessChain <$> obj .: "base" <*> (SV.fromList <$> obj .: "accesses"))
                v
        ("spanned", v) -> SpannedExpr <$> parseJSON v
        ("assign", v) ->
            withObject
                "Assign"
                (\obj -> Assign <$> obj .: "base" <*> obj .: "access" <*> obj .: "value")
                v
        (tag, _) -> fail $ "unknown expression tag: " <> T.unpack tag

instance FromJSON Visibility where
    parseJSON = withText "Visibility" $ \case
        "public" -> pure Public
        "private" -> pure Private
        other -> fail $ "unknown visibility: " <> T.unpack other

instance FromJSON RecordFields where
    parseJSON = parseTagged "RecordFields" $ \case
        ("named", value) -> Stmt.Named . SV.fromList <$> parseJSON value
        ("unnamed", value) -> Stmt.Unnamed . SV.fromList <$> parseJSON value
        ("variants", value) -> Stmt.Variants . SV.fromList <$> parseJSON value
        (tag, _) -> fail $ "unknown record fields tag: " <> T.unpack tag

instance FromJSON RecordKind where
    parseJSON = withText "RecordKind" $ \case
        "struct" -> pure StructKind
        "enum" -> pure EnumKind
        other -> fail $ "unknown record kind: " <> T.unpack other

instance FromJSON Record where
    parseJSON = withObject "Record" $ \obj ->
        Record
            <$> obj .: "name"
            <*> obj .: "ty_params"
            <*> obj .: "fields"
            <*> obj .: "kind"
            <*> obj .: "visibility"
            <*> obj .: "default_cap"

instance FromJSON Function where
    parseJSON = withObject "Function" $ \obj ->
        Function
            <$> obj .: "visibility"
            <*> obj .: "name"
            <*> obj .: "generics"
            <*> obj .: "params"
            <*> obj .: "return_type"
            <*> obj .: "is_regional"
            <*> obj .: "body"

instance FromJSON Stmt where
    parseJSON = parseTagged "Stmt" $ \case
        ("function", v) -> FunctionStmt <$> parseJSON v
        ("record", v) -> RecordStmt <$> parseJSON v
        ("extern_trampoline", v) ->
            withObject
                "ExternTrampolineStmt"
                ( \obj ->
                    ExternTrampolineStmt
                        <$> obj .: "name"
                        <*> obj .: "abi"
                        <*> obj .: "func"
                        <*> obj .: "func_ty_args"
                )
                v
        ("mod", v) ->
            withObject
                "ModStmt"
                (\obj -> ModStmt <$> obj .: "visibility" <*> obj .: "name")
                v
        ("spanned", v) -> SpannedStmt <$> parseJSON v
        (tag, _) -> fail $ "unknown statement tag: " <> T.unpack tag

parseTagged :: String -> ((T.Text, Value) -> Parser a) -> Value -> Parser a
parseTagged label parseCase = withObject label $ \obj ->
    case KeyMap.toList obj of
        [(key, value)] -> parseCase (Key.toText key, value)
        [] -> fail $ label <> " must not be an empty object"
        _ -> fail $ label <> " must have exactly one tag"

parseScientificText :: Value -> Parser Scientific
parseScientificText value =
    parseJSON value
        <|> (either fail pure . readScientificText =<< parseJSON value)
  where
    readScientificText txt =
        case reads (T.unpack txt) of
            [(n, "")] -> Right n
            _ -> Left $ "invalid scientific literal: " <> T.unpack txt
