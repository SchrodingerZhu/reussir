module Reussir.Core.Data.Full.Function where

import Data.Int (Int64)
import Reussir.Codegen.Context.Symbol (Symbol)
import Reussir.Parser.Types.Lexer (Identifier, Path)
import Reussir.Parser.Types.Stmt (Visibility)

import Data.HashTable.IO qualified as H

import Reussir.Core.Data.Full.Expr qualified as Full
import Reussir.Core.Data.Full.Type qualified as Full

data Function = Function
    { funcVisibility :: Visibility
    , funcName :: Symbol
    , funcRawPath :: Path
    , funcInstantiatedTyArgs :: [Full.Type]
    , funcParams :: [(Identifier, Full.Type)]
    , funcReturnType :: Full.Type
    , funcIsRegional :: Bool
    , funcBody :: Maybe Full.Expr
    , funcSpan :: Maybe (Int64, Int64)
    }
    deriving (Show, Eq)

type FunctionTable = H.CuckooHashTable Symbol Function
