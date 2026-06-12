{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | JSON serialization for the full syntax and the global full-elaboration
context.

Unlike the semi phase, almost all full-phase data ('Full.Function',
'Full.Record', 'Full.Expr', errors) is already pure, so it receives direct
'ToJSON'/'FromJSON' instances. Only the context's hash tables need to be
mirrored: 'snapshotFullContext' captures a 'FullContext' into a pure
'FullContextSnapshot' and 'restoreFullContext' rebuilds a functional context
from it. Semi records referenced by the full context are snapshotted via
"Reussir.Core.Serialization.Semi".

All hash table dumps are sorted by key (symbols by their text) so snapshots
of structurally equal contexts compare equal and serialize identically.
-}
module Reussir.Core.Serialization.Full (
    FullContextSnapshot (..),
    snapshotFullContext,
    restoreFullContext,
) where

import Data.Aeson (FromJSON (..), ToJSON (..), defaultOptions)
import Data.Aeson.TH (deriveJSON)
import Data.List (sortOn)
import Effectful (Eff, IOE, liftIO, (:>))
import Effectful.Prim (Prim)
import Reussir.Codegen.Context.Symbol (Symbol, symbolText)
import Reussir.Parser.Serialization ()
import Reussir.Parser.Types.Lexer (Path)

import Data.HashMap.Strict qualified as HashMap
import Data.HashTable.IO qualified as H
import Data.Text qualified as T

import Reussir.Core.Data.Full.Context (FullContext (..))
import Reussir.Core.Data.Full.Error (Error, ErrorKind)
import Reussir.Core.Data.Full.Expr (
    ClosureExpr,
    DTSwitchCases,
    DecisionTree,
    Expr,
    ExprKind,
    PatternVarRef (..),
 )
import Reussir.Core.Data.Full.Function (Function)
import Reussir.Core.Data.Full.Record (Record, RecordFields, RecordKind)
import Reussir.Core.Data.Full.Type (Type)
import Reussir.Core.Serialization.Orphans ()
import Reussir.Core.Serialization.Semi (
    SemiRecordSnapshot,
    StringUniqifierSnapshot,
    restoreSemiRecord,
    restoreStringUniqifier,
    snapshotSemiRecord,
    snapshotStringUniqifier,
 )

instance ToJSON PatternVarRef where
    toJSON = toJSON . unPatternVarRef
    toEncoding = toEncoding . unPatternVarRef

instance FromJSON PatternVarRef where
    parseJSON = fmap PatternVarRef . parseJSON

$(deriveJSON defaultOptions ''Type)

-- The expression types are mutually recursive, so they are derived in a
-- single declaration group.
$( concat
    <$> traverse
        (deriveJSON defaultOptions)
        [ ''ExprKind
        , ''Expr
        , ''DecisionTree
        , ''DTSwitchCases
        , ''ClosureExpr
        ]
 )

$(deriveJSON defaultOptions ''RecordKind)
$(deriveJSON defaultOptions ''RecordFields)
$(deriveJSON defaultOptions ''Record)
$(deriveJSON defaultOptions ''Function)
$(deriveJSON defaultOptions ''ErrorKind)
$(deriveJSON defaultOptions ''Error)

{- | Pure snapshot of the whole global full-elaboration context. This is the
unit of suspension between the full conversion and lowering.
-}
data FullContextSnapshot = FullContextSnapshot
    { fcFunctions :: [(Symbol, Function)]
    , fcRecords :: [(Symbol, Record)]
    , fcSemiRecords :: [(Path, SemiRecordSnapshot)]
    , fcErrors :: [Error]
    , fcStrings :: StringUniqifierSnapshot
    , fcFilePath :: FilePath
    , fcFlexible :: Bool
    , fcTrampolines :: [(Symbol, (T.Text, Symbol))]
    }
    deriving (Show, Eq)

$(deriveJSON defaultOptions ''FullContextSnapshot)

-- | Capture the whole global full context into a pure snapshot.
snapshotFullContext ::
    (IOE :> es, Prim :> es) => FullContext -> Eff es FullContextSnapshot
snapshotFullContext ctx = do
    functions <- liftIO $ H.toList (ctxFunctions ctx)
    records <- liftIO $ H.toList (ctxRecords ctx)
    semiRecords <- liftIO $ H.toList (ctxSemiRecords ctx)
    semiRecordSnaps <-
        traverse (\(path, r) -> (path,) <$> snapshotSemiRecord r) semiRecords
    strings <- snapshotStringUniqifier (ctxStringUniqifier ctx)
    pure
        FullContextSnapshot
            { fcFunctions = sortOn (symbolText . fst) functions
            , fcRecords = sortOn (symbolText . fst) records
            , fcSemiRecords = sortOn fst semiRecordSnaps
            , fcErrors = ctxErrors ctx
            , fcStrings = strings
            , fcFilePath = ctxFilePath ctx
            , fcFlexible = ctxFlexible ctx
            , fcTrampolines =
                sortOn (symbolText . fst) (HashMap.toList (ctxTrampolines ctx))
            }

-- | Rebuild a fully functional full context from a pure snapshot.
restoreFullContext ::
    (IOE :> es, Prim :> es) => FullContextSnapshot -> Eff es FullContext
restoreFullContext snap = do
    functionTable <- liftIO $ H.fromList (fcFunctions snap)
    recordTable <- liftIO $ H.fromList (fcRecords snap)
    semiRecords <-
        traverse (\(path, r) -> (path,) <$> restoreSemiRecord r) (fcSemiRecords snap)
    semiRecordTable <- liftIO $ H.fromList semiRecords
    uniqifier <- restoreStringUniqifier (fcStrings snap)
    pure
        FullContext
            { ctxFunctions = functionTable
            , ctxRecords = recordTable
            , ctxSemiRecords = semiRecordTable
            , ctxErrors = fcErrors snap
            , ctxStringUniqifier = uniqifier
            , ctxFilePath = fcFilePath snap
            , ctxFlexible = fcFlexible snap
            , ctxTrampolines = HashMap.fromList (fcTrampolines snap)
            }
