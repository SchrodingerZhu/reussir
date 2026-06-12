{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TupleSections #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | JSON serialization for the semi syntax and the global semi-elaboration
context.

The pure parts of the semi syntax ('Semi.Type', 'Semi.Expr', decision trees,
record fields, ...) receive direct 'ToJSON'/'FromJSON' instances. The mutable
parts of 'SemiContext' (hash tables, @IORef'@s) are mirrored by pure
@*Snapshot@ types: 'snapshotSemiContext' captures the full context into a
'SemiContextSnapshot' (which has JSON instances and structural equality) and
'restoreSemiContext' rebuilds a fresh, fully functional 'SemiContext' from a
snapshot. This is what allows the pipeline to be suspended after any semi
phase step and recovered later.

All hash table dumps are sorted by key so that snapshots taken from
structurally equal contexts compare equal and serialize identically.
-}
module Reussir.Core.Serialization.Semi (
    SemiFunctionSnapshot (..),
    SemiRecordSnapshot (..),
    ClassDAGSnapshot (..),
    GenericVarSnapshot (..),
    GenericStateSnapshot (..),
    StringUniqifierSnapshot,
    GenericSolutionSnapshot,
    SemiContextSnapshot (..),
    snapshotStringUniqifier,
    restoreStringUniqifier,
    snapshotClassDAG,
    restoreClassDAG,
    snapshotGenericState,
    restoreGenericState,
    snapshotGenericSolution,
    restoreGenericSolution,
    snapshotSemiRecord,
    restoreSemiRecord,
    snapshotSemiFunction,
    restoreSemiFunction,
    snapshotSemiContext,
    restoreSemiContext,
) where

import Data.Aeson (FromJSON (..), ToJSON (..), defaultOptions)
import Data.Aeson.TH (deriveJSON)
import Data.Array (bounds, elems, listArray)
import Data.Digest.XXHash.FFI (XXH3 (..))
import Data.Foldable (toList)
import Data.Int (Int64)
import Data.List (sort, sortOn)
import Effectful (Eff, IOE, liftIO, (:>))
import Effectful.Prim (Prim)
import Effectful.Prim.IORef.Strict (newIORef', readIORef')
import Reussir.Diagnostic.Report (Report)
import Reussir.Parser.Serialization ()
import Reussir.Parser.Types.Capability (Capability)
import Reussir.Parser.Types.Lexer (Identifier, Path)
import Reussir.Parser.Types.Stmt (Visibility)

import Data.HashMap.Strict qualified as HashMap
import Data.HashSet qualified as HashSet
import Data.HashTable.IO qualified as H
import Data.Sequence qualified as Seq
import Data.Text qualified as T
import Reussir.Bridge qualified as B

import Reussir.Core.Data.Class (
    ChainID,
    Class,
    ClassDAG (..),
    ClassNode,
 )
import Reussir.Core.Data.Generic (
    GenericSolution,
    GenericState (..),
    GenericVar (..),
 )
import Reussir.Core.Data.Semi.Context (SemiContext (..))
import Reussir.Core.Data.Semi.Expr (
    ClosureExpr,
    DTSwitchCases,
    DecisionTree,
    Expr,
    ExprKind,
    PatternVarRef (..),
 )
import Reussir.Core.Data.Semi.Function (
    FunctionProto (..),
    FunctionTable (..),
 )
import Reussir.Core.Data.Semi.Record (Record (..), RecordFields, RecordKind)
import Reussir.Core.Data.Semi.Type (
    Flexivity,
    Type,
    TypeClassTable (..),
 )
import Reussir.Core.Data.String (StringToken, StringUniqifier (..))
import Reussir.Core.Data.UniqueID (GenericID)
import Reussir.Core.Serialization.Orphans ()
import Reussir.Core.String (getAllStrings)

instance ToJSON PatternVarRef where
    toJSON = toJSON . unPatternVarRef
    toEncoding = toEncoding . unPatternVarRef

instance FromJSON PatternVarRef where
    parseJSON = fmap PatternVarRef . parseJSON

$(deriveJSON defaultOptions ''Flexivity)
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

-- | Pure snapshot of a semi-phase 'FunctionProto' (body pulled out of its ref).
data SemiFunctionSnapshot = SemiFunctionSnapshot
    { sfVisibility :: Visibility
    , sfName :: Identifier
    , sfPath :: Path
    , sfGenerics :: [(Identifier, GenericID)]
    , sfParams :: [(Identifier, Type)]
    , sfReturnType :: Type
    , sfIsRegional :: Bool
    , sfBody :: Maybe Expr
    , sfSpan :: Maybe (Int64, Int64)
    }
    deriving (Show, Eq)

-- | Pure snapshot of a semi-phase 'Record' (fields pulled out of their ref).
data SemiRecordSnapshot = SemiRecordSnapshot
    { srName :: Path
    , srTyParams :: [(Identifier, GenericID)]
    , srFields :: Maybe RecordFields
    , srKind :: RecordKind
    , srVisibility :: Visibility
    , srDefaultCap :: Capability
    , srSpan :: Maybe (Int64, Int64)
    }
    deriving (Show, Eq)

-- | Pure snapshot of the type class DAG.
data ClassDAGSnapshot = ClassDAGSnapshot
    { cdNameMap :: [(Class, Int)]
    , cdIdMap :: [(Int, Class)]
    , cdNodeMap :: [(Int, ClassNode)]
    , cdChainHeadMap :: [(ChainID, Int)]
    , cdCounter :: Int
    , cdFinalizedGraph :: Maybe ((Int, Int), [ClassNode])
    }
    deriving (Show, Eq)

-- | Pure snapshot of a generic variable, including its flow links.
data GenericVarSnapshot = GenericVarSnapshot
    { gvName :: Identifier
    , gvSpan :: Maybe (Int64, Int64)
    , gvLinks :: [(GenericID, Maybe Type)]
    , gvBounds :: [Path]
    }
    deriving (Show, Eq)

{- | Pure snapshot of the generic instantiation state. The position of each
variable in 'gsVars' is its 'GenericID', so the sequence order is preserved.
-}
data GenericStateSnapshot = GenericStateSnapshot
    { gsVars :: [GenericVarSnapshot]
    , gsConcreteFlow :: [(GenericID, [Type])]
    }
    deriving (Show, Eq)

-- | The interned strings, sorted by string content.
type StringUniqifierSnapshot = [(T.Text, StringToken)]

-- | Pure form of 'GenericSolution', sorted by generic ID.
type GenericSolutionSnapshot = [(GenericID, [Type])]

{- | Pure snapshot of the whole global semi-elaboration context. This is the
unit of suspension between pipeline steps of the semi phase.
-}
data SemiContextSnapshot = SemiContextSnapshot
    { scFile :: FilePath
    , scModulePath :: [Identifier]
    , scLogLevel :: B.LogLevel
    , scStrings :: StringUniqifierSnapshot
    , scReports :: [Report]
    , scHasFailed :: Bool
    , scClassDAG :: ClassDAGSnapshot
    , scTypeClassTable :: [(Type, [Class])]
    , scRecords :: [(Path, SemiRecordSnapshot)]
    , scFunctions :: [(Path, SemiFunctionSnapshot)]
    , scGenerics :: GenericStateSnapshot
    , scTrampolines :: [(Identifier, (Path, T.Text, [Type]))]
    }
    deriving (Show, Eq)

$(deriveJSON defaultOptions ''SemiFunctionSnapshot)
$(deriveJSON defaultOptions ''SemiRecordSnapshot)
$(deriveJSON defaultOptions ''ClassDAGSnapshot)
$(deriveJSON defaultOptions ''GenericVarSnapshot)
$(deriveJSON defaultOptions ''GenericStateSnapshot)
$(deriveJSON defaultOptions ''SemiContextSnapshot)

snapshotStringUniqifier ::
    (IOE :> es) => StringUniqifier -> Eff es StringUniqifierSnapshot
snapshotStringUniqifier = fmap (sortOn fst) . getAllStrings

restoreStringUniqifier ::
    (IOE :> es) => StringUniqifierSnapshot -> Eff es StringUniqifier
restoreStringUniqifier entries =
    StringUniqifier
        <$> liftIO (H.fromList [(XXH3 str, token) | (str, token) <- entries])

snapshotClassDAG ::
    (IOE :> es, Prim :> es) => ClassDAG -> Eff es ClassDAGSnapshot
snapshotClassDAG dag = do
    names <- liftIO $ H.toList (nameMap dag)
    ids <- liftIO $ H.toList (idMap dag)
    nodes <- liftIO $ H.toList (nodeMap dag)
    chainHeads <- liftIO $ H.toList (chainHeadMap dag)
    counterVal <- readIORef' (counter dag)
    finalized <- readIORef' (finalizedGraph dag)
    pure
        ClassDAGSnapshot
            { cdNameMap = sortOn snd names
            , cdIdMap = sortOn fst ids
            , cdNodeMap = sortOn fst nodes
            , cdChainHeadMap = sortOn fst chainHeads
            , cdCounter = counterVal
            , cdFinalizedGraph = fmap (\arr -> (bounds arr, elems arr)) finalized
            }

restoreClassDAG ::
    (IOE :> es, Prim :> es) => ClassDAGSnapshot -> Eff es ClassDAG
restoreClassDAG snap = do
    names <- liftIO $ H.fromList (cdNameMap snap)
    ids <- liftIO $ H.fromList (cdIdMap snap)
    nodes <- liftIO $ H.fromList (cdNodeMap snap)
    chainHeads <- liftIO $ H.fromList (cdChainHeadMap snap)
    counterRef <- newIORef' (cdCounter snap)
    finalizedRef <-
        newIORef' (fmap (\(bnds, xs) -> listArray bnds xs) (cdFinalizedGraph snap))
    pure
        ClassDAG
            { nameMap = names
            , idMap = ids
            , nodeMap = nodes
            , chainHeadMap = chainHeads
            , counter = counterRef
            , finalizedGraph = finalizedRef
            }

snapshotGenericVar :: (IOE :> es) => GenericVar -> Eff es GenericVarSnapshot
snapshotGenericVar var = do
    links <- liftIO $ H.toList (genericLinks var)
    pure
        GenericVarSnapshot
            { gvName = genericName var
            , gvSpan = genericSpan var
            , gvLinks = sortOn fst links
            , gvBounds = genericBounds var
            }

restoreGenericVar :: (IOE :> es) => GenericVarSnapshot -> Eff es GenericVar
restoreGenericVar snap = do
    links <- liftIO $ H.fromList (gvLinks snap)
    pure
        GenericVar
            { genericName = gvName snap
            , genericSpan = gvSpan snap
            , genericLinks = links
            , genericBounds = gvBounds snap
            }

snapshotGenericState ::
    (IOE :> es, Prim :> es) => GenericState -> Eff es GenericStateSnapshot
snapshotGenericState st = do
    vars <- readIORef' (getStateRef st)
    varSnaps <- traverse snapshotGenericVar (toList vars)
    flows <- liftIO $ H.toList (concreteFlow st)
    pure
        GenericStateSnapshot
            { gsVars = varSnaps
            , gsConcreteFlow = sortOn fst flows
            }

restoreGenericState ::
    (IOE :> es, Prim :> es) => GenericStateSnapshot -> Eff es GenericState
restoreGenericState snap = do
    vars <- traverse restoreGenericVar (gsVars snap)
    varsRef <- newIORef' (Seq.fromList vars)
    flows <- liftIO $ H.fromList (gsConcreteFlow snap)
    pure GenericState{getStateRef = varsRef, concreteFlow = flows}

snapshotGenericSolution ::
    (IOE :> es) => GenericSolution -> Eff es GenericSolutionSnapshot
snapshotGenericSolution sol = sortOn fst <$> liftIO (H.toList sol)

restoreGenericSolution ::
    (IOE :> es) => GenericSolutionSnapshot -> Eff es GenericSolution
restoreGenericSolution = liftIO . H.fromList

snapshotSemiFunction ::
    (Prim :> es) => FunctionProto -> Eff es SemiFunctionSnapshot
snapshotSemiFunction proto = do
    body <- readIORef' (funcBody proto)
    pure
        SemiFunctionSnapshot
            { sfVisibility = funcVisibility proto
            , sfName = funcName proto
            , sfPath = funcPath proto
            , sfGenerics = funcGenerics proto
            , sfParams = funcParams proto
            , sfReturnType = funcReturnType proto
            , sfIsRegional = funcIsRegional proto
            , sfBody = body
            , sfSpan = funcSpan proto
            }

restoreSemiFunction ::
    (Prim :> es) => SemiFunctionSnapshot -> Eff es FunctionProto
restoreSemiFunction snap = do
    bodyRef <- newIORef' (sfBody snap)
    pure
        FunctionProto
            { funcVisibility = sfVisibility snap
            , funcName = sfName snap
            , funcPath = sfPath snap
            , funcGenerics = sfGenerics snap
            , funcParams = sfParams snap
            , funcReturnType = sfReturnType snap
            , funcIsRegional = sfIsRegional snap
            , funcBody = bodyRef
            , funcSpan = sfSpan snap
            }

snapshotSemiRecord :: (Prim :> es) => Record -> Eff es SemiRecordSnapshot
snapshotSemiRecord record = do
    fields <- readIORef' (recordFields record)
    pure
        SemiRecordSnapshot
            { srName = recordName record
            , srTyParams = recordTyParams record
            , srFields = fields
            , srKind = recordKind record
            , srVisibility = recordVisibility record
            , srDefaultCap = recordDefaultCap record
            , srSpan = recordSpan record
            }

restoreSemiRecord :: (Prim :> es) => SemiRecordSnapshot -> Eff es Record
restoreSemiRecord snap = do
    fieldsRef <- newIORef' (srFields snap)
    pure
        Record
            { recordName = srName snap
            , recordTyParams = srTyParams snap
            , recordFields = fieldsRef
            , recordKind = srKind snap
            , recordVisibility = srVisibility snap
            , recordDefaultCap = srDefaultCap snap
            , recordSpan = srSpan snap
            }

-- | Capture the whole global semi context into a pure snapshot.
snapshotSemiContext ::
    (IOE :> es, Prim :> es) => SemiContext -> Eff es SemiContextSnapshot
snapshotSemiContext ctx = do
    strings <- snapshotStringUniqifier (stringUniqifier ctx)
    dag <- snapshotClassDAG (typeClassDAG ctx)
    typeClasses <- liftIO $ H.toList (unTypeClassTable (typeClassTable ctx))
    records <- liftIO (H.toList (knownRecords ctx))
    recordSnaps <- traverse (\(path, r) -> (path,) <$> snapshotSemiRecord r) records
    protos <- liftIO $ H.toList (functionProtos (functions ctx))
    protoSnaps <- traverse (\(path, p) -> (path,) <$> snapshotSemiFunction p) protos
    genericSnap <- snapshotGenericState (generics ctx)
    pure
        SemiContextSnapshot
            { scFile = currentFile ctx
            , scModulePath = currentModulePath ctx
            , scLogLevel = translationLogLevel ctx
            , scStrings = strings
            , scReports = translationReports ctx
            , scHasFailed = translationHasFailed ctx
            , scClassDAG = dag
            , scTypeClassTable =
                sortOn
                    (show . fst)
                    (map (fmap (sort . HashSet.toList)) typeClasses)
            , scRecords = sortOn fst recordSnaps
            , scFunctions = sortOn fst protoSnaps
            , scGenerics = genericSnap
            , scTrampolines = sortOn fst (HashMap.toList (trampolines ctx))
            }

-- | Rebuild a fully functional semi context from a pure snapshot.
restoreSemiContext ::
    (IOE :> es, Prim :> es) => SemiContextSnapshot -> Eff es SemiContext
restoreSemiContext snap = do
    uniqifier <- restoreStringUniqifier (scStrings snap)
    dag <- restoreClassDAG (scClassDAG snap)
    typeClasses <-
        liftIO $
            H.fromList
                [(ty, HashSet.fromList classes) | (ty, classes) <- scTypeClassTable snap]
    records <-
        traverse (\(path, r) -> (path,) <$> restoreSemiRecord r) (scRecords snap)
    recordTable <- liftIO $ H.fromList records
    protos <-
        traverse (\(path, p) -> (path,) <$> restoreSemiFunction p) (scFunctions snap)
    protoTable <- liftIO $ H.fromList protos
    genericState <- restoreGenericState (scGenerics snap)
    pure
        SemiContext
            { currentFile = scFile snap
            , currentModulePath = scModulePath snap
            , translationLogLevel = scLogLevel snap
            , stringUniqifier = uniqifier
            , translationReports = scReports snap
            , translationHasFailed = scHasFailed snap
            , typeClassDAG = dag
            , typeClassTable = TypeClassTable typeClasses
            , knownRecords = recordTable
            , functions = FunctionTable protoTable
            , generics = genericState
            , trampolines = HashMap.fromList (scTrampolines snap)
            }
