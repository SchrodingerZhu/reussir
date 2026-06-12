{-# LANGUAGE TemplateHaskell #-}

{- | Suspend/recover support for the frontend pipeline.

Every stage of the frontend can be serialized to JSON and recovered later:

* __Surface syntax__: the parsed program ('SurfaceProgram', one
  'SurfaceModule' per source file).
* __Semi syntax__: the global semi-elaboration context ('SemiContext'),
  suspendable after any semi step (statement scanning, record field
  population, function elaboration, generic solving), plus the
  'GenericSolution' produced by flow analysis.
* __Full syntax__: the global full-elaboration context ('FullContext')
  produced by the semi-to-full conversion.

For each stage there are @encode@/@decode@ (in-memory 'BSL.ByteString') and
@suspend@/@recover@ (file based) functions. Decoding rebuilds fully
functional, mutable contexts so the pipeline can continue from the recovered
state.
-}
module Reussir.Core.Serialization (
    -- * Surface syntax
    SurfaceModule (..),
    SurfaceProgram,
    encodeSurfaceProgram,
    decodeSurfaceProgram,
    suspendSurfaceProgram,
    recoverSurfaceProgram,

    -- * Semi syntax
    encodeSemiContext,
    decodeSemiContext,
    suspendSemiContext,
    recoverSemiContext,
    encodeGenericSolution,
    decodeGenericSolution,
    suspendGenericSolution,
    recoverGenericSolution,

    -- * Full syntax
    encodeFullContext,
    decodeFullContext,
    suspendFullContext,
    recoverFullContext,

    -- * Snapshots
    module Reussir.Core.Serialization.Semi,
    module Reussir.Core.Serialization.Full,
) where

import Data.Aeson (eitherDecode, encode)
import Data.Aeson.TH (deriveJSON)
import Effectful (Eff, IOE, liftIO, (:>))
import Effectful.Prim (Prim)
import Reussir.Parser.Serialization ()
import Reussir.Parser.Types.Lexer (Identifier)
import Reussir.Parser.Types.Stmt (Stmt)

import Data.Aeson qualified as Aeson
import Data.ByteString.Lazy qualified as BSL

import Reussir.Core.Data.Full.Context (FullContext)
import Reussir.Core.Data.Generic (GenericSolution)
import Reussir.Core.Data.Semi.Context (SemiContext)
import Reussir.Core.Serialization.Full
import Reussir.Core.Serialization.Orphans ()
import Reussir.Core.Serialization.Semi

-- | A parsed source file together with its module path inside the package.
data SurfaceModule = SurfaceModule
    { surfaceFile :: FilePath
    , surfaceModPath :: [Identifier]
    , surfaceStmts :: [Stmt]
    }
    deriving (Show, Eq)

$(deriveJSON Aeson.defaultOptions ''SurfaceModule)

-- | A whole parsed package: the input of the semi-elaboration phase.
type SurfaceProgram = [SurfaceModule]

encodeSurfaceProgram :: SurfaceProgram -> BSL.ByteString
encodeSurfaceProgram = encode

decodeSurfaceProgram :: BSL.ByteString -> Either String SurfaceProgram
decodeSurfaceProgram = eitherDecode

suspendSurfaceProgram :: (IOE :> es) => FilePath -> SurfaceProgram -> Eff es ()
suspendSurfaceProgram path = liftIO . Aeson.encodeFile path

recoverSurfaceProgram ::
    (IOE :> es) => FilePath -> Eff es (Either String SurfaceProgram)
recoverSurfaceProgram = liftIO . Aeson.eitherDecodeFileStrict

encodeSemiContext ::
    (IOE :> es, Prim :> es) => SemiContext -> Eff es BSL.ByteString
encodeSemiContext = fmap encode . snapshotSemiContext

decodeSemiContext ::
    (IOE :> es, Prim :> es) => BSL.ByteString -> Eff es (Either String SemiContext)
decodeSemiContext bytes = traverse restoreSemiContext (eitherDecode bytes)

suspendSemiContext ::
    (IOE :> es, Prim :> es) => FilePath -> SemiContext -> Eff es ()
suspendSemiContext path ctx = do
    snap <- snapshotSemiContext ctx
    liftIO $ Aeson.encodeFile path snap

recoverSemiContext ::
    (IOE :> es, Prim :> es) => FilePath -> Eff es (Either String SemiContext)
recoverSemiContext path = do
    snap <- liftIO $ Aeson.eitherDecodeFileStrict path
    traverse restoreSemiContext snap

encodeGenericSolution ::
    (IOE :> es) => GenericSolution -> Eff es BSL.ByteString
encodeGenericSolution = fmap encode . snapshotGenericSolution

decodeGenericSolution ::
    (IOE :> es) => BSL.ByteString -> Eff es (Either String GenericSolution)
decodeGenericSolution bytes = traverse restoreGenericSolution (eitherDecode bytes)

suspendGenericSolution ::
    (IOE :> es) => FilePath -> GenericSolution -> Eff es ()
suspendGenericSolution path sol = do
    snap <- snapshotGenericSolution sol
    liftIO $ Aeson.encodeFile path snap

recoverGenericSolution ::
    (IOE :> es) => FilePath -> Eff es (Either String GenericSolution)
recoverGenericSolution path = do
    snap <- liftIO $ Aeson.eitherDecodeFileStrict path
    traverse restoreGenericSolution snap

encodeFullContext ::
    (IOE :> es, Prim :> es) => FullContext -> Eff es BSL.ByteString
encodeFullContext = fmap encode . snapshotFullContext

decodeFullContext ::
    (IOE :> es, Prim :> es) => BSL.ByteString -> Eff es (Either String FullContext)
decodeFullContext bytes = traverse restoreFullContext (eitherDecode bytes)

suspendFullContext ::
    (IOE :> es, Prim :> es) => FilePath -> FullContext -> Eff es ()
suspendFullContext path ctx = do
    snap <- snapshotFullContext ctx
    liftIO $ Aeson.encodeFile path snap

recoverFullContext ::
    (IOE :> es, Prim :> es) => FilePath -> Eff es (Either String FullContext)
recoverFullContext path = do
    snap <- liftIO $ Aeson.eitherDecodeFileStrict path
    traverse restoreFullContext snap
