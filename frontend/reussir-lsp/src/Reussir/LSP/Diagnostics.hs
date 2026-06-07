{-# LANGUAGE OverloadedStrings #-}

{- |
  LSP Diagnostics Conversion
  ==========================

  Converts Reussir's internal diagnostic representation (byte-offset based)
  into LSP diagnostics (line/column based). Runs the parser and semi-elaboration
  pipeline to collect diagnostic reports.
-}
module Reussir.LSP.Diagnostics (
    -- * Diagnostic Conversion
    computeDiagnostics,

    -- * Shared Elaboration Pipeline
    elaborateFile,

    -- * Offset Conversion Utilities
    offsetToPosition,
    offsetsToRange,
) where

import Control.Monad (forM_)
import Data.Int (Int64)
import Effectful (Eff, IOE, inject, liftIO, (:>))
import Effectful.Prim (Prim)
import Effectful.State.Static.Local (runState)
import Reussir.Core.Data.Semi.Context (SemiContext (..))
import Reussir.Core.Semi.Context (
    emptySemiContext,
    populateRecordFields,
    scanStmt,
 )
import Reussir.Core.Semi.FlowAnalysis (solveAllGenerics)
import Reussir.Core.Semi.Tyck (checkFuncType)
import Reussir.Diagnostic.Report (
    CodeReference (..),
    Label (..),
    Report (..),
    TextWithFormat (..),
 )
import Reussir.Parser.Rust (parseProgIO)
import Reussir.Parser.Types.Lexer (WithSpan (..))

import Data.Text qualified as T
import Effectful.Log qualified as L
import Language.LSP.Protocol.Types qualified as LSP
import Reussir.Bridge qualified as B
import Reussir.Parser.Types.Stmt qualified as Syn

{- | Run the parser and semi-elaboration pipeline, returning both the context
and diagnostic reports. This is the shared entry point used by diagnostics,
hover, and semantic tokens.
-}
elaborateFile ::
    (IOE :> es, L.Log :> es, Prim :> es) =>
    -- | File path (URI-derived)
    FilePath ->
    -- | File content
    T.Text ->
    Eff es (Either T.Text (SemiContext, [Syn.Stmt], [Report]))
elaborateFile filePath content = do
    parsed <- liftIO $ parseProgIO filePath content
    case parsed of
        Left err -> return $ Left err
        Right prog -> do
            (_, finalState) <- do
                initState <- emptySemiContext B.LogWarning filePath []
                runState initState $ do
                    forM_ prog $ \stmt -> inject $ scanStmt stmt
                    forM_ prog $ \stmt -> inject $ populateRecordFields stmt
                    forM_ prog $ \stmt -> do
                        case unspanStmt stmt of
                            Syn.FunctionStmt f -> do
                                _ <- inject $ checkFuncType f
                                return ()
                            _ -> return ()
                    _ <- inject solveAllGenerics
                    return ()
            return $ Right (finalState, prog, translationReports finalState)
  where
    unspanStmt (Syn.SpannedStmt (WithSpan s _ _)) = unspanStmt s
    unspanStmt s = s

-- | Compute LSP diagnostics for a file by running the parser and semi-elaboration.
computeDiagnostics ::
    (IOE :> es, L.Log :> es, Prim :> es) =>
    -- | File path (URI-derived)
    FilePath ->
    -- | File content
    T.Text ->
    Eff es [LSP.Diagnostic]
computeDiagnostics filePath content = do
    result <- elaborateFile filePath content
    case result of
        Left errMsg ->
            return
                [ LSP.Diagnostic
                    { LSP._range = LSP.Range (LSP.Position 0 0) (LSP.Position 0 1)
                    , LSP._severity = Just LSP.DiagnosticSeverity_Error
                    , LSP._code = Nothing
                    , LSP._codeDescription = Nothing
                    , LSP._source = Just "reussir"
                    , LSP._message = errMsg
                    , LSP._tags = Nothing
                    , LSP._relatedInformation = Nothing
                    , LSP._data_ = Nothing
                    }
                ]
        Right (_, _, reports) ->
            return $ concatMap (reportToDiagnostics content) reports

-- | Convert a Reussir diagnostic Report into a list of LSP Diagnostics.
reportToDiagnostics :: T.Text -> Report -> [LSP.Diagnostic]
reportToDiagnostics content report = go LSP.DiagnosticSeverity_Error report
  where
    go _severity (Labeled label inner) =
        go (labelToSeverity label) inner
    go severity (CodeRef codeRef mAnnotation) =
        let range = offsetsToRange content (codeStartOffset codeRef) (codeEndOffset codeRef)
            msg = case mAnnotation of
                Just twf -> textContent twf
                Nothing -> "Error at this location"
         in [ LSP.Diagnostic
                { LSP._range = range
                , LSP._severity = Just severity
                , LSP._code = Nothing
                , LSP._codeDescription = Nothing
                , LSP._source = Just "reussir"
                , LSP._message = msg
                , LSP._tags = Nothing
                , LSP._relatedInformation = Nothing
                , LSP._data_ = Nothing
                }
            ]
    go severity (FormattedText texts) =
        let msg = T.concat $ map textContent texts
         in [ LSP.Diagnostic
                { LSP._range = LSP.Range (LSP.Position 0 0) (LSP.Position 0 1)
                , LSP._severity = Just severity
                , LSP._code = Nothing
                , LSP._codeDescription = Nothing
                , LSP._source = Just "reussir"
                , LSP._message = msg
                , LSP._tags = Nothing
                , LSP._relatedInformation = Nothing
                , LSP._data_ = Nothing
                }
            ]
    go severity (Nested inner) = go severity inner
    go _ ReportNil = []
    go severity (ReportSeq r1 r2) = go severity r1 ++ go severity r2

-- | Convert a diagnostic label to an LSP severity.
labelToSeverity :: Label -> LSP.DiagnosticSeverity
labelToSeverity Error = LSP.DiagnosticSeverity_Error
labelToSeverity Warning = LSP.DiagnosticSeverity_Warning
labelToSeverity Info = LSP.DiagnosticSeverity_Information
labelToSeverity Hint = LSP.DiagnosticSeverity_Hint

-- | Convert byte offsets (Int64) to an LSP Range using the file content.
offsetsToRange :: T.Text -> Int64 -> Int64 -> LSP.Range
offsetsToRange content startOff endOff =
    let startPos = offsetToPosition content (fromIntegral startOff)
        endPos = offsetToPosition content (fromIntegral endOff)
     in LSP.Range startPos endPos

-- | Convert a character offset to an LSP Position (0-indexed line and column).
offsetToPosition :: T.Text -> Int -> LSP.Position
offsetToPosition content offset =
    let prefix = T.take offset content
        linesBefore = T.count "\n" prefix
        lastNewline = case T.breakOnEnd "\n" prefix of
            ("", _) -> 0
            (before, _) -> T.length before
        col = T.length prefix - lastNewline
     in LSP.Position (fromIntegral linesBefore) (fromIntegral col)
