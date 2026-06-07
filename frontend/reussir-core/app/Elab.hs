module Main where

import Control.Monad (forM, forM_, when)
import Data.List (nub)
import Effectful (inject, liftIO, runEff)
import Effectful.Prim (runPrim)
import Effectful.State.Static.Local (runState)
import GHC.IO.Handle.FD (stderr)
import Log (LogLevel (..))
import Options.Applicative
import Prettyprinter (hardline, unAnnotate)
import Prettyprinter.Render.Terminal (putDoc)
import Reussir.Diagnostic (createRepository, displayReport)
import Reussir.Parser.Rust (parseProgIO)
import Reussir.Parser.Types.Lexer (Identifier (..), WithSpan (..))
import System.Exit (exitFailure, exitSuccess)
import System.IO (hIsTerminalDevice, hPutStrLn, stdout)

import Data.HashTable.IO qualified as H
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Effectful.Log qualified as L
import Effectful.State.Static.Local qualified as State
import Log.Backend.StandardOutput qualified as L
import Reussir.Bridge qualified as B
import Reussir.Parser.Types.Stmt qualified as Syn

import Reussir.Core.Data.Full.Context (FullContext (..))
import Reussir.Core.Data.Semi.Context (SemiContext (..))
import Reussir.Core.Data.Semi.Function (FunctionTable (..))
import Reussir.Core.Full.Context (reportAllErrors)
import Reussir.Core.Full.Conversion (convertCtx)
import Reussir.Core.Module (
    PackageInfo (..),
    discoverPackageFiles,
    renderDiscoveryError,
    validatePackageInfo,
 )
import Reussir.Core.Semi.Context (
    emptySemiContext,
    populateRecordFields,
    scanStmt,
    withModuleFile,
 )
import Reussir.Core.Semi.FlowAnalysis (solveAllGenerics)
import Reussir.Core.Semi.Trampoline
import Reussir.Core.Semi.Tyck (checkFuncType)

import Reussir.Core.Full.Pretty qualified as Full
import Reussir.Core.Semi.Pretty qualified as Semi

data ElabMode = SemiMode | FullMode
    deriving (Eq, Show)

data Args = Args
    { elabMode :: ElabMode
    , inputFile :: Maybe FilePath
    , argPackageRoot :: Maybe FilePath
    , argPackageName :: Maybe String
    }

elabModeReader :: ReadM ElabMode
elabModeReader = eitherReader $ \s ->
    case s of
        "semi" -> Right SemiMode
        "full" -> Right FullMode
        _ -> Left $ "Invalid mode '" ++ s ++ "'. Expected 'semi' or 'full'."

argsParser :: Parser Args
argsParser =
    Args
        <$> option
            elabModeReader
            ( long "mode"
                <> metavar "MODE"
                <> help "Elaboration mode: 'semi' or 'full'"
            )
        <*> optional (strArgument (metavar "FILE" <> help "Input file (single-file mode)"))
        <*> optional
            ( strOption
                ( long "package-root"
                    <> metavar "DIR"
                    <> help "Package root directory (multi-file mode)"
                )
            )
        <*> optional
            ( strOption
                ( long "package-name"
                    <> metavar "NAME"
                    <> help "Package name (required with --package-root)"
                )
            )

tyckBridgeLogLevel :: B.LogLevel
tyckBridgeLogLevel = B.LogWarning

toEffLogLevel :: B.LogLevel -> LogLevel
toEffLogLevel = \case
    B.LogError -> LogAttention
    B.LogWarning -> LogAttention
    B.LogInfo -> LogInfo
    B.LogDebug -> LogTrace
    B.LogTrace -> LogTrace

unspanStmt :: Syn.Stmt -> Syn.Stmt
unspanStmt (Syn.SpannedStmt (WithSpan s _ _)) = unspanStmt s
unspanStmt stmt = stmt

main :: IO ()
main = do
    args <- execParser opts
    case (argPackageRoot args, argPackageName args) of
        (Just root, Just pkgName) -> elabPackage args root pkgName
        (Just _, Nothing) -> do
            putStrLn "Error: --package-name is required when --package-root is specified"
            exitFailure
        (Nothing, Just _) -> do
            putStrLn "Error: --package-root is required when --package-name is specified"
            exitFailure
        (Nothing, Nothing) -> case inputFile args of
            Just fp -> elabSingleFile args fp
            Nothing -> do
                putStrLn "Error: either FILE or --package-root/--package-name must be specified"
                exitFailure
  where
    opts =
        info
            (argsParser <**> helper)
            ( fullDesc
                <> progDesc "Elaborate a Reussir file"
                <> header "reussir-elab - Reussir Elaborator"
            )

elabSingleFile :: Args -> FilePath -> IO ()
elabSingleFile args fp = do
    content <- TIO.readFile fp
    parsed <- parseProgIO fp content
    case parsed of
        Left err -> do
            TIO.putStrLn err
            exitFailure
        Right prog -> case elabMode args of
            SemiMode -> runSemiElab args [(fp, [], prog)]
            FullMode -> runFullElab args [(fp, [], prog)]

elabPackage :: Args -> FilePath -> String -> IO ()
elabPackage args root pkgName = do
    let rawPkgInfo = PackageInfo root (Identifier (T.pack pkgName))
    pkgInfo <- case validatePackageInfo rawPkgInfo of
        Left err -> putStrLn err >> exitFailure
        Right valid -> return valid
    discoveryResult <- discoverPackageFiles pkgInfo
    discoveredFiles <- case discoveryResult of
        Left err -> putStrLn (renderDiscoveryError err) >> exitFailure
        Right files -> return files
    when (null discoveredFiles) $ do
        putStrLn $ "Error: no .rr files found in " ++ root
        exitFailure
    parsedFiles <- forM discoveredFiles $ \(fp, modPath) -> do
        content <- TIO.readFile fp
        parsed <- parseProgIO fp content
        case parsed of
            Left err -> do
                TIO.putStrLn err
                exitFailure
            Right prog -> return (fp, modPath, prog)
    case elabMode args of
        SemiMode -> runSemiElab args parsedFiles
        FullMode -> runFullElab args parsedFiles

runSemiElab :: Args -> [(FilePath, [Identifier], [Syn.Stmt])] -> IO ()
runSemiElab _args files = do
    let allFilePaths = map (\(fp, _, _) -> fp) files
    let primaryFilePath = case files of
            ((fp, _, _) : _) -> fp
            [] -> "<unknown>"
    let primaryModPath = case files of
            ((_, mp, _) : _) -> mp
            [] -> []
    repository <- runEff $ createRepository allFilePaths
    ((), finalState) <- L.withStdOutLogger $ \logger -> do
        runEff $ L.runLog (T.pack "reussir-elab") logger (toEffLogLevel tyckBridgeLogLevel) $ do
            initState <-
                runPrim $ emptySemiContext tyckBridgeLogLevel primaryFilePath primaryModPath
            runPrim $ runState initState $ do
                -- Scan all statements
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $
                                \stmt -> inject $ scanStmt stmt

                -- Populate record fields
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $
                                \stmt -> inject $ populateRecordFields stmt

                -- Elaborate all functions
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $ \stmt -> do
                                case unspanStmt stmt of
                                    Syn.FunctionStmt f -> do
                                        _ <- inject $ checkFuncType f
                                        return ()
                                    Syn.ExternTrampolineStmt name abi target args' -> do
                                        inject $ resolveTrampoline name abi target args'
                                    _ -> return ()

                let putDoc' doc = do
                        isTTy <- liftIO $ hIsTerminalDevice stdout
                        if isTTy
                            then liftIO $ putDoc (doc <> hardline)
                            else liftIO $ putDoc (unAnnotate doc <> hardline)

                -- Solve generics
                slns <- inject solveAllGenerics
                case slns of
                    Just instances -> do
                        liftIO $ putStrLn "\n;; Instantiated Generics"
                        instList <- liftIO $ H.toList instances
                        forM_ instList $ \(gid, types) -> do
                            liftIO $ putStrLn $ "Generic " ++ show gid ++ " should be instantiated to:"
                            forM_ types $ \ty -> do
                                doc <- Semi.prettyColored ty
                                liftIO $ putDoc' (doc <> hardline)
                    Nothing -> pure ()

                liftIO $ putStrLn ";; Elaborated Records"
                knownRecs <- State.gets knownRecords
                recs <- liftIO $ H.toList knownRecs
                forM_ recs $ \(_, record) -> do
                    doc <- Semi.prettyColored record
                    liftIO $ putDoc' (doc <> hardline)

                liftIO $ putStrLn "\n;; Elaborated Functions"
                funcsTbl <- State.gets functions
                funcs <- liftIO $ H.toList (functionProtos funcsTbl)
                forM_ funcs $ \(_, proto) -> do
                    doc <- Semi.prettyColored proto
                    liftIO $ putDoc' (doc <> hardline)

    forM_ (nub $ translationReports finalState) $ \report -> do
        runEff $ displayReport report repository 0 stderr
        hPutStrLn stderr ""

    if null (translationReports finalState)
        then exitSuccess
        else exitFailure

runFullElab :: Args -> [(FilePath, [Identifier], [Syn.Stmt])] -> IO ()
runFullElab _args files = do
    let allFilePaths = map (\(fp, _, _) -> fp) files
    let primaryFilePath = case files of
            ((fp, _, _) : _) -> fp
            [] -> "<unknown>"
    let primaryModPath = case files of
            ((_, mp, _) : _) -> mp
            [] -> []
    repository <- runEff $ createRepository allFilePaths
    (result, finalSemiCtx) <- L.withStdOutLogger $ \logger -> do
        runEff $ L.runLog (T.pack "reussir-elab") logger (toEffLogLevel tyckBridgeLogLevel) $ do
            -- 1. Semi Elab
            initSemiState <-
                runPrim $ emptySemiContext tyckBridgeLogLevel primaryFilePath primaryModPath
            (slns, finalSemiState) <- runPrim $ runState initSemiState $ do
                -- Scan all statements
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $
                                \stmt -> inject $ scanStmt stmt

                -- Populate record fields
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $
                                \stmt -> inject $ populateRecordFields stmt

                -- Elaborate all functions
                forM_ files $ \(fp, modPath, prog) ->
                    inject $
                        withModuleFile fp modPath $
                            forM_ prog $ \stmt -> do
                                case unspanStmt stmt of
                                    Syn.FunctionStmt f -> do
                                        _ <- inject $ checkFuncType f
                                        return ()
                                    Syn.ExternTrampolineStmt name abi target args' -> do
                                        inject $ resolveTrampoline name abi target args'
                                    _ -> return ()

                -- Solve generics
                inject solveAllGenerics

            case slns of
                Nothing -> return (Nothing, finalSemiState)
                Just solutions -> do
                    -- 2. Full Conversion
                    fullCtx <- runPrim $ convertCtx finalSemiState solutions
                    return (Just fullCtx, finalSemiState)

    -- Report errors from Semi Elab
    forM_ (nub $ translationReports finalSemiCtx) $ \report -> do
        runEff $ displayReport report repository 0 stderr

    let putDoc' doc = do
            isTTy <- liftIO $ hIsTerminalDevice stdout
            if isTTy
                then liftIO $ putDoc (doc <> hardline)
                else liftIO $ putDoc (unAnnotate doc <> hardline)

    case result of
        Nothing -> do
            hPutStrLn stderr "Generic solution failed or other errors in Semi Elab."
            exitFailure
        Just finalFullCtx -> do
            -- Report errors from Full Elab
            runEff $ runPrim $ reportAllErrors finalFullCtx repository stderr

            case ctxErrors finalFullCtx of
                [] | null (translationReports finalSemiCtx) -> do
                    -- Pretty Print if no errors
                    putStrLn ";; Elaborated Full Records"
                    recs <- H.toList (ctxRecords finalFullCtx)
                    runEff $ runPrim $ forM_ recs $ \(_, record) -> do
                        doc <- Full.prettyColored record
                        liftIO $ putDoc' (doc <> hardline)

                    putStrLn "\n;; Elaborated Full Functions"
                    funcs <- H.toList (ctxFunctions finalFullCtx)
                    runEff $ runPrim $ forM_ funcs $ \(_, func) -> do
                        doc <- Full.prettyColored func
                        liftIO $ putDoc' (doc <> hardline)

                    exitSuccess
                _ -> exitFailure
