{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (forM, when)
import Data.Char (toLower)
import Effectful (liftIO, runEff)
import Effectful.Prim (runPrim)
import Log (LogLevel (..))
import Options.Applicative
import Reussir.Codegen.Context (TargetSpec (..))
import Reussir.Parser.Rust (parseProgIO)
import Reussir.Parser.Types.Lexer (Identifier (..))
import System.Exit (exitFailure)

import Data.Text qualified as T
import Data.Text.Encoding qualified as TE
import Data.Text.IO qualified as TIO
import Effectful.Log qualified as L
import Reussir.Bridge qualified as B
import Reussir.Codegen qualified as C

import Reussir.Core (translatePackageToModule, translateProgToModule)
import Reussir.Core.Module (
    PackageInfo (..),
    discoverPackageFiles,
    renderDiscoveryError,
    validatePackageInfo,
 )

data Args = Args
    { argInputFile :: Maybe FilePath
    , argOutputFile :: FilePath
    , argOptLevel :: B.OptOption
    , argOutputTarget :: OutputTarget
    , argLogLevel :: B.LogLevel
    , argModuleName :: String
    , argTargetTriple :: Maybe String
    , argTargetCPU :: Maybe String
    , argTargetFeatures :: Maybe String
    , argRelocationMode :: B.RelocationModel
    , argReuseTokenAcrossCall :: Bool
    , argTokenReuseHeuristic :: String
    , argDisableInvariantAnalysis :: Bool
    , argPackageRoot :: Maybe FilePath
    , argPackageName :: Maybe String
    }

argsParser :: Parser Args
argsParser =
    Args
        <$> optional (strArgument (metavar "INPUT" <> help "Input file (single-file mode)"))
        <*> strOption (long "output" <> short 'o' <> metavar "OUTPUT" <> help "Output file")
        <*> option
            parseOptLevel
            ( long "opt-level"
                <> short 'O'
                <> value B.OptDefault
                <> help "Optimization level (none, default, aggressive, size, tpde)"
            )
        <*> option
            parseOutputTarget
            ( long "target"
                <> short 't'
                <> value (Backend B.OutputObject)
                <> help "Output target (llvm-ir, asm, object)"
            )
        <*> option
            parseLogLevel
            ( long "log-level"
                <> short 'l'
                <> value B.LogWarning
                <> help "Log level (error, warning, info, debug, trace)"
            )
        <*> strOption
            (long "module-name" <> short 'm' <> value "main" <> help "Module name")
        <*> optional
            ( strOption
                ( long "target-triple"
                    <> metavar "TRIPLE"
                    <> help "Target triple (default: native)"
                )
            )
        <*> optional
            ( strOption
                (long "target-cpu" <> metavar "CPU" <> help "Target CPU (default: native)")
            )
        <*> optional
            ( strOption
                ( long "target-features"
                    <> metavar "FEATURES"
                    <> help "Target features (default: native)"
                )
            )
        <*> option
            parseRelocationMode
            ( long "relocation-mode"
                <> value B.RelocationModelDefault
                <> help
                    "Relocation mode (default, static, pic, dynamic-no-pic, ropi, rwpi, ropi-rwpi)"
            )
        <*> switch
            ( long "reuse-across-call"
                <> help "Allow token reuse across function calls (may increase peak heap usage)"
            )
        <*> strOption
            ( long "token-reuse-heuristic"
                <> value "default"
                <> metavar "HEURISTIC"
                <> help "Token reuse selection heuristic (default, nearest)"
            )
        <*> switch
            ( long "disable-invariant-analysis"
                <> help "Disable the invariant group analysis pass in backend lowering"
            )
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

parseOptLevel :: ReadM B.OptOption
parseOptLevel = eitherReader $ \s -> case s of
    "none" -> Right B.OptNone
    "default" -> Right B.OptDefault
    "aggressive" -> Right B.OptAggressive
    "size" -> Right B.OptSize
    "tpde" -> Right B.OptTPDE
    _ -> Left $ "Unknown optimization level: " ++ s

data OutputTarget = Backend B.OutputTarget | MLIR

parseOutputTarget :: ReadM OutputTarget
parseOutputTarget = eitherReader $ \s -> case s of
    "llvm-ir" -> Right $ Backend B.OutputLLVMIR
    "asm" -> Right $ Backend B.OutputASM
    "object" -> Right $ Backend B.OutputObject
    "mlir" -> Right MLIR
    _ -> Left $ "Unknown output target: " ++ s

parseLogLevel :: ReadM B.LogLevel
parseLogLevel = eitherReader $ \s -> case s of
    "error" -> Right B.LogError
    "warning" -> Right B.LogWarning
    "info" -> Right B.LogInfo
    "debug" -> Right B.LogDebug
    "trace" -> Right B.LogTrace
    _ -> Left $ "Unknown log level: " ++ s

parseRelocationMode :: ReadM B.RelocationModel
parseRelocationMode = eitherReader $ \s -> case map toLower s of
    "default" -> Right B.RelocationModelDefault
    "static" -> Right B.RelocationModelStatic
    "pic" -> Right B.RelocationModelPIC
    "dynamic" -> Right B.RelocationModelDynamic
    "dynamic-no-pic" -> Right B.RelocationModelDynamic
    "dynamic-nopic" -> Right B.RelocationModelDynamic
    "ropi" -> Right B.RelocationModelROPI
    "rwpi" -> Right B.RelocationModelRWPI
    "ropi-rwpi" -> Right B.RelocationModelROPI_RWPI
    "ropi_rwpi" -> Right B.RelocationModelROPI_RWPI
    _ -> Left $ "Unknown relocation mode: " ++ s

-- | Build a TargetSpec from the common arguments and a given file path.
makeTargetSpec :: Args -> FilePath -> TargetSpec
makeTargetSpec args filePath =
    TargetSpec
        { programName = T.pack (argModuleName args)
        , outputPath = argOutputFile args
        , optimization = argOptLevel args
        , outputTarget = case argOutputTarget args of
            MLIR -> B.OutputObject
            Backend t -> t
        , logLevel = argLogLevel args
        , moduleFilePath = filePath
        , targetTriple = T.pack <$> argTargetTriple args
        , targetCPU = T.pack <$> argTargetCPU args
        , targetFeatures = T.pack <$> argTargetFeatures args
        }

main :: IO ()
main = do
    args <- execParser opts
    case (argPackageRoot args, argPackageName args) of
        (Just root, Just pkgName) -> compilePackage args root pkgName
        (Just _, Nothing) -> do
            putStrLn "Error: --package-name is required when --package-root is specified"
            exitFailure
        (Nothing, Just _) -> do
            putStrLn "Error: --package-root is required when --package-name is specified"
            exitFailure
        (Nothing, Nothing) -> case argInputFile args of
            Just inputFile -> compileSingleFile args inputFile
            Nothing -> do
                putStrLn
                    "Error: either INPUT file or --package-root/--package-name must be specified"
                exitFailure
  where
    opts =
        info
            (argsParser <**> helper)
            ( fullDesc
                <> progDesc "Compile a Reussir file"
                <> header "reussir-compiler - Reussir Compiler"
            )

compileSingleFile :: Args -> FilePath -> IO ()
compileSingleFile args inputFile = do
    content <- TIO.readFile inputFile
    parsed <- parseProgIO inputFile content
    case parsed of
        Left err -> do
            TIO.putStrLn err
            exitFailure
        Right prog -> do
            let outputTarget = argOutputTarget args
            let spec = makeTargetSpec args inputFile

            B.withReussirLogger (argLogLevel args) "reussir-compiler" $ \logger -> do
                runEff $ L.runLog "reussir-compiler" logger (toEffLogLevel (argLogLevel args)) $ do
                    irModule <- runPrim $ translateProgToModule spec prog
                    case irModule of
                        Nothing -> liftIO exitFailure
                        Just module' -> case outputTarget of
                            MLIR -> do
                                mlirText <- C.emitModuleToText module'
                                liftIO $ TIO.writeFile (argOutputFile args) mlirText
                            Backend target -> do
                                mlirText <- C.emitModuleToText module'
                                liftIO $
                                    B.compileForTargetWithModels
                                        (TE.encodeUtf8 mlirText)
                                        (argModuleName args)
                                        (argOutputFile args)
                                        target
                                        (argOptLevel args)
                                        (argLogLevel args)
                                        (TE.encodeUtf8 . T.pack <$> argTargetTriple args)
                                        (TE.encodeUtf8 . T.pack <$> argTargetCPU args)
                                        (TE.encodeUtf8 . T.pack <$> argTargetFeatures args)
                                        B.CodeModelDefault
                                        (argRelocationMode args)
                                        (argReuseTokenAcrossCall args)
                                        (TE.encodeUtf8 . T.pack $ argTokenReuseHeuristic args)
                                        (not (argDisableInvariantAnalysis args))

compilePackage :: Args -> FilePath -> String -> IO ()
compilePackage args root pkgName = do
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
    let primaryFile = case discoveredFiles of
            ((fp, _) : _) -> fp
            [] -> root
    let outputTarget = argOutputTarget args
    let spec = makeTargetSpec args primaryFile

    B.withReussirLogger (argLogLevel args) "reussir-compiler" $ \logger -> do
        runEff $ L.runLog "reussir-compiler" logger (toEffLogLevel (argLogLevel args)) $ do
            irModule <- runPrim $ translatePackageToModule spec parsedFiles
            case irModule of
                Nothing -> liftIO exitFailure
                Just module' -> case outputTarget of
                    MLIR -> do
                        mlirText <- C.emitModuleToText module'
                        liftIO $ TIO.writeFile (argOutputFile args) mlirText
                    Backend target -> do
                        mlirText <- C.emitModuleToText module'
                        liftIO $
                            B.compileForTargetWithModels
                                (TE.encodeUtf8 mlirText)
                                (argModuleName args)
                                (argOutputFile args)
                                target
                                (argOptLevel args)
                                (argLogLevel args)
                                (TE.encodeUtf8 . T.pack <$> argTargetTriple args)
                                (TE.encodeUtf8 . T.pack <$> argTargetCPU args)
                                (TE.encodeUtf8 . T.pack <$> argTargetFeatures args)
                                B.CodeModelDefault
                                (argRelocationMode args)
                                (argReuseTokenAcrossCall args)
                                (TE.encodeUtf8 . T.pack $ argTokenReuseHeuristic args)
                                (not (argDisableInvariantAnalysis args))

toEffLogLevel :: B.LogLevel -> LogLevel
toEffLogLevel = \case
    B.LogError -> LogAttention
    B.LogWarning -> LogAttention
    B.LogInfo -> LogInfo
    B.LogDebug -> LogTrace
    B.LogTrace -> LogTrace
