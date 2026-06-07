{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.Rust (
    ParserBackend (..),
    findRustSyntaxLibrary,
    parseProgIO,
    parseStmtIO,
    parseExprIO,
    parseTypeIO,
    parseReplInputIO,
) where

import Control.Exception (SomeException, try)
import Data.Char (isSpace)
import Data.List (find)
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)
import System.FilePath ((</>))
import System.Info (os)
import Text.Megaparsec (errorBundlePretty, runParser)

import Data.Text qualified as T

import Reussir.Parser.Expr (parseExpr)
import Reussir.Parser.Prog (Prog, ReplInput (..), parseProg, parseReplInput)
import Reussir.Parser.Rust.FFI (
    RustSyntaxLibrary,
    parseExprWithRust,
    parseProgramWithRust,
    parseStmtWithRust,
    parseTypeWithRust,
    withRustSyntaxLibrary,
 )
import Reussir.Parser.Rust.JSON (RustParseResponse (..))
import Reussir.Parser.Stmt (parseStmt)
import Reussir.Parser.Type (parseType)
import Reussir.Parser.Types (Parser)
import Reussir.Parser.Types.Expr (Expr)
import Reussir.Parser.Types.Stmt (Stmt)
import Reussir.Parser.Types.Type (Type)

data ParserBackend
    = HaskellOnly
    | RustFirst
    | RustOnly
    deriving (Eq, Show)

parseProgIO :: FilePath -> T.Text -> IO (Either T.Text Prog)
parseProgIO = parseWithRustFirst parseProg parseProgramWithRust

parseStmtIO :: FilePath -> T.Text -> IO (Either T.Text Stmt)
parseStmtIO = parseWithRustFirst parseStmt parseStmtWithRust

parseExprIO :: FilePath -> T.Text -> IO (Either T.Text Expr)
parseExprIO = parseWithRustFirst parseExpr parseExprWithRust

parseTypeIO :: FilePath -> T.Text -> IO (Either T.Text Type)
parseTypeIO = parseWithRustFirst parseType parseTypeWithRust

parseReplInputIO :: FilePath -> T.Text -> IO (Either T.Text ReplInput)
parseReplInputIO fileName input
    | isIgnorableReplInput input = pure (Right EmptyLine)
    | otherwise = do
        backend <- readParserBackend
        case backend of
            HaskellOnly -> pure haskellRepl
            RustFirst -> parseReplRustFirst True
            RustOnly -> parseReplRustFirst False
  where
    haskellRepl = runHaskellParser parseReplInput fileName input
    parseReplRustFirst allowFallback = do
        libraryPath <- findRustSyntaxLibrary
        case libraryPath of
            Nothing
                | allowFallback -> pure haskellRepl
                | otherwise -> pure $ Left rustLibraryMissingMessage
            Just path -> do
                rustResult <- tryAny $ withRustSyntaxLibrary path $ \lib -> do
                    stmtResult <- parseStmtWithRust lib fileName input
                    case stmtResult of
                        RustParseOk stmt -> pure $ Right (ReplStmt stmt)
                        RustParseError stmtErr -> do
                            exprResult <- parseExprWithRust lib fileName input
                            pure $ case exprResult of
                                RustParseOk expr -> Right (ReplExpr expr)
                                RustParseError exprErr ->
                                    Left $ combineDiagnostics stmtErr exprErr
                case rustResult of
                    Right (Right value) -> pure (Right value)
                    Right (Left rustErr)
                        | allowFallback ->
                            pure $ either (Left . combineDiagnostics rustErr) Right haskellRepl
                        | otherwise -> pure (Left rustErr)
                    Left err
                        | allowFallback -> pure haskellRepl
                        | otherwise -> pure $ Left $ T.pack (show err)

parseWithRustFirst ::
    Parser a ->
    (RustSyntaxLibrary -> FilePath -> T.Text -> IO (RustParseResponse a)) ->
    FilePath ->
    T.Text ->
    IO (Either T.Text a)
parseWithRustFirst haskellParser rustParser fileName input = do
    backend <- readParserBackend
    case backend of
        HaskellOnly -> pure haskellResult
        RustFirst -> parseRust True
        RustOnly -> parseRust False
  where
    haskellResult = runHaskellParser haskellParser fileName input
    parseRust allowFallback = do
        libraryPath <- findRustSyntaxLibrary
        case libraryPath of
            Nothing
                | allowFallback -> pure haskellResult
                | otherwise -> pure $ Left rustLibraryMissingMessage
            Just path -> do
                rustResult <- tryAny $ withRustSyntaxLibrary path $ \lib ->
                    rustParser lib fileName input
                case rustResult of
                    Right (RustParseOk value) -> pure (Right value)
                    Right (RustParseError rustErr)
                        | allowFallback ->
                            pure $ either (Left . combineDiagnostics rustErr) Right haskellResult
                        | otherwise -> pure (Left rustErr)
                    Left err
                        | allowFallback -> pure haskellResult
                        | otherwise -> pure $ Left $ T.pack (show err)

runHaskellParser :: Parser a -> FilePath -> T.Text -> Either T.Text a
runHaskellParser parser fileName input =
    case runParser parser fileName input of
        Left err -> Left $ T.pack $ errorBundlePretty err
        Right value -> Right value

readParserBackend :: IO ParserBackend
readParserBackend = do
    backend <- lookupEnv "REUSSIR_PARSER_BACKEND"
    pure $ case fmap normalize backend of
        Just "haskell" -> HaskellOnly
        Just "haskell-only" -> HaskellOnly
        Just "rust-first" -> RustFirst
        Just "rust" -> RustOnly
        Just "rust-only" -> RustOnly
        _ -> RustOnly
  where
    normalize = map dash . map lower
    lower c
        | 'A' <= c && c <= 'Z' = toEnum (fromEnum c + 32)
        | otherwise = c
    dash '_' = '-'
    dash c = c

findRustSyntaxLibrary :: IO (Maybe FilePath)
findRustSyntaxLibrary = do
    explicit <- lookupEnv "REUSSIR_SYNTAX_LIB"
    case explicit of
        Just path -> do
            exists <- doesFileExist path
            pure $ if exists then Just path else Nothing
        Nothing -> findExisting candidatePaths

findExisting :: [FilePath] -> IO (Maybe FilePath)
findExisting paths = do
    tagged <-
        traverse
            ( \path -> do
                exists <- doesFileExist path
                pure (path, exists)
            )
            paths
    pure $ fst <$> find snd tagged

candidatePaths :: [FilePath]
candidatePaths =
    [prefix </> "build" </> "lib" </> libName | prefix <- ancestorPrefixes]
        <> [prefix </> "lib" </> libName | prefix <- ancestorPrefixes]
        <> [prefix </> "target" </> "debug" </> libName | prefix <- ancestorPrefixes]

ancestorPrefixes :: [FilePath]
ancestorPrefixes = scanl (</>) "." (replicate 8 "..")

libName :: FilePath
libName = case os of
    "mingw32" -> "reussir_syntax.dll"
    "darwin" -> "libreussir_syntax.dylib"
    _ -> "libreussir_syntax.so"

isIgnorableReplInput :: T.Text -> Bool
isIgnorableReplInput = all ignorableLine . T.lines
  where
    ignorableLine line =
        let stripped = T.dropWhile isSpace line
         in T.null stripped || "//" `T.isPrefixOf` stripped

tryAny :: IO a -> IO (Either SomeException a)
tryAny = try

rustLibraryMissingMessage :: T.Text
rustLibraryMissingMessage =
    "Rust parser backend requested, but libreussir_syntax was not found. "
        <> "Set REUSSIR_SYNTAX_LIB to the built shared library path."

combineDiagnostics :: T.Text -> T.Text -> T.Text
combineDiagnostics first second = first <> "\n\nFallback parser also failed:\n" <> second
