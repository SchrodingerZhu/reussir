{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.Rust.FFI (
    RustSyntaxLibrary,
    withRustSyntaxLibrary,
    parseProgramWithRust,
    parseStmtWithRust,
    parseExprWithRust,
    parseTypeWithRust,
) where

import Control.Exception (bracket, throwIO)
import Foreign.C.String (CString, withCString)
import Foreign.Ptr (FunPtr)
import System.Posix.DynamicLinker (
    DL,
    RTLDFlags (RTLD_LOCAL, RTLD_NOW),
    dlclose,
    dlopen,
    dlsym,
 )

import Data.ByteString qualified as BS
import Data.Text qualified as T
import Data.Text.Encoding qualified as TE

import Reussir.Parser.Prog (Prog)
import Reussir.Parser.Rust.JSON (
    RustParseResponse,
    decodeExprResponseText,
    decodeProgramResponseText,
    decodeStmtResponseText,
    decodeTypeResponseText,
 )
import Reussir.Parser.Types.Expr (Expr)
import Reussir.Parser.Types.Stmt (Stmt)
import Reussir.Parser.Types.Type (Type)

type ParseFn = CString -> CString -> IO CString

type FreeFn = CString -> IO ()

foreign import ccall "dynamic"
    mkParseFn :: FunPtr ParseFn -> ParseFn

foreign import ccall "dynamic"
    mkFreeFn :: FunPtr FreeFn -> FreeFn

data RustSyntaxLibrary = RustSyntaxLibrary
    { rslHandle :: DL
    , rslParseProgram :: ParseFn
    , rslParseStmt :: ParseFn
    , rslParseExpr :: ParseFn
    , rslParseType :: ParseFn
    , rslFreeString :: FreeFn
    }

withRustSyntaxLibrary :: FilePath -> (RustSyntaxLibrary -> IO a) -> IO a
withRustSyntaxLibrary path = bracket (loadRustSyntaxLibrary path) (dlclose . rslHandle)

parseProgramWithRust ::
    RustSyntaxLibrary -> FilePath -> T.Text -> IO (RustParseResponse Prog)
parseProgramWithRust lib fileName input =
    callRustParser
        decodeProgramResponseText
        (rslParseProgram lib)
        (rslFreeString lib)
        fileName
        input

parseStmtWithRust ::
    RustSyntaxLibrary -> FilePath -> T.Text -> IO (RustParseResponse Stmt)
parseStmtWithRust lib fileName input =
    callRustParser
        decodeStmtResponseText
        (rslParseStmt lib)
        (rslFreeString lib)
        fileName
        input

parseExprWithRust ::
    RustSyntaxLibrary -> FilePath -> T.Text -> IO (RustParseResponse Expr)
parseExprWithRust lib fileName input =
    callRustParser
        decodeExprResponseText
        (rslParseExpr lib)
        (rslFreeString lib)
        fileName
        input

parseTypeWithRust ::
    RustSyntaxLibrary -> FilePath -> T.Text -> IO (RustParseResponse Type)
parseTypeWithRust lib fileName input =
    callRustParser
        decodeTypeResponseText
        (rslParseType lib)
        (rslFreeString lib)
        fileName
        input

loadRustSyntaxLibrary :: FilePath -> IO RustSyntaxLibrary
loadRustSyntaxLibrary path = do
    handle <- dlopen path [RTLD_NOW, RTLD_LOCAL]
    freeString <- mkFreeFn <$> dlsym handle "reussir_syntax_string_free"
    RustSyntaxLibrary handle
        <$> loadParseFn handle "reussir_syntax_parse_program_json"
        <*> loadParseFn handle "reussir_syntax_parse_stmt_json"
        <*> loadParseFn handle "reussir_syntax_parse_expr_json"
        <*> loadParseFn handle "reussir_syntax_parse_type_json"
        <*> pure freeString
  where
    loadParseFn handle symbol = mkParseFn <$> dlsym handle symbol

callRustParser ::
    (T.Text -> Either String a) -> ParseFn -> FreeFn -> FilePath -> T.Text -> IO a
callRustParser decode parseFn freeFn fileName input =
    withCString (T.unpack input) $ \inputPtr ->
        withCString fileName $ \fileNamePtr -> do
            resultPtr <- parseFn inputPtr fileNamePtr
            resultBytes <- BS.packCString resultPtr
            freeFn resultPtr
            case decode (TE.decodeUtf8 resultBytes) of
                Right parsed -> pure parsed
                Left err -> throwIO $ userError ("failed to decode Rust parser JSON: " <> err)
