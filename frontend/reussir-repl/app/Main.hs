{-# LANGUAGE CApiFFI #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

-- Base libraries
import Control.Exception (SomeException, bracketOnError, catch)
import Control.Monad (forM_, when)
import Data.ByteString (ByteString)
import Data.Int (Int16, Int32, Int64, Int8)
import Data.Maybe (isNothing)
import Data.String (IsString (fromString))
import Data.Word (Word16, Word32, Word64, Word8)
-- Text

-- Megaparsec

-- Console

-- Reussir

import Effectful (runEff)
import Effectful.Prim (runPrim)
import Foreign (FunPtr, Ptr, Storable (..), alloca, nullPtr)
import Foreign.C (CChar, CSize)
import Foreign.C.String (peekCStringLen)
import Foreign.Ptr (castPtrToFunPtr)
import Options.Applicative
import Prettyprinter (defaultLayoutOptions, layoutPretty)
import Prettyprinter.Render.Terminal (renderStrict)
import Reussir.Bridge (ReussirJIT, addModule, lookupSymbol, withJIT)
import Reussir.Codegen.Context.Symbol (symbolText)
import Reussir.Core.REPL (
    ReplError (..),
    ReplState (..),
    ResultKind (..),
    addDefinition,
    compileExpression,
    initReplState,
 )
import Reussir.Parser.Prog (ReplInput (..))
import Reussir.Parser.Rust (parseReplInputIO)
import System.Console.Haskeline
import System.Console.Haskeline.IO

import Data.HashTable.IO qualified as H
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Reussir.Bridge qualified as B
import Reussir.Core.Data.Semi.Context qualified as Semi
import Reussir.Core.Data.Semi.Function qualified as SemiFunc
import Reussir.Core.Semi.Pretty qualified as SemiP
import Reussir.Parser.Types.Expr qualified as P

--------------------------------------------------------------------------------
-- Foreign function imports for different result types
--------------------------------------------------------------------------------

foreign import ccall "dynamic"
    callI64Func :: FunPtr (IO Int64) -> IO Int64

foreign import ccall "dynamic"
    callI32Func :: FunPtr (IO Int32) -> IO Int32

foreign import ccall "dynamic"
    callI16Func :: FunPtr (IO Int16) -> IO Int16

foreign import ccall "dynamic"
    callI8Func :: FunPtr (IO Int8) -> IO Int8

foreign import ccall "dynamic"
    callU64Func :: FunPtr (IO Word64) -> IO Word64

foreign import ccall "dynamic"
    callU32Func :: FunPtr (IO Word32) -> IO Word32

foreign import ccall "dynamic"
    callU16Func :: FunPtr (IO Word16) -> IO Word16

foreign import ccall "dynamic"
    callU8Func :: FunPtr (IO Word8) -> IO Word8

foreign import ccall "dynamic"
    callBoolFunc :: FunPtr (IO Word8) -> IO Word8

foreign import ccall "dynamic"
    callUnitFunc :: FunPtr (IO ()) -> IO ()

-- | Representation of the str type from Reussir (ptr, len pair)
data StrResult = StrResult {-# UNPACK #-} !(Ptr CChar) {-# UNPACK #-} !CSize

instance Storable StrResult where
    sizeOf _ = sizeOf (undefined :: Ptr CChar) + sizeOf (undefined :: CSize)
    alignment _ = alignment (undefined :: Ptr CChar)
    peek p = do
        ptr <- peekByteOff p 0
        len <- peekByteOff p $ sizeOf (undefined :: Ptr CChar)
        return $ StrResult ptr len
    poke p (StrResult ptr len) = do
        pokeByteOff p 0 ptr
        pokeByteOff p (sizeOf (undefined :: Ptr CChar)) len

-- | C helper function to call a JIT function returning a str type
foreign import capi "Reussir/Bridge.h reussir_bridge_call_str_func"
    c_reussir_bridge_call_str_func :: Ptr () -> Ptr StrResult -> IO ()

foreign import capi "Reussir/Bridge.h reussir_bridge_call_f32_func"
    c_reussir_bridge_call_f32_func :: Ptr () -> Ptr Float -> IO ()

foreign import capi "Reussir/Bridge.h reussir_bridge_call_f64_func"
    c_reussir_bridge_call_f64_func :: Ptr () -> Ptr Double -> IO ()

--------------------------------------------------------------------------------
-- Placeholder callback for lazy module loading
--------------------------------------------------------------------------------

{- | Placeholder callback for lazy module loading
This would be used if we want to do lazy compilation of ASTs
-}
placeholderCallback :: () -> IO ByteString
placeholderCallback _ = return ""

--------------------------------------------------------------------------------
-- Main REPL Entry Point
--------------------------------------------------------------------------------

-- | Command line arguments
data Args = Args
    { argOptLevel :: B.OptOption
    , argLogLevel :: B.LogLevel
    , argInputFile :: Maybe FilePath
    }

argsParser :: Parser Args
argsParser =
    Args
        <$> option
            parseOptLevel
            ( long "opt-level"
                <> short 'O'
                <> value B.OptTPDE
                <> help "Optimization level (none, default, aggressive, size, tpde)"
            )
        <*> option
            parseLogLevel
            ( long "log-level"
                <> short 'l'
                <> value B.LogWarning
                <> help "Log level (error, warning, info, debug, trace)"
            )
        <*> optional
            ( strOption
                (long "input" <> short 'i' <> help "Input file for line-by-line execution")
            )

parseOptLevel :: ReadM B.OptOption
parseOptLevel = eitherReader $ \s -> case s of
    "none" -> Right B.OptNone
    "default" -> Right B.OptDefault
    "aggressive" -> Right B.OptAggressive
    "size" -> Right B.OptSize
    "tpde" -> Right B.OptTPDE
    _ -> Left $ "Unknown optimization level: " ++ s

parseLogLevel :: ReadM B.LogLevel
parseLogLevel = eitherReader $ \s -> case s of
    "error" -> Right B.LogError
    "warning" -> Right B.LogWarning
    "info" -> Right B.LogInfo
    "debug" -> Right B.LogDebug
    "trace" -> Right B.LogTrace
    _ -> Left $ "Unknown log level: " ++ s

opts :: ParserInfo Args
opts =
    info
        (argsParser <**> helper)
        ( fullDesc
            <> progDesc "Reussir REPL"
            <> header "reussir-repl - Read-Eval-Print Loop for Reussir"
        )

-- | Main REPL entry point
main :: IO ()
main = do
    args <- execParser opts
    bracketOnError
        (initializeInput defaultSettings)
        cancelInput
        ( \hd -> do
            when (isNothing $ argInputFile args) $ do
                putStrLn "Reussir REPL v0.1.0"
                putStrLn "Type :help for available commands, :q to quit"
                putStrLn ""
            state <- initReplState (argLogLevel args) "<repl>"
            -- Use () as the AST type since we're not using lazy modules
            withJIT placeholderCallback (argOptLevel args) (argLogLevel args) $ \jit ->
                case argInputFile args of
                    Just inputFile -> do
                        content <- readFile inputFile
                        fileLoop jit state (lines content)
                    Nothing -> loop jit state hd >> closeInput hd
        )

--------------------------------------------------------------------------------
-- Command Handling
--------------------------------------------------------------------------------

-- | Check if input is a REPL command
isCommand :: String -> Bool
isCommand (':' : _) = True
isCommand _ = False

-- | Process REPL commands
processCommand ::
    ReussirJIT () ->
    ReplState ->
    String ->
    InputState ->
    IO (Maybe ReplState)
processCommand _jit state cmd hd = case words cmd of
    [":q"] -> return Nothing
    [":quit"] -> return Nothing
    [":help"] -> do
        queryInput hd $ outputStrLn helpText
        return $ Just state
    [":clear"] -> do
        -- Re-initialize the REPL state
        newState <- initReplState (replLogLevel state) (replFilePath state)
        queryInput hd $ outputStrLn "Context cleared"
        return $ Just newState
    [":dump", "context"] -> do
        dumpSemiContext state
        return $ Just state
    [":dump", "compiled"] -> do
        dumpCompiledFunctions state
        return $ Just state
    _ -> do
        queryInput hd $ outputStrLn $ "Unknown command: " ++ cmd
        queryInput hd $ outputStrLn "Type :help for available commands"
        return $ Just state

-- | Help text for REPL commands
helpText :: String
helpText =
    unlines
        [ "Available commands:"
        , "  :help           Show this help message"
        , "  :q, :quit       Exit the REPL"
        , "  :clear          Clear the context and start fresh"
        , "  :dump context   Dump the current semi-elaboration context"
        , "  :dump compiled  List compiled functions"
        , ""
        , "  :{              Begin multiline input"
        , "  }:              End multiline input"
        , ""
        , "Input is automatically parsed as either definitions or expressions."
        ]

--------------------------------------------------------------------------------
-- Main REPL Loop
--------------------------------------------------------------------------------

-- | The main REPL loop
loop :: ReussirJIT () -> ReplState -> InputState -> IO ()
loop jit state hd = do
    let prompt = "λ> "
    minput <- queryInput hd (getInputLine prompt)
    case minput of
        Nothing -> return ()
        Just "" -> loop jit state hd
        Just ":{" -> do
            multilineInput <- readMultiline hd []
            case multilineInput of
                Nothing -> return () -- EOF during multiline
                Just input -> do
                    result <- processInput jit state input
                    case result of
                        Left errMsg -> do
                            queryInput hd $ outputStrLn errMsg
                            loop jit state hd
                        Right (output, state') -> do
                            case output of
                                Just msg -> queryInput hd $ outputStrLn msg
                                Nothing -> return ()
                            loop jit state' hd
        Just input
            | isCommand input -> do
                mState <- processCommand jit state input hd
                case mState of
                    Nothing -> return () -- Exit
                    Just state' -> loop jit state' hd
            | otherwise -> do
                result <- processInput jit state input
                case result of
                    Left errMsg -> do
                        queryInput hd $ outputStrLn errMsg
                        loop jit state hd
                    Right (output, state') -> do
                        case output of
                            Just msg -> queryInput hd $ outputStrLn msg
                            Nothing -> return ()
                        loop jit state' hd

-- | Read multiline input until }:
readMultiline :: InputState -> [String] -> IO (Maybe String)
readMultiline hd acc = do
    let prompt = "| "
    minput <- queryInput hd (getInputLine prompt)
    case minput of
        Nothing -> return Nothing
        Just "}:" -> return $ Just $ unlines (reverse acc)
        Just line -> readMultiline hd (line : acc)

--------------------------------------------------------------------------------
-- File Input Loop
--------------------------------------------------------------------------------

fileLoop :: ReussirJIT () -> ReplState -> [String] -> IO ()
fileLoop _ _ [] = return ()
fileLoop jit state (line : rest)
    | null line = fileLoop jit state rest
    | line == ":{" = do
        let (block, rest') = collectMultiline rest []
        let input = unlines (reverse block)
        result <- processInput jit state input
        case result of
            Left errMsg -> do
                putStrLn errMsg
                fileLoop jit state rest'
            Right (output, state') -> do
                case output of
                    Just msg -> putStrLn msg
                    Nothing -> return ()
                fileLoop jit state' rest'
    | isCommand line = do
        -- We need a dummy InputState for processCommand, but it uses it for printing.
        -- Commands in file mode might output to stdout.
        -- Since processCommand uses `queryInput` which calls `withInterrupt . liftIO`,
        -- we can construct a specialized InputState or just mock it, OR rework processCommand.
        -- Since `processCommand` is tightly coupled with `InputState` (Haskeline), it's hard to reuse directly without a dummy.
        -- However, for simplicity, we can just execute logic directly or create a simplified handlers.

        -- IMPORTANT: processCommand uses `queryInput hd ...` to print.
        -- We should probably refactor processCommand to return IO (Maybe ReplState) and take a printer function?
        -- OR, simpler hack: we just handle commands that don't need input inside fileLoop/helper.
        -- But we want to reuse logic.

        -- Let's just handle specific commands here or reimplement simple dispatch for file mode
        -- since file mode shouldn't really be interactive.
        case words line of
            [":q"] -> return ()
            [":quit"] -> return ()
            [":help"] -> putStrLn helpText >> fileLoop jit state rest
            [":clear"] -> do
                newState <- initReplState (replLogLevel state) (replFilePath state)
                putStrLn "Context cleared"
                fileLoop jit newState rest
            [":dump", "context"] -> do
                dumpSemiContext state
                fileLoop jit state rest
            [":dump", "compiled"] -> do
                dumpCompiledFunctions state
                fileLoop jit state rest
            _ -> do
                putStrLn $ "Unknown/Unsupported command in file mode: " ++ line
                fileLoop jit state rest
    | otherwise = do
        result <- processInput jit state line
        case result of
            Left errMsg -> do
                putStrLn errMsg
                fileLoop jit state rest
            Right (output, state') -> do
                case output of
                    Just msg -> putStrLn msg
                    Nothing -> return ()
                fileLoop jit state' rest

-- | Collect lines until }: for multiline blocks in file mode
collectMultiline :: [String] -> [String] -> ([String], [String])
collectMultiline [] acc = (acc, [])
collectMultiline (l : ls) acc
    | l == "}:" = (acc, ls)
    | otherwise = collectMultiline ls (l : acc)

--------------------------------------------------------------------------------
-- Dump Helpers
--------------------------------------------------------------------------------

dumpSemiContext :: ReplState -> IO ()
dumpSemiContext state = do
    let semiCtx = replSemiContext state

    putStrLn "=== Records ==="
    records <- H.toList (Semi.knownRecords semiCtx)
    forM_ records $ \(_, rec') -> do
        doc <- runEff $ runPrim $ SemiP.prettyColored rec'
        TIO.putStrLn $ renderStrict $ layoutPretty defaultLayoutOptions doc

    putStrLn "\n=== Functions ==="
    funcs <- H.toList (SemiFunc.functionProtos $ Semi.functions semiCtx)
    forM_ funcs $ \(_, func) -> do
        doc <- runEff $ runPrim $ SemiP.prettyColored func
        TIO.putStrLn $ renderStrict $ layoutPretty defaultLayoutOptions doc

dumpCompiledFunctions :: ReplState -> IO ()
dumpCompiledFunctions state = do
    putStrLn "=== Compiled Functions ==="
    funcs <- H.toList (replCompiledFunctions state)
    let names = map (symbolText . fst) funcs
    mapM_ (TIO.putStrLn . (" - " <>)) names

--------------------------------------------------------------------------------
-- Input Processing
--------------------------------------------------------------------------------

-- | Process a single line of REPL input based on current mode
processInput ::
    ReussirJIT () ->
    ReplState ->
    String ->
    IO (Either String (Maybe String, ReplState))
processInput jit state input =
    catch
        (processAutoDetect jit state input)
        (\(e :: SomeException) -> return $ Left $ "Error: " ++ show e)

-- | Auto-detect whether input is a statement or expression
processAutoDetect ::
    ReussirJIT () ->
    ReplState ->
    String ->
    IO (Either String (Maybe String, ReplState))
processAutoDetect jit state input = do
    parsed <- parseReplInputIO "<repl>" (T.pack input)
    case parsed of
        Left err -> return $ Left $ T.unpack err
        Right (ReplStmt stmt) -> do
            result <- addDefinition state stmt
            case result of
                Left (TypeCheckError msg) -> return $ Left $ "Type error: " ++ msg
                Left (ParseError msg) -> return $ Left $ "Parse error: " ++ msg
                Left (CompilationError msg) -> return $ Left $ "Compilation error: " ++ msg
                Left (ElaborationError msg) -> return $ Left $ "Elaboration error: " ++ msg
                Right state' -> return $ Right (Just "Definition added.", state')
        Right (ReplExpr expr) -> do
            processExpressionParsed jit state input expr
        Right EmptyLine -> return $ Right (Nothing, state)

-- | Process a pre-parsed expression
processExpressionParsed ::
    ReussirJIT () ->
    ReplState ->
    String ->
    P.Expr ->
    IO (Either String (Maybe String, ReplState))
processExpressionParsed jit state input expr = do
    result <- compileExpression state (T.pack input) expr
    case result of
        Left (TypeCheckError msg) -> return $ Left $ "Type error: " ++ msg
        Left (ParseError msg) -> return $ Left $ "Parse error: " ++ msg
        Left (CompilationError msg) -> return $ Left $ "Compilation error: " ++ msg
        Left (ElaborationError msg) -> return $ Left $ "Elaboration error: " ++ msg
        Right (moduleBytes, state', resultKind) -> do
            -- JIT compile and execute
            -- Use replCounter - 1 since compileExpression increments after generating the name
            let exprCounter = fromIntegral (replCounter state' - 1)
            executeResult <- executeModule jit moduleBytes exprCounter resultKind
            case executeResult of
                Left msg -> return $ Left msg
                Right resultStr -> return $ Right (Just resultStr, state')

--------------------------------------------------------------------------------
-- JIT Execution
--------------------------------------------------------------------------------

-- | Execute a compiled module in the JIT and return the result
executeModule ::
    ReussirJIT () ->
    ByteString ->
    Int64 ->
    ResultKind ->
    IO (Either String String)
executeModule jit moduleBytes n resultKind = do
    let funcName = "__repl_expr_" ++ show n ++ "_trampoline"
    let packedName = fromString funcName
    success <- addModule jit moduleBytes
    if success
        then do
            sym <- lookupSymbol jit packedName False
            if sym /= nullPtr
                then do
                    result <- executeWithResultKind sym resultKind
                    return $ Right result
                else return $ Left $ "Symbol not found: " ++ funcName
        else return $ Left "Failed to add module to JIT"

-- | Execute a function pointer based on the result kind and format output
executeWithResultKind :: Ptr () -> ResultKind -> IO String
executeWithResultKind sym resultKind = case resultKind of
    ResultI8 -> do
        result <- callI8Func (castPtrToFunPtr sym)
        return $ show result ++ " : i8"
    ResultI16 -> do
        result <- callI16Func (castPtrToFunPtr sym)
        return $ show result ++ " : i16"
    ResultI32 -> do
        result <- callI32Func (castPtrToFunPtr sym)
        return $ show result ++ " : i32"
    ResultI64 -> do
        result <- callI64Func (castPtrToFunPtr sym)
        return $ show result ++ " : i64"
    ResultU8 -> do
        result <- callU8Func (castPtrToFunPtr sym)
        return $ show result ++ " : u8"
    ResultU16 -> do
        result <- callU16Func (castPtrToFunPtr sym)
        return $ show result ++ " : u16"
    ResultU32 -> do
        result <- callU32Func (castPtrToFunPtr sym)
        return $ show result ++ " : u32"
    ResultU64 -> do
        result <- callU64Func (castPtrToFunPtr sym)
        return $ show result ++ " : u64"
    ResultF16 -> do
        -- F16 is typically not directly supported in C ABI, use F32 storage.
        alloca $ \resultPtr -> do
            c_reussir_bridge_call_f32_func sym resultPtr
            result <- peek resultPtr
            return $ show result ++ " : f16"
    ResultF32 -> do
        alloca $ \resultPtr -> do
            c_reussir_bridge_call_f32_func sym resultPtr
            result <- peek resultPtr
            return $ show result ++ " : f32"
    ResultF64 -> do
        alloca $ \resultPtr -> do
            c_reussir_bridge_call_f64_func sym resultPtr
            result <- peek resultPtr
            return $ show result ++ " : f64"
    ResultBool -> do
        result <- callBoolFunc (castPtrToFunPtr sym)
        return $ (if result /= 0 then "true" else "false") ++ " : bool"
    ResultUnit -> do
        callUnitFunc (castPtrToFunPtr sym)
        return "()"
    ResultStr -> do
        -- String is returned as a struct { ptr, len }
        -- Use the C helper function to call the JIT'd function and capture the result
        alloca $ \resultPtr -> do
            c_reussir_bridge_call_str_func sym resultPtr
            StrResult ptrWord lenWord <- peek resultPtr
            strContent <- peekCStringLen (ptrWord, fromIntegral lenWord)
            return $ show strContent ++ " : str"
    ResultOther tyName -> do
        -- For non-primitive types, we can't easily print the value
        -- Just indicate the expression was evaluated
        _ <- callI64Func (castPtrToFunPtr sym) -- Execute to trigger side effects
        return $ "expr evaluated as " ++ T.unpack tyName
