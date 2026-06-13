{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

{- | Verify that a JSON surface AST (for example, one produced by the Rust
parser in @crates/reussir-syntax@) matches what the Haskell parser produces
for the same source file.

Both ASTs are normalized before comparison: @SpannedExpr@/@SpannedStmt@
wrappers are stripped and the remaining structural spans are zeroed, so the
comparison is about syntactic structure, not byte positions. ('Type' already
compares modulo @TypeSpanned@ wrappers.)

Usage: @reussir-surface-verify SOURCE.rr AST.json@. Prints @OK@ on a match;
prints the first mismatching statement and exits non-zero otherwise.
-}
module Main (main) where

import Control.Monad (forM_)
import Data.Aeson (eitherDecodeFileStrict)
import System.Environment (getArgs, getProgName)
import System.Exit (die, exitFailure)
import System.IO (hPutStrLn, stderr)
import Text.Megaparsec (errorBundlePretty, runParser)

import Data.Text.IO qualified as TIO
import Data.Vector.Strict qualified as V

import Reussir.Parser.Prog (Prog, parseProg)
import Reussir.Parser.Serialization ()
import Reussir.Parser.Types.Expr hiding (Named, Unnamed)
import Reussir.Parser.Types.Lexer (WithSpan (..))
import Reussir.Parser.Types.Stmt

main :: IO ()
main =
    getArgs >>= \case
        [srcPath, jsonPath] -> run srcPath jsonPath
        _ -> do
            name <- getProgName
            die ("usage: " <> name <> " SOURCE.rr AST.json")

run :: FilePath -> FilePath -> IO ()
run srcPath jsonPath = do
    content <- TIO.readFile srcPath
    reference <- case runParser parseProg srcPath content of
        Left err -> die (errorBundlePretty err)
        Right prog -> pure prog
    candidate :: Prog <-
        eitherDecodeFileStrict jsonPath >>= \case
            Left err -> die ("failed to decode " <> jsonPath <> ": " <> err)
            Right prog -> pure prog
    let ref = map normStmt reference
        cand = map normStmt candidate
    if ref == cand
        then putStrLn "OK"
        else do
            hPutStrLn stderr $
                "surface AST mismatch: parser produced "
                    <> show (length ref)
                    <> " statement(s), JSON contains "
                    <> show (length cand)
            forM_ (zip3 [0 :: Int ..] ref cand) $ \(idx, a, b) ->
                if a /= b
                    then do
                        hPutStrLn stderr ("first mismatch at statement " <> show idx)
                        hPutStrLn stderr ("  parser: " <> show a)
                        hPutStrLn stderr ("  json:   " <> show b)
                        exitFailure
                    else pure ()
            -- Same prefix but different lengths.
            exitFailure

zeroSpan :: WithSpan a -> WithSpan a
zeroSpan ws = ws{spanStartOffset = 0, spanEndOffset = 0}

normStmt :: Stmt -> Stmt
normStmt (SpannedStmt ws) = normStmt (spanValue ws)
normStmt (FunctionStmt f) = FunctionStmt f{funcBody = normExpr <$> funcBody f}
normStmt (RecordStmt r) = RecordStmt r{recordFields = normFields (recordFields r)}
normStmt s@(ModStmt _ _) = s
normStmt s@(ExternTrampolineStmt{}) = s

normFields :: RecordFields -> RecordFields
normFields (Named fs) = Named (V.map zeroSpan fs)
normFields (Unnamed fs) = Unnamed (V.map zeroSpan fs)
normFields (Variants vs) = Variants (V.map zeroSpan vs)

normExpr :: Expr -> Expr
normExpr (SpannedExpr ws) = normExpr (spanValue ws)
normExpr (ConstExpr c) = ConstExpr c
normExpr (BinOpExpr op a b) = BinOpExpr op (normExpr a) (normExpr b)
normExpr (UnaryOpExpr op e) = UnaryOpExpr op (normExpr e)
normExpr (If c t f) = If (normExpr c) (normExpr t) (normExpr f)
normExpr (Cast ty e) = Cast ty (normExpr e)
normExpr (Let name ty e) = Let (zeroSpan name) ty (normExpr e)
normExpr (ExprSeq es) = ExprSeq (map normExpr es)
normExpr (Lambda (LambdaExpr args body ret)) = Lambda (LambdaExpr args (normExpr body) ret)
normExpr (Match e arms) =
    Match (normExpr e) (V.map (\(p, b) -> (normPattern p, normExpr b)) arms)
normExpr (Var p) = Var p
normExpr (FuncCallExpr (FuncCall p tys args)) =
    FuncCallExpr (FuncCall p tys (map normExpr args))
normExpr (RegionalExpr e) = RegionalExpr (normExpr e)
normExpr (CtorCallExpr (CtorCall p tys args)) =
    CtorCallExpr (CtorCall p tys (map (fmap normExpr) args))
normExpr (CallExpr f args) = CallExpr (normExpr f) (map normExpr args)
normExpr (AccessChain e accs) = AccessChain (normExpr e) accs
normExpr (Assign a acc b) = Assign (normExpr a) acc (normExpr b)

normPattern :: Pattern -> Pattern
normPattern (Pattern kind guard) = Pattern kind (normExpr <$> guard)
