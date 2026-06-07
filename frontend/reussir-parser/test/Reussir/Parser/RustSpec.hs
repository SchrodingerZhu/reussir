{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.RustSpec (spec) where

import Control.Exception (bracket)
import Data.Either (isLeft, isRight)
import System.Environment (lookupEnv, setEnv, unsetEnv)
import Test.Hspec

import Reussir.Parser.Prog (ReplInput (..))
import Reussir.Parser.Rust (
    findRustSyntaxLibrary,
    parseProgIO,
    parseReplInputIO,
    parseTypeIO,
 )

import Reussir.Parser.Types.Expr qualified as Expr

spec :: Spec
spec = do
    describe "Rust parser facade" $ do
        it
            "falls back to the Haskell parser in explicit rust-first mode when the Rust library is unavailable" $
            withEnv "REUSSIR_PARSER_BACKEND" (Just "rust-first") $
                withEnv "REUSSIR_SYNTAX_LIB" (Just "/tmp/reussir-missing-syntax-library.so") $ do
                    result <- parseProgIO "mod.rr" "mod math;"
                    result `shouldSatisfy` isRight

        it "reports a missing Rust library when the Rust backend is forced" $
            withEnv "REUSSIR_PARSER_BACKEND" (Just "rust") $
                withEnv "REUSSIR_SYNTAX_LIB" (Just "/tmp/reussir-missing-syntax-library.so") $ do
                    result <- parseTypeIO "type.rr" "i32"
                    result `shouldSatisfy` isLeft

        it "reports a missing Rust library by default" $
            withEnv "REUSSIR_PARSER_BACKEND" Nothing $
                withEnv "REUSSIR_SYNTAX_LIB" (Just "/tmp/reussir-missing-syntax-library.so") $ do
                    result <- parseTypeIO "type.rr" "i32"
                    result `shouldSatisfy` isLeft

        it "uses the Rust parser by default when the syntax library is available" $ do
            mLib <- findRustSyntaxLibrary
            case mLib of
                Nothing -> pendingWith "build libreussir_syntax to run default Rust facade smoke tests"
                Just _ ->
                    withEnv "REUSSIR_PARSER_BACKEND" Nothing $ do
                        result <- parseReplInputIO "<repl>" "id(1).field"
                        case result of
                            Right (ReplExpr (Expr.AccessChain{})) -> pure ()
                            other -> expectationFailure $ "unexpected default Rust parse result: " <> show other

        it "treats comment-only REPL input as an empty line" $ do
            result <- parseReplInputIO "<repl>" "// CHECK: 1 : i64"
            result `shouldBe` Right EmptyLine

        it "parses a representative program when the Rust backend is forced" $ do
            mLib <- findRustSyntaxLibrary
            case mLib of
                Nothing -> pendingWith "build libreussir_syntax to run forced-Rust facade smoke tests"
                Just _ ->
                    withEnv "REUSSIR_PARSER_BACKEND" (Just "rust") $ do
                        result <-
                            parseProgIO
                                "program.rr"
                                "pub enum Option<T> { Some(T), None }\npub fn id(x: i32) -> i32 { x }\nmod math;"
                        result `shouldSatisfy` isRight

        it "parses match scrutinees without treating case braces as constructor arguments" $ do
            mLib <- findRustSyntaxLibrary
            case mLib of
                Nothing -> pendingWith "build libreussir_syntax to run forced-Rust facade smoke tests"
                Just _ ->
                    withEnv "REUSSIR_PARSER_BACKEND" (Just "rust") $ do
                        result <- parseProgIO "match.rr" "fn test(x: i32) -> i32 { match x { _ => 1 } }"
                        result `shouldSatisfy` isRight

        it "parses REPL expressions when the Rust backend is forced" $ do
            mLib <- findRustSyntaxLibrary
            case mLib of
                Nothing -> pendingWith "build libreussir_syntax to run forced-Rust facade smoke tests"
                Just _ ->
                    withEnv "REUSSIR_PARSER_BACKEND" (Just "rust") $ do
                        result <- parseReplInputIO "<repl>" "id(1).field"
                        case result of
                            Right (ReplExpr (Expr.AccessChain{})) -> pure ()
                            other -> expectationFailure $ "unexpected Rust REPL parse result: " <> show other

withEnv :: String -> Maybe String -> IO a -> IO a
withEnv name value = bracket save restore . const
  where
    save = do
        old <- lookupEnv name
        case value of
            Just new -> setEnv name new
            Nothing -> unsetEnv name
        pure old
    restore old = case old of
        Just oldValue -> setEnv name oldValue
        Nothing -> unsetEnv name
