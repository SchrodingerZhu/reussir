{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.Rust.FFISpec (spec) where

import System.Directory (doesFileExist)
import System.Environment (lookupEnv)
import Test.Hspec

import Reussir.Parser.Rust.FFI
import Reussir.Parser.Rust.JSON
import Reussir.Parser.Types.Lexer (WithSpan (..))
import Reussir.Parser.Types.Type

spec :: Spec
spec = do
    describe "Rust syntax FFI" $ do
        it
            "loads the Rust syntax library and decodes a type parse result when available"
            $ do
                mLib <- findSyntaxLibrary
                case mLib of
                    Nothing ->
                        pendingWith
                            "set REUSSIR_SYNTAX_LIB or build target/debug/libreussir_syntax.so to run this smoke test"
                    Just libPath ->
                        withRustSyntaxLibrary libPath $ \lib -> do
                            result <- parseTypeWithRust lib "type.rr" "i32 -> bool"
                            result
                                `shouldBe` RustParseOk
                                    ( TypeSpanned $
                                        WithSpan
                                            (TypeArrow [TypeIntegral (Signed 32)] TypeBool)
                                            0
                                            11
                                    )

findSyntaxLibrary :: IO (Maybe FilePath)
findSyntaxLibrary = do
    mEnv <- lookupEnv "REUSSIR_SYNTAX_LIB"
    let candidates =
            maybe [] pure mEnv
                <> [ "target/debug/libreussir_syntax.so"
                   , "../../target/debug/libreussir_syntax.so"
                   , "../../../target/debug/libreussir_syntax.so"
                   ]
    firstExisting candidates

firstExisting :: [FilePath] -> IO (Maybe FilePath)
firstExisting = foldr step (pure Nothing)
  where
    step path rest = do
        exists <- doesFileExist path
        if exists then pure (Just path) else rest
