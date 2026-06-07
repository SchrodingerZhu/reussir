{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.Rust.FFISpec (spec) where

import Test.Hspec

import Reussir.Parser.Rust (findRustSyntaxLibrary)
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
                mLib <- findRustSyntaxLibrary
                case mLib of
                    Nothing -> pure ()
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

