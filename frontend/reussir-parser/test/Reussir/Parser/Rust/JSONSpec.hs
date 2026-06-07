{-# LANGUAGE OverloadedStrings #-}

module Reussir.Parser.Rust.JSONSpec (spec) where

import Test.Hspec

import Reussir.Parser.Rust.JSON
import Reussir.Parser.Types.Expr
import Reussir.Parser.Types.Lexer (Identifier (..), Path (..), WithSpan (..))
import Reussir.Parser.Types.Stmt
import Reussir.Parser.Types.Type

spec :: Spec
spec = do
    describe "Rust parser JSON response decoding" $ do
        it "decodes a serialized Rust program into the Haskell parser AST" $ do
            let json =
                    "{\"ok\":true,\"value\":[{\"spanned\":{\"value\":{\"function\":{\"visibility\":\"public\",\"name\":\"id\",\"generics\":[],\"params\":[[\"x\",{\"spanned\":{\"value\":{\"integral\":{\"signed\":32}},\"span\":{\"start\":12,\"end\":15}}},false]],\"return_type\":[{\"spanned\":{\"value\":{\"integral\":{\"signed\":32}},\"span\":{\"start\":20,\"end\":23}}},false],\"is_regional\":false,\"body\":{\"seq\":[{\"spanned\":{\"value\":{\"var\":{\"basename\":\"x\",\"segments\":[]}},\"span\":{\"start\":26,\"end\":27}}}]}}},\"span\":{\"start\":0,\"end\":29}}}]}"
            decodeProgramResponseText json
                `shouldBe` Right
                    ( RustParseOk
                        [ SpannedStmt $
                            WithSpan
                                ( FunctionStmt $
                                    Function
                                        Public
                                        (Identifier "id")
                                        []
                                        [ (Identifier "x", TypeSpanned (WithSpan (TypeIntegral (Signed 32)) 12 15), False)
                                        ]
                                        (Just (TypeSpanned (WithSpan (TypeIntegral (Signed 32)) 20 23), False))
                                        False
                                        (Just (ExprSeq [SpannedExpr (WithSpan (Var (Path (Identifier "x") [])) 26 27)]))
                                )
                                0
                                29
                        ]
                    )

        it "decodes a serialized Rust arrow type" $ do
            let json =
                    "{\"ok\":true,\"value\":{\"spanned\":{\"value\":{\"arrow\":{\"args\":[{\"integral\":{\"signed\":32}}],\"ret\":\"bool\"}},\"span\":{\"start\":0,\"end\":11}}}}"
            decodeTypeResponseText json
                `shouldBe` Right
                    ( RustParseOk $
                        TypeSpanned $
                            WithSpan
                                (TypeArrow [TypeIntegral (Signed 32)] TypeBool)
                                0
                                11
                    )

        it "decodes a serialized Rust diagnostic response" $ do
            decodeProgramResponseText "{\"ok\":false,\"diagnostic\":\"syntax error\"}"
                `shouldBe` Right (RustParseError "syntax error")
