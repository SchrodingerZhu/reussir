module Main (main) where

import Test.Tasty
import Test.Tasty.Hspec

import Reussir.Parser.ExprSpec qualified as ExprSpec
import Reussir.Parser.LexerSpec qualified as LexerSpec
import Reussir.Parser.SerializationSpec qualified as SerializationSpec
import Reussir.Parser.StmtSpec qualified as StmtSpec
import Reussir.Parser.TypeSpec qualified as TypeSpec

main :: IO ()
main = do
    lexerSpec <- testSpec "Reussir.Parser.Lexer" LexerSpec.spec
    typeSpec <- testSpec "Reussir.Parser.Type" TypeSpec.spec
    stmtSpec <- testSpec "Reussir.Parser.Stmt" StmtSpec.spec
    exprSpec <- testSpec "Reussir.Parser.Expr" ExprSpec.spec
    serializationSpec <-
        testSpec "Reussir.Parser.Serialization" SerializationSpec.spec
    defaultMain
        ( testGroup
            "Reussir Parser Tests"
            [lexerSpec, typeSpec, stmtSpec, exprSpec, serializationSpec]
        )
