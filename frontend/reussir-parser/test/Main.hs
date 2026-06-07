module Main (main) where

import Test.Tasty
import Test.Tasty.Hspec (testSpec)

import Reussir.Parser.ExprSpec qualified as ExprSpec
import Reussir.Parser.LexerSpec qualified as LexerSpec
import Reussir.Parser.Rust.FFISpec qualified as RustFFISpec
import Reussir.Parser.Rust.JSONSpec qualified as RustJSONSpec
import Reussir.Parser.RustSpec qualified as RustSpec
import Reussir.Parser.StmtSpec qualified as StmtSpec
import Reussir.Parser.TypeSpec qualified as TypeSpec

main :: IO ()
main = do
    lexerSpec <- testSpec "Reussir.Parser.Lexer" LexerSpec.spec
    typeSpec <- testSpec "Reussir.Parser.Type" TypeSpec.spec
    stmtSpec <- testSpec "Reussir.Parser.Stmt" StmtSpec.spec
    exprSpec <- testSpec "Reussir.Parser.Expr" ExprSpec.spec
    rustSpec <- testSpec "Reussir.Parser.Rust" RustSpec.spec
    rustJSONSpec <- testSpec "Reussir.Parser.Rust.JSON" RustJSONSpec.spec
    rustFFISpec <- testSpec "Reussir.Parser.Rust.FFI" RustFFISpec.spec
    defaultMain
        ( testGroup
            "Reussir Parser Tests"
            [lexerSpec, typeSpec, stmtSpec, exprSpec, rustSpec, rustJSONSpec, rustFFISpec]
        )
