module Main where

import Test.Tasty
import Test.Tasty.Hspec (testSpec)

import Test.Reussir.Core.Class qualified as Class
import Test.Reussir.Core.Generic qualified as Generic
import Test.Reussir.Core.Semi.Mangle qualified as Mangle
import Test.Reussir.Core.Semi.ModuleResolution qualified as ModuleResolution
import Test.Reussir.Core.Semi.PatternMatch qualified as PatternMatch
import Test.Reussir.Core.Semi.TyckSpec qualified as TyckSpec
import Test.Reussir.Core.SerializationSpec qualified as SerializationSpec
import Test.Reussir.Core.String qualified as String

main :: IO ()
main = do
    tyckSpec <- testSpec "Reussir.Core.Semi.Tyck" TyckSpec.spec
    serializationSpec <-
        testSpec "Reussir.Core.Serialization" SerializationSpec.spec
    defaultMain (tests tyckSpec serializationSpec)

tests :: TestTree -> TestTree -> TestTree
tests tyckSpec serializationSpec =
    testGroup
        "Reussir.Core"
        [ Generic.tests
        , Class.tests
        , String.tests
        , Mangle.tests
        , ModuleResolution.tests
        , PatternMatch.tests
        , tyckSpec
        , serializationSpec
        ]
