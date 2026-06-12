{-# LANGUAGE OverloadedStrings #-}

-- | Roundtrip tests for the JSON serialization of the surface syntax.
module Reussir.Parser.SerializationSpec (spec) where

import Data.Aeson (eitherDecode, encode)
import Test.Hspec
import Text.Megaparsec (errorBundlePretty, runParser)

import Data.Text qualified as T

import Reussir.Parser.Prog (parseProg)
import Reussir.Parser.Serialization ()

{- | Parse a program and require that decoding its JSON encoding yields the
exact same AST (spans included).
-}
roundtrips :: T.Text -> Expectation
roundtrips src = case runParser parseProg "<test>" src of
    Left err -> expectationFailure (errorBundlePretty err)
    Right prog -> eitherDecode (encode prog) `shouldBe` Right prog

spec :: Spec
spec = do
    describe "surface syntax JSON roundtrip" $ do
        it "roundtrips functions with let/if and operators" $
            roundtrips
                "fn f(x: i32, y: i32) -> i32 {\n\
                \    let z = x * 2 + y % 3;\n\
                \    if (z >= 0 && !(z == 4)) { z } else { -z }\n\
                \}"

        it "roundtrips generic records, enums and pattern matching" $
            roundtrips
                "enum List<T> {\n\
                \    Nil,\n\
                \    Cons(T, List<T>)\n\
                \}\n\
                \struct Pair<A, B> { first: A, second: B }\n\
                \fn append(a : List<i32>, b : List<i32>) -> List<i32> {\n\
                \    match a {\n\
                \        List::Nil => b,\n\
                \        List::Cons(x, xs) => List::Cons{x, append(xs, b)}\n\
                \    }\n\
                \}"

        it "roundtrips closures and closure calls" $
            roundtrips
                "fn abstract() -> bool -> u64 {\n\
                \    let u = 1;\n\
                \    |x| if (x) { 0 } else { u }\n\
                \}\n\
                \fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }"

        it "roundtrips regional records and assignments" $
            roundtrips
                "struct [regional] DLLink<T> {\n\
                \    val: T,\n\
                \    next: [field] DLLink<T>,\n\
                \    prev: [field] DLLink<T>\n\
                \}\n\
                \regional fn new<T>(val: T) -> [flex] DLLink<T> {\n\
                \    DLLink { val: val, next: Nullable::Null {}, prev: Nullable::Null {} }\n\
                \}\n\
                \fn foo() -> DLLink<i32> {\n\
                \    regional {\n\
                \        let a = new(1);\n\
                \        a->next := Nullable::NonNull{a};\n\
                \        a\n\
                \    }\n\
                \}"

        it "roundtrips extern trampolines and module statements" $
            roundtrips
                "pub mod utils;\n\
                \fn fibonacci<T>(n: T) -> T { n }\n\
                \extern \"C\" trampoline \"fibonacci_ffi\" = fibonacci<u64>;"

        it "roundtrips constants, casts, strings and access chains" $
            roundtrips
                "fn g(b: bool) -> str {\n\
                \    match b {\n\
                \        true => \"yes\",\n\
                \        false => \"no\"\n\
                \    }\n\
                \}\n\
                \fn h(p: Pair<i32, f64>) -> f64 {\n\
                \    let x = p.first;\n\
                \    (x as f64) + 1.5\n\
                \}"

        it "roundtrips patterns with guards, wildcards and ellipsis" $
            roundtrips
                "fn classify(p: Pair<i32, i32>) -> i32 {\n\
                \    match p {\n\
                \        Pair { first: 0, .. } => 0,\n\
                \        Pair { first: x, second: y } if x > y => 1,\n\
                \        _ => 2\n\
                \    }\n\
                \}"
