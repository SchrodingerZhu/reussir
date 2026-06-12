{-# LANGUAGE MultilineStrings #-}
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
                """
                fn f(x: i32, y: i32) -> i32 {
                    let z = x * 2 + y % 3;
                    if (z >= 0 && !(z == 4)) { z } else { -z }
                }
                """

        it "roundtrips generic records, enums and pattern matching" $
            roundtrips
                """
                enum List<T> {
                    Nil,
                    Cons(T, List<T>)
                }
                struct Pair<A, B> { first: A, second: B }
                fn append(a : List<i32>, b : List<i32>) -> List<i32> {
                    match a {
                        List::Nil => b,
                        List::Cons(x, xs) => List::Cons{x, append(xs, b)}
                    }
                }
                """

        it "roundtrips closures and closure calls" $
            roundtrips
                """
                fn abstract() -> bool -> u64 {
                    let u = 1;
                    |x| if (x) { 0 } else { u }
                }
                fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }
                """

        it "roundtrips regional records and assignments" $
            roundtrips
                """
                struct [regional] DLLink<T> {
                    val: T,
                    next: [field] DLLink<T>,
                    prev: [field] DLLink<T>
                }
                regional fn new<T>(val: T) -> [flex] DLLink<T> {
                    DLLink { val: val, next: Nullable::Null {}, prev: Nullable::Null {} }
                }
                fn foo() -> DLLink<i32> {
                    regional {
                        let a = new(1);
                        a->next := Nullable::NonNull{a};
                        a
                    }
                }
                """

        it "roundtrips extern trampolines and module statements" $
            roundtrips
                """
                pub mod utils;
                fn fibonacci<T>(n: T) -> T { n }
                extern "C" trampoline "fibonacci_ffi" = fibonacci<u64>;
                """

        it "roundtrips constants, casts, strings and access chains" $
            roundtrips
                """
                fn g(b: bool) -> str {
                    match b {
                        true => "yes",
                        false => "no"
                    }
                }
                fn h(p: Pair<i32, f64>) -> f64 {
                    let x = p.first;
                    (x as f64) + 1.5
                }
                """

        it "roundtrips patterns with guards, wildcards and ellipsis" $
            roundtrips
                """
                fn classify(p: Pair<i32, i32>) -> i32 {
                    match p {
                        Pair { first: 0, .. } => 0,
                        Pair { first: x, second: y } if x > y => 1,
                        _ => 2
                    }
                }
                """
