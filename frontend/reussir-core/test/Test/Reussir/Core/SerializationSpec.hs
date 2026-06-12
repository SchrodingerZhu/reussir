{-# LANGUAGE OverloadedStrings #-}

{- | Roundtrip tests for the JSON serialization of the semi and full syntax.

The tests run the real elaboration pipeline on a sample program, snapshot the
resulting contexts, and check that

* decoding the JSON encoding of a snapshot yields the same snapshot, and
* restoring a context from a snapshot and re-snapshotting it yields the same
  snapshot (i.e. suspend/recover loses no information).
-}
module Test.Reussir.Core.SerializationSpec (spec) where

import Control.Monad (forM_)
import Data.Aeson (eitherDecode, encode)
import Data.Digest.XXHash.FFI (XXH3 (..))
import Effectful (inject, liftIO, runEff)
import Effectful.Prim (runPrim)
import Effectful.State.Static.Local (runState)
import Log (LogLevel (..))
import Reussir.Parser.Prog (Prog, parseProg)
import Reussir.Parser.Types.Lexer (WithSpan (..))
import System.Directory (getTemporaryDirectory, removeFile)
import Test.Hspec
import Text.Megaparsec (errorBundlePretty, runParser)

import Data.HashMap.Strict qualified as HashMap
import Data.Sequence qualified as Seq
import Data.Text qualified as T
import Effectful.Log qualified as L
import Log.Backend.StandardOutput qualified as L
import Reussir.Bridge qualified as B
import Reussir.Parser.Types.Stmt qualified as Syn

import Reussir.Core.Data.Semi.Expr (
    DTSwitchCases (..),
    DecisionTree (..),
    Expr (..),
    ExprKind (..),
    PatternVarRef (..),
 )
import Reussir.Core.Data.UniqueID (ExprID (..), VarID (..))
import Reussir.Core.Full.Conversion (convertCtx)
import Reussir.Core.Semi.Context (
    emptySemiContext,
    populateRecordFields,
    scanStmt,
    withModuleFile,
 )
import Reussir.Core.Semi.FlowAnalysis (solveAllGenerics)
import Reussir.Core.Semi.Tyck (checkFuncType)
import Reussir.Core.Serialization

import Reussir.Core.Data.Semi.Type qualified as Semi

sampleProgram :: T.Text
sampleProgram =
    "enum List<T> {\n\
    \    Nil,\n\
    \    Cons(T, List<T>)\n\
    \}\n\
    \struct Pair { first: i32, second: i32 }\n\
    \fn append(a : List<i32>, b : List<i32>) -> List<i32> {\n\
    \    match a {\n\
    \        List::Nil => b,\n\
    \        List::Cons(x, xs) => List::Cons{x, append(xs, b)}\n\
    \    }\n\
    \}\n\
    \fn apply(f: i32 -> i32, x: i32) -> i32 { f(x) }\n\
    \fn inc(x: i32) -> i32 {\n\
    \    apply(|v: i32| v + 1, x)\n\
    \}\n\
    \fn choose(b: bool) -> str {\n\
    \    if (b) { \"yes\" } else { \"no\" }\n\
    \}"

unspanStmt :: Syn.Stmt -> Syn.Stmt
unspanStmt (Syn.SpannedStmt (WithSpan s _ _)) = unspanStmt s
unspanStmt stmt = stmt

parseSample :: T.Text -> IO Prog
parseSample src = case runParser parseProg "<test>" src of
    Left err -> fail (errorBundlePretty err)
    Right prog -> pure prog

{- | Run the semi and full elaboration pipeline on a program and snapshot the
context after each stage.
-}
elaborateSnapshots ::
    T.Text ->
    IO (SemiContextSnapshot, GenericSolutionSnapshot, FullContextSnapshot)
elaborateSnapshots src = do
    prog <- parseSample src
    L.withStdOutLogger $ \logger ->
        runEff $ L.runLog "serialization-test" logger LogAttention $ do
            initState <- runPrim $ emptySemiContext B.LogWarning "<test>" []
            (slns, finalSemi) <- runPrim $ runState initState $ do
                inject $
                    withModuleFile "<test>" [] $
                        forM_ prog $
                            \stmt -> inject $ scanStmt stmt
                inject $
                    withModuleFile "<test>" [] $
                        forM_ prog $
                            \stmt -> inject $ populateRecordFields stmt
                inject $
                    withModuleFile "<test>" [] $
                        forM_ prog $ \stmt ->
                            case unspanStmt stmt of
                                Syn.FunctionStmt f -> do
                                    _ <- inject $ checkFuncType f
                                    return ()
                                _ -> return ()
                inject solveAllGenerics
            solutions <- maybe (liftIO $ fail "generic solving failed") pure slns
            fullCtx <- runPrim $ convertCtx finalSemi solutions
            runPrim $ do
                semiSnap <- snapshotSemiContext finalSemi
                slnSnap <- snapshotGenericSolution solutions
                fullSnap <- snapshotFullContext fullCtx
                pure (semiSnap, slnSnap, fullSnap)

spec :: Spec
spec = do
    (semiSnap, slnSnap, fullSnap) <- runIO (elaborateSnapshots sampleProgram)

    describe "sample elaboration" $ do
        it "elaborates without failures" $ do
            scHasFailed semiSnap `shouldBe` False
            scReports semiSnap `shouldBe` []
            fcErrors fullSnap `shouldBe` []

        it "produces a non-trivial context" $ do
            scFunctions semiSnap `shouldNotBe` []
            scRecords semiSnap `shouldNotBe` []
            fcFunctions fullSnap `shouldNotBe` []
            fcRecords fullSnap `shouldNotBe` []
            scStrings semiSnap `shouldNotBe` []

    describe "surface program suspension" $ do
        it "roundtrips through JSON" $ do
            prog <- parseSample sampleProgram
            let surface = [SurfaceModule "<test>" [] prog]
            decodeSurfaceProgram (encodeSurfaceProgram surface)
                `shouldBe` Right surface

    describe "semi context suspension" $ do
        it "snapshot roundtrips through JSON" $
            eitherDecode (encode semiSnap) `shouldBe` Right semiSnap

        it "restore/re-snapshot is lossless" $ do
            resnap <-
                runEff . runPrim $
                    restoreSemiContext semiSnap >>= snapshotSemiContext
            resnap `shouldBe` semiSnap

        it "suspends to and recovers from a file" $ do
            tmpDir <- getTemporaryDirectory
            let path = tmpDir <> "/reussir-semi-suspend-test.json"
            recovered <- runEff . runPrim $ do
                ctx <- restoreSemiContext semiSnap
                suspendSemiContext path ctx
                recoverSemiContext path
            resnap <-
                either
                    (fail . ("recovery failed: " <>))
                    (runEff . runPrim . snapshotSemiContext)
                    recovered
            removeFile path
            resnap `shouldBe` semiSnap

    describe "generic solution suspension" $ do
        it "snapshot roundtrips through JSON" $
            eitherDecode (encode slnSnap) `shouldBe` Right slnSnap

        it "restore/re-snapshot is lossless" $ do
            resnap <-
                runEff . runPrim $
                    restoreGenericSolution slnSnap >>= snapshotGenericSolution
            resnap `shouldBe` slnSnap

    describe "full context suspension" $ do
        it "snapshot roundtrips through JSON" $
            eitherDecode (encode fullSnap) `shouldBe` Right fullSnap

        it "restore/re-snapshot is lossless" $ do
            resnap <-
                runEff . runPrim $
                    restoreFullContext fullSnap >>= snapshotFullContext
            resnap `shouldBe` fullSnap

    describe "pipeline continuation after recovery" $ do
        it "full conversion from a recovered semi context matches the original" $ do
            fullResnap <- L.withStdOutLogger $ \logger ->
                runEff $ L.runLog "serialization-test" logger LogAttention $ runPrim $ do
                    ctx <- restoreSemiContext semiSnap
                    solutions <- restoreGenericSolution slnSnap
                    fullCtx <- convertCtx ctx solutions
                    snapshotFullContext fullCtx
            fullResnap `shouldBe` fullSnap

    describe "semi expression corner cases" $ do
        it "roundtrips decision trees with string switches" $ do
            let leaf = mkSemiExpr (Constant 1)
                tree =
                    DTSwitch
                        { dtSwitchVarRef = PatternVarRef (Seq.fromList [0, 1])
                        , dtSwitchCases =
                            DTSwitchString
                                { dtSwitchStringMap =
                                    HashMap.fromList
                                        [ (XXH3 "hello", DTLeaf leaf mempty)
                                        , (XXH3 "world", DTUnreachable)
                                        ]
                                , dtSwitchStringDefault = DTUncovered
                                }
                        }
                expr = mkSemiExpr (Match (mkSemiExpr (Var (VarID 0))) tree)
            eitherDecode (encode expr) `shouldBe` Right expr

mkSemiExpr :: ExprKind -> Expr
mkSemiExpr kind =
    Expr
        { exprKind = kind
        , exprSpan = Just (3, 14)
        , exprType = Semi.TypeStr
        , exprID = ExprID 42
        }
