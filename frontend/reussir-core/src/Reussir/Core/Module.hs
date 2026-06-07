{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{- |
Module      : Reussir.Core.Module
Description : Declaration-driven module resolution for packages.

This module implements Rust-like module resolution driven by @mod@ declarations
in the source code. Given a package root and name, it parses the root @lib.rr@,
collects @mod@ declarations, resolves them to file paths, and recursively
processes sub-modules.

== Module Path Convention

@
\/root\/lib.rr            → [pkgName]          (root file, always included)
mod math;  in lib.rr      → [pkgName, "math"]  (resolves math.rr or math\/mod.rr)
mod utils; in lib.rr      → [pkgName, "utils"] (resolves utils.rr or utils\/mod.rr)
mod sub;   in utils/mod.rr→ [pkgName, "utils", "sub"]
@
-}
module Reussir.Core.Module (
    DiscoveryError (..),
    PackageInfo (..),
    renderDiscoveryError,
    validatePackageInfo,
    computeModulePath,
    discoverPackageFiles,
) where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Except (ExceptT, runExceptT, throwE)
import Control.Monad.Trans.State.Strict (StateT, evalStateT, get, modify')
import Data.Int (Int64)
import Data.List (intercalate, sort)
import Reussir.Parser.Rust (parseProgIO)
import Reussir.Parser.Types.Lexer (Identifier (..), WithSpan (..))
import System.Directory (
    canonicalizePath,
    doesFileExist,
 )
import System.FilePath (
    dropExtension,
    makeRelative,
    splitDirectories,
    takeDirectory,
    (</>),
 )

import Data.Set qualified as Set
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Reussir.Parser.Types.Stmt qualified as Syn

-- | Information about a package being compiled.
data PackageInfo = PackageInfo
    { packageRoot :: FilePath
    -- ^ Absolute path to the package root directory
    , packageName :: Identifier
    -- ^ The root name of the package (e.g., "mylib")
    }

data DiscoveryError
    = MissingModuleFile
        { missingModuleName :: Identifier
        , missingModuleDeclaringFile :: FilePath
        , missingModuleDeclaringSpan :: Maybe (Int64, Int64)
        , missingModuleCandidates :: [FilePath]
        }
    | ModuleParseError
        { moduleParseFile :: FilePath
        , moduleParseDetails :: String
        }

renderDiscoveryError :: DiscoveryError -> String
renderDiscoveryError MissingModuleFile{..} =
    "Error: module '"
        <> T.unpack (unIdentifier missingModuleName)
        <> "' declared in "
        <> missingModuleDeclaringFile
        <> spanSuffix
        <> " has no source file. Expected one of: "
        <> intercalate ", " missingModuleCandidates
  where
    spanSuffix = case missingModuleDeclaringSpan of
        Just (start, end) -> " (byte offsets " <> show start <> "-" <> show end <> ")"
        Nothing -> ""
renderDiscoveryError ModuleParseError{..} =
    "Error: failed to parse discovered module file "
        <> moduleParseFile
        <> "\n"
        <> moduleParseDetails

-- | Reject package names that would collide with compiler-defined absolute paths.
validatePackageInfo :: PackageInfo -> Either String PackageInfo
validatePackageInfo pkg
    | packageName pkg == "core" =
        Left "Error: package name 'core' is reserved for the built-in core package"
    | otherwise = Right pkg

{- | Compute the module path segments for a source file given its package context.

The mapping from filesystem to module path follows Rust conventions:

  * @lib.rr@ at the root maps to @[packageName]@ (crate root)
  * @foo.rr@ at the root maps to @[packageName, "foo"]@
  * @foo\/mod.rr@ maps to @[packageName, "foo"]@
  * @foo\/bar.rr@ maps to @[packageName, "foo", "bar"]@

Both @filePath@ and @packageRoot@ should be canonicalized before calling.
-}
computeModulePath :: PackageInfo -> FilePath -> [Identifier]
computeModulePath pkg filePath =
    let rel = makeRelative (packageRoot pkg) filePath
        parts = splitDirectories (dropExtension rel)
        cleanParts = filter (/= ".") parts
        moduleSegs = case cleanParts of
            ["lib"] -> []
            segs
                | not (null segs) && last segs == "mod" -> init segs
                | otherwise -> segs
     in packageName pkg : map (Identifier . T.pack) moduleSegs

type DiscoveryM = StateT (Set.Set FilePath) (ExceptT DiscoveryError IO)

{- | Discover package files by following @mod@ declarations starting from @lib.rr@.

Parses the root file, collects @mod name;@ statements, resolves each to
@name.rr@ or @name\/mod.rr@ relative to the declaring file's directory, and
recurses. Canonical file paths are tracked during discovery so duplicate
declarations and import cycles do not recurse forever or produce duplicate
entries.
-}
discoverPackageFiles ::
    PackageInfo -> IO (Either DiscoveryError [(FilePath, [Identifier])])
discoverPackageFiles pkg = do
    root <- canonicalizePath (packageRoot pkg)
    let pkg' = pkg{packageRoot = root}
    let rootFile = root </> "lib.rr"
    rootExists <- doesFileExist rootFile
    if rootExists
        then do
            rootCanon <- canonicalizePath rootFile
            runExceptT $
                evalStateT
                    ( do
                        children <- resolveModDecls pkg' rootCanon
                        let all' = (rootCanon, computeModulePath pkg' rootCanon) : children
                        return $ sort all'
                    )
                    (Set.singleton rootCanon)
        else return $ Right []

{- | Parse a file and recursively resolve its @mod@ declarations.

For each @mod name;@ found, tries @name.rr@ then @name\/mod.rr@ relative
to the declaring file's directory. Successfully resolved modules are
recursively processed for their own @mod@ declarations.
-}
resolveModDecls ::
    PackageInfo -> FilePath -> DiscoveryM [(FilePath, [Identifier])]
resolveModDecls pkg filePath = do
    content <- lift $ lift $ TIO.readFile filePath
    parsed <- lift $ lift $ parseProgIO filePath content
    case parsed of
        Left err ->
            lift $
                throwE
                    ModuleParseError
                        { moduleParseFile = filePath
                        , moduleParseDetails = T.unpack err
                        }
        Right stmts -> do
            let modNames = collectModDecls stmts
            let parentDir = takeDirectory filePath
            concat <$> mapM (resolveOneModule pkg filePath parentDir) modNames

-- | Extract module names from @mod@ declarations in a list of statements.
collectModDecls :: [Syn.Stmt] -> [(Identifier, Maybe (Int64, Int64))]
collectModDecls = concatMap (go Nothing)
  where
    go span' (Syn.ModStmt _vis name) = [(name, span')]
    go _ (Syn.SpannedStmt (WithSpan s start end)) = go (Just (start, end)) s
    go _ _ = []

{- | Resolve a single @mod name;@ declaration to a file path.

Tries @parentDir\/name.rr@ first, then @parentDir\/name\/mod.rr@.
If found, computes the module path and recurses into the resolved file.
-}
resolveOneModule ::
    PackageInfo ->
    FilePath ->
    FilePath ->
    (Identifier, Maybe (Int64, Int64)) ->
    DiscoveryM [(FilePath, [Identifier])]
resolveOneModule pkg declaringFile parentDir (modName@(Identifier name), declSpan) = do
    let nameStr = T.unpack name
    let fileCandidate = parentDir </> nameStr ++ ".rr"
    let dirCandidate = parentDir </> nameStr </> "mod.rr"
    fileExists <- lift $ lift $ doesFileExist fileCandidate
    if fileExists
        then resolveResolvedModule pkg fileCandidate
        else do
            dirExists <- lift $ lift $ doesFileExist dirCandidate
            if dirExists
                then resolveResolvedModule pkg dirCandidate
                else
                    lift $
                        throwE
                            MissingModuleFile
                                { missingModuleName = modName
                                , missingModuleDeclaringFile = declaringFile
                                , missingModuleDeclaringSpan = declSpan
                                , missingModuleCandidates = [fileCandidate, dirCandidate]
                                }

resolveResolvedModule ::
    PackageInfo -> FilePath -> DiscoveryM [(FilePath, [Identifier])]
resolveResolvedModule pkg candidate = do
    canon <- lift $ lift $ canonicalizePath candidate
    visited <- get
    if Set.member canon visited
        then return []
        else do
            modify' (Set.insert canon)
            let modPath = computeModulePath pkg canon
            children <- resolveModDecls pkg canon
            return $ (canon, modPath) : children
