{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | Shared orphan 'ToJSON'/'FromJSON' instances for the building blocks used
by both the semi and the full syntax: unique identifiers, operators, interned
symbols, string tokens, diagnostic reports and the bridge log level.

See "Reussir.Parser.Serialization" for the encoding conventions shared by all
frontend serialization modules.
-}
module Reussir.Core.Serialization.Orphans () where

import Data.Aeson (
    FromJSON (..),
    FromJSONKey (..),
    FromJSONKeyFunction (..),
    ToJSON (..),
    ToJSONKey (..),
    defaultOptions,
    withText,
 )
import Data.Aeson.TH (deriveJSON)
import Data.Aeson.Types (toJSONKeyText)
import Data.Colour (Colour)
import Data.Colour.RGBSpace (RGB (..))
import Data.Digest.XXHash.FFI (XXH3 (..))
import Reussir.Codegen.Context.Symbol (Symbol, symbolText, verifiedSymbol)
import Reussir.Diagnostic.Report (
    CodeReference,
    Label,
    Report,
    TextWithFormat,
 )
import Reussir.Parser.Serialization ()

import Data.Colour.SRGB.Linear qualified as Linear
import Data.Text qualified as T
import Reussir.Bridge qualified as B
import Reussir.Codegen.Type.Data qualified as IR
import System.Console.ANSI.Types qualified as ANSI

import Reussir.Core.Data.Class (Class, ClassNode)
import Reussir.Core.Data.FP (FloatingPointType)
import Reussir.Core.Data.Integral (IntegralType)
import Reussir.Core.Data.Operator (ArithOp, CmpOp)
import Reussir.Core.Data.String (StringToken (..))
import Reussir.Core.Data.UniqueID (
    ExprID (..),
    GenericID (..),
    HoleID (..),
    VarID (..),
 )

instance ToJSON GenericID where
    toJSON = toJSON . genericIDValue
    toEncoding = toEncoding . genericIDValue

instance FromJSON GenericID where
    parseJSON = fmap GenericID . parseJSON

instance ToJSON VarID where
    toJSON = toJSON . unVarID
    toEncoding = toEncoding . unVarID

instance FromJSON VarID where
    parseJSON = fmap VarID . parseJSON

instance ToJSON ExprID where
    toJSON = toJSON . unExprID
    toEncoding = toEncoding . unExprID

instance FromJSON ExprID where
    parseJSON = fmap ExprID . parseJSON

instance ToJSON HoleID where
    toJSON = toJSON . holeIDValue
    toEncoding = toEncoding . holeIDValue

instance FromJSON HoleID where
    parseJSON = fmap HoleID . parseJSON

instance ToJSON StringToken where
    toJSON (StringToken ws) = toJSON ws
    toEncoding (StringToken ws) = toEncoding ws

instance FromJSON StringToken where
    parseJSON = fmap StringToken . parseJSON

instance ToJSON Symbol where
    toJSON = toJSON . symbolText
    toEncoding = toEncoding . symbolText

instance FromJSON Symbol where
    parseJSON = withText "Symbol" (pure . verifiedSymbol)

instance ToJSONKey Symbol where
    toJSONKey = toJSONKeyText symbolText

instance FromJSONKey Symbol where
    fromJSONKey = FromJSONKeyText verifiedSymbol

instance ToJSON (XXH3 T.Text) where
    toJSON (XXH3 t) = toJSON t
    toEncoding (XXH3 t) = toEncoding t

instance FromJSON (XXH3 T.Text) where
    parseJSON = fmap XXH3 . parseJSON

instance ToJSONKey (XXH3 T.Text) where
    toJSONKey = toJSONKeyText (\(XXH3 t) -> t)

instance FromJSONKey (XXH3 T.Text) where
    fromJSONKey = FromJSONKeyText XXH3

-- Colours are stored by their linear RGB channels, which is the internal
-- representation of 'Colour' and therefore roundtrips exactly.
instance ToJSON (Colour Float) where
    toJSON c = let RGB r g b = Linear.toRGB c in toJSON (r, g, b)
    toEncoding c = let RGB r g b = Linear.toRGB c in toEncoding (r, g, b)

instance FromJSON (Colour Float) where
    parseJSON v = do
        (r, g, b) <- parseJSON v
        pure (Linear.rgb r g b)

$(deriveJSON defaultOptions ''ArithOp)
$(deriveJSON defaultOptions ''CmpOp)
$(deriveJSON defaultOptions ''IntegralType)
$(deriveJSON defaultOptions ''FloatingPointType)
$(deriveJSON defaultOptions ''Class)
$(deriveJSON defaultOptions ''ClassNode)
$(deriveJSON defaultOptions ''B.LogLevel)
$(deriveJSON defaultOptions ''IR.Capability)

$(deriveJSON defaultOptions ''ANSI.Color)
$(deriveJSON defaultOptions ''ANSI.ColorIntensity)
$(deriveJSON defaultOptions ''ANSI.ConsoleLayer)
$(deriveJSON defaultOptions ''ANSI.ConsoleIntensity)
$(deriveJSON defaultOptions ''ANSI.Underlining)
$(deriveJSON defaultOptions ''ANSI.BlinkSpeed)
$(deriveJSON defaultOptions ''ANSI.SGR)

$(deriveJSON defaultOptions ''Label)
$(deriveJSON defaultOptions ''TextWithFormat)
$(deriveJSON defaultOptions ''CodeReference)
$(deriveJSON defaultOptions ''Report)
