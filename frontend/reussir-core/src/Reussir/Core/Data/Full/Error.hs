module Reussir.Core.Data.Full.Error where

import Data.Int (Int64)
import Reussir.Parser.Types.Lexer (Path (..))

import Data.Vector.Strict qualified as V
import Reussir.Parser.Types.Capability qualified as Syn

import Reussir.Core.Data.UniqueID (GenericID)

import Reussir.Core.Data.Semi.Type qualified as Semi

data Error = Error
    { errorSpan :: (Int64, Int64)
    , errorKind :: ErrorKind
    }
    deriving (Show, Eq)

data ErrorKind
    = InvalidRecordField
        { recordPath :: Path -- The path of the record
        , recordTypeArgs :: V.Vector Semi.Type -- The type arguments of the record
        , errorIndex :: Int -- The index of the field that is invalid
        }
    | InvalidNullableType Semi.Type
    | UnknownGeneric GenericID
    | InvalidCapability Path Syn.Capability
    | -- | Source of assignment is not Nullable<Rc<T, Flex>>
      InvalidAssignSourceCapability
    | -- | Source of assignment inner type is not a regional record
      InvalidAssignSourceNotRegional
    deriving (Show, Eq)
