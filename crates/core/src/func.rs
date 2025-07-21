use crate::types::Type;
use crate::{Location, Path};
use ustr::Ustr;
#[derive(Clone, Debug)]
pub struct Param {
    pub name: Option<Ustr>,
    pub location: Location,
    pub ty: Type,
}
#[derive(Clone, Debug)]
pub struct FunctionProto {
    pub path: Path,
    pub location: Location,
    pub type_args: Option<Box<[Ustr]>>,
    pub params: Box<[Param]>,
    pub return_type: Type,
    pub has_region: bool,
    pub is_public: bool,
    pub is_opaque: bool,
}
