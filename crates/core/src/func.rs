use ustr::Ustr;
use crate::{Location, Path};
use crate::types::Type;
#[derive(Clone, Debug)]
pub struct Param {
    pub name: Ustr,
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
    pub region: bool,
    pub public: bool,
    pub opaque: bool,
}