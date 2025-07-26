use crate::ir::Block;
use crate::types::Type;
use crate::{Location, Path};
use rustc_hash::FxHashMapRand;
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

pub struct FunctionDatabase {
    functions: FxHashMapRand<Path, FunctionProto>,
}

impl FunctionDatabase {
    pub fn new() -> Self {
        Self {
            functions: FxHashMapRand::default(),
        }
    }

    pub fn get(&self, path: &Path) -> Option<&FunctionProto> {
        self.functions.get(path)
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.functions.contains_key(path)
    }

    pub fn add_function(&mut self, proto: FunctionProto) {
        self.functions.insert(proto.path.clone(), proto);
    }
}
