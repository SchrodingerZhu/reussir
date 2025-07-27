use crate::{Context, Path, ir::Block, types::Type};

#[derive(Debug, Clone)]
pub struct ModuleInstance<'a> {
    pub ctx: &'a Context,
    pub functions: &'a [FunctionInstance<'a>],
}

#[derive(Debug, Clone)]
pub struct FunctionInstance<'a> {
    pub path: Path,
    pub type_params: Option<&'a [&'a Type]>,
    pub body: &'a Block<'a>,
}
