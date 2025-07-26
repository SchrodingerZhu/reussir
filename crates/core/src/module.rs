use crate::{Context, Path, ir::Block, types::Type};

#[derive(Debug, Clone)]
pub struct ModuleInstance<'a> {
    ctx: &'a Context,
    functions: &'a [FunctionInstance<'a>],
}

#[derive(Debug, Clone)]
pub struct FunctionInstance<'a> {
    path: Path,
    type_params: Option<&'a [&'a Type]>,
    body: &'a Block<'a>,
}
