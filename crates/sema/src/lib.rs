use reussir_front::{Module, RichError};
use thiserror::Error;

mod builder;
mod expr;

#[derive(Debug, Clone, Error)]
pub enum Error<'a> {
    #[error("Syntax error detected")]
    SyntaxError(Box<[RichError<'a>]>),
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;

pub fn populate_module(ctx: &mut reussir_core::Context, module: &Module) {
    for i in module.statements.iter() {
        match &***i {
            reussir_front::stmt::Stmt::TypeDecl(_) => todo!(),
            reussir_front::stmt::Stmt::FunctionDecl(function) => {
                let proto = function.proto.clone();
                ctx.functions_mut().add_function(proto);
            }
        }
    }
}
