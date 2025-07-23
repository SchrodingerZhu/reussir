use reussir_front::RichError;
use thiserror::Error;

mod builder;
mod expr;

#[derive(Debug, Clone, Error)]
pub enum Error<'a> {
    #[error("Syntax error detected")]
    SyntaxError(Box<[RichError<'a>]>),
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;
