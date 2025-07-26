use std::collections::HashMap;

use reussir_core::{Path, types::TypeDatabase};
use reussir_front::RichError;
use rustc_hash::FxHashMapRand;
use thiserror::Error;

mod builder;
mod expr;

#[derive(Debug, Clone, Error)]
pub enum Error<'a> {
    #[error("Syntax error detected")]
    SyntaxError(Box<[RichError<'a>]>),
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;

pub struct Context {
    bump: bumpalo::Bump,
    type_database: TypeDatabase,
}
