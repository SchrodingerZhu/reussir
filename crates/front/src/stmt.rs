use reussir_core::func::Function;
use crate::{SpanBox, types::TypeDecl};

pub type StmtBox = SpanBox<Stmt>;

#[derive(Debug, Clone)]
pub enum Stmt {
    TypeDecl(TypeDecl),
    FunctionDecl(Function),
}