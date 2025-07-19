use crate::{SpanBox, types::TypeDecl};

pub type StmtBox = SpanBox<Stmt>;

pub enum Stmt {
    TypeDecl(TypeDecl),
    FunctionDecl,
}
