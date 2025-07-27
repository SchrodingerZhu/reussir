use reussir_core::module::ModuleInstance;
use reussir_front::{Module, RichError};
use thiserror::Error;

use crate::builder::ModuleBuilder;

pub mod builder;
pub mod expr;

#[derive(Debug, Clone, Error)]
pub enum Error<'a> {
    #[error("Syntax error detected")]
    SyntaxError(Box<[RichError<'a>]>),
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;

pub fn populate_module<'a>(
    ctx: &'a mut reussir_core::Context,
    module: &Module,
    source_code: &str,
) -> Option<ModuleInstance<'a>> {
    // Pass 1: record all functions and types.
    for i in module.statements.iter() {
        match &***i {
            reussir_front::stmt::Stmt::TypeDecl(_) => todo!(),
            reussir_front::stmt::Stmt::FunctionDecl(function) => {
                let proto = function.proto.clone();
                ctx.functions_mut().add_function(proto);
            }
        }
    }
    ctx.functions_mut().build_fuzzy_indices();
    let mut builder = ModuleBuilder::new(ctx);
    // Pass 2: instantiate public functions with type parameters
    // TODO: Populate their dependencies along the way.
    for i in module.statements.iter() {
        match &***i {
            reussir_front::stmt::Stmt::TypeDecl(_) => todo!(),
            reussir_front::stmt::Stmt::FunctionDecl(function) => {
                if function.proto.is_public
                    && function.proto.type_args.is_none()
                    && let Some(body) = &function.body
                {
                    builder.define_function(&function.proto, body);
                }
            }
        }
    }
    // TODO: pass 3: instantiate dependencies.

    // Pass 4: return errors and build the module instance.
    if builder.report_errors(source_code, module.input_file) {
        return None;
    }
    Some(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reussir_core::path;
    use reussir_front::lexer::Token;
    use ustr::Ustr;

    macro_rules! test_module_codegen {
        ($input:literal, $name:ident, $has_error:expr) => {
            #[test]
            fn $name() {
                use chumsky::prelude::*;
                let mut ctx = reussir_core::Context::new(path!("test"));
                let mut parser_state = reussir_front::ParserState::new(path!("test"), "<stdin>");
                let source = $input;
                let parser = reussir_front::module();
                let token_stream = Token::stream(Ustr::from("<stdin>"), source);
                let res = parser
                    .parse_with_state(token_stream, &mut parser_state)
                    .unwrap();
                let module = populate_module(&mut ctx, &res, source);
                if $has_error {
                    assert!(
                        module.is_none(),
                        "Module should not be created due to errors"
                    );
                } else {
                    assert!(module.is_some(), "Module should be created successfully");
                    println!("{:#?}", module.unwrap());
                }
            }
        };
    }

    test_module_codegen!("pub fn main() { }", test_main_function, false);
    test_module_codegen!(
        "pub fn add_one(n: u64) -> u64 { n + 1u64 }",
        test_add_one_function,
        false
    );
    test_module_codegen!(
        "pub fn fibonacci(n: u64) -> u64 { if n < 2u64 { n } else { fibonacci(n - 1u64) + fibonacci(n - 2u64) } }",
        test_fibonacci_function,
        false
    );

    test_module_codegen!(
        r#"
        pub fn test_name_x() -> u64 { test_name_y() }
        pub fn test_name_1() -> u64 { 42 }
        pub fn test_name_2() -> u64 { 42 }
        pub fn test_name_3() -> u64 { 42 }
        "#,
        test_fuzzy_search,
        true
    );
}
