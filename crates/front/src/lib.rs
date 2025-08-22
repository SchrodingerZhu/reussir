pub mod expr;
pub mod lexer;
pub mod stmt;
pub mod types;
pub use chumsky;
use std::ops::Deref;

use ariadne::{Report, sources};
use chumsky::{
    ParseResult, Parser,
    container::Container,
    error::Rich,
    input::{Checkpoint, Cursor, Input, MapExtra, ValueInput},
    inspector::Inspector,
};
use lexer::Token;
use reussir_core::{Location, Path};
use smallvec::SmallVec;
use stmt::StmtBox;
use ustr::Ustr;

pub type RichError<'a> = Rich<'a, lexer::Token<'a>, Location>;
type ParserExtra<'a> = chumsky::extra::Full<RichError<'a>, ParserState, ()>;

#[derive(Debug, Clone)]
pub struct Module {
    pub path: Path,
    pub input_file: Ustr,
    pub statements: Box<[StmtBox]>,
}

pub struct ParserState {
    pub module_path: Path,
    pub input_file: Ustr,
}

impl ParserState {
    pub fn new<S: Into<Ustr>>(module_path: Path, input_file: S) -> Self {
        ParserState {
            module_path,
            input_file: input_file.into(),
        }
    }
    pub fn print_result<T>(&self, result: &ParseResult<T, RichError<'_>>, source: &str) {
        let _lock_out = std::io::stdout().lock();
        let _lock_err = std::io::stderr().lock();
        result
            .errors()
            .cloned()
            .map(|e| e.map_token(|tk| format!("{tk:?}")))
            .for_each(|e| {
                Report::build(ariadne::ReportKind::Error, *e.span())
                    .with_config(
                        ariadne::Config::default()
                            .with_index_type(ariadne::IndexType::Char)
                            .with_color(true),
                    )
                    .with_message(e.reason())
                    .with_label(ariadne::Label::new(*e.span()).with_color(ariadne::Color::Red))
                    .finish()
                    .eprint(sources([(self.input_file, source)]))
                    .unwrap();
            });
    }
}

impl<'src, I: Input<'src>> Inspector<'src, I> for ParserState {
    type Checkpoint = ();
    #[inline(always)]
    fn on_token(&mut self, _: &<I as Input<'src>>::Token) {}
    #[inline(always)]
    fn on_save<'parse>(&self, _: &Cursor<'src, 'parse, I>) -> Self::Checkpoint {}
    #[inline(always)]
    fn on_rewind<'parse>(&mut self, _: &Checkpoint<'src, 'parse, I, Self::Checkpoint>) {}
}

struct SmallCollector<T, const N: usize>(SmallVec<T, N>);

impl<T, const N: usize> Default for SmallCollector<T, N> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T, const N: usize> Container<T> for SmallCollector<T, N> {
    fn with_capacity(n: usize) -> Self {
        Self(SmallVec::with_capacity(n))
    }

    fn push(&mut self, item: T) {
        self.0.push(item);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct WithSpan<T>(T, Location);

pub type SpanBox<T> = Box<WithSpan<T>>;

impl<T: PartialEq> PartialEq for WithSpan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> WithSpan<T> {
    pub fn new(value: T, location: Location) -> Self {
        WithSpan(value, location)
    }
    pub fn value(&self) -> &T {
        &self.0
    }

    pub fn location(&self) -> Location {
        self.1
    }
}

impl<T: Eq> Eq for WithSpan<T> {}

impl<T> std::hash::Hash for WithSpan<T>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Deref for WithSpan<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn path<'a, I>() -> impl Parser<'a, I, Path, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    use chumsky::prelude::*;
    let ident = select! {
        Token::Ident(x) => x
    };
    let prefix = ident
        .map(Ustr::from)
        .then_ignore(just(Token::PathSep))
        .repeated()
        .collect::<SmallCollector<_, 4>>();

    prefix
        .then(ident)
        .map(|(prefix, basename)| Path::new(basename.into(), prefix.0))
        .labelled("path")
}

pub(crate) fn make_spanbox<T>(value: T, location: Location) -> SpanBox<T> {
    Box::new(WithSpan::new(value, location))
}

pub(crate) fn make_spanbox_with<'src, 'b, T, I>(
    value: T,
    extra: &mut MapExtra<'src, 'b, I, ParserExtra<'src>>,
) -> SpanBox<T>
where
    I: ValueInput<'src, Token = Token<'src>, Span = Location>,
{
    make_spanbox(value, extra.span())
}

pub fn module<'a, I>() -> impl Parser<'a, I, Module, ParserExtra<'a>> + Clone
where
    I: ValueInput<'a, Token = Token<'a>, Span = Location>,
{
    use chumsky::prelude::*;
    let statements = stmt::stmt()
        .repeated()
        .collect::<Vec<_>>()
        .map(Vec::into_boxed_slice)
        .then_ignore(end());

    statements.map_with(|statements, extra| Module {
        path: extra.state().module_path.clone(),
        input_file: extra.state().input_file,
        statements,
    })
}

#[cfg(test)]
mod tests {
    use reussir_core::path;

    use super::*;

    #[test]
    fn test_path_parser() {
        let source = "foo::bar::baz";
        let mut state = ParserState::new(path!("test"), "<stdin>");
        let parser = path();
        let token_stream = Token::stream(Ustr::from("<stdin>"), source);
        let result = parser.parse_with_state(token_stream, &mut state).unwrap();
        println!("{:?}", result);
    }

    macro_rules! parse_file {
        (@ $input:literal, $test_name:ident, $success:literal) => {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap();
            let canonical_path = path.join("misc").join($input);
            let source = std::fs::read_to_string(&canonical_path).unwrap();
            let canonical_path = canonical_path.to_string_lossy().to_owned();
            let mut state =
                ParserState::new(path!(stringify!($test_name)), canonical_path.clone());
            let parser = module();
            let token_stream = Token::stream(Ustr::from(&canonical_path), &source);
            let res = parser.parse_with_state(token_stream, &mut state);
            state.print_result(&res, &source);
            if $success {
                println!("{:#?}", res.unwrap());
            } else {
                assert!(res.has_errors(), "Parsing succeeded unexpectedly for {}", $input);
            }
        };
        ([success] $input:literal, $test_name:ident) => {
            #[test]
            fn $test_name() {
                parse_file!(@ $input, $test_name, true);
            }
        };
        ([failure] $input:literal, $test_name:ident) => {
            #[test]
            fn $test_name() {
                parse_file!(@ $input, $test_name, false);
            }
        };
    }

    parse_file!([success] "simple_test.rr", simple_test);
    parse_file!([success] "fibonacci.rr", fibonacci);
    parse_file!([failure] "simple_failure_test_1.rr", simple_failure_test_1);
    parse_file!([failure] "simple_failure_test_2.rr", simple_failure_test_2);
}
