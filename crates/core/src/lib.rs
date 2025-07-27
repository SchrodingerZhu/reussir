use chumsky::span::Span;
use std::fmt::Debug;
use std::ops::Range;
use ustr::Ustr;

pub mod func;
pub mod ir;
pub mod literal;
pub mod module;
pub mod types;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Path(Box<[Ustr]>);

impl Path {
    pub fn new(basename: Ustr, prefix: impl IntoIterator<Item = Ustr>) -> Self {
        let mut all_segments = prefix.into_iter().collect::<Vec<_>>();
        all_segments.push(basename);
        Path(all_segments.into_boxed_slice())
    }
    pub fn segments(&self) -> &[Ustr] {
        &self.0
    }
    pub fn prefix(&self) -> &[Ustr] {
        &self.0[..self.0.len() - 1]
    }
    pub fn basename(&self) -> Ustr {
        self.0
            .last()
            .cloned()
            .expect("Path must have at least one segment")
    }
    pub fn append(self, segment: Ustr) -> Self {
        let mut new_segments = self.0.into_vec();
        new_segments.push(segment);
        Path(new_segments.into_boxed_slice())
    }
    pub fn consolidate(&self) -> Box<str> {
        let mut result = String::new();
        for (i, segment) in self.0.iter().enumerate() {
            if i > 0 {
                result.push_str("::");
            }
            result.push_str(segment.as_str());
        }
        result.into_boxed_str()
    }
}

impl Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for segment in self.0.iter().enumerate() {
            if segment.0 > 0 {
                f.write_str("::")?;
            }
            f.write_str(segment.1.as_str())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    file: Ustr,
    span: (u32, u32),
}

impl Debug for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.file, self.span.0, self.span.1)
    }
}

impl Location {
    pub fn new(file: Ustr, span: (u32, u32)) -> Self {
        Location { file, span }
    }

    pub fn file(&self) -> Ustr {
        self.file
    }

    pub fn span(&self) -> (u32, u32) {
        self.span
    }
}

impl Span for Location {
    type Context = Ustr;
    type Offset = u32;

    fn new(context: Self::Context, range: Range<Self::Offset>) -> Self {
        Location {
            file: context,
            span: (range.start, range.end),
        }
    }

    fn context(&self) -> Self::Context {
        self.file
    }

    fn start(&self) -> Self::Offset {
        self.span.0
    }

    fn end(&self) -> Self::Offset {
        self.span.1
    }
}

impl ariadne::Span for Location {
    type SourceId = Ustr;

    fn source(&self) -> &Self::SourceId {
        &self.file
    }

    fn start(&self) -> usize {
        self.span.0 as usize
    }

    fn end(&self) -> usize {
        self.span.1 as usize
    }
}

#[macro_export]
macro_rules! path {
        ($basename:expr $(, $prefix:expr)*) => {
            $crate::Path::new(
                ustr::Ustr::from($basename),
                [$($prefix.into()),*],
            )
        }
    }

#[derive(Debug)]
pub struct Context {
    bump: bumpalo::Bump,
    module: Path,
    type_database: types::TypeDatabase,
    function_database: func::FunctionDatabase,
}

impl Context {
    pub fn new(module: Path) -> Self {
        Self {
            bump: bumpalo::Bump::new(),
            module,
            type_database: types::TypeDatabase::new(),
            function_database: func::FunctionDatabase::new(),
        }
    }

    pub fn module(&self) -> &Path {
        &self.module
    }

    pub fn bump(&self) -> &bumpalo::Bump {
        &self.bump
    }

    pub fn types(&self) -> &types::TypeDatabase {
        &self.type_database
    }

    pub fn functions(&self) -> &func::FunctionDatabase {
        &self.function_database
    }

    pub fn types_mut(&mut self) -> &mut types::TypeDatabase {
        &mut self.type_database
    }

    pub fn functions_mut(&mut self) -> &mut func::FunctionDatabase {
        &mut self.function_database
    }
}
