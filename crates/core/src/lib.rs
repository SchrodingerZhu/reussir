use std::fmt::Debug;
use ustr::Ustr;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location {
    file: Ustr,
    line: u32,
    column: u32,
}

impl Location {
    pub fn new(file: Ustr, line: u32, column: u32) -> Self {
        Location { file, line, column }
    }

    pub fn file(&self) -> Ustr {
        self.file
    }

    pub fn line(&self) -> u32 {
        self.line
    }

    pub fn column(&self) -> u32 {
        self.column
    }
}

#[macro_export]
macro_rules! type_path {
        ($basename:expr $(, $prefix:expr)*) => {
            Path::new(
                ustr::Ustr::from($basename),
                [$($prefix.into()),*],
            )
        }
    }
