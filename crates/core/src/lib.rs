use ustr::Ustr;

pub mod types;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        self.0.last().cloned().expect("Path must have at least one segment")
    }
}