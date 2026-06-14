use crate::option::Option;
use crate::rc::{Rc, RcRef};
type StdVec<T> = std::vec::Vec<T>;

#[derive(Clone)]
#[repr(transparent)]
pub struct Vec<T: Clone>(Rc<StdVec<T>>);

impl<T: Clone> Default for Vec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Vec<T> {
    pub fn new() -> Self {
        Self(Rc::new(StdVec::new()))
    }
    pub fn push(mut self, value: T) -> Self {
        self.0.make_mut().push(value);
        self
    }
    pub fn pop(mut self) -> Self {
        self.0.make_mut().pop();
        self
    }
    pub fn len(this: RcRef<Self>) -> usize {
        this.0.len()
    }
    pub fn is_empty(this: RcRef<Self>) -> bool {
        this.0.is_empty()
    }
    pub fn clear(mut self) -> Self {
        self.0.make_mut().clear();
        self
    }
    pub fn get(this: RcRef<Self>, index: usize) -> Option<T> {
        this.0.get(index).cloned().into()
    }
    pub fn set(mut self, index: usize, value: T) -> Self {
        self.0.make_mut().insert(index, value);
        self
    }
}
