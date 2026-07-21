use crate::rc::Rc;
type StdVec<T> = std::vec::Vec<T>;

/// A functional (copy-on-write, rc-boxed) vector: the standard exposed
/// collection for Reussir's polymorphic FFI. Owning operations take `self`
/// linearly and return the updated vector (`make_mut` copies only when the
/// box is shared); read-only operations borrow. The `#[repr(transparent)]`
/// wrapper over [`Rc`] is what the FFI contract requires: the compiler
/// treats the value as an rc pointer whose count sits at offset 0.
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
    pub fn len(&self) -> usize {
        self.0.data_ref().len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.data_ref().is_empty()
    }
    pub fn clear(mut self) -> Self {
        self.0.make_mut().clear();
        self
    }
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.data_ref().get(index).cloned()
    }
    pub fn set(mut self, index: usize, value: T) -> Self {
        self.0.make_mut()[index] = value;
        self
    }
}
