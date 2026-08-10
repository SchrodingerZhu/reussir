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
    /// Inserts at `index`, shifting the tail right. Copy-on-write like
    /// every other owning operation: a shared box is cloned once, then the
    /// single shift happens in place — the node-editing primitive a
    /// bitmap-compressed trie needs (`set` alone cannot grow a node).
    pub fn insert_at(mut self, index: usize, value: T) -> Self {
        self.0.make_mut().insert(index, value);
        self
    }
    /// Removes at `index`, shifting the tail left; the counterpart of
    /// [`Self::insert_at`].
    pub fn remove_at(mut self, index: usize) -> Self {
        self.0.make_mut().remove(index);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_remove_shift_around_index() {
        let v = Vec::new().push(1).push(3);
        let v = v.insert_at(1, 2);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(1));
        assert_eq!(v.get(1), Some(2));
        assert_eq!(v.get(2), Some(3));
        let v = v.remove_at(0);
        assert_eq!(v.len(), 2);
        assert_eq!(v.get(0), Some(2));
        assert_eq!(v.get(1), Some(3));
    }

    #[test]
    fn insert_at_copies_a_shared_box() {
        let a = Vec::new().push(10).push(30);
        let b = a.clone().insert_at(1, 20);
        // The snapshot is untouched; only the edited copy grew.
        assert_eq!(a.len(), 2);
        assert_eq!(a.get(1), Some(30));
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(1), Some(20));
    }
}
