use std::collections::VecDeque as StdVecDeque;

use crate::rc::Rc;

/// A functional (copy-on-write, rc-boxed) double-ended queue for Reussir's
/// polymorphic FFI. The same contract as
/// [`crate::collections::vec::Vec`]: owning operations take `self` linearly
/// and return the updated deque, read-only operations borrow.
#[derive(Clone)]
#[repr(transparent)]
pub struct VecDeque<T: Clone>(Rc<StdVecDeque<T>>);

impl<T: Clone> Default for VecDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> VecDeque<T> {
    pub fn new() -> Self {
        Self(Rc::new(StdVecDeque::new()))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(mut self) -> Self {
        if !self.0.is_empty() {
            self.0.make_mut().clear();
        }
        self
    }

    pub fn push_front(mut self, value: T) -> Self {
        self.0.make_mut().push_front(value);
        self
    }

    pub fn push_back(mut self, value: T) -> Self {
        self.0.make_mut().push_back(value);
        self
    }

    /// Drops the front element; a no-op on an empty deque (which never
    /// takes the copy-on-write clone).
    pub fn pop_front(mut self) -> Self {
        if !self.0.is_empty() {
            self.0.make_mut().pop_front();
        }
        self
    }

    /// Drops the back element; a no-op on an empty deque.
    pub fn pop_back(mut self) -> Self {
        if !self.0.is_empty() {
            self.0.make_mut().pop_back();
        }
        self
    }

    pub fn front(&self) -> Option<T> {
        self.0.front().cloned()
    }

    pub fn back(&self) -> Option<T> {
        self.0.back().cloned()
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.0.get(index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deque_operations() {
        let deque = VecDeque::new().push_back(2).push_back(3).push_front(1);

        assert_eq!(deque.len(), 3);
        assert!(!deque.is_empty());
        assert_eq!(deque.front(), Some(1));
        assert_eq!(deque.back(), Some(3));
        assert_eq!(deque.get(1), Some(2));
        assert_eq!(deque.get(9), None);

        let deque = deque.pop_front().pop_back();
        assert_eq!(deque.len(), 1);
        assert_eq!(deque.front(), Some(2));
        assert_eq!(deque.back(), Some(2));

        let deque = deque.pop_front().pop_front();
        assert!(deque.is_empty());
        assert_eq!(deque.front(), None);
        assert!(deque.push_back(4).clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = VecDeque::new().push_back(1).push_back(2);
        let changed = original.clone().pop_front().push_back(3);

        assert_eq!(original.len(), 2);
        assert_eq!(original.front(), Some(1));
        assert_eq!(original.back(), Some(2));
        assert_eq!(changed.len(), 2);
        assert_eq!(changed.front(), Some(2));
        assert_eq!(changed.back(), Some(3));
    }
}
