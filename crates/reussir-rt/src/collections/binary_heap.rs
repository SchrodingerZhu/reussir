use std::collections::BinaryHeap as StdBinaryHeap;

use crate::rc::Rc;

/// A functional (copy-on-write, rc-boxed) max-priority queue for Reussir's
/// polymorphic FFI, ordering through the compiler-emitted `Ord` bridge.
/// This is Rust's `BinaryHeap`, so it is a *max*-heap: `peek` and `pop`
/// see the greatest element; a min-heap comes from a reversed `Ord`.
#[derive(Clone)]
#[repr(transparent)]
pub struct BinaryHeap<T: Ord + Clone>(Rc<StdBinaryHeap<T>>);

impl<T: Ord + Clone> Default for BinaryHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> BinaryHeap<T> {
    pub fn new() -> Self {
        Self(Rc::new(StdBinaryHeap::new()))
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

    pub fn push(mut self, value: T) -> Self {
        self.0.make_mut().push(value);
        self
    }

    /// Drops the greatest element; a no-op on an empty heap (which never
    /// takes the copy-on-write clone). Retrieve it with [`Self::peek`]
    /// first.
    pub fn pop(mut self) -> Self {
        if !self.0.is_empty() {
            self.0.make_mut().pop();
        }
        self
    }

    pub fn peek(&self) -> Option<T> {
        self.0.peek().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_operations() {
        let heap = BinaryHeap::new().push(2).push(5).push(1).push(5);

        assert_eq!(heap.len(), 4);
        assert!(!heap.is_empty());
        assert_eq!(heap.peek(), Some(5));

        let heap = heap.pop();
        assert_eq!(heap.peek(), Some(5));
        let heap = heap.pop();
        assert_eq!(heap.peek(), Some(2));
        let heap = heap.pop().pop();
        assert!(heap.is_empty());
        assert_eq!(heap.peek(), None);
        let heap = heap.pop();
        assert!(heap.is_empty());
        assert!(heap.push(3).clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = BinaryHeap::new().push(1).push(3);
        let changed = original.clone().pop().push(7);

        assert_eq!(original.len(), 2);
        assert_eq!(original.peek(), Some(3));
        assert_eq!(changed.len(), 2);
        assert_eq!(changed.peek(), Some(7));
    }
}
