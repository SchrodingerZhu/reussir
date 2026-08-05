use std::collections::BTreeSet as StdBTreeSet;

use crate::rc::Rc;

/// A functional, copy-on-write ordered set for Reussir's polymorphic FFI.
#[derive(Clone)]
#[repr(transparent)]
pub struct BTreeSet<T: Ord + Clone>(Rc<StdBTreeSet<T>>);

impl<T: Ord + Clone> Default for BTreeSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> BTreeSet<T> {
    pub fn new() -> Self {
        Self(Rc::new(StdBTreeSet::new()))
    }

    pub fn insert(mut self, value: T) -> Self {
        self.0.make_mut().insert(value);
        self
    }

    pub fn remove(mut self, value: T) -> Self {
        self.0.make_mut().remove(&value);
        self
    }

    pub fn clear(mut self) -> Self {
        self.0.make_mut().clear();
        self
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, value: &T) -> bool {
        self.0.contains(value)
    }

    pub fn first(&self) -> Option<T> {
        self.0.first().cloned()
    }

    pub fn last(&self) -> Option<T> {
        self.0.last().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_set_operations() {
        let set = BTreeSet::new().insert(2).insert(1).insert(3).insert(2);

        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        assert!(set.contains(&2));
        assert_eq!(set.first(), Some(1));
        assert_eq!(set.last(), Some(3));

        let set = set.remove(2);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(&2));
        assert!(set.clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = BTreeSet::new().insert(1).insert(2);
        let changed = original.clone().insert(3).remove(1);

        assert_eq!(original.len(), 2);
        assert!(original.contains(&1));
        assert!(!original.contains(&3));
        assert_eq!(changed.len(), 2);
        assert!(!changed.contains(&1));
        assert!(changed.contains(&3));
    }
}
