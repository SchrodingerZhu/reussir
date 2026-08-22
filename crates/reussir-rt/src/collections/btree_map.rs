use std::collections::BTreeMap as StdBTreeMap;

use crate::rc::Rc;

/// A functional (copy-on-write, rc-boxed) ordered map for Reussir's
/// polymorphic FFI. Keys order through the compiler-emitted `Ord` bridge;
/// owning operations take `self` linearly and return the updated map
/// (`make_mut` copies only when the box is shared), read-only operations
/// borrow.
#[derive(Clone)]
#[repr(transparent)]
pub struct BTreeMap<K: Ord + Clone, V: Clone>(Rc<StdBTreeMap<K, V>>);

impl<K: Ord + Clone, V: Clone> Default for BTreeMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone, V: Clone> BTreeMap<K, V> {
    pub fn new() -> Self {
        Self(Rc::new(StdBTreeMap::new()))
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

    /// Inserts a key-value pair; an existing equal key keeps its stored key
    /// and has its value replaced.
    pub fn insert(mut self, key: K, value: V) -> Self {
        self.0.make_mut().insert(key, value);
        self
    }

    /// Removes an entry if present. Probes the shared map first so a miss
    /// never takes the copy-on-write clone.
    pub fn remove(mut self, key: K) -> Self {
        if self.0.contains_key(&key) {
            self.0.make_mut().remove(&key);
        }
        self
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.0.contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.0.get(key).cloned()
    }

    pub fn first_key(&self) -> Option<K> {
        self.0.first_key_value().map(|(k, _)| k.clone())
    }

    pub fn last_key(&self) -> Option<K> {
        self.0.last_key_value().map(|(k, _)| k.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_map_operations() {
        let map = BTreeMap::new().insert(2, 20).insert(1, 10).insert(3, 30);

        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
        assert_eq!(map.get(&2), Some(20));
        assert_eq!(map.get(&7), None);
        assert!(map.contains_key(&1));
        assert_eq!(map.first_key(), Some(1));
        assert_eq!(map.last_key(), Some(3));

        let map = map.insert(2, 21);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(21));

        let map = map.remove(1);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), None);
        let map = map.remove(9);
        assert_eq!(map.len(), 2);
        assert!(map.clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = BTreeMap::new().insert(1, 10).insert(2, 20);
        let changed = original.clone().insert(3, 30).remove(1);

        assert_eq!(original.len(), 2);
        assert_eq!(original.get(&1), Some(10));
        assert_eq!(original.get(&3), None);
        assert_eq!(changed.len(), 2);
        assert_eq!(changed.get(&1), None);
        assert_eq!(changed.get(&3), Some(30));
    }
}
