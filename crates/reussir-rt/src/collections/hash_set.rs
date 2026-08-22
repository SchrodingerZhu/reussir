use hashbrown::HashTable;

use crate::rc::Rc;

/// A functional (copy-on-write, rc-boxed) hash set for Reussir's polymorphic
/// FFI, built on hashbrown's low-level `HashTable`. The same contract as
/// [`crate::collections::hash_map::HashMap`]: the Reussir side supplies every
/// hash, each entry caches its digest so clone and resize never rehash, and
/// equality is the only callback.
#[derive(Clone)]
#[repr(transparent)]
pub struct HashSet<T: Eq + Clone>(Rc<HashTable<(u64, T)>>);

impl<T: Eq + Clone> Default for HashSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Clone> HashSet<T> {
    pub fn new() -> Self {
        Self(Rc::new(HashTable::new()))
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

    /// Inserts under a caller-computed hash. An existing equal value is kept.
    pub fn insert(mut self, hash: u64, value: T) -> Self {
        let table = self.0.make_mut();
        match table.entry(hash, |e| e.0 == hash && e.1 == value, |e| e.0) {
            hashbrown::hash_table::Entry::Occupied(_) => {}
            hashbrown::hash_table::Entry::Vacant(vacant) => {
                vacant.insert((hash, value));
            }
        }
        self
    }

    pub fn contains(&self, hash: u64, value: &T) -> bool {
        self.0
            .find(hash, |e| e.0 == hash && e.1 == *value)
            .is_some()
    }

    /// Removes a value if present. Probes the shared table first so a miss
    /// never takes the copy-on-write clone.
    pub fn remove(mut self, hash: u64, value: T) -> Self {
        if self.contains(hash, &value) {
            let table = self.0.make_mut();
            if let Ok(occupied) = table.find_entry(hash, |e| e.0 == hash && e.1 == value) {
                occupied.remove();
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A poor hash on purpose, so bucket collisions are exercised.
    fn hash(value: i64) -> u64 {
        (value % 3) as u64
    }

    fn insert(set: HashSet<i64>, value: i64) -> HashSet<i64> {
        set.insert(hash(value), value)
    }

    #[test]
    fn set_operations() {
        let set = insert(insert(insert(insert(HashSet::new(), 1), 2), 4), 2);

        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        assert!(set.contains(hash(1), &1));
        assert!(set.contains(hash(2), &2));
        assert!(set.contains(hash(4), &4));
        assert!(!set.contains(hash(7), &7));

        let set = set.remove(hash(2), 2);
        assert_eq!(set.len(), 2);
        assert!(!set.contains(hash(2), &2));
        let set = set.remove(hash(9), 9);
        assert_eq!(set.len(), 2);
        assert!(set.clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = insert(insert(HashSet::new(), 1), 2);
        let changed = insert(original.clone().remove(hash(1), 1), 3);

        assert_eq!(original.len(), 2);
        assert!(original.contains(hash(1), &1));
        assert!(!original.contains(hash(3), &3));
        assert_eq!(changed.len(), 2);
        assert!(!changed.contains(hash(1), &1));
        assert!(changed.contains(hash(3), &3));
    }
}
