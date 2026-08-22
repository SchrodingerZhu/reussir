use hashbrown::HashTable;

use crate::rc::Rc;

/// A functional (copy-on-write, rc-boxed) hash map for Reussir's polymorphic
/// FFI, built on hashbrown's low-level `HashTable`.
///
/// The Reussir side hashes keys with its own pure hashers and passes the
/// digest across the boundary, so the table never hashes anything itself:
/// each entry caches its full 64-bit hash, resizing and the copy-on-write
/// clone reuse the cached value, and key equality is the only callback into
/// Reussir (`Bridge<K>: Eq`), reached only after the cached hashes match.
/// Owning operations take `self` linearly and return the updated map
/// (`make_mut` copies only when the box is shared); read-only operations
/// borrow. The `#[repr(transparent)]` wrapper over [`Rc`] is what the FFI
/// contract requires: the compiler treats the value as an rc pointer whose
/// count sits at offset 0.
#[derive(Clone)]
#[repr(transparent)]
pub struct HashMap<K: Eq + Clone, V: Clone>(Rc<HashTable<(u64, K, V)>>);

impl<K: Eq + Clone, V: Clone> Default for HashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Clone, V: Clone> HashMap<K, V> {
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

    /// Inserts under a caller-computed hash. An existing equal key keeps its
    /// stored key and has its value replaced.
    pub fn insert(mut self, hash: u64, key: K, value: V) -> Self {
        let table = self.0.make_mut();
        match table.entry(hash, |e| e.0 == hash && e.1 == key, |e| e.0) {
            hashbrown::hash_table::Entry::Occupied(mut occupied) => {
                occupied.get_mut().2 = value;
            }
            hashbrown::hash_table::Entry::Vacant(vacant) => {
                vacant.insert((hash, key, value));
            }
        }
        self
    }

    pub fn contains(&self, hash: u64, key: &K) -> bool {
        self.0.find(hash, |e| e.0 == hash && e.1 == *key).is_some()
    }

    pub fn get(&self, hash: u64, key: &K) -> Option<V> {
        self.0
            .find(hash, |e| e.0 == hash && e.1 == *key)
            .map(|e| e.2.clone())
    }

    /// Removes an entry if present. Probes the shared table first so a miss
    /// never takes the copy-on-write clone.
    pub fn remove(mut self, hash: u64, key: K) -> Self {
        if self.contains(hash, &key) {
            let table = self.0.make_mut();
            if let Ok(occupied) = table.find_entry(hash, |e| e.0 == hash && e.1 == key) {
                occupied.remove();
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tests hash with an intentionally poor function so collisions are
    // exercised: distinct keys share buckets, and equality decides.
    fn hash(key: i64) -> u64 {
        (key % 3) as u64
    }

    fn insert(map: HashMap<i64, i64>, key: i64, value: i64) -> HashMap<i64, i64> {
        map.insert(hash(key), key, value)
    }

    #[test]
    fn map_operations() {
        let map = insert(insert(insert(HashMap::new(), 1, 10), 2, 20), 4, 40);

        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
        assert_eq!(map.get(hash(1), &1), Some(10));
        assert_eq!(map.get(hash(2), &2), Some(20));
        assert_eq!(map.get(hash(4), &4), Some(40));
        assert_eq!(map.get(hash(7), &7), None);
        assert!(map.contains(hash(4), &4));
        assert!(!map.contains(hash(7), &7));

        let map = insert(map, 2, 21);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(hash(2), &2), Some(21));

        let map = map.remove(hash(1), 1);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(hash(1), &1), None);
        let map = map.remove(hash(9), 9);
        assert_eq!(map.len(), 2);
        assert!(map.clear().is_empty());
    }

    #[test]
    fn mutation_is_copy_on_write() {
        let original = insert(insert(HashMap::new(), 1, 10), 2, 20);
        let changed = insert(original.clone().remove(hash(1), 1), 3, 30);

        assert_eq!(original.len(), 2);
        assert_eq!(original.get(hash(1), &1), Some(10));
        assert_eq!(original.get(hash(3), &3), None);
        assert_eq!(changed.len(), 2);
        assert_eq!(changed.get(hash(1), &1), None);
        assert_eq!(changed.get(hash(3), &3), Some(30));
    }

    #[test]
    fn growth_keeps_cached_hashes_consistent() {
        let mut map = HashMap::new();
        for key in 0..1000 {
            map = map.insert(key as u64, key, key * 2);
        }
        assert_eq!(map.len(), 1000);
        for key in 0..1000 {
            assert_eq!(map.get(key as u64, &key), Some(key * 2));
        }
    }
}
