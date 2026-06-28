//! A small-set-optimized bitset over `u32`.
//!
//! Most live-variable / index sets in a compiler are small and dense (a handful
//! of low ids), the regime where a flat word bitset is cheapest; a few are large
//! or sparse, where a compressed bitmap wins. [`HybridBitSet`] starts dense — a
//! [`FixedBitSet`] sized to one platform register, grown on demand (a register at
//! a time) up to [`MAX_DENSE_BITS`] — and switches **once** to a [`RoaringBitmap`]
//! the first time an element at or beyond that bound is inserted. The switch is
//! one-way: the set never demotes back to dense.
//!
//! Equality is by *contents*, independent of representation, so a `HybridBitSet`
//! can be used as a value in the usual way.

use std::fmt;

use fixedbitset::FixedBitSet;
use roaring::RoaringBitmap;

/// The dense representation's cap, in bits. Elements `< MAX_DENSE_BITS` live in a
/// [`FixedBitSet`]; inserting one at or beyond it promotes to a [`RoaringBitmap`].
/// 512 bits is 8 registers on a 64-bit target — small enough to stay cheap, large
/// enough to cover the common case without ever allocating a bitmap.
const MAX_DENSE_BITS: u32 = 512;

/// A set of `u32`s that is dense for small domains and compressed for large ones.
/// See the module docs.
#[derive(Clone)]
pub enum HybridBitSet {
    /// Every element is `< MAX_DENSE_BITS`. Capacity begins at one platform
    /// register ([`usize::BITS`]) and grows a register at a time, never past the
    /// cap.
    Dense(FixedBitSet),
    /// At least one element reached `MAX_DENSE_BITS`, so the whole set moved to a
    /// compressed bitmap.
    Sparse(RoaringBitmap),
}

impl HybridBitSet {
    /// An empty set, dense, sized to one platform register.
    pub fn new() -> Self {
        HybridBitSet::Dense(FixedBitSet::with_capacity(usize::BITS as usize))
    }

    /// Insert `value`. A dense set grows to fit (rounding capacity up to a whole
    /// platform register, up to [`MAX_DENSE_BITS`]); a `value` at or beyond the cap
    /// promotes the set to a [`RoaringBitmap`].
    pub fn insert(&mut self, value: u32) {
        match self {
            HybridBitSet::Sparse(bitmap) => {
                bitmap.insert(value);
            }
            HybridBitSet::Dense(dense) if value < MAX_DENSE_BITS => {
                dense.grow(register_aligned_len(value));
                dense.insert(value as usize);
            }
            // Beyond the cap: promote self by inserting into a bitmap copy.
            this @ HybridBitSet::Dense(_) => {
                let mut bitmap = this.to_bitmap();
                bitmap.insert(value);
                *this = HybridBitSet::Sparse(bitmap);
            }
        }
    }

    /// Is `value` in the set?
    pub fn contains(&self, value: u32) -> bool {
        match self {
            // `FixedBitSet::contains` treats out-of-capacity bits as unset.
            HybridBitSet::Dense(dense) => dense.contains(value as usize),
            HybridBitSet::Sparse(bitmap) => bitmap.contains(value),
        }
    }

    /// In place: `self ← self ∪ other`. If `other` is compressed (so it may hold
    /// elements beyond the cap) `self` is promoted to match.
    pub fn union_with(&mut self, other: &HybridBitSet) {
        match (self, other) {
            (HybridBitSet::Dense(a), HybridBitSet::Dense(b)) => a.union_with(b),
            (HybridBitSet::Sparse(a), HybridBitSet::Sparse(b)) => *a |= b,
            (HybridBitSet::Sparse(a), HybridBitSet::Dense(b)) => {
                a.extend(b.ones().map(|i| i as u32));
            }
            // Dense ∪ Sparse: `other` may hold elements past the cap, so promote
            // self by folding its bits into a copy of the compressed `b`.
            (this @ HybridBitSet::Dense(_), HybridBitSet::Sparse(b)) => {
                let mut bitmap = this.to_bitmap();
                bitmap |= b;
                *this = HybridBitSet::Sparse(bitmap);
            }
        }
    }

    /// Remove every member of `other` (set difference, in place). Never changes
    /// representation — a difference only removes elements.
    pub fn subtract(&mut self, other: &HybridBitSet) {
        match (self, other) {
            (HybridBitSet::Dense(a), HybridBitSet::Dense(b)) => a.difference_with(b),
            (HybridBitSet::Sparse(a), HybridBitSet::Sparse(b)) => *a -= b,
            (HybridBitSet::Sparse(a), HybridBitSet::Dense(b)) => {
                for i in b.ones() {
                    a.remove(i as u32);
                }
            }
            (HybridBitSet::Dense(a), HybridBitSet::Sparse(b)) => {
                let drop: Vec<usize> = a.ones().filter(|&i| b.contains(i as u32)).collect();
                for i in drop {
                    a.remove(i);
                }
            }
        }
    }

    /// The members in ascending order (both backends iterate sorted, so emission
    /// order is deterministic).
    pub fn iter(&self) -> Iter<'_> {
        match self {
            HybridBitSet::Dense(dense) => Iter::Dense(dense.ones()),
            HybridBitSet::Sparse(bitmap) => Iter::Sparse(bitmap.iter()),
        }
    }

    /// Materialize the current contents as a [`RoaringBitmap`] (used when a dense
    /// set has to promote).
    fn to_bitmap(&self) -> RoaringBitmap {
        match self {
            HybridBitSet::Dense(dense) => dense.ones().map(|i| i as u32).collect(),
            HybridBitSet::Sparse(bitmap) => bitmap.clone(),
        }
    }
}

/// The dense length needed to hold `value`, rounded up to a whole platform
/// register so growth happens a register at a time rather than bit by bit.
fn register_aligned_len(value: u32) -> usize {
    (value + 1).next_multiple_of(usize::BITS) as usize
}

impl Default for HybridBitSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Contents equality, independent of representation (a dense and a compressed set
/// holding the same elements compare equal).
impl PartialEq for HybridBitSet {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for HybridBitSet {}

impl fmt::Debug for HybridBitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<u32> for HybridBitSet {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut set = HybridBitSet::new();
        for v in iter {
            set.insert(v);
        }
        set
    }
}

/// Ascending iterator over a [`HybridBitSet`]'s members, over whichever backend
/// the set currently uses.
pub enum Iter<'a> {
    Dense(fixedbitset::Ones<'a>),
    Sparse(roaring::bitmap::Iter<'a>),
}

impl Iterator for Iter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        match self {
            Iter::Dense(ones) => ones.next().map(|i| i as u32),
            Iter::Sparse(bitmap) => bitmap.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference set built straight on a `BTreeSet<u32>` to check against.
    fn check(values: &[u32]) {
        use std::collections::BTreeSet;
        let reference: BTreeSet<u32> = values.iter().copied().collect();
        let set: HybridBitSet = values.iter().copied().collect();
        for &v in values {
            assert!(set.contains(v), "missing {v}");
        }
        assert!(!set.contains(99_999));
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            reference.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn dense_small() {
        check(&[0, 1, 5, 63, 64, 200, 511]);
    }

    #[test]
    fn promotes_past_the_cap() {
        // 511 stays dense, 512 forces a promotion; both must read back.
        let mut set = HybridBitSet::new();
        set.insert(511);
        assert!(matches!(set, HybridBitSet::Dense(_)));
        set.insert(512);
        assert!(matches!(set, HybridBitSet::Sparse(_)));
        assert!(set.contains(511) && set.contains(512));
        check(&[1, 511, 512, 1_000, 70_000]);
    }

    #[test]
    fn dense_capacity_is_register_aligned() {
        let reg = usize::BITS as usize;
        let mut set = HybridBitSet::new();
        let HybridBitSet::Dense(d) = &set else {
            unreachable!()
        };
        assert_eq!(d.len(), reg); // one register up front

        set.insert(reg as u32); // first bit past the initial register
        let HybridBitSet::Dense(d) = &set else {
            unreachable!()
        };
        assert_eq!(d.len(), 2 * reg); // grown by exactly one more register
    }

    #[test]
    fn union_across_representations() {
        // dense ∪ sparse must promote and keep both.
        let mut a: HybridBitSet = [1, 2, 511].into_iter().collect();
        let b: HybridBitSet = [2, 600].into_iter().collect();
        a.union_with(&b);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1, 2, 511, 600]);

        // sparse ∪ dense keeps the sparse backing.
        let mut c: HybridBitSet = [700].into_iter().collect();
        let d: HybridBitSet = [3, 4].into_iter().collect();
        c.union_with(&d);
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![3, 4, 700]);
    }

    #[test]
    fn subtract_across_representations() {
        let mut a: HybridBitSet = [1, 2, 3, 600].into_iter().collect();
        let b: HybridBitSet = [2, 600].into_iter().collect();
        a.subtract(&b);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1, 3]);

        let mut c: HybridBitSet = [1, 2, 3].into_iter().collect();
        let d: HybridBitSet = [2, 900].into_iter().collect();
        c.subtract(&d);
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn equality_is_representation_independent() {
        let dense: HybridBitSet = [1, 2, 3].into_iter().collect();
        let mut promoted: HybridBitSet = [1, 2, 3, 9_999].into_iter().collect();
        promoted.subtract(&[9_999].into_iter().collect());
        // `promoted` is Sparse but holds {1,2,3}, same as the dense set.
        assert!(matches!(promoted, HybridBitSet::Sparse(_)));
        assert_eq!(dense, promoted);
    }
}
