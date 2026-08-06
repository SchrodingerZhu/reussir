//! rapidhash-backed hashing for Reussir's `std::hash`.
//!
//! Two hashers are exposed, differing only in rapidhash's `AVALANCHE`
//! setting: [`FastHasher`] wraps `rapidhash::fast::RapidHasher` (avalanche
//! off — tuned for hash-table throughput) and [`StrongHasher`] wraps
//! `rapidhash::quality::RapidHasher` (avalanche on — tuned for output
//! quality). Both produce output bit-identical to the upstream hasher of
//! the same name and seed; nothing about the algorithm is reimplemented
//! here, only marshalled.
//!
//! Reussir's `Hasher` trait is state-passing — each write consumes the
//! hasher and returns the advanced one — so the state survives across
//! calls behind the opaque-FFI contract: a `#[repr(transparent)]`
//! wrapper over [`Rc`], so the compiler manages it as an ordinary
//! rc object. Writes go in place while the box is uniquely owned, which
//! it is when the hasher is threaded linearly; `make_mut` copies if it
//! has been shared.
//!
//! One-shot entry points sit alongside for hashing a single value, where
//! building a hasher would be pure overhead.

use std::hash::Hasher as _;

use rapidhash::{fast, quality};

use crate::rc::Rc;

/// The seed both hashers use when none is given: rapidhash's own default,
/// so an unseeded Reussir hasher agrees with an unseeded upstream one.
pub const DEFAULT_SEED: u64 = fast::RapidHasher::DEFAULT_SEED;

/// Streaming hasher tuned for speed (rapidhash `fast`: avalanche off).
#[derive(Clone)]
#[repr(transparent)]
pub struct FastHasher(Rc<fast::RapidHasher<'static>>);

impl Default for FastHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl FastHasher {
    pub fn new() -> Self {
        Self(Rc::new(fast::RapidHasher::default_const()))
    }

    pub fn with_seed(seed: u64) -> Self {
        Self(Rc::new(fast::RapidHasher::new(seed)))
    }

    pub fn write_u64(mut self, value: u64) -> Self {
        self.0.make_mut().write_u64(value);
        self
    }

    pub fn write_u32(mut self, value: u32) -> Self {
        self.0.make_mut().write_u32(value);
        self
    }

    pub fn write_u16(mut self, value: u16) -> Self {
        self.0.make_mut().write_u16(value);
        self
    }

    pub fn write_u8(mut self, value: u8) -> Self {
        self.0.make_mut().write_u8(value);
        self
    }

    pub fn finish(&self) -> u64 {
        self.0.data_ref().finish()
    }
}

/// Streaming hasher tuned for output quality (rapidhash `quality`:
/// avalanche on).
#[derive(Clone)]
#[repr(transparent)]
pub struct StrongHasher(Rc<quality::RapidHasher<'static>>);

impl Default for StrongHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl StrongHasher {
    pub fn new() -> Self {
        Self(Rc::new(quality::RapidHasher::default_const()))
    }

    pub fn with_seed(seed: u64) -> Self {
        Self(Rc::new(quality::RapidHasher::new(seed)))
    }

    pub fn write_u64(mut self, value: u64) -> Self {
        self.0.make_mut().write_u64(value);
        self
    }

    pub fn write_u32(mut self, value: u32) -> Self {
        self.0.make_mut().write_u32(value);
        self
    }

    pub fn write_u16(mut self, value: u16) -> Self {
        self.0.make_mut().write_u16(value);
        self
    }

    pub fn write_u8(mut self, value: u8) -> Self {
        self.0.make_mut().write_u8(value);
        self
    }

    pub fn finish(&self) -> u64 {
        self.0.data_ref().finish()
    }
}

/// One-shot: the `fast` digest of a single 64-bit value under the default
/// seed. Equivalent to creating a hasher, writing, and finishing, without
/// the allocation.
pub fn fast_hash_u64(value: u64) -> u64 {
    let mut h = fast::RapidHasher::default_const();
    h.write_u64(value);
    h.finish()
}

/// One-shot `fast` digest under an explicit seed.
pub fn fast_hash_u64_seeded(value: u64, seed: u64) -> u64 {
    let mut h = fast::RapidHasher::new(seed);
    h.write_u64(value);
    h.finish()
}

/// One-shot: the `quality` digest of a single 64-bit value under the
/// default seed.
pub fn strong_hash_u64(value: u64) -> u64 {
    let mut h = quality::RapidHasher::default_const();
    h.write_u64(value);
    h.finish()
}

/// One-shot `quality` digest under an explicit seed.
pub fn strong_hash_u64_seeded(value: u64, seed: u64) -> u64 {
    let mut h = quality::RapidHasher::new(seed);
    h.write_u64(value);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrappers must agree with the upstream hasher they wrap; if
    /// they ever diverge, the marshalling is at fault, not rapidhash.
    #[test]
    fn matches_upstream_fast() {
        let mut upstream = fast::RapidHasher::default_const();
        upstream.write_u64(42);
        upstream.write_u64(7);
        let wrapped = FastHasher::new().write_u64(42).write_u64(7);
        assert_eq!(wrapped.finish(), upstream.finish());
    }

    #[test]
    fn matches_upstream_quality() {
        let mut upstream = quality::RapidHasher::default_const();
        upstream.write_u64(42);
        upstream.write_u64(7);
        let wrapped = StrongHasher::new().write_u64(42).write_u64(7);
        assert_eq!(wrapped.finish(), upstream.finish());
    }

    #[test]
    fn one_shot_matches_streaming() {
        assert_eq!(fast_hash_u64(1234), FastHasher::new().write_u64(1234).finish());
        assert_eq!(
            strong_hash_u64(1234),
            StrongHasher::new().write_u64(1234).finish()
        );
    }

    /// A shared hasher must not be advanced by a write to one of its
    /// users: `make_mut` has to copy when the box is shared. Reussir's
    /// ownership analysis makes sharing possible, so this is a real case.
    #[test]
    fn sharing_copies_rather_than_aliasing() {
        let base = FastHasher::new().write_u64(1);
        let branch_a = base.clone().write_u64(2);
        let branch_b = base.clone().write_u64(3);
        assert_ne!(branch_a.finish(), branch_b.finish());
        // The original is unchanged by either branch.
        assert_eq!(base.finish(), FastHasher::new().write_u64(1).finish());
    }

    /// The two variants differ (avalanche on vs off), so a caller
    /// choosing between them is choosing between distinct digests.
    #[test]
    fn fast_and_strong_differ() {
        assert_ne!(fast_hash_u64(99), strong_hash_u64(99));
    }

    #[test]
    fn seeding_changes_the_digest() {
        assert_ne!(fast_hash_u64_seeded(5, 1), fast_hash_u64_seeded(5, 2));
        assert_ne!(strong_hash_u64_seeded(5, 1), strong_hash_u64_seeded(5, 2));
        assert_eq!(fast_hash_u64_seeded(5, DEFAULT_SEED), fast_hash_u64(5));
    }
}
