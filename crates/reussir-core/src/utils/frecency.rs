//! A generic frecency model over keys.
//!
//! "Frecency" = frequency × recency: a key used often *and* recently scores
//! higher than one used long ago or rarely. The Reussir elaborator feeds this
//! into its fuzzy suggester ([`crate::utils::fuzzy`]) so that, among several
//! plausible typo corrections, the name the program actually leans on wins —
//! e.g. after a body resolves `List` twenty times, a stray `Lst` is far more
//! likely a slip for `List` than for some rarely-touched `Last`.
//!
//! Recency is measured on a logical clock that ticks once per recorded use
//! (there is no wall clock during elaboration). Each entry decays with a fixed
//! half-life (in ticks) and is folded forward lazily — we never sweep the whole
//! table — using the standard exponential model atuin uses for shell history:
//! `score · 2^(-Δticks / half_life)`.

use std::hash::Hash;

use rustc_hash::FxHashMap;

/// Number of subsequent uses after which a past use's weight halves. Large
/// enough that a key stays "warm" across a whole function body, small enough
/// that a key only touched in the prelude fades by the time later bodies fail
/// to resolve something.
const HALF_LIFE: f32 = 50.0;

#[derive(Clone, Copy)]
struct Entry {
    /// Accumulated, decay-folded weight as of `last`.
    weight: f32,
    /// Logical tick at which `weight` was last updated.
    last: u64,
}

/// A frecency accumulator keyed by `K`. `K` is typically a cheap interned id
/// (e.g. a token key), so it is taken and stored by value.
pub struct Frecency<K> {
    clock: u64,
    entries: FxHashMap<K, Entry>,
}

// Derived `Default` would demand `K: Default`; an empty table needs no such
// bound, so implement it by hand.
impl<K> Default for Frecency<K> {
    fn default() -> Self {
        Self {
            clock: 0,
            entries: FxHashMap::default(),
        }
    }
}

impl<K: Eq + Hash> Frecency<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exponential decay factor for `dt` elapsed ticks.
    fn decay(dt: u64) -> f32 {
        // 2^(-dt / HALF_LIFE); dt as f32 is exact for any realistic tick count.
        (-(dt as f32) / HALF_LIFE).exp2()
    }

    /// Record one use of `key`, advancing the logical clock. Folds the existing
    /// weight forward to now before adding this use, so frequency compounds while
    /// recency decays.
    pub fn record(&mut self, key: K) {
        let now = self.clock;
        let entry = self.entries.entry(key).or_insert(Entry {
            weight: 0.0,
            last: now,
        });
        entry.weight = entry.weight * Self::decay(now - entry.last) + 1.0;
        entry.last = now;
        self.clock += 1;
    }

    /// The current frecency of `key` (0 if never used), decayed to the present
    /// tick. Reading does not advance the clock.
    pub fn score(&self, key: K) -> f32 {
        match self.entries.get(&key) {
            Some(e) => e.weight * Self::decay(self.clock - e.last),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unused_key_scores_zero() {
        let f: Frecency<u32> = Frecency::new();
        assert_eq!(f.score(1), 0.0);
    }

    #[test]
    fn frequency_compounds() {
        let mut f: Frecency<u32> = Frecency::new();
        let one = f.score(1);
        f.record(1);
        let after_one = f.score(1);
        f.record(1);
        let after_two = f.score(1);
        assert!(after_one > one);
        assert!(after_two > after_one);
    }

    #[test]
    fn recency_breaks_a_frequency_tie() {
        // Both keys are used, but `b` is used much more recently; its activity
        // must outweigh `a`'s single early use after heavy decay.
        let mut f: Frecency<u32> = Frecency::new();
        f.record(1);
        for _ in 0..100 {
            f.record(2);
        }
        assert!(f.score(2) > f.score(1));
    }

    #[test]
    fn a_recent_single_use_can_lose_to_heavy_recent_use() {
        let mut f: Frecency<u32> = Frecency::new();
        for _ in 0..20 {
            f.record(2);
        }
        f.record(1);
        assert!(f.score(2) > f.score(1));
    }
}
