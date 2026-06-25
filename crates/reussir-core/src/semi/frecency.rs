//! A frecency model over interned names, accumulated during type checking.
//!
//! "Frecency" = frequency × recency: a name used often *and* recently scores
//! higher than one used long ago or rarely. We feed this score into the fuzzy
//! suggester ([`crate::semi::fuzzy`]) so that, among several plausible typo
//! corrections, the one the program actually leans on wins — e.g. after a body
//! resolves `List` twenty times, a stray `Lst` is far more likely a slip for
//! `List` than for some rarely-touched `Last`.
//!
//! There is no wall clock during elaboration, so "recency" is measured on a
//! logical clock that ticks once per recorded use. Each entry decays with a
//! fixed half-life (in ticks) and is folded forward lazily — we never sweep the
//! whole table — using the standard exponential model atuin uses for shell
//! history: `score · 2^(-Δticks / half_life)`.

use rustc_hash::FxHashMap;

use reussir_syntax::kind::TokenKey;

/// Number of subsequent uses after which a past use's weight halves. Large
/// enough that a name stays "warm" across a whole function body, small enough
/// that a name only touched in the prelude fades by the time later bodies fail
/// to resolve something.
const HALF_LIFE: f32 = 50.0;

#[derive(Clone, Copy)]
struct Entry {
    /// Accumulated, decay-folded weight as of `last`.
    weight: f32,
    /// Logical tick at which `weight` was last updated.
    last: u64,
}

/// Per-name frecency accumulator, keyed by interned [`TokenKey`].
#[derive(Default)]
pub struct Frecency {
    clock: u64,
    entries: FxHashMap<TokenKey, Entry>,
}

impl Frecency {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exponential decay factor for `dt` elapsed ticks.
    fn decay(dt: u64) -> f32 {
        // 2^(-dt / HALF_LIFE); dt as f32 is exact for any realistic tick count.
        (-(dt as f32) / HALF_LIFE).exp2()
    }

    /// Record one use of `name`, advancing the logical clock. Folds the existing
    /// weight forward to now before adding this use, so frequency compounds while
    /// recency decays.
    pub fn record(&mut self, name: TokenKey) {
        let now = self.clock;
        let entry = self.entries.entry(name).or_insert(Entry {
            weight: 0.0,
            last: now,
        });
        entry.weight = entry.weight * Self::decay(now - entry.last) + 1.0;
        entry.last = now;
        self.clock += 1;
    }

    /// The current frecency of `name` (0 if never used), decayed to the present
    /// tick. Reading does not advance the clock.
    pub fn score(&self, name: TokenKey) -> f32 {
        match self.entries.get(&name) {
            Some(e) => e.weight * Self::decay(self.clock - e.last),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reussir_syntax::kind::InternKey;

    fn k(n: u32) -> TokenKey {
        TokenKey::try_from_u32(n).expect("nonzero key")
    }

    #[test]
    fn unused_name_scores_zero() {
        let f = Frecency::new();
        assert_eq!(f.score(k(1)), 0.0);
    }

    #[test]
    fn frequency_compounds() {
        let mut f = Frecency::new();
        let one = f.score(k(1));
        f.record(k(1));
        let after_one = f.score(k(1));
        f.record(k(1));
        let after_two = f.score(k(1));
        assert!(after_one > one);
        assert!(after_two > after_one);
    }

    #[test]
    fn recency_breaks_a_frequency_tie() {
        // Both names are used once, but `b` is used much more recently. With a
        // single use each, the more recent one must score at least as high.
        let mut f = Frecency::new();
        let a = k(1);
        let b = k(2);
        f.record(a);
        for _ in 0..100 {
            f.record(b);
        }
        // `a`'s single early use has decayed well below `b`'s recent activity.
        assert!(f.score(b) > f.score(a));
    }

    #[test]
    fn a_recent_single_use_can_lose_to_heavy_recent_use() {
        let mut f = Frecency::new();
        let rare = k(1);
        let common = k(2);
        for _ in 0..20 {
            f.record(common);
        }
        f.record(rare);
        assert!(f.score(common) > f.score(rare));
    }
}
