//! A frecency-weighted fuzzy-name index for "did you mean" suggestions.
//!
//! When a path fails to resolve, surfacing the closest in-scope spelling turns
//! an opaque "unknown name" into an actionable hint. This module is the search
//! tool behind those hints: build a [`FuzzyIndex`] from the names that *were*
//! in scope (each paired with its [frecency] from the type-checking pass), then
//! ask it for the best correction of the name the user wrote.
//!
//! Ranking blends three signals:
//!
//! 1. **Case-only match** (`foo` vs `Foo`) is the strongest signal and wins
//!    outright — it is almost always the intended name.
//! 2. **Fuzzy subsequence score** from `atuin-nucleo-matcher` (atuin's fork of
//!    helix's `nucleo`, an fzf-quality matcher). This is what ranks the bulk of
//!    near-misses, exactly as a fuzzy finder would.
//! 3. **Edit distance** (Damerau-Levenshtein, via `strsim`) as a fallback for
//!    the typo class nucleo's subsequence model misses — adjacent
//!    transpositions like `valeu` → `value`, where the swapped letters break
//!    the left-to-right subsequence nucleo requires.
//!
//! Within signals 2 and 3 the raw match strength is multiplied by a frecency
//! factor (see [`Frecency`](crate::semi::frecency::Frecency)), so that among
//! equally-plausible spellings the name the program actually uses most — and
//! most recently — is preferred. The acceptance threshold for the edit-distance
//! fallback follows rustc's `find_best_match_for_name`: within roughly a third
//! of the longer spelling, so genuine typos are offered but unrelated names are
//! not.
//!
//! [frecency]: crate::semi::frecency

use atuin_nucleo_matcher::{Config, Matcher, Utf32Str};
use strsim::damerau_levenshtein;

/// How much frecency may tilt the ranking. A factor of `1 + WEIGHT·(frec/max)`
/// means the most-used candidate gets up to a `WEIGHT` fractional boost over an
/// otherwise-equal but never-used one. Kept modest so a clearly-better textual
/// match still wins over a popular-but-distant one.
const FRECENCY_WEIGHT: f32 = 0.5;

/// The largest edit distance at which a candidate is still a plausible typo of a
/// query, given the two spellings' lengths. Mirrors rustc: about a third of the
/// longer spelling, with a floor of one edit so very short names still tolerate
/// a single slip.
fn edit_threshold(a: usize, b: usize) -> usize {
    a.max(b).max(3) / 3
}

struct Candidate<'a> {
    name: &'a str,
    frecency: f32,
}

/// An index of candidate names searchable by approximate spelling, weighted by
/// how heavily each name is used. Holds borrowed names, so it allocates nothing
/// beyond its backing vector; build it at an error site and drop it once the
/// hint is formatted.
#[derive(Default)]
pub struct FuzzyIndex<'a> {
    candidates: Vec<Candidate<'a>>,
}

impl<'a> FuzzyIndex<'a> {
    /// An empty index.
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Add a candidate `name` with the given `frecency` weight (use `0.0` for
    /// names that carry no usage history, e.g. built-in trait names).
    pub fn insert(&mut self, name: &'a str, frecency: f32) {
        self.candidates.push(Candidate { name, frecency });
    }

    /// The single best correction of `query`, or `None` when nothing is close
    /// enough. The query's own exact spelling is never suggested back — if it
    /// were in the index the lookup would not have failed.
    pub fn suggest(&self, query: &str) -> Option<&'a str> {
        if let Some(hit) = self.case_only_match(query) {
            return Some(hit);
        }
        self.best_fuzzy(query).or_else(|| self.best_edit(query))
    }

    /// A candidate identical to `query` up to ASCII case.
    fn case_only_match(&self, query: &str) -> Option<&'a str> {
        self.candidates
            .iter()
            .find(|c| c.name != query && c.name.eq_ignore_ascii_case(query))
            .map(|c| c.name)
    }

    /// The largest frecency among candidates, used to normalize the boost into
    /// a bounded multiplier.
    fn max_frecency(&self) -> f32 {
        self.candidates
            .iter()
            .map(|c| c.frecency)
            .fold(0.0_f32, f32::max)
    }

    /// `1 + WEIGHT·(frecency / max)`, or `1` when nothing has been used.
    fn frecency_factor(&self, frecency: f32, max: f32) -> f32 {
        if max <= 0.0 {
            1.0
        } else {
            1.0 + FRECENCY_WEIGHT * (frecency / max)
        }
    }

    /// Best nucleo subsequence match, frecency-weighted. `None` if no candidate
    /// even fuzzy-matches the query.
    fn best_fuzzy(&self, query: &str) -> Option<&'a str> {
        let mut matcher = Matcher::new(Config::DEFAULT);
        // With `Config::DEFAULT.ignore_case`, nucleo folds the *haystack* itself
        // but requires the needle to be pre-case-folded by the caller, so match
        // against a lowercased query.
        let lowered = query.to_lowercase();
        let mut needle_buf = Vec::new();
        let needle = Utf32Str::new(&lowered, &mut needle_buf);
        let max = self.max_frecency();

        let mut best: Option<(f32, &'a str)> = None;
        for cand in &self.candidates {
            if cand.name == query {
                continue;
            }
            let mut hay_buf = Vec::new();
            let haystack = Utf32Str::new(cand.name, &mut hay_buf);
            let Some(score) = matcher.fuzzy_match(haystack, needle) else {
                continue;
            };
            let weighted = f32::from(score) * self.frecency_factor(cand.frecency, max);
            if best.is_none_or(|(b, _)| weighted > b) {
                best = Some((weighted, cand.name));
            }
        }
        best.map(|(_, name)| name)
    }

    /// Best edit-distance match within the typo threshold, frecency-weighted.
    /// Catches transpositions and substitutions nucleo's subsequence model skips.
    fn best_edit(&self, query: &str) -> Option<&'a str> {
        let max = self.max_frecency();
        let mut best: Option<(f32, &'a str)> = None;
        for cand in &self.candidates {
            if cand.name == query {
                continue;
            }
            let limit = edit_threshold(query.len(), cand.name.len());
            let dist = damerau_levenshtein(query, cand.name);
            if dist > limit {
                continue;
            }
            // Closer (smaller distance) ⇒ higher base score; then weight it.
            let base = (limit + 1 - dist) as f32;
            let weighted = base * self.frecency_factor(cand.frecency, max);
            if best.is_none_or(|(b, _)| weighted > b) {
                best = Some((weighted, cand.name));
            }
        }
        best.map(|(_, name)| name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an index from names with no usage history.
    fn index<'a>(names: &[&'a str]) -> FuzzyIndex<'a> {
        let mut idx = FuzzyIndex::new();
        for &n in names {
            idx.insert(n, 0.0);
        }
        idx
    }

    #[test]
    fn suggests_a_close_typo() {
        let idx = index(&["List", "Option", "Vec"]);
        assert_eq!(idx.suggest("Lst"), Some("List"));
    }

    #[test]
    fn fallback_catches_a_transposition() {
        // `valeu` is `value` with the last two letters swapped: nucleo's
        // subsequence model can't match it, but the edit-distance fallback can.
        let idx = index(&["value", "length"]);
        assert_eq!(idx.suggest("valeu"), Some("value"));
    }

    #[test]
    fn prefers_a_case_only_match() {
        let idx = index(&["Foo", "Food"]);
        assert_eq!(idx.suggest("foo"), Some("Foo"));
    }

    #[test]
    fn case_only_match_beats_the_distance_floor() {
        // `ID` vs `id` is distance 2, above the edit threshold of 1; the
        // dedicated case-insensitive check still finds it.
        let idx = index(&["id"]);
        assert_eq!(idx.suggest("ID"), Some("id"));
    }

    #[test]
    fn rejects_a_wildly_different_name() {
        let idx = index(&["List", "Option"]);
        assert_eq!(idx.suggest("HashMap"), None);
    }

    #[test]
    fn never_suggests_the_query_itself() {
        let idx = index(&["List"]);
        assert_eq!(idx.suggest("List"), None);
    }

    #[test]
    fn empty_index_suggests_nothing() {
        let idx = FuzzyIndex::new();
        assert_eq!(idx.suggest("anything"), None);
    }

    #[test]
    fn frecency_breaks_a_tie() {
        // `lst` fuzzy-matches both `List` and `Last` equally well; the one with
        // higher frecency must win.
        let mut idx = FuzzyIndex::new();
        idx.insert("List", 10.0);
        idx.insert("Last", 0.0);
        assert_eq!(idx.suggest("lst"), Some("List"));

        let mut idx = FuzzyIndex::new();
        idx.insert("List", 0.0);
        idx.insert("Last", 10.0);
        assert_eq!(idx.suggest("lst"), Some("Last"));
    }
}
