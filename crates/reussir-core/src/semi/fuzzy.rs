//! A small fuzzy-name index for "did you mean" suggestions.
//!
//! When a path fails to resolve, surfacing the closest in-scope spelling turns
//! an opaque "unknown name" into an actionable hint. This module is the search
//! tool behind those hints: build a [`FuzzyIndex`] from the names that *were*
//! in scope, then ask it for the nearest neighbour of the name the user wrote.
//!
//! Closeness is Damerau-Levenshtein edit distance, computed with the `strsim`
//! crate (the de-facto string-similarity library in the Rust ecosystem).
//! Damerau-Levenshtein counts an adjacent transposition (`valeu` → `value`) as
//! a single edit, which matters because transpositions are one of the most
//! common typo classes. The acceptance threshold follows rustc's
//! `find_best_match_for_name`: a candidate is only proposed when its distance is
//! within roughly a third of the longer spelling, so a typo like `Lst` → `List`
//! is suggested but a wholly different name is not. A name that differs only by
//! letter case is always preferred — it is overwhelmingly the intended one.

use strsim::damerau_levenshtein;

/// The largest edit distance at which a candidate is still a plausible typo of
/// a query, given the two spellings' lengths.
///
/// Mirrors rustc: about a third of the longer spelling, with a floor of one
/// edit so very short names (`a`, `Id`) still tolerate a single slip.
fn threshold(a: usize, b: usize) -> usize {
    a.max(b).max(3) / 3
}

/// An index of candidate names searchable by approximate spelling.
///
/// Holds borrowed names, so it allocates nothing beyond its backing vector;
/// build it from the resolved names in scope at an error site and drop it once
/// the hint is formatted.
#[derive(Default)]
pub struct FuzzyIndex<'a> {
    names: Vec<&'a str>,
}

impl<'a> FuzzyIndex<'a> {
    /// An empty index.
    pub fn new() -> Self {
        Self { names: Vec::new() }
    }

    /// Add a candidate name to the index.
    pub fn insert(&mut self, name: &'a str) {
        self.names.push(name);
    }

    /// The single closest candidate to `query`, or `None` when nothing is
    /// within the typo threshold.
    ///
    /// A candidate that matches `query` case-insensitively wins outright (the
    /// most likely intent); otherwise the smallest in-threshold Levenshtein
    /// distance wins, ties broken by insertion order. The query's own exact
    /// spelling is never suggested back — if it were in the index the lookup
    /// would not have failed.
    pub fn suggest(&self, query: &str) -> Option<&'a str> {
        // A case-only difference (`foo` vs `Foo`) is the strongest signal and
        // can exceed the edit-distance threshold for short names, so check it
        // first and independently.
        if let Some(&hit) = self
            .names
            .iter()
            .find(|c| **c != query && c.eq_ignore_ascii_case(query))
        {
            return Some(hit);
        }

        let mut best: Option<(usize, &'a str)> = None;
        for &cand in &self.names {
            if cand == query {
                continue;
            }
            let dist = damerau_levenshtein(query, cand);
            if dist > threshold(query.len(), cand.len()) {
                continue;
            }
            if best.is_none_or(|(d, _)| dist < d) {
                best = Some((dist, cand));
            }
        }
        best.map(|(_, name)| name)
    }
}

impl<'a> FromIterator<&'a str> for FuzzyIndex<'a> {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        Self {
            names: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggests_a_close_typo() {
        let index: FuzzyIndex = ["List", "Option", "Vec"].into_iter().collect();
        assert_eq!(index.suggest("Lst"), Some("List"));
        assert_eq!(index.suggest("Optoin"), Some("Option"));
    }

    #[test]
    fn treats_a_transposition_as_one_edit() {
        // `valeu` is `value` with the last two letters swapped: a single
        // Damerau-Levenshtein edit, within the threshold of 1 for a 5-char name.
        let index: FuzzyIndex = ["value", "length"].into_iter().collect();
        assert_eq!(index.suggest("valeu"), Some("value"));
    }

    #[test]
    fn prefers_a_case_only_match() {
        let index: FuzzyIndex = ["Foo", "Food"].into_iter().collect();
        // `foo` is one edit from both `Foo` (case) and `Food` (case + insert);
        // the case-only match must win even though distances tie at the limit.
        assert_eq!(index.suggest("foo"), Some("Foo"));
    }

    #[test]
    fn case_only_match_beats_the_distance_floor() {
        // `ID` vs `id` is distance 2, above threshold(2,2)=1; the dedicated
        // case-insensitive check still finds it.
        let index: FuzzyIndex = ["id"].into_iter().collect();
        assert_eq!(index.suggest("ID"), Some("id"));
    }

    #[test]
    fn rejects_a_wildly_different_name() {
        let index: FuzzyIndex = ["List", "Option"].into_iter().collect();
        assert_eq!(index.suggest("HashMap"), None);
    }

    #[test]
    fn never_suggests_the_query_itself() {
        let index: FuzzyIndex = ["List"].into_iter().collect();
        assert_eq!(index.suggest("List"), None);
    }

    #[test]
    fn picks_the_nearest_of_several() {
        let index: FuzzyIndex = ["foobar", "foobaz", "foo"].into_iter().collect();
        // `fooba` is one edit from both `foobar` and `foobaz`; insertion order
        // breaks the tie toward `foobar`.
        assert_eq!(index.suggest("fooba"), Some("foobar"));
    }

    #[test]
    fn empty_index_suggests_nothing() {
        let index = FuzzyIndex::new();
        assert_eq!(index.suggest("anything"), None);
    }
}
