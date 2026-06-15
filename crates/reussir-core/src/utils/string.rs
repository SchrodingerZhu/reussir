//! String interning tokens and their Rust v0 symbol mangling.
//!
//! A [`StringToken`] is the 256-bit BLAKE3 digest of a string literal, kept as
//! four 64-bit words. The digest both uniquifies the literal and gives us a
//! stable, content-addressed name we can mangle into a linker symbol.

use rapidhash::RapidHashMap;

use crate::utils::b62;

/// A content hash of a string literal, stored as four 64-bit words in
/// big-to-little significance order (word `0` is the most significant).
///
/// This mirrors the layout produced by `bit_cast`-ing a 32-byte BLAKE3 digest
/// into four little-endian `u64`s, so tokens built here match those produced by
/// the C++ bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringToken([u64; 4]);

impl StringToken {
    /// Builds a token from the four words directly. The first word is treated as
    /// the most significant when mangling.
    pub fn from_words(words: [u64; 4]) -> Self {
        Self(words)
    }

    /// Hashes raw bytes with BLAKE3 and packs the 32-byte digest into a token.
    ///
    /// The digest is split into four little-endian `u64`s, matching the
    /// `bit_cast` the C++ bridge performs on `llvm::BLAKE3::hash<32>`.
    pub fn hash(bytes: &[u8]) -> Self {
        let digest = blake3::hash(bytes);
        let bytes = digest.as_bytes();
        let word = |i: usize| {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            u64::from_le_bytes(chunk)
        };
        Self([word(0), word(1), word(2), word(3)])
    }

    /// Hashes a string's UTF-8 bytes into a token. Convenience wrapper over
    /// [`StringToken::hash`].
    pub fn from_text(s: &str) -> Self {
        Self::hash(s.as_bytes())
    }

    /// The four 64-bit words backing this token, most significant first.
    pub fn words(&self) -> [u64; 4] {
        self.0
    }

    /// Mangles the token into a Rust v0 linker symbol of the form
    /// `_RNvC22REUSSIR_STRING_LITERAL<len><sep><digits>`.
    ///
    /// `<digits>` is the base-62 encoding of the 256-bit token, `<len>` is its
    /// length, and `<sep>` is a single `_` inserted only when the leading digit
    /// is numeric — the v0 grammar forbids an identifier starting with a digit.
    pub fn mangle(&self) -> String {
        let encoded = b62::encode(self.0);
        let sep = if encoded.leading_is_numeric { "_" } else { "" };
        format!(
            "_RNvC22REUSSIR_STRING_LITERAL{}{sep}{}",
            encoded.len(),
            encoded.digits
        )
    }
}

/// Interns string literals, handing back a stable [`StringToken`] for each
/// distinct string and remembering the originals.
///
/// Strings are deduplicated through a rapidhash-keyed map, so a literal seen
/// more than once is hashed only the first time.
#[derive(Debug, Default)]
pub struct StringUniqifier {
    storage: RapidHashMap<String, StringToken>,
}

impl StringUniqifier {
    /// Creates an empty uniqifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the token for `s`, hashing and recording it the first time it is
    /// seen and reusing the stored token on subsequent calls.
    pub fn allocate(&mut self, s: &str) -> StringToken {
        if let Some(&token) = self.storage.get(s) {
            return token;
        }
        let token = StringToken::from_text(s);
        self.storage.insert(s.to_owned(), token);
        token
    }

    /// Iterates over every interned string paired with its token, in arbitrary
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, StringToken)> {
        self.storage.iter().map(|(s, &token)| (s.as_str(), token))
    }

    /// The number of distinct strings interned so far.
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Whether no strings have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangle_has_expected_prefix() {
        let mangled = StringToken::from_words([1, 2, 3, 4]).mangle();
        assert!(mangled.starts_with("_RNvC22REUSSIR_STRING_LITERAL"));
    }

    #[test]
    fn mangle_separator_when_leading_digit_numeric() {
        let token = StringToken::from_words([1, 2, 3, 4]);
        let encoded = b62::encode(token.words());
        let mangled = token.mangle();
        let sep = if encoded.leading_is_numeric { "_" } else { "" };
        assert!(mangled.ends_with(&format!("{sep}{}", encoded.digits)));
        assert_eq!(
            encoded.digits.chars().next().unwrap().is_ascii_digit(),
            encoded.leading_is_numeric
        );
    }

    #[test]
    fn hash_matches_bitcast_layout() {
        // The most significant word is the first little-endian u64 of the digest.
        let token = StringToken::from_text("hello");
        let digest = blake3::hash(b"hello");
        let bytes = digest.as_bytes();
        let mut first = [0u8; 8];
        first.copy_from_slice(&bytes[0..8]);
        assert_eq!(token.words()[0], u64::from_le_bytes(first));
    }

    #[test]
    fn mangle_golden_vectors() {
        // Pinned to catch any endianness, word-order, or base-62 regression in
        // the full 256-bit hash -> mangle chain.
        assert_eq!(
            StringToken::from_text("hello world").mangle(),
            "_RNvC22REUSSIR_STRING_LITERAL43wgb3YCIRgEGO74ZP4LIXuJT1KMGLHopSegm5ygtxszP"
        );
        assert_eq!(
            StringToken::from_text("hello, world").mangle(),
            "_RNvC22REUSSIR_STRING_LITERAL43JqNctOZ6XzATvx85LLRAVM34hr1wyMynWh7Iym06JhW"
        );
    }

    #[test]
    fn mangle_is_deterministic() {
        assert_eq!(
            StringToken::from_text("foo").mangle(),
            StringToken::from_text("foo").mangle()
        );
        assert_ne!(
            StringToken::from_text("foo").mangle(),
            StringToken::from_text("bar").mangle()
        );
    }

    #[test]
    fn uniqifier_dedups_and_matches_hash() {
        let mut uniq = StringUniqifier::new();
        assert!(uniq.is_empty());

        let first = uniq.allocate("hello");
        let again = uniq.allocate("hello");
        assert_eq!(first, again);
        assert_eq!(first, StringToken::from_text("hello"));
        assert_eq!(uniq.len(), 1);

        let other = uniq.allocate("world");
        assert_ne!(first, other);
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn uniqifier_iter_recovers_all_strings() {
        let mut uniq = StringUniqifier::new();
        uniq.allocate("a");
        uniq.allocate("b");
        uniq.allocate("a");

        let mut pairs: Vec<_> = uniq.iter().collect();
        pairs.sort_by_key(|(s, _)| *s);
        assert_eq!(
            pairs,
            vec![
                ("a", StringToken::from_text("a")),
                ("b", StringToken::from_text("b")),
            ]
        );
    }
}
