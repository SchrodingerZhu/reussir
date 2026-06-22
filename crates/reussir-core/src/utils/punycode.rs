//! RFC 3492 Punycode encoding of Unicode identifiers.
//!
//! The v0 symbol mangling encodes a non-ASCII identifier as its Punycode form
//! (a `u` tag, a length, then the Bootstring output) so the resulting linker
//! symbol is pure ASCII. We only ever *encode* (identifiers are mangled, never
//! demangled here), so this is the Bootstring encoder of RFC 3492 specialized to
//! Punycode's parameters — no decode path and no host/label splitting.
//!
//! The output is the raw Punycode (e.g. `gödel` → `gdel-5qa`); the mangler is
//! responsible for the `-`→`_` rewrite the symbol grammar wants.

use ftree::FenwickTree;

// Bootstring parameters for Punycode (RFC 3492 §5).
//
// The Bootstring arithmetic is done in `u64`. RFC 3492 specifies that an encoder
// must *signal* overflow rather than wrap; the accumulator `delta` can in
// principle exceed `u32` for a pathologically long non-ASCII string (the term
// `(m - n) * (handled + 1)` grows with the code-point gap times the length).
// Widening to `u64` makes overflow unreachable for any valid Unicode input —
// code points are ≤ 0x10FFFF and lengths are bounded by `usize`, so `delta`
// stays far below `2^64` — so no explicit overflow check is needed.
const BASE: u64 = 36;
const TMIN: u64 = 1;
const TMAX: u64 = 26;
const SKEW: u64 = 38;
const DAMP: u64 = 700;
const INITIAL_BIAS: u64 = 72;
const INITIAL_N: u64 = 128;

/// Maps a Bootstring digit (`0..36`) to its ASCII character: `0-25` → `a-z`,
/// `26-35` → `0-9`.
fn digit_to_char(d: u64) -> char {
    debug_assert!(d < BASE);
    if d < 26 {
        (b'a' + d as u8) as char
    } else {
        (b'0' + (d - 26) as u8) as char
    }
}

/// The bias adaptation of RFC 3492 §6.1.
fn adapt(mut delta: u64, num_points: u64, first_time: bool) -> u64 {
    delta = if first_time { delta / DAMP } else { delta / 2 };
    delta += delta / num_points;
    let mut k = 0;
    while delta > ((BASE - TMIN) * TMAX) / 2 {
        delta /= BASE - TMIN;
        k += BASE;
    }
    k + (((BASE - TMIN + 1) * delta) / (delta + SKEW))
}

/// Emits `delta` as RFC 3492's generalized variable-length integer under the
/// current `bias`. Shared by the production encoder and the test oracle so the
/// two serialize deltas identically.
fn push_delta(output: &mut String, delta: u64, bias: u64) {
    let mut q = delta;
    let mut k = BASE;
    loop {
        let t = if k <= bias {
            TMIN
        } else if k >= bias + TMAX {
            TMAX
        } else {
            k - bias
        };
        if q < t {
            break;
        }
        output.push(digit_to_char(t + ((q - t) % (BASE - t))));
        q = (q - t) / (BASE - t);
        k += BASE;
    }
    output.push(digit_to_char(q));
}

/// Encodes a string as Punycode (RFC 3492 §6.3) in `O(n log n)`.
///
/// The input may be any Unicode text; the output is the Bootstring encoding,
/// with the basic (ASCII) code points first, a `-` delimiter when any basic
/// code point was emitted, and then the encoded non-basic code points.
///
/// The textbook encoder is `O(n²)`: for each distinct code-point value it
/// rescans the whole input twice — a min-find for the next-smallest value, then
/// a delta pass. We instead drive the outer loop over `(value, position)` pairs
/// sorted by value, so the min-find is just a linear walk, and replace the delta
/// pass with a Fenwick tree. The only quantity that pass needs per occurrence is
/// `C(p, m)` — the number of already-placed code points (value `< m`) lying left
/// of input position `p`. Visiting values in ascending order, every such code
/// point is "placed" before the value that needs it, so `C(p, m)` is a Fenwick
/// prefix-count over positions: `O(log n)` per occurrence.
pub fn encode(input: &str) -> String {
    let code_points: Vec<u32> = input.chars().map(|c| c as u32).collect();
    let mut output = String::new();

    // Basic (ASCII) code points are emitted verbatim, in input order.
    let mut basic_count: u64 = 0;
    for &c in &code_points {
        if u64::from(c) < INITIAL_N {
            output.push(c as u8 as char);
            basic_count += 1;
        }
    }

    // The delimiter separates the literal basic prefix from the encoded tail; it
    // is only present when there is a basic prefix to delimit.
    if basic_count > 0 {
        output.push('-');
    }

    // The non-basic code points as `(value, input-position)` pairs, sorted so the
    // Bootstring loop visits values ascending without rescanning for the next
    // smallest. Equal values keep input-position order — the order in which their
    // occurrences must be encoded.
    let mut pairs: Vec<(u32, usize)> = code_points
        .iter()
        .enumerate()
        .filter(|&(_, &c)| u64::from(c) >= INITIAL_N)
        .map(|(i, &c)| (c, i))
        .collect();
    pairs.sort_unstable();

    // Pure-ASCII input has no tail to encode.
    if pairs.is_empty() {
        return output;
    }

    // Fenwick tree over input positions: position `p` holds 1 once its code point
    // has been "placed" (its value is below the value now being encoded), so
    // `prefix_sum(p, 0)` counts placed code points strictly left of `p`. Every
    // basic code point sits below any non-basic value, so all are placed up front
    // — seed the tree with a 1 at each basic position. Building straight from the
    // indicator iterator does this in one `O(n)` pass (`FromIterator` folds each
    // slot into its Fenwick parent in place), rather than a zero-fill followed by
    // an `O(log n)` `add_at` for every basic code point.
    let mut placed = FenwickTree::from_iter(
        code_points
            .iter()
            .map(|&c| u64::from(u64::from(c) < INITIAL_N)),
    );

    let mut n = INITIAL_N;
    let mut delta: u64 = 0;
    let mut bias = INITIAL_BIAS;
    let mut handled = basic_count;

    let mut i = 0;
    while i < pairs.len() {
        let m = u64::from(pairs[i].0);

        // Advance `n` to the next present value; each skipped value costs
        // `handled + 1` decoder states (u64, so it cannot overflow for any valid
        // Unicode input).
        delta += (m - n) * (handled + 1);
        n = m;

        // At the start of the group `handled` is exactly the number of code points
        // with value `< m` (all lower values are already encoded) — the count the
        // trailing carry needs.
        let placed_below_m = handled;
        let group_start = i;
        let mut prev_c = 0u64;

        // Encode every occurrence of `m` in input-position order. An occurrence's
        // delta is the number of placed code points lying between it and the
        // previous occurrence — a difference of two Fenwick prefix counts.
        while i < pairs.len() && u64::from(pairs[i].0) == m {
            let p = pairs[i].1;
            let c_p = placed.prefix_sum(p, 0);
            delta += c_p - prev_c;
            push_delta(&mut output, delta, bias);
            bias = adapt(delta, handled + 1, handled == basic_count);
            delta = 0;
            prev_c = c_p;
            handled += 1;
            i += 1;
        }

        // Carry into the next value: the placed code points to the right of the
        // last occurrence (`placed_below_m - prev_c`) plus RFC 3492's per-value
        // increment.
        delta = (placed_below_m - prev_c) + 1;
        n += 1;

        // The whole group now counts as placed for every higher value.
        for &(_, p) in &pairs[group_start..i] {
            placed.add_at(p, 1);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3492_vectors() {
        // The commonly-cited German example: `bücher` → `bcher-kva`.
        assert_eq!(encode("bücher"), "bcher-kva");
    }

    #[test]
    fn mangle_vector() {
        // The identifier the v0 mangler golden-tests: `gödel` punycodes to
        // `gdel-5qa` (the mangler then rewrites `-` to `_`).
        assert_eq!(encode("gödel"), "gdel-5qa");
    }

    #[test]
    fn all_basic_has_trailing_delimiter() {
        // An all-ASCII input is its own basic prefix plus the delimiter; the
        // mangler never calls this path (it punycodes only non-ASCII names), but
        // the encoder must still follow the spec.
        assert_eq!(encode("abc"), "abc-");
    }

    #[test]
    fn pure_non_basic_has_no_delimiter() {
        // No basic code points ⇒ no delimiter, just the encoded tail.
        assert_eq!(encode("ü"), "tda");
    }

    /// Conformance vectors for the Punycode *encoder*: the RFC 3492 §7.1 sample
    /// strings plus the punycode.js corpus, as curated by the `idna` crate's
    /// `tests/punycode_tests.json` (MIT). Only the encode direction is exercised
    /// (we never decode), and every vector is exact-match safe: the basic prefix
    /// preserves case while our generated tail is always lowercase, so no
    /// case-folding fixup is needed (unlike the raw RFC vectors, some of which
    /// use the optional uppercase-flag feature we deliberately do not emit).
    #[test]
    fn rfc3492_and_punycodejs_conformance() {
        const VECTORS: &[(&str, &str)] = &[
            ("", ""),
            ("Bach", "Bach-"),
            ("\u{fc}", "tda"),
            ("\u{fc}\u{eb}\u{e4}\u{f6}\u{2665}", "4can8av2009b"),
            ("b\u{fc}cher", "bcher-kva"),
            (
                "Willst du die Bl\u{fc}the des fr\u{fc}hen, die Fr\u{fc}chte des sp\u{e4}teren Jahres",
                "Willst du die Blthe des frhen, die Frchte des spteren Jahres-x9e96lkal",
            ),
            // Arabic (Egyptian)
            (
                "\u{644}\u{64a}\u{647}\u{645}\u{627}\u{628}\u{62a}\u{643}\u{644}\u{645}\u{648}\u{634}\u{639}\u{631}\u{628}\u{64a}\u{61f}",
                "egbpdaj6bu4bxfgehfvwxn",
            ),
            // Chinese (simplified)
            (
                "\u{4ed6}\u{4eec}\u{4e3a}\u{4ec0}\u{4e48}\u{4e0d}\u{8bf4}\u{4e2d}\u{6587}",
                "ihqwcrb4cv8a8dqg056pqjye",
            ),
            // Chinese (traditional)
            (
                "\u{4ed6}\u{5011}\u{7232}\u{4ec0}\u{9ebd}\u{4e0d}\u{8aaa}\u{4e2d}\u{6587}",
                "ihqwctvzc91f659drss3x8bo0yb",
            ),
            // Czech
            (
                "Pro\u{10d}prost\u{11b}nemluv\u{ed}\u{10d}esky",
                "Proprostnemluvesky-uyb24dma41a",
            ),
            // Hebrew
            (
                "\u{5dc}\u{5de}\u{5d4}\u{5d4}\u{5dd}\u{5e4}\u{5e9}\u{5d5}\u{5d8}\u{5dc}\u{5d0}\u{5de}\u{5d3}\u{5d1}\u{5e8}\u{5d9}\u{5dd}\u{5e2}\u{5d1}\u{5e8}\u{5d9}\u{5ea}",
                "4dbcagdahymbxekheh6e0a7fei0b",
            ),
            // Hindi (Devanagari)
            (
                "\u{92f}\u{939}\u{932}\u{94b}\u{917}\u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940}\u{915}\u{94d}\u{92f}\u{94b}\u{902}\u{928}\u{939}\u{940}\u{902}\u{92c}\u{94b}\u{932}\u{938}\u{915}\u{924}\u{947}\u{939}\u{948}\u{902}",
                "i1baa7eci9glrd9b2ae1bj0hfcgg6iyaf8o0a1dig0cd",
            ),
            // Japanese (kanji and hiragana)
            (
                "\u{306a}\u{305c}\u{307f}\u{3093}\u{306a}\u{65e5}\u{672c}\u{8a9e}\u{3092}\u{8a71}\u{3057}\u{3066}\u{304f}\u{308c}\u{306a}\u{3044}\u{306e}\u{304b}",
                "n8jok5ay5dzabd5bym9f0cm5685rrjetr6pdxa",
            ),
            // Korean (Hangul syllables)
            (
                "\u{c138}\u{acc4}\u{c758}\u{baa8}\u{b4e0}\u{c0ac}\u{b78c}\u{b4e4}\u{c774}\u{d55c}\u{ad6d}\u{c5b4}\u{b97c}\u{c774}\u{d574}\u{d55c}\u{b2e4}\u{ba74}\u{c5bc}\u{b9c8}\u{b098}\u{c88b}\u{c744}\u{ae4c}",
                "989aomsvi5e83db1d2a355cv1e0vak1dwrv93d5xbh15a0dt30a5jpsd879ccm6fea98c",
            ),
            // Russian (Cyrillic)
            (
                "\u{43f}\u{43e}\u{447}\u{435}\u{43c}\u{443}\u{436}\u{435}\u{43e}\u{43d}\u{438}\u{43d}\u{435}\u{433}\u{43e}\u{432}\u{43e}\u{440}\u{44f}\u{442}\u{43f}\u{43e}\u{440}\u{443}\u{441}\u{441}\u{43a}\u{438}",
                "b1abfaaepdrnnbgefbadotcwatmq2g4l",
            ),
            // Spanish
            (
                "Porqu\u{e9}nopuedensimplementehablarenEspa\u{f1}ol",
                "PorqunopuedensimplementehablarenEspaol-fmd56a",
            ),
            // Vietnamese
            (
                "T\u{1ea1}isaoh\u{1ecd}kh\u{f4}ngth\u{1ec3}ch\u{1ec9}n\u{f3}iti\u{1ebf}ngVi\u{1ec7}t",
                "TisaohkhngthchnitingVit-kjcr8268qyxafd2f1b9g",
            ),
            // Mixed scripts with embedded ASCII digits / hyphens
            (
                "3\u{5e74}B\u{7d44}\u{91d1}\u{516b}\u{5148}\u{751f}",
                "3B-ww4c5e180e575a65lsy2b",
            ),
            (
                "\u{5b89}\u{5ba4}\u{5948}\u{7f8e}\u{6075}-with-SUPER-MONKEYS",
                "-with-SUPER-MONKEYS-pc58ag80a8qai00g7n9n",
            ),
            (
                "Hello-Another-Way-\u{305d}\u{308c}\u{305e}\u{308c}\u{306e}\u{5834}\u{6240}",
                "Hello-Another-Way--fc4qua05auwb3674vfr0b",
            ),
            (
                "\u{3072}\u{3068}\u{3064}\u{5c4b}\u{6839}\u{306e}\u{4e0b}2",
                "2-u9tlzr9756bt3uc0v",
            ),
            (
                "Maji\u{3067}Koi\u{3059}\u{308b}5\u{79d2}\u{524d}",
                "MajiKoi5-783gue6qz075azm5e",
            ),
            (
                "\u{30d1}\u{30d5}\u{30a3}\u{30fc}de\u{30eb}\u{30f3}\u{30d0}",
                "de-jg4avhby1noc0d",
            ),
            (
                "\u{305d}\u{306e}\u{30b9}\u{30d4}\u{30fc}\u{30c9}\u{3067}",
                "d9juau41awczczp",
            ),
            // Pure ASCII with punctuation: still gets a trailing delimiter.
            ("-> $1.00 <-", "-> $1.00 <--"),
        ];

        for &(decoded, encoded) in VECTORS {
            assert_eq!(encode(decoded), encoded, "encoding {decoded:?}");
        }
    }
}
