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

// Bootstring parameters for Punycode (RFC 3492 §5).
const BASE: u32 = 36;
const TMIN: u32 = 1;
const TMAX: u32 = 26;
const SKEW: u32 = 38;
const DAMP: u32 = 700;
const INITIAL_BIAS: u32 = 72;
const INITIAL_N: u32 = 128;

/// Maps a Bootstring digit (`0..36`) to its ASCII character: `0-25` → `a-z`,
/// `26-35` → `0-9`.
fn digit_to_char(d: u32) -> char {
    debug_assert!(d < BASE);
    if d < 26 {
        (b'a' + d as u8) as char
    } else {
        (b'0' + (d - 26) as u8) as char
    }
}

/// The bias adaptation of RFC 3492 §6.1.
fn adapt(mut delta: u32, num_points: u32, first_time: bool) -> u32 {
    delta = if first_time { delta / DAMP } else { delta / 2 };
    delta += delta / num_points;
    let mut k = 0;
    while delta > ((BASE - TMIN) * TMAX) / 2 {
        delta /= BASE - TMIN;
        k += BASE;
    }
    k + (((BASE - TMIN + 1) * delta) / (delta + SKEW))
}

/// Encodes a string as Punycode (RFC 3492 §6.3).
///
/// The input may be any Unicode text; the output is the Bootstring encoding,
/// with the basic (ASCII) code points first, a `-` delimiter when any basic
/// code point was emitted, and then the encoded non-basic code points.
pub fn encode(input: &str) -> String {
    let code_points: Vec<u32> = input.chars().map(|c| c as u32).collect();
    let mut output = String::new();

    // Emit all basic (ASCII) code points up front, in order.
    let mut basic_count: u32 = 0;
    for &c in &code_points {
        if c < INITIAL_N {
            output.push(c as u8 as char);
            basic_count += 1;
        }
    }

    // The delimiter separates the literal basic prefix from the encoded tail; it
    // is only present when there is a basic prefix to delimit.
    if basic_count > 0 {
        output.push('-');
    }

    let mut n = INITIAL_N;
    let mut delta: u32 = 0;
    let mut bias = INITIAL_BIAS;
    let mut handled = basic_count;
    let total = code_points.len() as u32;

    while handled < total {
        // The smallest non-basic code point not yet handled becomes the next `n`.
        let m = code_points
            .iter()
            .copied()
            .filter(|&c| c >= n)
            .min()
            .expect("handled < total implies an unhandled code point remains");

        // Advancing `n` to `m` costs `(m - n) * (handled + 1)` in delta. The
        // inputs are bounded by valid Unicode, so u32 cannot overflow here.
        delta += (m - n) * (handled + 1);
        n = m;

        for &c in &code_points {
            if c < n {
                delta += 1;
            }
            if c == n {
                // Represent `delta` as a generalized variable-length integer.
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
                bias = adapt(delta, handled + 1, handled == basic_count);
                delta = 0;
                handled += 1;
            }
        }

        delta += 1;
        n += 1;
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
}
