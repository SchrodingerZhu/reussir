//! Base-62 encoding of 256-bit numbers.
//!
//! The alphabet is `0-9`, then `a-z`, then `A-Z`. We encode the number as a
//! 256-bit big integer held in four 64-bit words (most significant first),
//! dividing by 62 one word at a time so no wide-integer type is needed.

/// The result of base-62 encoding a number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    /// The digits, most significant first.
    pub digits: String,
    /// Whether the leading (most significant) digit is numeric (`0-9`).
    ///
    /// Callers that splice the digits into an identifier use this to decide
    /// whether a separator is needed, since identifiers may not start with a
    /// digit.
    pub leading_is_numeric: bool,
}

impl Encoded {
    /// The number of digits. The alphabet is ASCII, so this is the byte length.
    pub fn len(&self) -> usize {
        self.digits.len()
    }

    /// Whether the encoding is empty. It never is — every number encodes to at
    /// least one digit — but clippy asks for this alongside [`Encoded::len`].
    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }
}

/// Divides a 64-bit limb (with an incoming `carry` in `0..62`) by 62, returning
/// the quotient contribution and the remainder.
fn div_mod_62_word(carry: u64, w: u64) -> (u64, u64) {
    // 2^64 = 62 * 297528130221121800 + 16, so each unit of `carry` contributes
    // this fixed quotient and remainder before folding in the current word.
    const Q_CONST: u64 = 297528130221121800;
    const R_CONST: u64 = 16;

    let q_part1 = carry * Q_CONST;
    let r_part1 = carry * R_CONST;

    let q_part2 = w / 62;
    let r_part2 = w % 62;

    let r_sum = r_part1 + r_part2;
    let (q_adj, r_final) = if r_sum >= 62 {
        (r_sum / 62, r_sum % 62)
    } else {
        (0, r_sum)
    };

    (q_part1 + q_part2 + q_adj, r_final)
}

/// Divides a 256-bit number (most-significant word first) by 62, returning the
/// quotient and the single base-62 digit remainder.
fn div_mod_62(words: [u64; 4]) -> ([u64; 4], usize) {
    let (q0, r0) = div_mod_62_word(0, words[0]);
    let (q1, r1) = div_mod_62_word(r0, words[1]);
    let (q2, r2) = div_mod_62_word(r1, words[2]);
    let (q3, r3) = div_mod_62_word(r2, words[3]);
    ([q0, q1, q2, q3], r3 as usize)
}

/// Maps a base-62 digit to its character, reporting whether it is numeric.
fn digit_to_char(d: usize) -> (char, bool) {
    if d < 10 {
        ((b'0' + d as u8) as char, true)
    } else if d < 36 {
        ((b'a' + (d - 10) as u8) as char, false)
    } else {
        ((b'A' + (d - 36) as u8) as char, false)
    }
}

/// Encodes a 256-bit number, given as four words most-significant first.
pub fn encode(mut n: [u64; 4]) -> Encoded {
    let mut digits = Vec::new();
    let mut leading_is_numeric;
    loop {
        let (q, r) = div_mod_62(n);
        let (c, is_num) = digit_to_char(r);
        digits.push(c);
        leading_is_numeric = is_num;
        n = q;
        if n == [0, 0, 0, 0] {
            break;
        }
    }
    // Digits were produced least-significant first; reverse for the final string.
    let digits = digits.into_iter().rev().collect();
    Encoded {
        digits,
        leading_is_numeric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes a small value packed into the least-significant word so results
    /// can be checked against hand-computed base-62.
    fn b62(value: u64) -> String {
        encode([0, 0, 0, value]).digits
    }

    #[test]
    fn small_values() {
        assert_eq!(b62(0), "0");
        assert_eq!(b62(9), "9");
        assert_eq!(b62(10), "a");
        assert_eq!(b62(35), "z");
        assert_eq!(b62(36), "A");
        assert_eq!(b62(61), "Z");
        assert_eq!(b62(62), "10");
        assert_eq!(b62(62 * 62), "100");
    }

    #[test]
    fn leading_numeric_flag() {
        assert!(encode([0, 0, 0, 0]).leading_is_numeric);
        assert!(encode([0, 0, 0, 9]).leading_is_numeric);
        assert!(!encode([0, 0, 0, 10]).leading_is_numeric);
        assert!(!encode([0, 0, 0, 61]).leading_is_numeric);
    }

    #[test]
    fn len_matches_digits() {
        let e = encode([0, 0, 0, 62 * 62]);
        assert_eq!(e.len(), 3);
        assert_eq!(e.len(), e.digits.len());
    }

    #[test]
    fn carries_across_word_boundary() {
        // 2^64 = 62 * 297528130221121800 + 16. The low digit of encoding 2^64 is
        // therefore 16, which maps to 'g' (16 - 10 + 'a').
        let e = encode([0, 0, 1, 0]);
        assert_eq!(e.digits.chars().last().unwrap(), 'g');
    }
}
