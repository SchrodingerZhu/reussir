//! Arbitrary-precision numeric literals.
//!
//! A numeric literal is *exact* at the source level: an integer literal is an
//! integer, and a float literal `m.f e±x` is exactly the rational
//! `mantissa × 10^exp10` (a decimal literal is always a finite decimal). Both
//! are carried unrounded — [`Integer`] and [`FloatLit`] — from the surface AST
//! through HIR and MIR. The single lossy step, correctly-rounded conversion
//! into the ground IEEE format, happens once the type is known
//! ([`FloatLit::to_ieee_bits`], used by codegen and by the range diagnostics
//! at monomorphization). This avoids the classic double rounding
//! (decimal → `f64` → `f16` can differ from decimal → `f16` by an ULP) and
//! keeps values outside `i64`/`f64` representable until we know whether they
//! fit their target.
//!
//! Printing is lossless by construction: HIR/MIR never hold a binary float,
//! only the exact decimal, so the textual IR round-trips numerics through
//! plain decimal forms ([`FloatLit`]'s `Display` is canonical and its
//! `FromStr` inverts it exactly).

use std::fmt;
use std::str::FromStr;

use malachite_base::num::arithmetic::traits::{DivMod, DivRound, Pow};
use malachite_base::num::basic::traits::One;
use malachite_base::num::conversion::traits::FromStringBase;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_base::rounding_modes::RoundingMode;
pub use malachite_nz::integer::Integer;
pub use malachite_nz::natural::Natural;

use crate::semi::ty::{FpTy, IntTy};

/// Parse an integer literal's token text into its exact value. Accepts the
/// surface forms: decimal digit runs, `0x`/`0o`/`0b` radix prefixes, and `_`
/// digit separators. The lexer has validated the shape, so this is total.
pub fn parse_int(text: &str) -> Integer {
    let text: String = text.chars().filter(|&c| c != '_').collect();
    let natural = if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Natural::from_string_base(16, &hex.to_ascii_lowercase())
    } else if let Some(oct) = text.strip_prefix("0o").or_else(|| text.strip_prefix("0O")) {
        Natural::from_string_base(8, oct)
    } else if let Some(bin) = text.strip_prefix("0b").or_else(|| text.strip_prefix("0B")) {
        Natural::from_string_base(2, bin)
    } else {
        Natural::from_str(&text).ok()
    };
    Integer::from(natural.expect("integer literal validated by the lexer"))
}

/// Parse a float literal's token text (`1.5`, `1e3`, `2.5e-7`, with optional
/// `_` separators) into its exact decimal value. Total for lexer-validated
/// input.
pub fn parse_float(text: &str) -> FloatLit {
    let text: String = text.chars().filter(|&c| c != '_').collect();
    text.parse().expect("float literal validated by the lexer")
}

/// An exact decimal floating-point literal: the value `mantissa × 10^exp10`.
///
/// Canonical form (maintained by the constructor): the mantissa of a non-zero
/// value is not divisible by 10, and zero is `(0, 0)`. Derived equality is
/// therefore semantic equality, and `Display`/`FromStr` round-trip exactly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FloatLit {
    mantissa: Integer,
    exp10: i64,
}

/// The literal does not fit its target format: rounding it would produce an
/// infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpOverflow;

impl FloatLit {
    /// Build from a (possibly non-canonical) decimal pair, normalizing.
    pub fn new(mut mantissa: Integer, mut exp10: i64) -> FloatLit {
        if mantissa == 0 {
            return FloatLit { mantissa, exp10: 0 };
        }
        let ten = Integer::from(10);
        loop {
            let (q, r) = (&mantissa).div_mod(&ten);
            if r != 0 {
                break;
            }
            mantissa = q;
            exp10 += 1;
        }
        FloatLit { mantissa, exp10 }
    }

    pub fn mantissa(&self) -> &Integer {
        &self.mantissa
    }

    /// The negated value (used when folding `Negate(lit)` into a constant).
    pub fn neg(&self) -> FloatLit {
        FloatLit {
            mantissa: -self.mantissa.clone(),
            exp10: self.exp10,
        }
    }

    pub fn exp10(&self) -> i64 {
        self.exp10
    }

    /// Round to the nearest (ties-to-even) value of the IEEE binary format
    /// with `exp_bits` exponent bits and `mant_bits` explicit mantissa bits
    /// (an implicit leading bit is assumed; total width `1 + exp_bits +
    /// mant_bits` must be at most 128). Returns the format's raw bit pattern.
    ///
    /// This is the *only* decimal→binary conversion in the pipeline, and it
    /// never goes through a host float, so formats without a Rust encoding
    /// (`bf16`, `f16`, `f128`) round exactly once like everything else.
    /// Overflow to infinity is an error (a literal can never mean `inf`);
    /// underflow rounds to a subnormal or zero, as IEEE prescribes.
    pub fn to_ieee_bits(&self, exp_bits: u32, mant_bits: u32) -> Result<u128, FpOverflow> {
        assert!(
            (1..=30).contains(&exp_bits) && 1 + exp_bits + mant_bits <= 128,
            "unsupported format ({exp_bits}, {mant_bits})"
        );
        let sign_bit = if self.mantissa < 0 {
            1u128 << (exp_bits + mant_bits)
        } else {
            0
        };
        if self.mantissa == 0 {
            return Ok(0);
        }
        let a: &Natural = self.mantissa.unsigned_abs_ref();
        let bias = (1i64 << (exp_bits - 1)) - 1;
        let emax = bias;
        let emin = 1 - bias;
        let p = i64::from(mant_bits) + 1;

        // Cheap decimal-magnitude guards so absurd exponents (`1e999999999`)
        // resolve without materializing 10^e. log10(a) lies within
        // [(bits-1)·log10(2), bits·log10(2) + 1); ±6000 clears every format up
        // to binary128 (max ≈ 1.19e4932, min subnormal ≈ 6.5e-4966).
        let bits = a.significant_bits() as i64;
        let mag_hi = (bits * 302) / 1000 + 1 + self.exp10;
        let mag_lo = ((bits - 1) * 301) / 1000 + self.exp10;
        if mag_lo > 6000 {
            return Err(FpOverflow);
        }
        if mag_hi < -6000 {
            return Ok(sign_bit); // underflows to (signed) zero
        }

        // The exact value as n/d with d a power of ten.
        let (n, d) = if self.exp10 >= 0 {
            (
                a * Natural::from(10u32).pow(self.exp10 as u64),
                Natural::ONE,
            )
        } else {
            (
                a.clone(),
                Natural::from(10u32).pow(self.exp10.unsigned_abs()),
            )
        };

        // e2 = floor(log2(n/d)): estimate from bit lengths, then correct so
        // that d·2^e2 <= n < d·2^(e2+1).
        let mut e2 = n.significant_bits() as i64 - d.significant_bits() as i64;
        if shifted_gt(&d, e2, &n) {
            e2 -= 1;
        } else if !shifted_gt(&d, e2 + 1, &n) {
            e2 += 1;
        }
        debug_assert!(!shifted_gt(&d, e2, &n) && shifted_gt(&d, e2 + 1, &n));

        // Round the significand: q = round((n/d) · 2^(p-1-target_exp)),
        // ties to even. Clamping the exponent at emin makes the same formula
        // produce subnormals.
        let mut target_exp = e2.max(emin);
        let s = p - 1 - target_exp;
        let mut q = if s >= 0 {
            (n << s).div_round(&d, RoundingMode::Nearest).0
        } else {
            n.div_round(&(d << (-s) as u64), RoundingMode::Nearest).0
        };
        // Rounding can carry into the next power of two (q = 2^p exactly).
        if q.significant_bits() as i64 > p {
            q >>= 1u32;
            target_exp += 1;
        }

        let low = if q.significant_bits() as i64 == p {
            // Normal: strip the implicit leading bit, store the biased exponent.
            if target_exp > emax {
                return Err(FpOverflow);
            }
            let frac = q - (Natural::ONE << (p - 1) as u64);
            (((target_exp + bias) as u128) << mant_bits) | natural_to_u128(&frac)
        } else {
            // Subnormal (or zero after underflow rounding): biased exponent 0.
            debug_assert!(target_exp == emin);
            natural_to_u128(&q)
        };
        Ok(sign_bit | low)
    }
}

/// `d · 2^k > n`, with `k` possibly negative.
fn shifted_gt(d: &Natural, k: i64, n: &Natural) -> bool {
    if k >= 0 {
        (d << k as u64) > *n
    } else {
        *d > (n << (-k) as u64)
    }
}

fn natural_to_u128(n: &Natural) -> u128 {
    u128::try_from(n).expect("value bounded by the format width")
}

impl fmt::Display for FloatLit {
    /// Canonical, lossless, and lexable: always contains `.` or `e`, plain
    /// decimal for short values (`1.1`, `0.00123`, `250.0`) and scientific
    /// otherwise (`2.5e300`, `5e-7`). `FromStr` inverts it exactly.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mantissa == 0 {
            return write!(f, "0.0");
        }
        if self.mantissa < 0 {
            write!(f, "-")?;
        }
        let digits = self.mantissa.unsigned_abs_ref().to_string();
        let n = digits.len() as i64;
        // The scientific exponent: value = d.ddd × 10^sci.
        let sci = self.exp10 + n - 1;
        if self.exp10 >= 0 && sci <= 16 {
            // A short integral value: digits, trailing zeros, `.0`.
            write!(f, "{digits}")?;
            for _ in 0..self.exp10 {
                write!(f, "0")?;
            }
            write!(f, ".0")
        } else if self.exp10 < 0 && sci >= -5 {
            if sci >= 0 {
                // Point inside the digits.
                let split = (sci + 1) as usize;
                write!(f, "{}.{}", &digits[..split], &digits[split..])
            } else {
                // 0.00ddd
                write!(f, "0.")?;
                for _ in 0..(-sci - 1) {
                    write!(f, "0")?;
                }
                write!(f, "{digits}")
            }
        } else if n == 1 {
            write!(f, "{digits}e{sci}")
        } else {
            write!(f, "{}.{}e{}", &digits[..1], &digits[1..], sci)
        }
    }
}

/// Parse the decimal forms `digits`, `digits.digits`, with an optional
/// `e`/`E` exponent and optional leading `-`. (The bare-integer form is
/// accepted for robustness even though the printers never emit it.)
impl FromStr for FloatLit {
    type Err = ();

    fn from_str(s: &str) -> Result<FloatLit, ()> {
        let (sign, s) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let (base, exp) = match s.find(['e', 'E']) {
            Some(i) => {
                let exp_text = &s[i + 1..];
                let exp_digits = exp_text.strip_prefix(['+', '-']).unwrap_or(exp_text);
                if exp_digits.is_empty() || exp_digits.contains(|c: char| !c.is_ascii_digit()) {
                    return Err(());
                }
                // Saturate exponents beyond i64: the magnitude guards in
                // `to_ieee_bits` turn them into overflow/zero as appropriate.
                let exp = exp_text.parse::<i64>().unwrap_or_else(|_| {
                    if exp_text.starts_with('-') {
                        i64::MIN / 2
                    } else {
                        i64::MAX / 2
                    }
                });
                (&s[..i], exp)
            }
            None => (s, 0),
        };
        let (int_part, frac_part) = match base.find('.') {
            Some(i) => (&base[..i], &base[i + 1..]),
            None => (base, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(());
        }
        if int_part.contains(|c: char| !c.is_ascii_digit())
            || frac_part.contains(|c: char| !c.is_ascii_digit())
        {
            return Err(());
        }
        let mut digits = String::with_capacity(int_part.len() + frac_part.len());
        digits.push_str(int_part);
        digits.push_str(frac_part);
        let mantissa = Natural::from_str(&digits).map_err(|_| ())?;
        let mantissa = if sign {
            -Integer::from(mantissa)
        } else {
            Integer::from(mantissa)
        };
        Ok(FloatLit::new(
            mantissa,
            exp.saturating_sub(frac_part.len() as i64),
        ))
    }
}

/// The `(exponent bits, explicit mantissa bits)` of a semi float type, or
/// `None` for formats without a settled encoding (`float8` today).
pub fn ieee_params(fp: FpTy) -> Option<(u32, u32)> {
    match fp {
        FpTy::Ieee(16) => Some((5, 10)),
        FpTy::Ieee(32) => Some((8, 23)),
        FpTy::Ieee(64) => Some((11, 52)),
        FpTy::Ieee(128) => Some((15, 112)),
        FpTy::BFloat16 => Some((8, 7)),
        FpTy::Ieee(_) | FpTy::Float8 => None,
    }
}

/// Whether `n` is representable in the integer type `ty`.
pub fn int_fits(n: &Integer, ty: IntTy) -> bool {
    match ty {
        IntTy::Unsigned(w) => *n >= 0 && n.unsigned_abs_ref().significant_bits() <= u64::from(w),
        IntTy::Signed(w) => {
            if w == 0 {
                return false;
            }
            let bound = Natural::ONE << u64::from(w - 1);
            if *n >= 0 {
                *n.unsigned_abs_ref() < bound
            } else {
                *n.unsigned_abs_ref() <= bound
            }
        }
    }
}

/// The low 64 bits of `n`, two's complement — the value MLIR's
/// `DenseI64ArrayAttribute` and friends want for a key that fits its type.
pub fn low_u64(n: &Integer) -> u64 {
    use malachite_base::num::conversion::traits::WrappingFrom;
    u64::wrapping_from(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(s: &str) -> FloatLit {
        parse_float(s)
    }

    #[test]
    fn integer_parsing_covers_radixes_and_separators() {
        assert_eq!(parse_int("0"), Integer::from(0));
        assert_eq!(parse_int("1_000_000"), Integer::from(1_000_000));
        assert_eq!(parse_int("0xFF"), Integer::from(255));
        assert_eq!(parse_int("0xff_ff"), Integer::from(0xFFFF));
        assert_eq!(parse_int("0o777"), Integer::from(511));
        assert_eq!(parse_int("0b1010_1010"), Integer::from(0xAA));
        // Beyond i64/u64/i128: exact.
        assert_eq!(
            parse_int("340282366920938463463374607431768211456").to_string(),
            "340282366920938463463374607431768211456" // 2^128
        );
        assert_eq!(
            parse_int("0x1_0000_0000_0000_0000").to_string(),
            "18446744073709551616" // 2^64
        );
    }

    #[test]
    fn float_parsing_is_exact_and_normalized() {
        assert_eq!(lit("1.1"), FloatLit::new(Integer::from(11), -1));
        assert_eq!(lit("2.50"), FloatLit::new(Integer::from(25), -1));
        assert_eq!(lit("1e3"), FloatLit::new(Integer::from(1), 3));
        assert_eq!(lit("2.5e-7"), FloatLit::new(Integer::from(25), -8));
        assert_eq!(lit("0.0"), FloatLit::new(Integer::from(0), 0));
        assert_eq!(lit("1_000.5"), FloatLit::new(Integer::from(10_005), -1));
    }

    #[test]
    fn display_round_trips_and_always_lexes_as_a_float() {
        for s in [
            "0.0",
            "1.0",
            "1.1",
            "0.00123",
            "250.0",
            "2.5e300",
            "5e-7",
            "1e17",
            "123.456",
            "9.999999999e100",
            "1e-4966",
        ] {
            let v = lit(s);
            let printed = v.to_string();
            assert!(
                printed.contains(['.', 'e']),
                "`{printed}` must re-lex as a float"
            );
            assert_eq!(printed.parse::<FloatLit>().unwrap(), v, "via `{printed}`");
        }
        // Canonical spellings.
        assert_eq!(lit("1.1").to_string(), "1.1");
        assert_eq!(lit("1e3").to_string(), "1000.0");
        assert_eq!(lit("2.5e300").to_string(), "2.5e300");
        assert_eq!(lit("0.0000005").to_string(), "5e-7");
    }

    /// Differential check against Rust's own correctly-rounded parser.
    #[test]
    fn ieee_rounding_matches_rust_for_f32_and_f64() {
        let cases = [
            "0.0",
            "1.0",
            "0.1",
            "1.1",
            "2.5",
            "3.14159265358979323846",
            "1e-40",
            "1.17549435e-38",
            "3.4028235e38",
            "1.7976931348623157e308",
            "4.9e-324",
            "2.2250738585072014e-308",
            "1e-45",
            "7e-46",
            "0.500000000000000166533453693773481063544750213623046875",
            "123456789.123456789",
            "9007199254740993.0",
            "1e-5000",
        ];
        for s in cases {
            let v = lit(s);
            let f64_bits = v.to_ieee_bits(11, 52).unwrap() as u64;
            assert_eq!(f64_bits, s.parse::<f64>().unwrap().to_bits(), "f64 `{s}`");
            // Rust rounds overflow to inf where we hard-error; compare only
            // finite results.
            let rust_f32: f32 = s.parse().unwrap();
            match v.to_ieee_bits(8, 23) {
                Ok(bits) => assert_eq!(bits as u32, rust_f32.to_bits(), "f32 `{s}`"),
                Err(FpOverflow) => assert!(rust_f32.is_infinite(), "f32 `{s}`"),
            }
        }
    }

    /// Randomized differential check against Rust's correctly-rounded parser:
    /// random digit strings across the whole f32/f64 exponent range (normals,
    /// subnormals, underflow, overflow).
    #[test]
    fn ieee_rounding_differential_fuzz() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let digits = (next() % 25 + 1) as usize;
            let mut text = String::with_capacity(digits + 6);
            for _ in 0..digits {
                text.push(char::from(b'0' + (next() % 10) as u8));
            }
            let exp = (next() % 700) as i64 - 350;
            text.push('e');
            text.push_str(&exp.to_string());
            let v = lit(&text);
            let rust_f64: f64 = text.parse().unwrap();
            match v.to_ieee_bits(11, 52) {
                Ok(bits) => assert_eq!(bits as u64, rust_f64.to_bits(), "f64 `{text}`"),
                Err(FpOverflow) => assert!(rust_f64.is_infinite(), "f64 `{text}`"),
            }
            let rust_f32: f32 = text.parse().unwrap();
            match v.to_ieee_bits(8, 23) {
                Ok(bits) => assert_eq!(bits as u32, rust_f32.to_bits(), "f32 `{text}`"),
                Err(FpOverflow) => assert!(rust_f32.is_infinite(), "f32 `{text}`"),
            }
        }
    }

    /// The reason this module exists: rounding decimal → f64 → f16 differs
    /// from decimal → f16 on halfway-straddling values.
    #[test]
    fn single_rounding_beats_double_rounding_for_f16() {
        // 1 + 2^-11 = 1.00048828125 is exactly halfway between the f16 values
        // 1.0 (0x3C00) and 1 + 2^-10 (0x3C01). Nudge it up by 1e-30 — far
        // below f64's resolution here — and the true value is above the tie,
        // so f16 must round UP. But f64 first rounds the nudge away, landing
        // exactly on the tie, and the second rounding then goes to even: DOWN.
        let s = "1.000488281250000000000000000001";
        let direct = lit(s).to_ieee_bits(5, 10).unwrap() as u16;
        let via_f64 = half_from_f64(s.parse::<f64>().unwrap());
        assert_eq!(direct, 0x3C01, "single rounding goes up");
        assert_eq!(
            via_f64, 0x3C00,
            "double rounding collapses to the tie, then even"
        );
    }

    /// f64 → f16 narrowing with round-to-nearest-even, for the test above.
    fn half_from_f64(x: f64) -> u16 {
        // Round via the exact decimal of the f64 value: f64 is exact binary,
        // so FloatLit::new over its exact decimal expansion is exact.
        let bits = x.to_bits();
        let exp = ((bits >> 52) & 0x7FF) as i64 - 1023 - 52;
        let mant = (bits & ((1 << 52) - 1)) | (1 << 52);
        // value = mant * 2^exp; 2^exp = 5^-exp * 10^exp for exp < 0.
        assert!(exp < 0);
        let m = Integer::from(mant) * Integer::from(5).pow((-exp) as u64);
        FloatLit::new(m, exp).to_ieee_bits(5, 10).unwrap() as u16
    }

    #[test]
    fn bf16_and_f128_round_without_a_host_type() {
        // 1.0 in formats with no Rust encoding.
        assert_eq!(lit("1.0").to_ieee_bits(8, 7).unwrap(), 0x3F80); // bf16
        assert_eq!(lit("1.0").to_ieee_bits(15, 112).unwrap(), 0x3FFFu128 << 112);
        // bf16 0.1 — correctly rounded: 0x3DCD (mantissa 0b1001101).
        assert_eq!(lit("0.1").to_ieee_bits(8, 7).unwrap(), 0x3DCD);
        // f128 max finite is ~1.19e4932; just below fits, above overflows.
        assert!(lit("1.18e4932").to_ieee_bits(15, 112).is_ok());
        assert_eq!(lit("1.2e4932").to_ieee_bits(15, 112), Err(FpOverflow));
    }

    #[test]
    fn overflow_is_an_error_and_underflow_rounds_to_zero_or_subnormal() {
        assert_eq!(lit("1e400").to_ieee_bits(11, 52), Err(FpOverflow));
        assert_eq!(lit("65520.0").to_ieee_bits(5, 10), Err(FpOverflow)); // rounds to inf in f16
        assert!(lit("65504.0").to_ieee_bits(5, 10).is_ok()); // f16 max finite
        assert_eq!(lit("1e-1000").to_ieee_bits(11, 52).unwrap(), 0); // underflow → +0
        assert_eq!(lit("1e-9999").to_ieee_bits(15, 112).unwrap(), 0);
        // Smallest f64 subnormal neighborhood: 4.9e-324 rounds to bit pattern 1.
        assert_eq!(lit("4.9e-324").to_ieee_bits(11, 52).unwrap(), 1);
    }

    #[test]
    fn int_fits_matches_the_type_ranges() {
        use IntTy::*;
        let n = |s: &str| parse_int(s);
        assert!(int_fits(&n("255"), Unsigned(8)));
        assert!(!int_fits(&n("256"), Unsigned(8)));
        assert!(int_fits(&n("127"), Signed(8)));
        assert!(!int_fits(&n("128"), Signed(8)));
        assert!(int_fits(&-n("128"), Signed(8)));
        assert!(!int_fits(&-n("129"), Signed(8)));
        assert!(int_fits(&n("18446744073709551615"), Unsigned(64)));
        assert!(!int_fits(&n("18446744073709551615"), Signed(64)));
        assert!(!int_fits(&n("18446744073709551616"), Unsigned(64)));
        assert!(!int_fits(&-n("1"), Unsigned(64)));
        assert!(int_fits(
            &n("170141183460469231731687303715884105727"),
            Signed(128)
        ));
        assert!(!int_fits(
            &n("170141183460469231731687303715884105728"),
            Signed(128)
        ));
    }

    #[test]
    fn low_u64_is_twos_complement() {
        assert_eq!(low_u64(&parse_int("18446744073709551615")), u64::MAX);
        assert_eq!(low_u64(&-parse_int("1")), u64::MAX);
        assert_eq!(low_u64(&parse_int("42")), 42);
    }
}
