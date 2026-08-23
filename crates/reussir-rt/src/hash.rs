//! Byte-stream continuations of `std::hash`'s hashers.
//!
//! The hashers' byte protocols read the buffer in word-sized chunks,
//! which needs raw-pointer access the surface language cannot express
//! yet; these functions continue each hasher's algorithm from its current
//! state. The constants and update shapes mirror
//! `library/std/src/hash.rr` and must stay in lockstep with it.
//!
//! TODO: retire this module once bare-pointer unsafe intrinsics let the
//! surface language express the chunked reads natively.

/// aHash's multiplier, `library/std/src/hash.rr`'s `folded_multiply`
/// constant.
const MULTIPLE: u64 = 6364136223846793005;

/// Fx's multiplier, `hash.rr`'s `fast_mix` constant.
const FX_MULTIPLE: u64 = 0x517cc1b727220a95;

#[inline]
fn folded_multiply(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

#[inline]
fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"))
}

/// `FastHasher`'s byte protocol: every full 8-byte little-endian word
/// through the Fx mix, the remaining tail as one zero-extended word, and
/// the byte length last — so a zero-padded tail cannot collide with the
/// longer input it pads to.
pub fn fx_write_bytes(state: u64, bytes: &[u8]) -> u64 {
    let mix =
        |state: u64, value: u64| (state.rotate_left(5) ^ value).wrapping_mul(FX_MULTIPLE);
    let mut state = state;
    let mut rest = bytes;
    while rest.len() >= 8 {
        state = mix(state, le_u64(rest));
        rest = &rest[8..];
    }
    if !rest.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        state = mix(state, u64::from_le_bytes(tail));
    }
    mix(state, bytes.len() as u64)
}

/// The overlapping word pair a short buffer (≤ 8 bytes) reads as, per
/// aHash's `read_small`.
fn read_small(bytes: &[u8]) -> (u64, u64) {
    match bytes.len() {
        0 => (0, 0),
        1 => (u64::from(bytes[0]), u64::from(bytes[0])),
        2 | 3 => (
            u64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            u64::from(bytes[bytes.len() - 1]),
        ),
        len => (
            u64::from(u32::from_le_bytes(bytes[..4].try_into().expect("four bytes"))),
            u64::from(u32::from_le_bytes(
                bytes[len - 4..].try_into().expect("four bytes"),
            )),
        ),
    }
}

/// `StrongHasher`'s byte protocol — aHash's byte-buffer path (as in the
/// LLVM libc port): fold the length into the buffer, then feed word pairs
/// through `large_update` (`hash.rr`'s spelling of the pair update, rotate
/// 23) — the possibly-overlapping tail pair first, then every full
/// 16-byte block; short buffers read as one overlapping pair. `pad`,
/// `key0`, and `key1` are the hasher's immutable state; the returned value
/// is the advanced `buffer`.
pub fn strong_write_bytes(buffer: u64, pad: u64, key0: u64, key1: u64, bytes: &[u8]) -> u64 {
    let update = |buffer: u64, low: u64, high: u64| {
        let combined = folded_multiply(low ^ key0, high ^ key1);
        (buffer.wrapping_add(pad) ^ combined).rotate_left(23)
    };
    let size = bytes.len();
    let mut buffer = buffer.wrapping_add(size as u64).wrapping_mul(MULTIPLE);
    if size > 8 {
        if size > 16 {
            buffer = update(
                buffer,
                le_u64(&bytes[size - 16..]),
                le_u64(&bytes[size - 8..]),
            );
            let mut rest = bytes;
            while rest.len() > 16 {
                buffer = update(buffer, le_u64(rest), le_u64(&rest[8..]));
                rest = &rest[16..];
            }
        } else {
            buffer = update(buffer, le_u64(bytes), le_u64(&bytes[size - 8..]));
        }
    } else {
        let (low, high) = read_small(bytes);
        buffer = update(buffer, low, high);
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference spelling of the Fx protocol, word by word, to pin the
    // chunk walk in `fx_write_bytes`.
    fn fx_reference(state: u64, bytes: &[u8]) -> u64 {
        let mix =
            |s: u64, v: u64| (s.rotate_left(5) ^ v).wrapping_mul(FX_MULTIPLE);
        let mut state = state;
        let words = bytes.len() / 8;
        for w in 0..words {
            state = mix(state, le_u64(&bytes[w * 8..]));
        }
        let tail = &bytes[words * 8..];
        if !tail.is_empty() {
            let mut padded = [0u8; 8];
            padded[..tail.len()].copy_from_slice(tail);
            state = mix(state, u64::from_le_bytes(padded));
        }
        mix(state, bytes.len() as u64)
    }

    #[test]
    fn fx_chunks_match_the_reference() {
        let data: Vec<u8> = (0u8..64).collect();
        for len in [0usize, 1, 7, 8, 9, 16, 17, 63] {
            assert_eq!(
                fx_write_bytes(99, &data[..len]),
                fx_reference(99, &data[..len]),
                "len {len}"
            );
        }
    }

    #[test]
    fn equal_bytes_hash_equal_and_prefixes_differ() {
        let a: Vec<u8> = (0u8..40).collect();
        for len in [0usize, 1, 2, 3, 4, 8, 9, 16, 17, 40] {
            assert_eq!(
                strong_write_bytes(1, 2, 3, 4, &a[..len]),
                strong_write_bytes(1, 2, 3, 4, &a[..len]),
                "len {len}"
            );
            if len > 0 {
                assert_ne!(
                    strong_write_bytes(1, 2, 3, 4, &a[..len]),
                    strong_write_bytes(1, 2, 3, 4, &a[..len - 1]),
                    "len {len}"
                );
                assert_ne!(
                    fx_write_bytes(0, &a[..len]),
                    fx_write_bytes(0, &a[..len - 1]),
                    "len {len}"
                );
            }
        }
    }
}
