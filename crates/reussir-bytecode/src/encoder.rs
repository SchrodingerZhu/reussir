//! The low-level byte emitter for the MLIR bytecode container.
//!
//! This module knows nothing about MLIR semantics. It provides the primitive
//! encodings the bytecode format is built from — the `PrefixVarInt` variable
//! width integer, nul-terminated strings, and nestable length-prefixed
//! sections — plus a deduplicating string table.
//!
//! The encodings mirror MLIR's own `EncodingEmitter` byte-for-byte, so the
//! output is readable by any stock `mlir-opt` of a compatible bytecode version.
//! The grammar fragments in this crate's doc comments use a small EBNF (`*`
//! zero-or-more, `?` optional, `|` alternation); the productions are attached to
//! the code that emits them. This module covers the file frame and primitives;
//! the higher-level section grammars live in [`crate::writer`] and the value
//! numbering in [`crate::numbering`].
//!
//! # Wire format: file frame
//!
//! ```text
//! file     = magic version producer section*
//! magic    = 0x4D 0x4C 0xEF 0x52        // "ML\xefR" (MAGIC)
//! version  = varint                     // this crate emits 4; current MLIR is 6
//! producer = nul_string
//! ```
//!
//! Sections appear at most once each and may come in any order; readers index
//! them by id. See [`Emitter::section`] for the per-section frame and
//! [`crate::writer::write_module`] for the order this crate writes them in.

/// The four magic bytes every MLIR bytecode file starts with: `"ML\xefR"`.
///
/// First production of the file frame: `file = magic version producer section*`.
pub const MAGIC: [u8; 4] = [0x4D, 0x4C, 0xEF, 0x52];

/// Numeric identifiers for the container's sections (`mlir::bytecode::Section::ID`).
///
/// Only [`DIALECT`](section::DIALECT), [`ATTR_TYPE_OFFSET`](section::ATTR_TYPE_OFFSET),
/// [`ATTR_TYPE`](section::ATTR_TYPE), [`IR`](section::IR), and
/// [`STRING`](section::STRING) are emitted by this crate; the rest are listed for
/// completeness. The id occupies the low 7 bits of a section's first byte; the
/// high bit signals a present alignment field, which this crate never sets (see
/// [`Emitter::section`]).
pub mod section {
    pub const STRING: u8 = 0;
    pub const DIALECT: u8 = 1;
    pub const ATTR_TYPE: u8 = 2;
    pub const ATTR_TYPE_OFFSET: u8 = 3;
    pub const IR: u8 = 4;
    pub const RESOURCE: u8 = 5;
    pub const RESOURCE_OFFSET: u8 = 6;
    pub const DIALECT_VERSIONS: u8 = 7;
    pub const PROPERTIES: u8 = 8;
}

/// Bit flags marking which components of an operation are present in its
/// encoding (`mlir::bytecode::OpEncodingMask`).
///
/// The byte is the second field of an operation (after its name index) and is
/// the OR of the bits below. It gates the optional operation fields, which —
/// when present — appear in mask-bit order. See [`crate::writer`]'s `write_op`
/// for the full operation grammar. `HAS_PROPERTIES` is never set by this crate:
/// at bytecode version 4 properties are folded into the attribute dictionary.
pub mod op_mask {
    pub const HAS_ATTRS: u8 = 0b0000_0001;
    pub const HAS_RESULTS: u8 = 0b0000_0010;
    pub const HAS_OPERANDS: u8 = 0b0000_0100;
    pub const HAS_SUCCESSORS: u8 = 0b0000_1000;
    pub const HAS_INLINE_REGIONS: u8 = 0b0001_0000;
    pub const HAS_USE_LIST_ORDERS: u8 = 0b0010_0000;
    pub const HAS_PROPERTIES: u8 = 0b0100_0000;
}

/// A growable buffer that accumulates encoded bytecode.
///
/// The emitter appends primitives to an in-memory buffer and supports
/// back-patching a previously written byte (used for operation encoding masks,
/// whose value is only known after the operation body has been measured).
#[derive(Default)]
pub struct Emitter {
    buf: Vec<u8>,
}

impl Emitter {
    /// Create an empty emitter.
    pub fn new() -> Self {
        Emitter { buf: Vec::new() }
    }

    /// The number of bytes written so far. Also the offset the next byte lands
    /// at, which callers snapshot before emitting a placeholder to patch later.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Consume the emitter and return the accumulated bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the accumulated bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Append a single byte.
    pub fn byte(&mut self, b: u8) {
        self.buf.push(b);
    }

    /// Append a slice of raw bytes.
    pub fn bytes(&mut self, bs: &[u8]) {
        self.buf.extend_from_slice(bs);
    }

    /// Overwrite a single, already-emitted byte. The offset must refer to a byte
    /// that was previously written into this emitter.
    pub fn patch_byte(&mut self, offset: usize, value: u8) {
        self.buf[offset] = value;
    }

    /// Emit an unsigned integer using MLIR's `PrefixVarInt` encoding (the
    /// `varint` production every other field is built from).
    ///
    /// The low bits of the first byte hold a prefix — a run of zero bits
    /// terminated by a one bit — whose zero count is the number of *additional*
    /// bytes. The remaining bits, little-endian, carry the value:
    ///
    /// ```text
    /// value < 2^7  : 1 byte   xxxxxxx1            // (value << 1) | 1
    /// value < 2^14 : 2 bytes  xxxxxx10 xxxxxxxx
    /// value < 2^21 : 3 bytes  xxxxx100 ...
    /// ...
    /// value < 2^56 : 8 bytes  10000000 + 7 payload bytes
    /// otherwise    : 9 bytes  00000000 + 8 raw little-endian bytes
    /// ```
    pub fn var_int(&mut self, value: u64) {
        if value >> 7 == 0 {
            self.byte(((value << 1) | 1) as u8);
            return;
        }
        // Find the smallest 2..=8 byte width whose payload holds the value.
        let mut it = value >> 7;
        let mut num_bytes = 2usize;
        while num_bytes < 9 {
            it >>= 7;
            if it == 0 {
                let encoded = ((value << 1) | 1) << (num_bytes - 1);
                let le = encoded.to_le_bytes();
                self.bytes(&le[..num_bytes]);
                return;
            }
            num_bytes += 1;
        }
        // 64 significant bits: escape marker then the raw word.
        self.byte(0);
        self.bytes(&value.to_le_bytes());
    }

    /// Emit a varint whose low bit carries a boolean flag.
    ///
    /// `varint_flag(v, f) = varint((v << 1) | (f ? 1 : 0))`. Used pervasively to
    /// pack a one-bit discriminator alongside a count or index (block argument
    /// count + has-args, region count + isolated-from-above, an attr/type offset
    /// + has-custom-encoding, and so on).
    pub fn var_int_with_flag(&mut self, value: u64, flag: bool) {
        self.var_int((value << 1) | (flag as u64));
    }

    /// Emit a signed integer with zigzag mapping then a [`varint`](Emitter::var_int).
    ///
    /// `svarint(v) = varint((v << 1) ^ (v >> 63))`. (Unused by this crate's
    /// current output, but part of the format's primitive set.)
    pub fn signed_var_int(&mut self, value: i64) {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        self.var_int(zigzag);
    }

    /// Append a string's bytes with no terminator.
    pub fn str_raw(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    /// Append a nul-terminated string: `nul_string = byte* 0x00`.
    ///
    /// This is the form used for the producer string, string-section entries,
    /// and the textual-fallback attribute/type entries.
    pub fn str_nul(&mut self, s: &str) {
        self.str_raw(s);
        self.byte(0);
    }

    /// Append a nested, length-prefixed section.
    ///
    /// ```text
    /// section = id_byte varint(len) data[len]
    /// id_byte = (id & 0x7F) | (has_alignment << 7)
    /// ```
    ///
    /// This crate never aligns a section, so the high bit of `id_byte` is always
    /// clear and no alignment field follows. The body is produced into a
    /// separate [`Emitter`] and spliced in whole.
    pub fn section(&mut self, id: u8, body: &Emitter) {
        self.byte(id);
        self.var_int(body.len() as u64);
        self.bytes(body.as_bytes());
    }
}

/// A deduplicating table of strings referenced elsewhere in the bytecode, and
/// the builder for the String section (id 0).
///
/// Strings are referenced by the index returned from [`StringTable::insert`].
/// Indices are assigned in first-insertion order; inserting an equal string
/// again returns the original index. The section body is:
///
/// ```text
/// string_section = varint(count) string_len{count} string_data{count}
/// string_len     = varint(byte_len + 1)   // emitted in REVERSE order
/// string_data    = nul_string             // emitted in forward order
/// ```
///
/// Emitting the lengths in reverse lets a reader recover each string's start
/// without a separate offset table. See [`StringTable::write`].
#[derive(Default)]
pub struct StringTable {
    // Insertion-ordered unique strings.
    strings: Vec<String>,
    index_of: rustc_hash::FxHashMap<String, usize>,
}

impl StringTable {
    /// Create an empty string table.
    pub fn new() -> Self {
        StringTable::default()
    }

    /// Intern `s`, returning its stable index within the table.
    pub fn insert(&mut self, s: &str) -> usize {
        if let Some(&i) = self.index_of.get(s) {
            return i;
        }
        let i = self.strings.len();
        self.strings.push(s.to_owned());
        self.index_of.insert(s.to_owned(), i);
        i
    }

    /// Write the string section body: the count, every string's length
    /// (including its terminator) in reverse order, then the strings forward.
    pub fn write(&self, emitter: &mut Emitter) {
        emitter.var_int(self.strings.len() as u64);
        for s in self.strings.iter().rev() {
            emitter.var_int((s.len() + 1) as u64);
        }
        for s in &self.strings {
            emitter.str_nul(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Encode a single varint and return the bytes.
    fn vi(v: u64) -> Vec<u8> {
        let mut e = Emitter::new();
        e.var_int(v);
        e.into_bytes()
    }

    #[test]
    fn varint_single_byte() {
        // (v << 1) | 1 for small values, as seen in real mlir-opt output:
        // version 4 -> 0x09, section size 12 -> 0x19.
        assert_eq!(vi(0), vec![0x01]);
        assert_eq!(vi(4), vec![0x09]);
        assert_eq!(vi(12), vec![0x19]);
        assert_eq!(vi(127), vec![0xFF]);
    }

    #[test]
    fn varint_two_bytes() {
        // 128 needs two bytes: prefix has one trailing zero bit.
        // encoded = ((128<<1)|1) << 1 = 0x201 -> little-endian 0x02 0x02? verify
        // against the documented scheme: low bit set after one zero => byte0 even.
        let b = vi(128);
        assert_eq!(b.len(), 2);
        // Round-trips through the reference decoder below.
        assert_eq!(decode(&b), (128, 2));
        assert_eq!(decode(&vi(300)), (300, 2));
        assert_eq!(decode(&vi(16383)), (16383, 2));
    }

    #[test]
    fn varint_widths() {
        for &v in &[16384u64, 1 << 20, 1 << 21, 1 << 28, 1 << 35, u64::MAX] {
            let b = vi(v);
            assert_eq!(decode(&b).0, v, "roundtrip {v}");
        }
    }

    #[test]
    fn varint_with_flag() {
        let mut e = Emitter::new();
        e.var_int_with_flag(5, true);
        assert_eq!(decode(&e.into_bytes()).0, 11); // (5<<1)|1
    }

    // A reference PrefixVarInt decoder mirroring the MLIR reader, used to verify
    // the emitter round-trips.
    fn decode(bytes: &[u8]) -> (u64, usize) {
        let first = bytes[0];
        if first == 0 {
            let mut v = [0u8; 8];
            v.copy_from_slice(&bytes[1..9]);
            return (u64::from_le_bytes(v), 9);
        }
        let num_extra = first.trailing_zeros() as usize;
        let total = num_extra + 1;
        let mut buf = [0u8; 8];
        buf[..total].copy_from_slice(&bytes[..total]);
        let raw = u64::from_le_bytes(buf);
        (raw >> (num_extra + 1), total)
    }

    #[test]
    fn string_table_dedup_and_order() {
        let mut t = StringTable::new();
        assert_eq!(t.insert("a"), 0);
        assert_eq!(t.insert("bb"), 1);
        assert_eq!(t.insert("a"), 0);
        let mut e = Emitter::new();
        t.write(&mut e);
        let bytes = e.into_bytes();
        // count = 2, then sizes in reverse ("bb"=3, "a"=2), then "a\0bb\0".
        assert_eq!(decode(&bytes), (2, 1));
        assert_eq!(decode(&bytes[1..]), (3, 1));
        assert_eq!(decode(&bytes[2..]), (2, 1));
        assert_eq!(&bytes[3..], b"a\0bb\0");
    }

    #[test]
    fn section_framing() {
        let mut body = Emitter::new();
        body.bytes(&[0xAA, 0xBB]);
        let mut e = Emitter::new();
        e.section(section::STRING, &body);
        assert_eq!(e.as_bytes()[0], section::STRING);
        assert_eq!(decode(&e.as_bytes()[1..]), (2, 1));
        assert_eq!(&e.as_bytes()[2..], &[0xAA, 0xBB]);
    }
}
