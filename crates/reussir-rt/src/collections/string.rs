use std::{marker::PhantomData, string::String as StdString};

use crate::rc::{Rc, RcRef};

/// A functional (copy-on-write, rc-boxed) string: the exposed string
/// collection for Reussir's polymorphic FFI, crossing the boundary the same
/// way [`crate::collections::vec::Vec`] does. Owning operations take `self`
/// linearly and return the updated string (`make_mut` copies only when the
/// box is shared); read-only operations borrow. The `#[repr(transparent)]`
/// wrapper over [`Rc`] is what the FFI contract requires: the compiler
/// treats the value as an rc pointer whose count sits at offset 0.
#[derive(Clone)]
#[repr(transparent)]
pub struct String(Rc<StdString>);

impl Default for String {
    fn default() -> Self {
        Self::new()
    }
}

impl String {
    pub fn new() -> Self {
        Self(Rc::new(StdString::new()))
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Rc::new(StdString::with_capacity(capacity)))
    }
    pub fn make_mut(&mut self) -> &mut StdString {
        self.0.make_mut()
    }
    pub fn push(mut self, ch: char) -> Self {
        self.make_mut().push(ch);
        self
    }
    pub fn to_string(&self) -> String {
        String(self.0.clone())
    }
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
    pub fn push_str(mut self, s: Str) -> Self {
        self.make_mut().push_str(&s);
        self
    }
    pub fn len(&self) -> usize {
        self.0.data_ref().len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.data_ref().is_empty()
    }
    pub fn as_ref(&self) -> StringRef<'_> {
        StringRef(self.0.as_ref())
    }
}

impl std::ops::Deref for String {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct StringRef<'a>(RcRef<'a, StdString>);

impl<'a> std::ops::Deref for StringRef<'a> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> StringRef<'a> {
    pub fn len(self) -> usize {
        self.0.len()
    }
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }
}

/// The FFI rendering of Reussir's `str`: bit-identical to the lowered
/// `{ptr, len}` pair, so a `str` parameter of an `#[ffi(import)]` arrives
/// here directly. Values built by the compiler point at `'static` literal
/// storage and are valid UTF-8; Reussir's `slice` intrinsic is
/// byte-granular, so a sliced `str` may start inside a multi-byte
/// character — [`Self::new_checked`] is the guard when that matters.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Str<'a> {
    ptr: *const u8,
    len: usize,
    _marker: PhantomData<&'a str>,
}

impl<'a> Str<'a> {
    // still unsafe since ptr can be invalid
    pub unsafe fn new_checked(ptr: *const u8, len: usize) -> Option<Self> {
        let data = std::str::from_utf8(unsafe { std::slice::from_raw_parts(ptr, len) }).ok()?;
        Some(Self {
            ptr: data.as_ptr(),
            len: data.len(),
            _marker: PhantomData,
        })
    }
    pub unsafe fn new_unchecked(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }
}

impl<'a> Str<'a> {
    /// The raw bytes. Unlike [`Deref`], this makes no UTF-8 assumption, so
    /// it is the safe view of a byte-granular slice.
    pub fn as_bytes(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// A well-mixed digest of the byte content, stable within a process.
    /// Backs `std::hash::Hash for str`: a proper hasher chunks the byte
    /// stream into words, which the surface language cannot express until
    /// bare-pointer intrinsics land, so the chunking happens here and the
    /// digest feeds the Reussir-side hasher.
    pub fn content_hash(&self) -> u64 {
        use std::hash::{DefaultHasher, Hasher};
        let mut hasher = DefaultHasher::new();
        hasher.write(self.as_bytes());
        hasher.finish()
    }
}

impl<'a> std::ops::Deref for Str<'a> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)) }
    }
}

// Comparisons and hashing are byte-wise over `as_bytes`, matching the
// dialect's `str.equal`/`str.compare` semantics (lexicographic bytes,
// length as the shared-prefix tiebreak) without assuming UTF-8 — this is
// what lets a Reussir `str` key a Rust-backed container.
impl PartialEq for Str<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for Str<'_> {}

impl PartialOrd for Str<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Str<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl std::hash::Hash for Str<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string() {
        let mut s = String::new();
        s = s.push('a');
        assert_eq!(s.as_str(), "a");
    }
}
