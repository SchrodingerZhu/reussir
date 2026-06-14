use std::{marker::PhantomData, string::String as StdString};

use crate::rc::{Rc, RcRef};

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

impl<'a> std::ops::Deref for Str<'a> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)) }
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
