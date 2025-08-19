use cxx::{ExternType, type_id};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::os::raw::{c_char, c_void};
use std::str::Utf8Error;
use std::string::FromUtf8Error;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringView<'a> {
    repr: MaybeUninit<[*const c_void; 2]>,
    borrow: PhantomData<&'a [c_char]>,
}

unsafe impl<'a> ExternType for StringView<'a> {
    type Id = type_id!("std::string_view");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("string_view.h");

        #[namespace = "std"]
        #[cxx_name = "string_view"]
        type StringView<'a> = super::StringView<'a>;

        fn string_view_from_str<'a>(s: &'a str) -> StringView<'a>;
        fn string_view_as_bytes<'a>(s: StringView<'a>) -> &'a [c_char];
    }
}

impl<'a> StringView<'a> {
    pub fn new(s: &'a str) -> Self {
        ffi::string_view_from_str(s)
    }

    pub fn as_bytes(self) -> &'a [c_char] {
        ffi::string_view_as_bytes(self)
    }

    pub fn to_string(self) -> Result<String, FromUtf8Error> {
        let bytes = self
            .as_bytes()
            .iter()
            .map(|x| (*x) as u8)
            .collect::<Vec<u8>>();
        String::from_utf8(bytes)
    }
    pub fn to_str(self) -> Result<&'a str, Utf8Error> {
        let bytes = self.as_bytes();
        let u8_bytes =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u8, bytes.len()) };
        std::str::from_utf8(u8_bytes)
    }
}

impl<'a> Deref for StringView<'a> {
    type Target = [c_char];
    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_view() {
        let sv = StringView::new("Hello, world!");
        assert_eq!(sv.to_string().unwrap(), "Hello, world!");
        assert_eq!(sv.to_str().unwrap(), "Hello, world!");
    }
}
