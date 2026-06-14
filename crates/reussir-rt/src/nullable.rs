use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::option::Option as StdOption;
use std::ptr::NonNull;

/// A nullable pointer. This usually should be combined with rigid rc or rc pointer.
#[repr(transparent)]
pub struct Nullable<T> {
    ptr: StdOption<NonNull<u8>>,
    ty: PhantomData<T>,
}
impl<T> Nullable<T> {
    pub fn new(value: T) -> Self {
        debug_assert!(std::mem::size_of::<T>() == std::mem::size_of::<Self>());
        debug_assert!(std::mem::align_of::<T>() == std::mem::align_of::<Self>());
        let value = ManuallyDrop::new(value);
        let mut uninitialized = std::mem::MaybeUninit::<Nullable<T>>::uninit();
        unsafe {
            uninitialized
                .as_mut_ptr()
                .cast::<ManuallyDrop<T>>()
                .write(value);
        }
        unsafe { uninitialized.assume_init() }
    }
    pub fn null() -> Self {
        Self {
            ptr: None,
            ty: PhantomData,
        }
    }
    pub fn is_null(&self) -> bool {
        self.ptr.is_none()
    }
    pub fn is_non_null(&self) -> bool {
        self.ptr.is_some()
    }
    pub fn unwrap(self) -> T {
        self.to_option().unwrap()
    }
    pub fn to_option(self) -> StdOption<T> {
        let moved = ManuallyDrop::new(self);
        moved
            .ptr
            .map(|ptr| unsafe { NonNull::from(&ptr).cast::<T>().read() })
    }
    pub fn take(&mut self) -> StdOption<T> {
        let result = std::mem::replace(self, Self::null());
        result.to_option()
    }
    pub fn as_ref(&self) -> Option<&T> {
        match self.ptr {
            Some(ptr) => Some(unsafe { NonNull::from(&ptr).cast::<T>().as_ref() }),
            None => None,
        }
    }
    pub fn as_deref<U>(&self) -> Option<&U>
    where
        T: std::ops::Deref<Target = U>,
    {
        self.as_ref().map(|v| v.deref())
    }
}

impl<T> Drop for Nullable<T> {
    fn drop(&mut self) {
        self.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nullable() {
        let data = Nullable::new(crate::rc::Rc::new(123));
        assert!(data.is_non_null());
    }
}
