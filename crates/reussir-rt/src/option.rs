use std::mem::MaybeUninit;
type StdOption<T> = std::option::Option<T>;

#[repr(C)]
pub struct Option<T> {
    tag: usize,
    data: MaybeUninit<T>,
}

impl<T> Option<T> {
    pub fn new(value: StdOption<T>) -> Self {
        match value {
            Some(value) => Self {
                tag: 1,
                data: MaybeUninit::new(value),
            },
            None => Self {
                tag: 0,
                data: MaybeUninit::uninit(),
            },
        }
    }
    pub fn as_std_ref(&self) -> StdOption<&T> {
        if self.tag == 0 {
            None
        } else {
            unsafe { Some(self.data.assume_init_ref()) }
        }
    }
}

impl<T> Drop for Option<T> {
    fn drop(&mut self) {
        if self.tag == 1 {
            unsafe { self.data.assume_init_drop() };
        }
    }
}

impl<T> From<StdOption<T>> for Option<T> {
    fn from(value: StdOption<T>) -> Self {
        Self::new(value)
    }
}
