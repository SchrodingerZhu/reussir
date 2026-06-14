use std::alloc::Layout;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_allocate(align: usize, size: usize) -> *mut u8 {
    unsafe {
        // The language makes sure the alignment is valid. This will still be checked
        // in debug mode even though it is 'unchecked' API.
        let layout = Layout::from_size_align(size, align).unwrap_unchecked();
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            crate::panic!("allocation failed");
        }
        ptr
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_deallocate(ptr: *mut u8, align: usize, size: usize) {
    if !ptr.is_null() {
        let layout = unsafe { Layout::from_size_align(size, align).unwrap_unchecked() };
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

// TODO: maybe this can be optimized further since we do not need copy the data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_reallocate(
    ptr: *mut u8,
    old_align: usize,
    old_size: usize,
    new_align: usize,
    new_size: usize,
) -> *mut u8 {
    unsafe {
        if ptr.is_null() || old_align != new_align {
            if !ptr.is_null() {
                __reussir_deallocate(ptr, old_align, old_size);
            }
            return __reussir_allocate(new_align, new_size);
        }
        let layout = Layout::from_size_align(old_size, old_align).unwrap_unchecked();
        let ptr = std::alloc::realloc(ptr, layout, new_size);
        if ptr.is_null() {
            crate::panic!("reallocation failed");
        }
        ptr
    }
}
