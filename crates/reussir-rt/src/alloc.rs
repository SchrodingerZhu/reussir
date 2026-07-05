use std::alloc::Layout;

/// A `GlobalAlloc` over mimalloc's raw entry points.
///
/// The `mimalloc` crate routes every allocation through `mi_malloc_aligned`,
/// and mimalloc v3 serves an aligned request from its natural size bins only
/// when the rounded block size is a power of two; everything else takes the
/// over-allocating aligned path, so e.g. a 48-byte rc box occupies a 64-byte
/// block — growing every heap line the program touches — and pays the generic
/// path's extra instructions on each call. Dispatch on the layout instead: a
/// natural bin already satisfies `align` whenever `align <= 16`
/// (`MI_MAX_ALIGN_SIZE`: bin sizes are multiples of 8 and, above 16, of 16,
/// and a page's first block is 16-aligned) and `size` is a multiple of
/// `align`, so such requests — every rc box among them — take the plain fast
/// path.
#[cfg(all(feature = "mimalloc", not(miri)))]
pub mod mimalloc {
    use std::alloc::{GlobalAlloc, Layout};

    use libmimalloc_sys as ffi;

    /// `MI_MAX_ALIGN_SIZE`: the alignment mimalloc's plain entry points
    /// guarantee for naturally aligned sizes.
    const MI_MAX_ALIGN_SIZE: usize = 16;

    #[inline]
    const fn naturally_aligned(size: usize, align: usize) -> bool {
        align <= MI_MAX_ALIGN_SIZE && size % align == 0
    }

    pub struct MiMalloc;

    unsafe impl GlobalAlloc for MiMalloc {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe {
                if naturally_aligned(layout.size(), layout.align()) {
                    ffi::mi_malloc(layout.size()).cast()
                } else {
                    ffi::mi_malloc_aligned(layout.size(), layout.align()).cast()
                }
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            unsafe {
                if naturally_aligned(layout.size(), layout.align()) {
                    ffi::mi_zalloc(layout.size()).cast()
                } else {
                    ffi::mi_zalloc_aligned(layout.size(), layout.align()).cast()
                }
            }
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            unsafe { ffi::mi_free(ptr.cast()) }
        }

        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // The result only has to satisfy the (unchanged) alignment, so
            // dispatch on the new size; mi_realloc accepts blocks from either
            // path.
            unsafe {
                if naturally_aligned(new_size, layout.align()) {
                    ffi::mi_realloc(ptr.cast(), new_size).cast()
                } else {
                    ffi::mi_realloc_aligned(ptr.cast(), new_size, layout.align()).cast()
                }
            }
        }
    }
}

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
