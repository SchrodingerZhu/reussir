//! Direct allocator entry points for the Reussir language heap.
//!
//! The runtime no longer routes the language heap through Rust's `GlobalAlloc`.
//! `GlobalAlloc::dealloc`/`realloc` require the block's `Layout`, but an
//! unpinned decrement of a non-uniform per-constructor variant frees a
//! `token<?>` whose size is not known at the call site. Instead the heap talks
//! to a *size-recovering* allocator directly (mimalloc: `mi_free`/`mi_realloc`
//! recover the block from the pointer; dlmalloc later). That lets a `token<?>`
//! be a bare pointer with no carried size — no fat `{ptr,size}` value and no
//! per-arm size computation at the free site. The linked allocator is always
//! size-recovering; there is no `Layout`-tracking fallback.
//!
//! Two ABIs coexist: the *sized* `__reussir_deallocate`/`__reussir_reallocate`
//! (static tokens still pass their known size, ignored by the backend) and the
//! *unsized* `__reussir_dealloc_unsized`/`__reussir_realloc_unsized` (dynamic
//! `token<?>`), which pass only the pointer (and, for realloc, the new
//! alignment + size).

/// mimalloc backend (production). Its free/realloc recover the block size from
/// the pointer, so no size is needed to free or grow.
#[cfg(all(feature = "mimalloc", not(miri)))]
mod backend {
    use libmimalloc_sys as ffi;

    /// `MI_MAX_ALIGN_SIZE`: the alignment mimalloc's plain entry points
    /// guarantee for naturally aligned sizes. Must match the value mimalloc is
    /// built with (`.cargo/config.toml` sets `-DMI_MAX_ALIGN_SIZE=8`): the 8
    /// unlocks 8-granular size-classes so a 24-byte variant box stays 24
    /// bytes. A natural bin already satisfies `align` whenever
    /// `align <= 8` and `size` is a multiple of `align`, so such requests
    /// (every rc box among them — Reussir boxes are <=8-aligned) take the
    /// plain fast path; anything over-aligned goes through the aligned path.
    const MI_MAX_ALIGN_SIZE: usize = 8;

    #[inline]
    const fn naturally_aligned(size: usize, align: usize) -> bool {
        align <= MI_MAX_ALIGN_SIZE && size % align == 0
    }

    #[inline]
    pub unsafe fn alloc(align: usize, size: usize) -> *mut u8 {
        unsafe {
            if naturally_aligned(size, align) {
                ffi::mi_malloc(size).cast()
            } else {
                ffi::mi_malloc_aligned(size, align).cast()
            }
        }
    }

    #[inline]
    pub unsafe fn free(ptr: *mut u8) {
        unsafe { ffi::mi_free(ptr.cast()) }
    }

    #[inline]
    pub unsafe fn realloc(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
        // The result only has to satisfy the (new) alignment; mi_realloc accepts
        // blocks from either path and recovers the old size itself.
        unsafe {
            if naturally_aligned(new_size, new_align) {
                ffi::mi_realloc(ptr.cast(), new_size).cast()
            } else {
                ffi::mi_realloc_aligned(ptr.cast(), new_size, new_align).cast()
            }
        }
    }
}

/// Fallback backend for builds without the mimalloc backend: the sanitizer
/// runtime variants (built `--no-default-features`), `miri` (mimalloc's C can't
/// run in the interpreter), and lean embeddings like reussir-jit. It calls the
/// platform C allocator directly — size-recovering (so the unsized `token<?>`
/// ABI works), *interposed by the sanitizers* (so ASan/LSan/MSan see the
/// runtime's allocations, which they cannot through mimalloc's or dlmalloc's
/// private heaps), and shimmed by miri. Never used in production.
///
/// The allocator family differs by platform (see the inner impls): Unix uses
/// `malloc`/`posix_memalign` (both released by the same `free`), while Windows
/// uses the UCRT `_aligned_*` family (whose blocks must be released with
/// `_aligned_free`). Either way the unsized free needs only the pointer.
#[cfg(any(not(feature = "mimalloc"), miri))]
mod backend {
    // Unix: `malloc` for <=16-aligned requests, `posix_memalign` for the
    // (never-hit-at-runtime — Reussir boxes are <=8-aligned) over-aligned case.
    // Both are released by the same `free`, so the pointer-only unsized free
    // stays unambiguous.
    #[cfg(not(windows))]
    mod imp {
        use core::ffi::c_void;

        /// libc `malloc` guarantees `max_align_t` (16-byte) alignment; anything
        /// larger goes through `posix_memalign`.
        const MALLOC_ALIGN: usize = 16;

        #[inline]
        pub unsafe fn alloc(align: usize, size: usize) -> *mut u8 {
            // `malloc` already returns `max_align_t` (16-byte) aligned, so it
            // covers every request up to 16. Larger alignments go through
            // `posix_memalign`; those are always powers of two >= 32, hence
            // valid multiples of `sizeof(void*)`.
            if align <= MALLOC_ALIGN {
                unsafe { libc::malloc(size) as *mut u8 }
            } else {
                let mut ptr: *mut c_void = core::ptr::null_mut();
                if unsafe { libc::posix_memalign(&mut ptr, align, size) } != 0 {
                    return core::ptr::null_mut();
                }
                ptr as *mut u8
            }
        }

        #[inline]
        pub unsafe fn free(ptr: *mut u8) {
            unsafe { libc::free(ptr as *mut c_void) }
        }

        #[inline]
        pub unsafe fn realloc(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
            // `realloc` preserves malloc's `max_align_t` alignment, which covers
            // every Reussir box (all <=8-aligned); reussir never grows to a
            // larger alignment.
            let _ = new_align;
            unsafe { libc::realloc(ptr as *mut c_void, new_size) as *mut u8 }
        }
    }

    // Windows (MSVC/UCRT): the CRT has neither `posix_memalign` nor a
    // `free`-compatible C11 `aligned_alloc` — its only aligned family is
    // `_aligned_malloc`/`_aligned_realloc`/`_aligned_free`, and those blocks
    // *must* be released with `_aligned_free` (calling `free` on them, or
    // `_aligned_free` on a `malloc` block, is undefined). The unsized free is
    // handed only a pointer and cannot tell the two apart, so route *every*
    // allocation through the aligned family uniformly.
    #[cfg(windows)]
    mod imp {
        use core::ffi::c_void;

        // Declared directly rather than via `libc` so this fallback never
        // depends on which of the UCRT aligned entry points the `libc` crate
        // happens to surface for the MSVC target.
        unsafe extern "C" {
            fn _aligned_malloc(size: usize, alignment: usize) -> *mut c_void;
            fn _aligned_free(memblock: *mut c_void);
            fn _aligned_realloc(
                memblock: *mut c_void,
                size: usize,
                alignment: usize,
            ) -> *mut c_void;
        }

        /// `_aligned_malloc` requires a non-zero power-of-two alignment. Every
        /// Reussir request is already a power of two >= 8; clamp for safety.
        const MIN_ALIGN: usize = 8;

        #[inline]
        pub unsafe fn alloc(align: usize, size: usize) -> *mut u8 {
            unsafe { _aligned_malloc(size, align.max(MIN_ALIGN)) as *mut u8 }
        }

        #[inline]
        pub unsafe fn free(ptr: *mut u8) {
            unsafe { _aligned_free(ptr as *mut c_void) }
        }

        #[inline]
        pub unsafe fn realloc(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
            unsafe {
                _aligned_realloc(ptr as *mut c_void, new_size, new_align.max(MIN_ALIGN)) as *mut u8
            }
        }
    }

    pub use imp::{alloc, free, realloc};
}

/// The runtime's `#[global_allocator]`: forwards Rust's own allocations
/// (`Vec`/`Box`, the region headers freed through `std::alloc`, hashbrown, …)
/// to the *same* backend the language heap uses. This is essential, not
/// cosmetic: runtime-internal allocations and `__reussir_*` blocks cross — a
/// region cell is allocated by generated code via `__reussir_allocate` yet its
/// memory is released through `std::alloc::dealloc` — so both sides must be one
/// allocator.
///
/// It uses the same allocator but *not* the language heap's 8-byte-natural
/// convention: the process global allocator must honor `max_align_t`
/// (16-byte) alignment even for small requests, because platform code that
/// shares this heap assumes it (notably macOS system frameworks and parts of
/// Rust's std). So every allocation is forced to at least 16-byte alignment
/// via the backend's aligned path. `__reussir_*` keep the 8-byte convention
/// (reussir boxes request their own alignment and never escape to that code),
/// preserving the 8-granular per-constructor win; they also call the backend
/// *directly*, so the unsized `token<?>` ABI needs no `Layout`.
pub struct ReussirGlobalAlloc;

/// `max_align_t` — the alignment a C/Rust global allocator is expected to
/// provide for any allocation.
const GLOBAL_MAX_ALIGN: usize = 16;

unsafe impl std::alloc::GlobalAlloc for ReussirGlobalAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        unsafe { backend::alloc(layout.align().max(GLOBAL_MAX_ALIGN), layout.size()) }
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: std::alloc::Layout) {
        unsafe { backend::free(ptr) }
    }
    #[inline]
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: std::alloc::Layout,
        new_size: usize,
    ) -> *mut u8 {
        unsafe { backend::realloc(ptr, layout.align().max(GLOBAL_MAX_ALIGN), new_size) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_allocate(align: usize, size: usize) -> *mut u8 {
    let ptr = unsafe { backend::alloc(align, size) };
    if ptr.is_null() {
        unsafe { crate::panic!("allocation failed") };
    }
    ptr
}

/// Free a statically sized token. The size/align are the caller's record of the
/// block; a size-recovering backend ignores them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_deallocate(ptr: *mut u8, _align: usize, _size: usize) {
    if !ptr.is_null() {
        unsafe { backend::free(ptr) };
    }
}

/// Free a dynamic `token<?>`: only the pointer is known — the backend recovers
/// the block size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_dealloc_unsized(ptr: *mut u8) {
    if !ptr.is_null() {
        unsafe { backend::free(ptr) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_reallocate(
    ptr: *mut u8,
    _old_align: usize,
    _old_size: usize,
    new_align: usize,
    new_size: usize,
) -> *mut u8 {
    unsafe { __reussir_realloc_unsized(ptr, new_align, new_size) }
}

/// Resize a dynamic `token<?>` to `new_size`/`new_align`: the old size is not
/// supplied — the backend recovers it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_realloc_unsized(
    ptr: *mut u8,
    new_align: usize,
    new_size: usize,
) -> *mut u8 {
    if ptr.is_null() {
        return unsafe { __reussir_allocate(new_align, new_size) };
    }
    let ptr = unsafe { backend::realloc(ptr, new_align, new_size) };
    if ptr.is_null() {
        unsafe { crate::panic!("reallocation failed") };
    }
    ptr
}
