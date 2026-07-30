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
//!
//! The language reallocs do NOT preserve the block's contents (#362): every
//! `token.realloc` resizes a *dead* donor whose storage the acceptor is about
//! to overwrite with a freshly constructed object — the compiler already
//! declares the entry points `allockind("realloc,uninitialized,...")`. So the
//! backend resizes in place when the existing block satisfies the new layout
//! and otherwise does a plain free + alloc, never a copying `realloc`. Only
//! `ReussirGlobalAlloc::realloc` keeps the copying path — Rust's
//! `GlobalAlloc::realloc` contract requires it.

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
        align <= MI_MAX_ALIGN_SIZE && size.is_multiple_of(align)
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

    /// Allocate a *small, naturally aligned* block: the compiler only emits
    /// calls with a constant `size <= MI_SMALL_SIZE_MAX` (1024) whose
    /// requested alignment is natural (`align <= 8` and dividing `size`), so
    /// the generic wrapper's alignment/divisibility checks and mimalloc's
    /// own size classification are all discharged statically. `mi_malloc_small`
    /// indexes the heap's direct small-page table without a bounds check in
    /// release builds — the size precondition is load-bearing.
    #[inline]
    pub unsafe fn alloc_small(size: usize) -> *mut u8 {
        debug_assert!(size <= ffi::MI_SMALL_SIZE_MAX);
        unsafe { ffi::mi_malloc_small(size).cast() }
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

    /// Resize a block whose contents are dead — the language-heap realloc.
    /// Keeps the block when it already satisfies the new layout (mimalloc
    /// recovers the usable size from the pointer, so a same-bin resize — the
    /// pairing TokenReuse prefers — is pointer-identity); otherwise free +
    /// alloc, skipping mi_realloc's copy of the dead contents. The
    /// `usable / 2` floor mirrors mi_realloc's shrink policy: a much smaller
    /// object moves to a right-sized block instead of squatting on a big one.
    #[inline]
    pub unsafe fn realloc_dead(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
        unsafe {
            let usable = ffi::mi_usable_size(ptr.cast());
            if (ptr as usize).is_multiple_of(new_align)
                && new_size <= usable
                && new_size >= usable / 2
            {
                return ptr;
            }
            ffi::mi_free(ptr.cast());
            alloc(new_align, new_size)
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
            // `realloc` preserves malloc's `max_align_t` alignment, which
            // covers every Reussir box (all <=8-aligned; reussir never grows
            // to a larger alignment) and every runtime-internal type. An
            // over-16-aligned realloc CANNOT be honored here (plain `realloc`
            // would mis-align it, and without the old size the block cannot be
            // moved by hand) — assert the precondition instead of silently
            // returning under-aligned memory.
            debug_assert!(
                new_align <= MALLOC_ALIGN,
                "the libc fallback cannot realloc to alignment {new_align} > {MALLOC_ALIGN}"
            );
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
        // happens to surface for the MSVC target. (msvc targets resolve these
        // from ucrt.lib; mingw's msvcrt also exports them, but the gnu target
        // is untested in CI.)
        unsafe extern "C" {
            fn _aligned_malloc(size: usize, alignment: usize) -> *mut c_void;
            fn _aligned_free(memblock: *mut c_void);
            fn _aligned_realloc(
                memblock: *mut c_void,
                size: usize,
                alignment: usize,
            ) -> *mut c_void;
        }

        /// `_aligned_realloc` forbids changing a block's alignment, so alloc
        /// and realloc must normalize `align` the SAME way. Clamping every
        /// request up to 16 makes all of today's traffic uniform — the
        /// language heap asks <= 8 (verifier-enforced power of two, possibly
        /// < 8) and the global allocator exactly 16 — so a realloc can never
        /// observe a different alignment than the allocation. A > 16 request
        /// keeps its own alignment; that too is realloc-stable because a
        /// box's alignment is a property of its type and does not change
        /// across `token.realloc`.
        const MIN_ALIGN: usize = 16;

        #[inline]
        fn norm_align(align: usize) -> usize {
            align.max(MIN_ALIGN)
        }

        #[inline]
        pub unsafe fn alloc(align: usize, size: usize) -> *mut u8 {
            unsafe { _aligned_malloc(size, norm_align(align)) as *mut u8 }
        }

        #[inline]
        pub unsafe fn free(ptr: *mut u8) {
            unsafe { _aligned_free(ptr as *mut c_void) }
        }

        #[inline]
        pub unsafe fn realloc(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
            unsafe {
                _aligned_realloc(ptr as *mut c_void, new_size, norm_align(new_align)) as *mut u8
            }
        }
    }

    pub use imp::{alloc, free, realloc};

    /// The small-block entry under the fallback backend: the platform
    /// allocators have no small fast path, so this is the plain (sanitizer-
    /// interposable) allocation at natural alignment. The compiler-side
    /// contract (size <= 1024, natural alignment) still holds; it is simply
    /// not needed here.
    #[inline]
    pub unsafe fn alloc_small(size: usize) -> *mut u8 {
        unsafe { imp::alloc(8, size) }
    }

    /// Resize a block whose contents are dead — the language-heap realloc.
    /// The platform allocators expose no portable usable-size query, so this
    /// is always free + alloc. That is not just simple but *diagnostic*: the
    /// result is always a fresh block, so generated code that wrongly reads
    /// the donor's contents after a `token.realloc` fails loudly under the
    /// sanitizers/miri instead of being masked by an in-place resize. It also
    /// honors any alignment, unlike the copying realloc above.
    #[inline]
    pub unsafe fn realloc_dead(ptr: *mut u8, new_align: usize, new_size: usize) -> *mut u8 {
        unsafe {
            imp::free(ptr);
            imp::alloc(new_align, new_size)
        }
    }
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
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
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

/// Allocate a block whose layout the compiler proved *small and naturally
/// aligned*: constant `size <= 1024` (mimalloc's `MI_SMALL_SIZE_MAX` — the
/// shared contract in `include/Reussir/Support/AllocatorBinModel.h`) with
/// `align <= 8` dividing it. This is the sized fast path of
/// `__reussir_allocate` with every runtime check discharged at compile time;
/// the OOM contract is identical.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_allocate_small(size: usize) -> *mut u8 {
    let ptr = unsafe { backend::alloc_small(size) };
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
/// supplied — the backend recovers it. The block's contents are NOT preserved
/// (see the module docs): the donor is dead, so a cross-bin resize is a plain
/// free + alloc instead of a copying realloc.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_realloc_unsized(
    ptr: *mut u8,
    new_align: usize,
    new_size: usize,
) -> *mut u8 {
    if ptr.is_null() {
        return unsafe { __reussir_allocate(new_align, new_size) };
    }
    let ptr = unsafe { backend::realloc_dead(ptr, new_align, new_size) };
    if ptr.is_null() {
        unsafe { crate::panic!("reallocation failed") };
    }
    ptr
}

#[cfg(test)]
mod tests {
    /// Alloc → grow (realloc) → free, straight through the selected backend.
    /// Asserts only what the ABI promises: non-null, requested alignment, and
    /// that the result is a writable block freeable by the pointer-only free.
    /// Deliberately NO content-preservation check — the language-heap realloc
    /// resizes a *dead* donor's storage and does not copy (#362).
    /// Cfg-agnostic on purpose: under the default feature
    /// this exercises the mimalloc backend, under `--no-default-features` the
    /// platform fallback (Unix `malloc`/`posix_memalign`, Windows
    /// `_aligned_*`), and under miri the shimmed fallback — the same paths the
    /// language heap and `ReussirGlobalAlloc` use in each configuration.
    /// Realloc alignments stop at 16: every caller's ceiling today (language
    /// heap <= 8, `ReussirGlobalAlloc` exactly 16) and the libc fallback's
    /// documented realloc limit.
    #[test]
    fn backend_alloc_realloc_free() {
        for align in [1usize, 8, 16] {
            unsafe {
                let p = super::backend::alloc(align, 24);
                assert!(!p.is_null());
                assert_eq!(p as usize % align.max(1), 0, "align {align}");
                let q = super::backend::realloc(p, align, 40);
                assert!(!q.is_null());
                assert_eq!(q as usize % align.max(1), 0, "align {align}");
                // Touch the full grown extent (miri/ASan validate the block).
                for i in 0..40 {
                    q.add(i).write(i as u8);
                }
                super::backend::free(q);
            }
        }
    }

    /// Over-aligned allocation (no realloc — see the fallback's realloc
    /// limit): the aligned path on every backend.
    #[test]
    fn backend_over_aligned_alloc() {
        unsafe {
            let p = super::backend::alloc(64, 24);
            assert!(!p.is_null());
            assert_eq!(p as usize % 64, 0);
            super::backend::free(p);
        }
    }

    /// The dead-contents resize behind the language reallocs. Asserts the ABI
    /// (non-null, aligned, writable, pointer-only-freeable) across a within-bin
    /// grow, a cross-bin grow, and a large shrink; contents are dead by
    /// contract so none are checked.
    #[test]
    fn backend_realloc_dead() {
        unsafe {
            let p = super::backend::alloc(8, 24);
            assert!(!p.is_null());
            // A resize the block already fits (same bin): mimalloc keeps it,
            // so this must be pointer-identity there. The fallback always
            // moves — no identity assert for it.
            let q = super::backend::realloc_dead(p, 8, 24);
            assert!(!q.is_null());
            #[cfg(all(feature = "mimalloc", not(miri)))]
            assert_eq!(q, p, "a fitting resize stays in place under mimalloc");
            // Cross-bin grow: relocates without copying the dead contents.
            let r = super::backend::realloc_dead(q, 8, 4096);
            assert!(!r.is_null());
            assert_eq!(r as usize % 8, 0);
            for i in 0..4096 {
                r.add(i).write(i as u8);
            }
            // Large shrink: moves to a right-sized block (mi_realloc's shrink
            // policy) rather than squatting on the big one.
            let s = super::backend::realloc_dead(r, 8, 16);
            assert!(!s.is_null());
            assert_eq!(s as usize % 8, 0);
            s.cast::<u64>().write(42);
            super::backend::free(s);
        }
    }

    /// The unsized entry points: free and realloc recover the block from the
    /// pointer alone. Again no content assertion — only that the resized
    /// block is valid and pointer-only-freeable.
    #[test]
    fn unsized_abi_roundtrip() {
        unsafe {
            let p = super::__reussir_allocate(8, 16);
            assert!(!p.is_null());
            let q = super::__reussir_realloc_unsized(p, 8, 32);
            assert!(!q.is_null());
            q.cast::<u64>().write(0xDEAD_BEEF_u64);
            super::__reussir_dealloc_unsized(q);
        }
    }
}
