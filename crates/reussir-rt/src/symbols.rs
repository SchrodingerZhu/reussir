//! The runtime's C-ABI symbol table, for embedding the runtime into an
//! in-process JIT.
//!
//! A JIT that links this crate (rather than loading the runtime shared library
//! from disk) can register these `(name, address)` pairs directly with its
//! execution session, so JIT'd code resolves its `__reussir_*` calls against the
//! linked-in runtime. Referencing the function items through
//! [`exported_symbols`] also keeps them from being dead-stripped out of a
//! statically linked binary that never calls them from Rust.

use core::ffi::c_void;

use crate::alloc::{
    __reussir_allocate, __reussir_allocate_small, __reussir_dealloc_unsized, __reussir_deallocate,
    __reussir_realloc_unsized, __reussir_reallocate,
};
use crate::instrument::__reussir_report_nonlinear_usage;
use crate::panic::__reussir_panic;
use crate::region::{
    __reussir_acquire_rigid_object, __reussir_cleanup_region, __reussir_freeze_flex_object,
    __reussir_release_rigid_object,
};

/// A runtime function exported under a stable C symbol name, paired with its
/// address in the current process.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeSymbol {
    /// The C symbol name JIT'd code references, e.g. `__reussir_allocate`.
    pub name: &'static str,
    /// The function's address in the current process.
    pub address: *const c_void,
}

// SAFETY: `address` is the address of a `'static` function; sharing it across
// threads is sound.
unsafe impl Send for RuntimeSymbol {}
unsafe impl Sync for RuntimeSymbol {}

/// Returns every runtime function exported under a stable C symbol, paired with
/// its address in the current process.
///
/// An in-process JIT can define these symbols in its execution session to
/// resolve `__reussir_*` calls against the linked-in runtime — no runtime shared
/// library to locate or load. The returned names match the `#[no_mangle]` C
/// symbols exactly.
pub fn exported_symbols() -> &'static [RuntimeSymbol] {
    macro_rules! symbols {
        ($($f:ident),* $(,)?) => {
            &[$(RuntimeSymbol { name: stringify!($f), address: $f as *const c_void }),*]
        };
    }
    symbols![
        __reussir_allocate,
        __reussir_allocate_small,
        __reussir_deallocate,
        __reussir_dealloc_unsized,
        __reussir_reallocate,
        __reussir_realloc_unsized,
        __reussir_panic,
        __reussir_report_nonlinear_usage,
        __reussir_freeze_flex_object,
        __reussir_cleanup_region,
        __reussir_acquire_rigid_object,
        __reussir_release_rigid_object,
    ]
}
