#![allow(clippy::missing_safety_doc)]

// The language heap talks to a size-recovering allocator directly through the
// `__reussir_*` entry points in `alloc` (so a `token<?>` frees without a
// `Layout`). Rust's own allocations go through `ReussirGlobalAlloc`, which
// forwards to that *same* backend — runtime-internal memory and `__reussir_*`
// blocks cross (region headers are allocated by `__reussir_allocate` and freed
// through `std::alloc`), so they must be one allocator. Skipped under miri,
// where the backend cannot service the interpreter's own allocations.
#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: alloc::ReussirGlobalAlloc = alloc::ReussirGlobalAlloc;

pub mod alloc;
pub mod bridge;
pub mod collections;
pub mod hash;
pub mod nullable;
pub mod option;
pub mod panic;
pub mod rc;
pub mod region;
pub mod symbols;
use panic::panic;

// The `sync` dialect's futex slow paths: lowered lock-guarded cell code calls
// mlir_sync's `#[no_mangle]` entry points (`mlir_sync_*_slow_path`), so the
// runtime links the crate and thereby exports them from every artifact.
pub use mlir_sync;
