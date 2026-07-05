#![allow(clippy::missing_safety_doc)]

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: alloc::mimalloc::MiMalloc = alloc::mimalloc::MiMalloc;

// mimalloc is in the default feature set, so an additive `--features snmalloc`
// would otherwise enable both and declare two global allocators.
#[cfg(all(feature = "snmalloc", not(feature = "mimalloc"), not(miri)))]
#[global_allocator]
static GLOBAL: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

pub mod alloc;
pub mod collections;
pub mod nullable;
pub mod option;
pub mod panic;
pub mod rc;
pub mod region;
pub mod symbols;
use panic::panic;
