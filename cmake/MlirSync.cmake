include(FetchContent)

# reussir-lang/mlir-sync provides the `sync` dialect: portable synchronization
# primitives (mutex, reader-writer lock, flat-combining lock, once) with an
# inlined fast path and a futex-style runtime slow path, plus conversions that
# split their control flow into `scf`/`func` (`convert-sync-to-std`) and lower
# the remaining bridge operations to LLVM (`convert-sync-to-llvm`). Reussir's
# lock-guarded cells (mutex/flatlock/rwlock) are lowered onto this dialect.
#
# mlir-sync discovers LLVM/MLIR through the same LLVM_DIR/MLIR_DIR this build
# already resolved, so no extra configuration is needed. Its own tests build a
# `sync-opt` tool that we do not need here.
set(SYNC_ENABLE_TESTS OFF CACHE BOOL "Build sync dialect integration tests" FORCE)

# Default the sync backend to the preserve_most calling convention on the
# futex slow paths: ConvertSyncToSTD annotates the runtime declarations and
# calls with #llvm.cconv<preserve_mostcc>, keeping lock fast paths free of
# caller-saved register spills. The matching runtime side is unconditional —
# reussir-rt always builds mlir_sync with its `nightly` feature (see
# crates/reussir-rt/Cargo.toml), so a preserve_most-annotated call always
# lands on a preserve_most callee. Turning this OFF only drops the caller
# annotation, which is still ABI-safe (a preserve_most callee saves a
# superset of what a C-convention caller expects).
#
# Windows is the exception and must stay OFF: rustc lowers `extern
# "rust-cold"` to the plain C calling convention on Windows targets (LLVM's
# preserve_most is broken there), so the runtime slow paths are Win64-C no
# matter how the crate is built. That flips the mismatch into the unsafe
# direction — a preserve_most-annotated call site keeps live values in
# RCX/RDX/R8-R10 across the call, exactly the registers a Win64-C callee is
# free to clobber. Under contention this corrupts the inlined lock fast
# paths (the rwlock OpenMP e2e test deadlocked the Windows CI). With the
# annotation off, caller and callee agree on the C convention.
#
# aarch64 must also stay OFF, for a subtler reason. There rustc really does
# lower `extern "rust-cold"` to preserve_most (unlike Windows), and the
# CSR_AArch64_RT_MostRegs callee-saved set is the same across the LLVM
# versions the fast path (opt/llc) and the runtime (rustc) are built with, so
# on paper caller and callee agree. Empirically they do not hold up under
# contention: with the annotation ON, the mutex and rwlock OpenMP e2e drivers
# drop exactly one increment on aarch64-linux (mutex 159999/160000, rwlock
# 79999/80000) while the combining-lock and atomic-rc drivers — which keep
# little or nothing live across their cold call — pass. That is the signature
# of the inlined fast path trusting a value to survive the preserve_most
# slow-path call across a boundary where, on aarch64, it does not. Dropping
# the caller annotation is ABI-safe (the runtime callee still saves a
# superset of what a C-convention caller expects), so force it OFF here too:
# the fast path then treats the cold call as a plain C call and keeps nothing
# live in the disputed registers across it.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mlir_sync_processor)
if(WIN32 OR _mlir_sync_processor MATCHES "^(aarch64|arm64)$")
  set(MLIR_SYNC_ENABLE_NIGHTLY_FEATURE OFF CACHE BOOL
    "Annotate sync runtime slow-path declarations and calls with preserve_most (unsupported on Windows and aarch64)"
    FORCE)
else()
  option(MLIR_SYNC_ENABLE_NIGHTLY_FEATURE
    "Annotate sync runtime slow-path declarations and calls with preserve_most"
    ON)
endif()

FetchContent_Declare(
  mlirsync
  GIT_REPOSITORY https://github.com/reussir-lang/mlir-sync.git
  GIT_TAG 3b9412ea23464e262fcfb8f5f4d2538ddb17fadc
)

FetchContent_MakeAvailable(mlirsync)

# The sync dialect libraries rely on the directory-scoped `include_directories`
# of their own project, so linking against them does not propagate the header
# search paths. Expose both the source headers and the tablegen-generated
# `.h.inc` files to the rest of the Reussir build.
set(MLIR_SYNC_INCLUDE_DIRS
    ${mlirsync_SOURCE_DIR}/include
    ${mlirsync_BINARY_DIR}/include
    CACHE INTERNAL "mlir-sync include directories")

# The `sync` lowering's contended paths call the futex slow-path runtime
# (`mlir_sync_*_slow_path`). It rides inside reussir_rt: the `reussir-rt`
# crate depends on mlir-sync's `mlir_sync` crate (pinned to the same revision
# as the FetchContent above — keep the two in sync) and re-exports its
# `#[no_mangle]` entry points, so anything that links the runtime already has
# the slow paths on every platform, with exactly one Rust runtime per image.
