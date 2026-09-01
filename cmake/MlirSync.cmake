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

# The futex slow-path calling convention needs no configuration: the runtime
# exports its slow paths as plain extern "C", and ConvertSyncToSTD reaches
# them through internal preserve_most trampolines it generates itself, so
# lock fast paths keep their live registers while the caller-saved spilling
# happens once inside the cold trampoline. There is no ABI coupling between
# the backend and the runtime build on any platform. (mlir-sync's
# MLIR_SYNC_ENABLE_NIGHTLY_FEATURE option only opts its own test runtime
# into nightly-gated optimizations, and that runtime is not built here.)

FetchContent_Declare(
  mlirsync
  GIT_REPOSITORY https://github.com/reussir-lang/mlir-sync.git
  GIT_TAG c6a8493003e4cf32441d715a16f270b253ab3fe1
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
