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

FetchContent_Declare(
  mlirsync
  GIT_REPOSITORY https://github.com/reussir-lang/mlir-sync.git
  GIT_TAG 47ef8aae13eeed41400822ffd53213b78a6abcea
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
