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

# The `sync` lowering's contended paths call the futex slow-path runtime
# (`mlir_sync_*_slow_path`), provided by mlir-sync's `mlir_sync` Rust crate as
# a static library. Build it with cargo so end-to-end tests can link and run
# real lock behavior; regular compilation never needs it (the symbols are only
# referenced by fully lowered modules at link time).
set(MLIR_SYNC_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/mlir-sync-cargo)
set(MLIR_SYNC_RUNTIME_LIB
    ${MLIR_SYNC_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}mlir_sync${CMAKE_STATIC_LIBRARY_SUFFIX}
    CACHE INTERNAL "mlir-sync runtime static library")
add_custom_target(mlir-sync-runtime
  COMMAND ${CMAKE_COMMAND} -E env
      CARGO_TARGET_DIR=${MLIR_SYNC_CARGO_TARGET_DIR}
      RUSTFLAGS=-Awarnings
      cargo build -p mlir_sync --release --quiet
  WORKING_DIRECTORY ${mlirsync_SOURCE_DIR}
  BYPRODUCTS ${MLIR_SYNC_RUNTIME_LIB}
  COMMENT "Building mlir-sync runtime with cargo (release profile)"
)
