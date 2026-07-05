---
name: sanitizer-testing
description: Build and test Reussir with ASan, LSan, MSan, or TSan, especially when validating generated LLVM IR against the Rust runtime allocator path and future multithreaded code.
license: MPL-2.0
---

Use this skill when the task is about sanitizer correctness, sanitizer-enabled
test builds, or checking whether generated Reussir code interacts correctly with
the runtime.

## Standard Procedure

1. Keep the compiler and tools unsanitized. Build dedicated runtime variants
   instead:

   ```bash
   cmake -S . -B build-sanitizers -G Ninja \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DREUSSIR_ENABLE_TESTS=ON \
     -DREUSSIR_RUNTIME_SANITIZERS="address;leak;memory;thread"
   ```

2. Build the normal project plus the sanitizer runtimes:

   ```bash
   ninja -C build-sanitizers reussir-ut check check-sanitizer
   ctest --test-dir build-sanitizers --output-on-failure
   ```

3. For integration tests of generated Reussir programs, prefer this pipeline:

   ```bash
   %rrc input.rr -o %t.ll --emit llvm-ir ...
   %cc %asan_flags -c -x ir %t.ll -o %t.o
   %cc %asan_flags %t.o harness.c %reussir_rt_asan -o %t.exe \
     %rpath_flag %extra_sys_libs
   %asan_env %t.exe
   ```

   Replace `asan` with `lsan`, `msan`, or `tsan` as needed. This lets Clang
   instrument the generated LLVM IR while the runtime comes from a dedicated
   `libreussir_rt.<san>.so` build.

## Reussir-Specific Rules

- Sanitized runtime builds must not use the runtime's custom global allocator.
  The runtime CMake rules build sanitizer variants with `--no-default-features`
  so ASan/LSan can see heap allocation and free events.
- Prefer `check-sanitizer` for sanitizer-specific lit cases under
  `tests/integration/sanitizer/`.

## Accuracy Expectations

- `address` / `leak`:
  Accurate heap reports require the runtime allocator path to stay visible to
  compiler-rt. If allocator customization is reintroduced in sanitized runtime
  builds,
  expect heap-use-after-free and leak reports to become incomplete.
- `memory`:
  MSan requires all participating code to be instrumented. Reussir-generated
  LLVM IR and the Rust runtime can be instrumented, but external libraries and
  system runtimes may still limit report quality.
- `thread`:
  TSan does not need allocator-specific hooks for ordinary memory accesses, but
  future Reussir thread primitives or custom mutex/queue implementations must be
  annotated with ThreadSanitizer synchronization APIs such as
  `__tsan_acquire`, `__tsan_release`, or mutex annotations.

## Test Patterns

- Add one leak-clean sample that either returns a scalar result or uses a known
  correct ownership handoff in the C harness.
- Add one negative leak sample that intentionally omits release and uses `%not`
  under `REQUIRES: lsan`.
- Reuse existing benchmark-like programs such as `rbtree.rr` and `rbtree2.rr`
  so sanitizer coverage stays close to realistic allocation and ownership
  patterns.
