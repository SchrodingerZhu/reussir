// REQUIRES: asan, openmp
// RUN: %reussir-opt %S/../conversion/atomic_rc_omp_e2e.mlir --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1},convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %cc %asan_flags -c -x ir %t.ll -o %t.o
// RUN: %cc %asan_flags %openmp_flags %t.o %S/../conversion/atomic_rc_omp_e2e_main.c %reussir_rt_asan -o %t.exe %rpath_flag %rpath_san_flag %extra_sys_libs
// RUN: %asan_env LSAN_OPTIONS=suppressions=%S/Inputs/lsan-libomp.supp %t.exe

// ASan leg of the atomic rc OpenMP e2e: a duplicated decrement frees the box
// while other threads still read it (heap-use-after-free); a lost one leaks
// it (LeakSanitizer rides along on Linux ASan). libomp's own never-freed
// worker bookkeeping is suppressed — the patterns cannot match a leaked box,
// which allocates through __reussir_allocate.
