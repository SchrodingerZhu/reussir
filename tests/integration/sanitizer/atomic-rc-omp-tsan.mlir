// REQUIRES: tsan, openmp
// RUN: %reussir-opt %S/../conversion/atomic_rc_omp_e2e.mlir --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1},convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %cc %tsan_flags -c -x ir %t.ll -o %t.o
// RUN: %cc %tsan_flags %openmp_flags %t.o %S/../conversion/atomic_rc_omp_e2e_main.c %reussir_rt_tsan -o %t.exe %rpath_flag %rpath_san_flag %extra_sys_libs
// RUN: TSAN_OPTIONS=abort_on_error=1:halt_on_error=1:suppressions=%S/Inputs/tsan-libomp.supp %t.exe

// TSan leg of the atomic rc OpenMP e2e: any plain (non-atomic) access to the
// count word — the exact bug class the acquire/release lowering exists to
// prevent — is reported as a data race. The driver keeps all cross-thread
// ordering of user data on the count's acquire/release chain, so even a
// too-weak ordering shows up (payload reads racing the final free). Only
// libomp-internal reports are suppressed: the runtime is not instrumented,
// and its thread-management noise carries libomp/__kmp frames that a real
// refcount race never has.
