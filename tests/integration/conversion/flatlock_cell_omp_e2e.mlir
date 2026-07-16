// REQUIRES: openmp
// On Windows `-lreussir_rt` resolves to the Rust *staticlib*, so this test
// would statically combine two Rust runtimes (reussir_rt's bundled std and
// the no_std mlir_sync archive both define `rust_eh_personality` /
// `rust_begin_unwind`) and lld-link fails with LNK2005; see
// mutex_cell_omp_e2e.mlir. ELF/Mach-O link reussir_rt as a shared library.
// UNSUPPORTED: windows
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// %sync_runtime_lib must precede -lreussir_rt: the staticlib bundles its own
// Rust core, and only archive-first ordering lets the linker resolve its
// internal core symbols from the archive itself (see mutex_cell_omp_e2e.mlir).
// RUN: %cc %openmp_flags %t.o %S/flatlock_cell_omp_e2e_main.c %sync_runtime_lib -o %t.exe -L%library_path -lreussir_rt %rpath_flag %extra_sys_libs
// RUN: %t.exe

// A flatlock cell under real contention: an OpenMP driver
// (flatlock_cell_omp_e2e_main.c) shares one flat-combining-lock-guarded i64
// counter across threads, each performing region-form read-modify-write
// increments in a hot loop. Increments run either inline under the acquired
// status flag or as outlined requests served by the combining thread, so both
// paths of the combining protocol are exercised: a lost wake, a dropped
// request, or a torn update shows up as a wrong final count.

!flatlock_i64 = !reussir.rc<!reussir.cell<i64 flatlock> atomic>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  func.func @flatlock_counter_create(%initial: i64) -> !flatlock_i64 {
    %cell = reussir.cell.create value(%initial : i64) : !flatlock_i64
    return %cell : !flatlock_i64
  }

  // Region-form rmw: add `delta` under the lock and return the old value
  // (routed through the captured out slot the lowering materializes).
  func.func @flatlock_counter_fetch_add(%cell: !flatlock_i64, %delta: i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !flatlock_i64) -> i64 {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  func.func @flatlock_counter_get(%cell: !flatlock_i64) -> i64 {
    %value = reussir.cell.get(%cell : !flatlock_i64) : i64
    return %value : i64
  }

  func.func @flatlock_counter_set(%cell: !flatlock_i64, %value: i64) {
    reussir.cell.set(%value : i64, %cell : !flatlock_i64)
    return
  }

  // The box is `{i32 count, {{ptr tail, i8 status}, i64 payload}}`: 32 bytes
  // at 8-byte alignment, the lock-aware layout token instantiation computes.
  func.func @flatlock_counter_drop(%cell: !flatlock_i64) {
    %t = reussir.rc.dec (%cell : !flatlock_i64) : !reussir.nullable<!reussir.token<align: 8, size: 32>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 32>>)
    return
  }
}
