// REQUIRES: openmp
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// The futex slow paths (`mlir_sync_*_slow_path`) ride inside reussir_rt,
// which depends on mlir-sync's `mlir_sync` crate — one Rust runtime per
// image, no separate sync archive to link.
// RUN: %cc %openmp_flags %t.o %S/mutex_cell_omp_e2e_main.c -o %t.exe -L%library_path -lreussir_rt %rpath_flag %extra_sys_libs
// RUN: %t.exe

// A mutex cell under real contention: an OpenMP driver
// (mutex_cell_omp_e2e_main.c) shares one mutex-guarded i64 counter across
// threads, each performing region-form read-modify-write increments in a hot
// loop. Every increment runs inside the mutex critical section, so a lost or
// torn update — a fast path that fails to exclude, an unlock that misses a
// wake, a slow path that misparks — shows up as a wrong final count. The
// driver also exercises get/set around the parallel region and drops the
// final reference, freeing the box through the mutex-aware token layout.

!mutex_i64 = !reussir.rc<!reussir.cell<i64 mutex> atomic>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  func.func @mutex_counter_create(%initial: i64) -> !mutex_i64 {
    %cell = reussir.cell.create value(%initial : i64) : !mutex_i64
    return %cell : !mutex_i64
  }

  // Region-form rmw: add `delta` under the lock and return the old value.
  func.func @mutex_counter_fetch_add(%cell: !mutex_i64, %delta: i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !mutex_i64) -> i64 {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  func.func @mutex_counter_get(%cell: !mutex_i64) -> i64 {
    %value = reussir.cell.get(%cell : !mutex_i64) : i64
    return %value : i64
  }

  func.func @mutex_counter_set(%cell: !mutex_i64, %value: i64) {
    reussir.cell.set(%value : i64, %cell : !mutex_i64)
    return
  }

  // The box is `{i32 count, {i32 state, i64 payload}}`: 24 bytes at 8-byte
  // alignment, the mutex-aware layout token instantiation also computes.
  func.func @mutex_counter_drop(%cell: !mutex_i64) {
    %t = reussir.rc.dec (%cell : !mutex_i64) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }
}
