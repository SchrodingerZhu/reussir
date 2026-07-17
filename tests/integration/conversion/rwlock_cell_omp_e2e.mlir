// REQUIRES: openmp
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},func.func(reussir-token-reuse),reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// The futex slow paths ride inside reussir_rt (see mutex_cell_omp_e2e.mlir).
// RUN: %cc %openmp_flags %t.o %S/rwlock_cell_omp_e2e_main.c -o %t.exe -L%library_path -lreussir_rt %rpath_flag %extra_sys_libs
// RUN: %t.exe

// An rwlock cell under real reader/writer contention: an OpenMP driver
// (rwlock_cell_omp_e2e_main.c) shares one rwlock-guarded i64 counter across
// threads. Writer threads increment through the region-form rmw (a write
// critical section); reader threads snapshot through `cell.rdlock` (a read
// critical section) concurrently. A lost or torn update shows up as a wrong
// final count; a broken read path shows up as a reader observing a value
// outside the range of counts any prefix of increments could have produced,
// or going backwards between two snapshots of the same thread.

!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

// The real target layout (as `reussir-attach-native-target` emits it), so
// `rc.dec`'s explicit token type is verified against the layout the box is
// actually allocated with — parse-time verification runs before any pass,
// and MLIR's default layout (i64 align 4) would reject the align-8 token.
// The pipeline's attach-native-target pass overwrites these attributes with
// the actual host's, so the sizes below stay honest on every LP64 target.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  func.func @rwlock_counter_create(%initial: i64) -> !rwlock_i64 {
    %cell = reussir.cell.create value(%initial : i64) : !rwlock_i64
    return %cell : !rwlock_i64
  }

  // Region-form rmw: add `delta` under the write lock and return the old
  // value.
  func.func @rwlock_counter_fetch_add(%cell: !rwlock_i64, %delta: i64) -> i64 {
    %old = reussir.cell.rmw(%cell : !rwlock_i64) -> i64 {
      ^bb0(%current: i64):
        %next = arith.addi %current, %delta : i64
        reussir.cell.yield(%next : i64) output(%current : i64)
    }
    return %old : i64
  }

  // A read transaction under the shared lock: the body sees the element and
  // yields only its result. Returning `current * 2` (halved by the driver)
  // keeps real work inside the section so the lock is held across it.
  func.func @rwlock_counter_snapshot(%cell: !rwlock_i64) -> i64 {
    %doubled = reussir.cell.rdlock(%cell : !rwlock_i64) -> i64 {
      ^bb0(%current: i64):
        %two = arith.constant 2 : i64
        %scaled = arith.muli %current, %two : i64
        reussir.scf.yield %scaled : i64
    }
    return %doubled : i64
  }

  // Whole-element get (read critical section) and set (write critical
  // section).
  func.func @rwlock_counter_get(%cell: !rwlock_i64) -> i64 {
    %value = reussir.cell.get(%cell : !rwlock_i64) : i64
    return %value : i64
  }

  func.func @rwlock_counter_set(%cell: !rwlock_i64, %value: i64) {
    reussir.cell.set(%value : i64, %cell : !rwlock_i64)
    return
  }

  // The box is `{i32 count, {{i32, i32} header, i64 payload}}`: 24 bytes at
  // 8-byte alignment, the rwlock-aware layout token instantiation also
  // computes.
  func.func @rwlock_counter_drop(%cell: !rwlock_i64) {
    %t = reussir.rc.dec (%cell : !rwlock_i64) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }
}
