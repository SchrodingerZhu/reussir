// REQUIRES: openmp
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1},convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %openmp_flags %t.o %S/atomic_rc_omp_e2e_main.c -o %t.exe -L%library_path -lreussir_rt %rpath_flag %extra_sys_libs
// RUN: %t.exe

// The atomic rc lifecycle under real contention: an OpenMP driver
// (atomic_rc_omp_e2e_main.c) shares one atomic box across threads, each of
// which clones (rc.inc), reads, and drops (rc.dec) it in a hot loop, and the
// main thread releases the final reference afterwards. A lost decrement
// leaks the box; a duplicated one frees it while other threads still read —
// the sanitizer legs of this module (tests/integration/sanitizer/
// atomic-rc-omp-{asan,tsan}.mlir) turn either into a hard failure.

!pair = !reussir.record<compound "Pair" {i64, i64}>
!apair = !reussir.rc<!pair shared atomic>
!refpair = !reussir.ref<!pair shared atomic>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  func.func @atomic_pair_create(%x: i64, %y: i64) -> !apair attributes { llvm.passthrough = ["sanitize_thread", "sanitize_address"] } {
    %p = reussir.record.compound(%x, %y : i64, i64) : !pair
    %rc = reussir.rc.create value(%p : !pair) : !apair
    return %rc : !apair
  }

  func.func @atomic_pair_clone(%rc: !apair) attributes { llvm.passthrough = ["sanitize_thread", "sanitize_address"] } {
    reussir.rc.inc (%rc : !apair)
    return
  }

  func.func @atomic_pair_sum(%rc: !apair) -> i64 attributes { llvm.passthrough = ["sanitize_thread", "sanitize_address"] } {
    %ref = reussir.rc.borrow (%rc : !apair) : !refpair
    %s0 = reussir.ref.project (%ref : !refpair) [0] : !reussir.ref<i64 shared atomic>
    %x = reussir.ref.load (%s0 : !reussir.ref<i64 shared atomic>) : i64
    %s1 = reussir.ref.project (%ref : !refpair) [1] : !reussir.ref<i64 shared atomic>
    %y = reussir.ref.load (%s1 : !reussir.ref<i64 shared atomic>) : i64
    %sum = arith.addi %x, %y : i64
    return %sum : i64
  }

  func.func @atomic_pair_drop(%rc: !apair) attributes { llvm.passthrough = ["sanitize_thread", "sanitize_address"] } {
    %t = reussir.rc.dec (%rc : !apair) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }
}
