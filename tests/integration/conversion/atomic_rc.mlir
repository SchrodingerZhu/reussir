// RUN: %reussir-opt %s --reussir-rc-decrement-expansion | %FileCheck %s --check-prefix=EXPAND
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion{expand-decrement=1},convert-scf-to-cf,convert-to-llvm)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

// The atomic rc lifecycle. An atomic box is created like a normal one (the
// initial count store happens before the box is published), incremented with
// a monotonic `atomicrmw add`, and decremented through ONE acquire-release
// `atomicrmw sub` returning the previous count: the subtraction is already
// published, so — unlike the nonatomic plain fetch/cmp/set scheme — the
// shared path stores nothing, and the final decrementer (previous count 1)
// destroys the box under the acquire half's ordering. The uniqueness probe
// `rc.fetch` loads with acquire so copy-on-write stays sound across threads.

!point = !reussir.record<compound "Point" {i64, i64}>
!apoint = !reussir.rc<!point shared atomic>

module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // The expansion decrements with `rc.fetch_sub` and leaves the shared branch
  // empty — no plain fetch, no `rc.set`.
  // EXPAND-LABEL: @atomic_dec
  // EXPAND: %[[OLD:[0-9a-z]+]] = reussir.rc.fetch_sub
  // EXPAND-NOT: reussir.rc.fetch (
  // EXPAND-NOT: reussir.rc.set
  // EXPAND: arith.cmpi eq, %[[OLD]]
  // EXPAND: scf.if
  // EXPAND: reussir.rc.reinterpret
  // EXPAND-NOT: reussir.rc.set

  // LLVM-LABEL: define void @atomic_dec(ptr %0)
  // LLVM: %{{[0-9]+}} = atomicrmw sub ptr %0, i32 1 acq_rel, align 4
  // LLVM-NOT: store i32
  func.func @atomic_dec(%rc: !apoint) {
    %t = reussir.rc.dec (%rc : !apoint) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }

  // LLVM-LABEL: define void @atomic_inc(ptr %0)
  // LLVM: atomicrmw add ptr %{{[0-9]+}}, i32 1 monotonic, align 4
  func.func @atomic_inc(%rc: !apoint) {
    reussir.rc.inc (%rc : !apoint)
    return
  }

  // LLVM-LABEL: define{{.*}}@atomic_fetch(ptr %0)
  // LLVM: load atomic i32, ptr %0 acquire, align 4
  func.func @atomic_fetch(%rc: !apoint) -> index {
    %c = reussir.rc.fetch (%rc : !apoint) : index
    return %c : index
  }

  // The create is a plain allocation + ordinary stores: the box is not yet
  // published, so the initial count needs no atomic ordering.
  // LLVM-LABEL: define ptr @atomic_create(i64 %0, i64 %1)
  // LLVM: call{{.*}}ptr @__reussir_allocate
  // LLVM: store i32 1, ptr
  // LLVM-NOT: atomicrmw
  func.func @atomic_create(%x: i64, %y: i64) -> !apoint {
    %p = reussir.record.compound(%x, %y : i64, i64) : !point
    %rc = reussir.rc.create value(%p : !point) : !apoint
    return %rc : !apoint
  }
}
