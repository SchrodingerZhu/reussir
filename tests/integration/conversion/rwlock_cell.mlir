// RUN: %reussir-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %reussir-opt %s --reussir-lowering-scf-ops | %FileCheck %s --check-prefix=SCF
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-acquire-drop-expansion,reussir-lowering-scf-ops,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>

module {
  // The read_with region prints back verbatim, handing its body a zero-ranked
  // memref view of the payload.
  // ROUNDTRIP-LABEL: func.func @rwlock_read
  // ROUNDTRIP: reussir.cell.read_with(%{{.*}} : !reussir.rc<!reussir.cell<i64 rwlock> atomic>) -> i64
  // ROUNDTRIP: ^bb0(%{{.*}}: memref<i64>):
  // ROUNDTRIP: reussir.scf.yield

  // SCF lowering expands read_with into a sync read critical section that is
  // further lowered to the rwlock read-lock fast path (an scf.while CAS loop)
  // plus the payload load, releasing with the read-unlock fast path. The raw
  // sync fast-path/bridge ops survive for the LLVM lowering.
  // SCF-LABEL: func.func @rwlock_read
  // SCF: reussir.ref.as_memref(%{{.*}}) : memref<!sync.rwlock<i64>>
  // SCF: sync.rwlock.get_raw_rwlock
  // SCF: scf.while
  // SCF: sync.raw_rwlock.cmpxchg_state
  // SCF: sync.rwlock.get_payload
  // SCF: memref.load
  // SCF: sync.raw_rwlock.read_unlock_fast
  // SCF-NOT: reussir.cell.read_with

  // The whole thing bottoms out in a cmpxchg-based read lock, an inline
  // payload load, and an atomicrmw-based read unlock, with the runtime slow
  // paths called on contention.
  // LLVM-LABEL: define i64 @rwlock_read
  // LLVM: cmpxchg ptr
  // LLVM: call void @mlir_sync_rwlock_read_lock_slow_path
  // LLVM: atomicrmw sub ptr
  // LLVM: call void @mlir_sync_rwlock_unlock_slow_path
  func.func @rwlock_read(%cell: !rwlock_i64) -> i64 {
    %v = reussir.cell.read_with(%cell : !rwlock_i64) -> i64 {
      ^bb0(%view: memref<i64>):
        %loaded = memref.load %view[] : memref<i64>
        reussir.scf.yield %loaded : i64
    }
    return %v : i64
  }
}
