// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --reussir-convert-to-std --convert-scf-to-cf --reussir-lowering-basic-ops --convert-to-llvm | %FileCheck %s --check-prefix=LLVM

module {
  func.func @read_locked(%lock: memref<!sync.rwlock<i64>>) -> i64 {
    %result = sync.rwlock.read_critical_section %lock
        : memref<!sync.rwlock<i64>> -> i64 {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      sync.yield %value : i64
    }
    return %result : i64
  }
}

// Reussir's ConvertToSTD pass owns sync's structured expansion. It removes
// the high-level critical section but deliberately leaves the straight-line
// bridge operations for the later LLVM conversion.
// STD-LABEL: func.func @read_locked
// STD-NOT: sync.rwlock.read_critical_section
// STD: sync.rwlock.get_raw_rwlock
// STD: scf.while
// STD: sync.raw_rwlock.load_state
// STD: sync.raw_rwlock.cmpxchg_state
// STD: func.call @mlir_sync_rwlock_read_lock_slow_path{{.*}} {CConv = #llvm.cconv<preserve_mostcc>}
// STD: sync.rwlock.get_payload
// STD: memref.load
// STD: sync.raw_rwlock.read_unlock_fast
// STD-NOT: sync.rwlock.read_critical_section

// The retained sync-to-LLVM interface lowers those bridge operations during
// the normal convert-to-llvm step.
// LLVM-LABEL: llvm.func @read_locked
// LLVM-NOT: sync.
// LLVM: llvm.load
// LLVM: llvm.cmpxchg
// LLVM: llvm.call preserve_mostcc @mlir_sync_rwlock_read_lock_slow_path
// LLVM: llvm.atomicrmw sub
// LLVM-NOT: sync.
