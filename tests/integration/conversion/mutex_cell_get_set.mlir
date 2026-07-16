// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=STD
// RUN: %reussir-opt %s --reussir-convert-to-std --reussir-acquire-drop-expansion | %FileCheck %s --check-prefix=MANAGE
// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,reussir-acquire-drop-expansion{expand-decrement=1 outline-record=1},reussir-convert-to-std,convert-scf-to-cf,reussir-lowering-basic-ops,convert-to-llvm,reconcile-unrealized-casts,canonicalize,cse)' | %reussir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVM

!inner = !reussir.rc<i64 atomic>
!mutex_i64 = !reussir.rc<!reussir.cell<i64 mutex> atomic>
!mutex_rc = !reussir.rc<!reussir.cell<!inner mutex> atomic>

module {
  // STD-LABEL: func.func @get_i64
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.mutex<i64>>
  // STD: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // STD: sync.raw_mutex.try_lock %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD: sync.raw_mutex.unlock_fast %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: return %[[VALUE]]
  // LLVM-LABEL: define i64 @get_i64
  // LLVM: cmpxchg
  // LLVM: load i64
  // LLVM: atomicrmw xchg
  // LLVM: ret i64
  func.func @get_i64(%cell: !mutex_i64) -> i64 {
    %value = reussir.cell.get(%cell : !mutex_i64) : i64
    return %value : i64
  }

  // STD-LABEL: func.func @set_i64
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.mutex<i64>>
  // STD: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // STD: sync.raw_mutex.try_lock %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<i64>) : !reussir.ref<i64 field atomic>
  // STD: reussir.ref.drop(%[[SLOT]]
  // STD: reussir.ref.store(%[[SLOT]]
  // STD: sync.raw_mutex.unlock_fast %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: return
  // LLVM-LABEL: define void @set_i64
  // LLVM: cmpxchg
  // LLVM: store i64
  // LLVM: atomicrmw xchg
  // LLVM: ret void
  func.func @set_i64(%value: i64, %cell: !mutex_i64) {
    reussir.cell.set(%value : i64, %cell : !mutex_i64)
    return
  }

  // STD-LABEL: func.func @get_rc
  // STD: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.mutex<!reussir.rc<i64 atomic>>>
  // STD: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // STD: sync.raw_mutex.try_lock %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // STD: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<!reussir.rc<i64 atomic>>) : !reussir.ref<!reussir.rc<i64 atomic> field atomic>
  // STD: %[[VALUE:.+]] = reussir.ref.load(%[[SLOT]]
  // STD-NEXT: reussir.rc.inc(%[[VALUE]]
  // STD: sync.raw_mutex.unlock_fast %[[RAW]]
  // STD-NOT: reussir.expect
  // STD-NOT: reussir.panic
  // STD: return %[[VALUE]]
  // LLVM-LABEL: define ptr @get_rc
  // LLVM: %[[VALUE:.+]] = load ptr
  // LLVM: atomicrmw add ptr %{{.+}}, i32 1
  // LLVM: atomicrmw xchg
  // LLVM: ret ptr %[[VALUE]]
  func.func @get_rc(%cell: !mutex_rc) -> !inner {
    %value = reussir.cell.get(%cell : !mutex_rc) : !inner
    return %value : !inner
  }

  // MANAGE-LABEL: func.func @set_rc(
  // MANAGE-SAME: %[[NEW:[a-zA-Z0-9_]+]]: !reussir.rc<i64 atomic>
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: %[[VIEW:.+]] = reussir.ref.to_memref{{.*}} : memref<!sync.mutex<!reussir.rc<i64 atomic>>>
  // MANAGE: %[[RAW:.+]] = sync.mutex.get_raw_mutex %[[VIEW]]
  // MANAGE: sync.raw_mutex.try_lock %[[RAW]]
  // MANAGE-NOT: reussir.expect
  // MANAGE-NOT: reussir.panic
  // MANAGE: %[[PAYLOAD:.+]] = sync.mutex.get_payload %[[VIEW]]
  // MANAGE: %[[SLOT:.+]] = reussir.ref.from_memref(%[[PAYLOAD]] : memref<!reussir.rc<i64 atomic>>) : !reussir.ref<!reussir.rc<i64 atomic> field atomic>
  // MANAGE: %[[OLD:.+]] = reussir.ref.load(%[[SLOT]]
  // MANAGE-NEXT: %{{.+}} = reussir.rc.dec(%[[OLD]] : !reussir.rc<i64 atomic>)
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: reussir.ref.store(%[[SLOT]]{{.*}}) (%[[NEW]] : !reussir.rc<i64 atomic>)
  // MANAGE: sync.raw_mutex.unlock_fast %[[RAW]]
  // MANAGE-NOT: reussir.expect
  // MANAGE-NOT: reussir.panic
  // MANAGE-NOT: reussir.rc.inc
  // MANAGE: return
  // LLVM-LABEL: define void @set_rc(
  // LLVM-SAME: ptr %[[NEW:.+]], ptr
  // LLVM: %[[SLOT:.+]] = getelementptr { i32, ptr }, ptr %{{.+}}, i32 0, i32 1
  // LLVM: %[[OLD:.+]] = load ptr, ptr %[[SLOT]]
  // LLVM: atomicrmw sub ptr %[[OLD]], i32 1
  // LLVM: store ptr %[[NEW]], ptr %[[SLOT]]
  // LLVM: atomicrmw xchg
  // LLVM: ret void
  func.func @set_rc(%value: !inner, %cell: !mutex_rc) {
    reussir.cell.set(%value : !inner, %cell : !mutex_rc)
    return
  }
}
