// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

// `reussir.cell.rdlock` runs a read transaction over an rwlock cell: the body
// receives a clone of the element, shares the read lock with concurrent gets
// and other read transactions, and yields only the optional result. This test
// pins down the assembly roundtrip of both forms.

!inner = !reussir.rc<i64 atomic>
!rwlock_i64 = !reussir.rc<!reussir.cell<i64 rwlock> atomic>
!rwlock_rc = !reussir.rc<!reussir.cell<!inner rwlock> atomic>

module {
  // CHECK-LABEL: func.func @rdlock_with_output
  // CHECK: %{{.+}} = reussir.cell.rdlock(%{{.+}} : !reussir.rc<!reussir.cell<i64 rwlock> atomic>) -> i64 {
  // CHECK: ^bb0(%[[CURRENT:.+]]: i64):
  // CHECK: reussir.scf.yield %{{.+}} : i64
  func.func @rdlock_with_output(%cell: !rwlock_i64) -> i64 {
    %doubled = reussir.cell.rdlock(%cell : !rwlock_i64) -> i64 {
      ^bb0(%current: i64):
        %two = arith.constant 2 : i64
        %result = arith.muli %current, %two : i64
        reussir.scf.yield %result : i64
    }
    return %doubled : i64
  }

  // CHECK-LABEL: func.func @rdlock_no_output
  // CHECK: reussir.cell.rdlock(%{{.+}} : !reussir.rc<!reussir.cell<!reussir.rc<i64 atomic> rwlock> atomic>) {
  // CHECK: reussir.scf.yield
  func.func private @consume(!inner) -> ()
  func.func @rdlock_no_output(%cell: !rwlock_rc) {
    reussir.cell.rdlock(%cell : !rwlock_rc) {
      ^bb0(%current: !inner):
        func.call @consume(%current) : (!inner) -> ()
        reussir.scf.yield
    }
    return
  }
}
