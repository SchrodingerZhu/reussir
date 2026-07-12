// RUN: %reussir-opt %s --inline | %FileCheck %s

// Calls nested in reussir regions inline like anywhere else; the one
// structural exception is a body carrying `region.run`, which must not
// become nested under another `region.run`.

!cell_ = !reussir.record<variant "Cell" incomplete>
!cell_some = !reussir.record<compound "Cell::Some" [value] {i64}>
!cell_none = !reussir.record<compound "Cell::None" [value] {}>
!cell = !reussir.record<variant "Cell" {!cell_some, !cell_none}>
!rc_cell = !reussir.rc<!cell>

func.func private @bump(%x: i64) -> i64 {
  %one = arith.constant 1 : i64
  %r = arith.addi %x, %one : i64
  return %r : i64
}

// The callee inlines into the dispatch arm: no calls remain.
// CHECK-LABEL: func.func @dispatch_arm_call(
// CHECK-NOT: call @bump
func.func @dispatch_arm_call(%c: !rc_cell) -> i64 {
  %ref = reussir.rc.borrow (%c : !rc_cell) : !reussir.ref<!cell>
  %r = reussir.record.dispatch (%ref : !reussir.ref<!cell>) -> i64 {
    [0] -> {
      ^bb0(%payload: !reussir.ref<!cell_some>):
      %slot = reussir.ref.project (%payload : !reussir.ref<!cell_some>) [0] : !reussir.ref<i64>
      %v = reussir.ref.load (%slot : !reussir.ref<i64>) : i64
      %b = func.call @bump(%v) : (i64) -> i64
      reussir.scf.yield %b : i64
    }
    [1] -> {
      ^bb1(%payload: !reussir.ref<!cell_none>):
      %zero = arith.constant 0 : i64
      reussir.scf.yield %zero : i64
    }
  }
  return %r : i64
}
