// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s

// Dispatching through a reference borrowed from an atomic rc box: the
// lowering re-derives each singleton arm's coerced reference from the
// scrutinee, and must propagate the scrutinee reference's atomic kind — a
// `record.coerce` producing a plain `shared` reference would leave an
// unresolved materialization against the atomic block argument.

!option_some = !reussir.record<compound "AOption::Some" [value] {i32}>
!option_none = !reussir.record<compound "AOption::None" [value] {}>
!option = !reussir.record<variant "AOption" {!option_some, !option_none}>

module {
  // CHECK-LABEL: func.func @dispatch_atomic
  // CHECK: reussir.record.coerce
  // CHECK-SAME: !reussir.ref<!reussir.record<compound "AOption::Some" [value] {i32}> shared atomic>
  // CHECK-NOT: reussir.record.dispatch
  func.func @dispatch_atomic(%opt_ref : !reussir.ref<!option shared atomic>) -> i32 {
    %result = reussir.record.dispatch(%opt_ref : !reussir.ref<!option shared atomic>) -> i32 {
      [0] -> {
        ^bb0(%some_ref : !reussir.ref<!option_some shared atomic>):
          %field_ref = reussir.ref.project(%some_ref : !reussir.ref<!option_some shared atomic>) [0] : !reussir.ref<i32 shared>
          %extracted = reussir.ref.load(%field_ref : !reussir.ref<i32 shared>) : i32
          reussir.scf.yield %extracted : i32
      }
      [1] -> {
        ^bb0(%none_ref : !reussir.ref<!option_none shared atomic>):
          %default = arith.constant -1 : i32
          reussir.scf.yield %default : i32
      }
    }
    func.return %result : i32
  }
}
