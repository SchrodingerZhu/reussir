// RUN: %reussir-opt %s --reussir-rc-decrement-expansion --reussir-acquire-drop-expansion --reussir-acquire-drop-expansion='expand-decrement=1 outline-record=1' | %FileCheck %s

// CHECK-DAG: func.func private @_RINvNvC4core9intrinsic13drop_in_placeListE
// CHECK-DAG-SAME: attributes {llvm.linkage = #llvm.linkage<linkonce_odr>}

// CHECK-DAG: reussir.rc.fetch
// CHECK-DAG: reussir.rc.set
// CHECK-DAG: reussir.expect
// CHECK-DAG: scf.if

// CHECK-DAG: reussir.record.dispatch

// CHECK-DAG: func.call @_RINvNvC4core9intrinsic13drop_in_placeListE

// CHECK-DAG: func.func @test_rc_dec_expansion

!list_ = !reussir.record<variant "List" incomplete>
!list_nil = !reussir.record<compound "List::Nil" [value] { }>
!list_cons = !reussir.record<compound "List::Cons" [value] { i64, !list_ }>
!list = !reussir.record<variant "List" { !list_nil, !list_cons }>

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> }  {
  func.func @test_rc_dec_expansion(%arg0: !reussir.rc<!list>) {
    %token = reussir.rc.dec (%arg0 : !reussir.rc<!list>) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
    reussir.token.free (%token :  !reussir.nullable<!reussir.token<align: 8, size: 24>>)
    return
  }
}
