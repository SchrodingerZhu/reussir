// RUN: %reussir-opt %s --reussir-convert-to-std -convert-scf-to-cf -reussir-lowering-basic-ops -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s

// Test closure uniqify operation with closure passed in and passed out
// No create operation is used - closure is provided as function argument
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  func.func @test_closure_uniqify_with_input(
    %input_closure : !reussir.rc<!reussir.closure<(i32) -> i32>>
  ) -> !reussir.rc<!reussir.closure<(i32) -> i32>> {
    // Test uniqify operation - should be rewritten to SCF if-else with rc.is_unique check
    %uniqified = reussir.closure.uniqify (%input_closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    return %uniqified : !reussir.rc<!reussir.closure<(i32) -> i32>>
  }
}

// CHECK: llvm.func @test_closure_uniqify_with_input(%[[INPUT:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:   %[[GEP_REFCNT:.*]] = llvm.getelementptr %[[INPUT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, struct<(ptr, ptr, i32)>)>
// CHECK:   %[[REFCNT:.*]] = llvm.load %[[GEP_REFCNT]] : !llvm.ptr -> i32
// CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   %[[IS_UNIQUE:.*]] = llvm.icmp "eq" %[[REFCNT]], %[[ONE]] : i32
// CHECK:   llvm.cond_br %[[IS_UNIQUE]], ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:  // pred: ^bb0
// CHECK:   llvm.br ^[[JOIN:bb[0-9]+]](%[[INPUT]] : !llvm.ptr)
// CHECK: ^[[ELSE]]:  // pred: ^bb0
// CHECK:   %[[GEP_VTABLE:.*]] = llvm.getelementptr %[[INPUT]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, struct<(ptr, ptr, i32)>)>
// CHECK:   %[[VTABLE:.*]] = llvm.load %[[GEP_VTABLE]] invariant_group : !llvm.ptr -> !llvm.ptr
// CHECK:   %[[GEP_CLONE:.*]] = llvm.getelementptr %[[VTABLE]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, ptr)>
// CHECK:   %[[CLONE_FUNC:.*]] = llvm.load %[[GEP_CLONE]] invariant_group : !llvm.ptr -> !llvm.ptr
// CHECK:   %[[CLONED:.*]] = llvm.call %[[CLONE_FUNC]](%[[INPUT]]) : !llvm.ptr, (!llvm.ptr) -> !llvm.ptr
// CHECK:   %[[GEP_VTABLE2:.*]] = llvm.getelementptr %[[INPUT]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, struct<(ptr, ptr, i32)>)>
// CHECK:   %[[VTABLE2:.*]] = llvm.load %[[GEP_VTABLE2]] invariant_group : !llvm.ptr -> !llvm.ptr
// CHECK:   %[[GEP_DROP:.*]] = llvm.getelementptr %[[VTABLE2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, ptr)>
// CHECK:   %[[DROP_FUNC:.*]] = llvm.load %[[GEP_DROP]] invariant_group : !llvm.ptr -> !llvm.ptr
// CHECK:   llvm.call %[[DROP_FUNC]](%[[INPUT]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK:   llvm.br ^[[JOIN]](%[[CLONED]] : !llvm.ptr)
// CHECK: ^[[JOIN]](%[[JOIN_ARG:[0-9]+]]: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
// CHECK:   llvm.br ^[[FINAL:bb[0-9]+]]
// CHECK: ^[[FINAL]]:  // pred: ^[[JOIN]]
// CHECK:   llvm.return %[[JOIN_ARG]] : !llvm.ptr
// CHECK: }
