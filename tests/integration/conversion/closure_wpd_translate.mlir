// RUN: %reussir-translate %s --reussir-to-llvmir | %FileCheck %s

// The Reussir dialect's LLVM translation interface materializes the closure
// WPD artifacts the LLVM dialect cannot express (docs/design/closure-wpd.md):
//
//  * `reussir.wpd.type_id` on a vtable global becomes a
//    `!type {i64 0, !"<id>"}` entry plus translation-unit
//    `!vcall_visibility` (`!{i64 2}`) — exactly what LLVM's
//    WholeProgramDevirt consumes;
//  * `reussir.closure.wpd_test` becomes `llvm.type.test(%vtable, !"<id>")`
//    feeding `llvm.assume`.

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>>} {
  llvm.mlir.global internal constant @vtable() {reussir.wpd.type_id = "reussir.closure.wpd.root"} : !llvm.struct<(ptr, ptr, ptr)> {
    %0 = llvm.mlir.zero : !llvm.struct<(ptr, ptr, ptr)>
    llvm.return %0 : !llvm.struct<(ptr, ptr, ptr)>
  }

  llvm.func @use(%box : !llvm.ptr) -> !llvm.ptr {
    %vtable = llvm.load %box : !llvm.ptr -> !llvm.ptr
    reussir.closure.wpd_test (%vtable : !llvm.ptr) id ("reussir.closure.wpd.root")
    llvm.return %vtable : !llvm.ptr
  }
}

// CHECK: @vtable = internal constant { ptr, ptr, ptr } zeroinitializer, !type ![[T0:[0-9]+]], !vcall_visibility ![[VIS:[0-9]+]]

// CHECK-LABEL: define ptr @use(
// CHECK: %[[VT:[0-9a-z]+]] = load ptr, ptr %{{[0-9a-z]+}}
// CHECK-NEXT: %[[HIT:[0-9a-z]+]] = call i1 @llvm.type.test(ptr %[[VT]], metadata !"reussir.closure.wpd.root")
// CHECK-NEXT: call void @llvm.assume(i1 %[[HIT]])

// CHECK-DAG: ![[T0]] = !{i64 0, !"reussir.closure.wpd.root"}
// CHECK-DAG: ![[VIS]] = !{i64 2}
