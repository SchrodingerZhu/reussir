// RUN: %reussir-opt %s --reussir-convert-to-std --convert-to-llvm | %FileCheck %s

module {
  // CHECK-LABEL: @test_str_ops
  func.func @test_str_ops(%str: !reussir.str<local>) -> i8 {
    %idx = arith.constant 0 : index
    // CHECK: %[[LEN:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, i64)> 
    // CHECK: %[[COND:.*]] = llvm.icmp "ugt" %[[LEN]], %{{.*}} : i64
    // CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i8) {
    // CHECK:   %[[PTR:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, i64)>
    // CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    // CHECK:   %[[VAL:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i8
    // CHECK:   scf.yield %[[VAL]] : i8
    // CHECK: } else {
    // CHECK:   %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
    // CHECK:   scf.yield %[[ZERO]] : i8
    // CHECK: }
    %byte = reussir.str.byte_at (%str : !reussir.str<local>) [%idx : index] : i8
    return %byte : i8
  }
}
