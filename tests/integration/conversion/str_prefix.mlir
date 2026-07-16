// RUN: %reussir-opt %s --reussir-convert-to-std --convert-to-llvm | %FileCheck %s

module {
  // CHECK-LABEL: @test_startswith_safe
  // CHECK-SAME: %[[STR:.*]]: !llvm.struct<(ptr, i64)>
  func.func @test_startswith_safe(%str: !reussir.str<local>) -> i1 {
    // CHECK: %[[LEN:.*]] = llvm.extractvalue %[[STR]][1] : !llvm.struct<(ptr, i64)>
    // CHECK: %[[PREFIX_LEN:.*]] = llvm.mlir.constant(3 : index) : i64
    // CHECK: %[[COND:.*]] = llvm.icmp "uge" %[[LEN]], %[[PREFIX_LEN]] : i64
    // CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i1) {
    // CHECK:   %[[PTR_SAFE:.*]] = llvm.extractvalue %[[STR]][0] : !llvm.struct<(ptr, i64)>
    // CHECK:   %[[PREFIX_ADDR_SAFE:.*]] = llvm.mlir.addressof @{{.*}} : !llvm.ptr
    // CHECK:   %[[PREFIX_LEN_SAFE:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK:   %[[CMP_SAFE:.*]] = llvm.call @memcmp(%[[PTR_SAFE]], %[[PREFIX_ADDR_SAFE]], %[[PREFIX_LEN_SAFE]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    // CHECK:   %[[ZERO_SAFE:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:   %[[UNSAFE_RES:.*]] = llvm.icmp "eq" %[[CMP_SAFE]], %[[ZERO_SAFE]] : i32
    // CHECK:   scf.yield %[[UNSAFE_RES]] : i1
    // CHECK: } else {
    // CHECK:   %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
    // CHECK:   scf.yield %[[FALSE]] : i1
    // CHECK: }
    %res = reussir.str.startswith (%str : !reussir.str<local>) "abc" : i1
    return %res : i1
  }

  // CHECK-LABEL: @test_unsafe_startswith_short
  // CHECK-SAME: %[[STR:.*]]: !llvm.struct<(ptr, i64)>
  func.func @test_unsafe_startswith_short(%str: !reussir.str<local>) -> i1 {
    // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[STR]][0] : !llvm.struct<(ptr, i64)>
    // CHECK: %[[PREFIX_ADDR:.*]] = llvm.mlir.addressof @{{.*}} : !llvm.ptr
    // CHECK: %[[PREFIX_LEN:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[CMP:.*]] = llvm.call @memcmp(%[[PTR]], %[[PREFIX_ADDR]], %[[PREFIX_LEN]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[RES:.*]] = llvm.icmp "eq" %[[CMP]], %[[ZERO]] : i32
    // CHECK: llvm.return %[[RES]] : i1
    %res = reussir.str.unsafe_startswith (%str : !reussir.str<local>) "abc" : i1
    return %res : i1
  }

  // CHECK-LABEL: @test_unsafe_startswith_long
  // CHECK-SAME: %[[STR:.*]]: !llvm.struct<(ptr, i64)>
  func.func @test_unsafe_startswith_long(%str: !reussir.str<local>) -> i1 {
    // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[STR]][0] : !llvm.struct<(ptr, i64)>
    // CHECK: %[[PREFIX_ADDR:.*]] = llvm.mlir.addressof @{{.*}} : !llvm.ptr
    // CHECK: %[[PREFIX_LEN:.*]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK: %[[CMP:.*]] = llvm.call @memcmp(%[[PTR]], %[[PREFIX_ADDR]], %[[PREFIX_LEN]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[RES:.*]] = llvm.icmp "eq" %[[CMP]], %[[ZERO]] : i32
    // CHECK: return %[[RES]] : i1
    %res = reussir.str.unsafe_startswith (%str : !reussir.str<local>) "12345678901234567890123456789012" : i1
    return %res : i1
  }
}
