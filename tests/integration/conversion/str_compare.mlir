// RUN: %reussir-opt %s --convert-to-llvm | %FileCheck %s

// The dynamic string comparisons expand branchlessly: one memcmp over
// min(len_lhs, len_rhs) — always in bounds — then selects. `equal` is a
// length check AND-ed with the byte check; `compare` follows the memcmp
// convention (negative/zero/positive) with length as the tiebreak.

module @test {
  func.func @str_eq(%a : !reussir.str<local>, %b : !reussir.str<local>) -> i1 {
    %eq = reussir.str.equal(%a : !reussir.str<local>, %b : !reussir.str<local>) : i1
    return %eq : i1
  }

  func.func @str_cmp(%a : !reussir.str<local>, %b : !reussir.str<local>) -> i32 {
    %ord = reussir.str.compare(%a : !reussir.str<local>, %b : !reussir.str<local>) : i32
    return %ord : i32
  }
}

// CHECK-DAG: llvm.func @memcmp(!llvm.ptr, !llvm.ptr, i64) -> i32

// CHECK-LABEL: @str_eq
// CHECK-DAG: %[[LLEN:.+]] = llvm.extractvalue %{{.+}}[1]
// CHECK-DAG: %[[RLEN:.+]] = llvm.extractvalue %{{.+}}[1]
// CHECK: %[[LENEQ:.+]] = llvm.icmp "eq" %[[LLEN]], %[[RLEN]]
// CHECK: %[[SHORTER:.+]] = llvm.icmp "ult" %[[LLEN]], %[[RLEN]]
// CHECK: %[[MIN:.+]] = llvm.select %[[SHORTER]], %[[LLEN]], %[[RLEN]]
// CHECK: %[[CMP:.+]] = llvm.call @memcmp(%{{.+}}, %{{.+}}, %[[MIN]])
// CHECK: %[[BYTESEQ:.+]] = llvm.icmp "eq" %[[CMP]]
// CHECK: llvm.and %[[LENEQ]], %[[BYTESEQ]]

// CHECK-LABEL: @str_cmp
// CHECK: %[[MIN2:.+]] = llvm.select
// CHECK: %[[CMP2:.+]] = llvm.call @memcmp(%{{.+}}, %{{.+}}, %[[MIN2]])
// CHECK: %[[NE:.+]] = llvm.icmp "ne" %[[CMP2]]
// CHECK: llvm.select %[[NE]], %[[CMP2]]
