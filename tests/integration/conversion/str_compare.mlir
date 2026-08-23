// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s
// RUN: %reussir-opt %s --reussir-convert-to-std --convert-scf-to-cf -reussir-lowering-basic-ops --convert-to-llvm --reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// The dynamic string comparisons expand inline at the scf level: `scf.if`
// carries only the fast paths — `str.ref_eq` view identity (identical
// `{ptr, len}` pairs answer without a byte scan) and, for equality, the
// length gate — while the main comparison stays a `memcmp` call through
// the straight-line `str.unsafe_memcmp` residue, lowered at the LLVM
// level.

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

// Equality: ref_eq fast path, length gate, then the memcmp residue.
// CHECK-LABEL: func.func @str_eq
// CHECK: %[[SAME:.+]] = reussir.str.ref_eq
// CHECK: scf.if %[[SAME]]
// CHECK: reussir.str.len
// CHECK: reussir.str.len
// CHECK: %[[LENEQ:.+]] = arith.cmpi eq
// CHECK: scf.if %[[LENEQ]]
// CHECK: reussir.str.unsafe_memcmp
// CHECK-NOT: scf.while

// Three-way: ref_eq → 0, memcmp over min(len), length tiebreak selects.
// CHECK-LABEL: func.func @str_cmp
// CHECK: %[[SAME2:.+]] = reussir.str.ref_eq
// CHECK: scf.if %[[SAME2]]
// CHECK: arith.minui
// CHECK: reussir.str.unsafe_memcmp
// CHECK: arith.select
// CHECK-NOT: scf.while

// The residues at the LLVM level: the view-identity two-field compare and
// the real `memcmp`.
// LLVM-DAG: llvm.func @memcmp(!llvm.ptr, !llvm.ptr, i64) -> i32
// LLVM-DAG: llvm.icmp "eq" {{.*}} : !llvm.ptr
// LLVM-DAG: llvm.call @memcmp
