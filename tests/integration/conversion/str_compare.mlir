// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=EQFN
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=CMPFN
// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s --check-prefix=CALLS
// RUN: %reussir-opt %s --reussir-convert-to-std --convert-scf-to-cf -reussir-lowering-basic-ops --convert-to-llvm --reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// The dynamic string comparisons expand at the scf level: each call site
// becomes a `func.call` into an outlined helper whose body short-circuits
// through the `str.ref_eq` view-identity fast path (and, for `equal`, the
// length mismatch) before an `scf.while` byte scan. Only the straight-line
// residues (`ref_eq`, `len`, `unsafe_byte_at`) reach the LLVM conversion.

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

// The outlined equality helper: ref_eq fast path, length gate, byte loop.
// EQFN-LABEL: func.func private @_RNvC20REUSSIR_STRING_EQUAL
// EQFN: reussir.str.ref_eq
// EQFN: scf.if
// EQFN: reussir.str.len
// EQFN: reussir.str.len
// EQFN: arith.cmpi eq
// EQFN: scf.while
// EQFN: reussir.str.unsafe_byte_at
// EQFN: reussir.str.unsafe_byte_at

// The three-way helper: ref_eq → 0, then a byte scan over min(len) with
// the length tiebreak.
// CMPFN-LABEL: func.func private @_RNvC22REUSSIR_STRING_COMPARE
// CMPFN: reussir.str.ref_eq
// CMPFN: arith.minui
// CMPFN: scf.while
// CMPFN: arith.subi

// The call sites route through the helpers.
// CALLS-LABEL: func.func @str_eq
// CALLS: call @_RNvC20REUSSIR_STRING_EQUAL
// CALLS-LABEL: func.func @str_cmp
// CALLS: call @_RNvC22REUSSIR_STRING_COMPARE

// The view-identity residue lowers to a two-field compare, and no libc
// call is involved anywhere.
// LLVM-NOT: memcmp
// LLVM: llvm.icmp "eq" {{.*}} : !llvm.ptr
// LLVM: llvm.icmp "eq" {{.*}} : i64
