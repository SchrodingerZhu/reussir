// RUN: %reussir-opt %s -verify-diagnostics

// LLVM cmpxchg requires byte-addressable scalar storage, so bool/i1 is not an
// atomic-cell arithmetic primitive even though it is represented as an MLIR
// integer.
// expected-error @+1 {{atomic cell element must have a byte-addressable power-of-two bit width, got 'i1'}}
!bad_atomic_bool = !reussir.cell<i1 atomic>

module {
}
