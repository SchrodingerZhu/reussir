// RUN: %reussir-opt %s -verify-diagnostics -split-input-file

func.func @non_field_capability(%expected: i64, %desired: i64,
                                %slot: !reussir.ref<i64>) {
  // expected-error @+1 {{target reference must have field capability, got: unspecified}}
  %observed, %ok = reussir.ref.cmpxchg(%expected : i64, %desired : i64,
      %slot : !reussir.ref<i64>) : i64, i1
  return
}

// -----

func.func @non_atomic_element(%expected: i1, %desired: i1,
                              %slot: !reussir.ref<i1 field>) {
  // expected-error @+1 {{cmpxchg element must have a byte-addressable power-of-two bit width, got 'i1'}}
  %observed, %ok = reussir.ref.cmpxchg(%expected : i1, %desired : i1,
      %slot : !reussir.ref<i1 field>) : i1, i1
  return
}

// -----

func.func @element_type_mismatch(%expected: i32, %desired: i32,
                                 %slot: !reussir.ref<i64 field>) {
  // expected-error @+1 {{expected, desired, and observed types must match the reference element type 'i64'}}
  %observed, %ok = reussir.ref.cmpxchg(%expected : i32, %desired : i32,
      %slot : !reussir.ref<i64 field>) : i32, i1
  return
}

// -----

func.func @invalid_ordering(%expected: i64, %desired: i64,
                            %slot: !reussir.ref<i64 field>) {
  // expected-error @+1 {{atomic ordering 'unordered' is invalid for an atomic read-modify-write}}
  %observed, %ok = reussir.ref.cmpxchg(%expected : i64, %desired : i64,
      %slot : !reussir.ref<i64 field>) ordering(unordered) : i64, i1
  return
}
