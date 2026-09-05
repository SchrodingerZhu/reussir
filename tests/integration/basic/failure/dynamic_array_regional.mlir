// RUN: %reussir-opt %s --split-input-file --verify-diagnostics

// Regional boxes have state/next/vtable fields instead of the shared
// strided header. Reject both construction types and borrowed references.
module {
  // expected-error @+1 {{dynamic-extent arrays require shared RC capability}}
  func.func private @flex() -> !reussir.rc<!reussir.array<? x i32> flex>
}

// -----

module {
  // expected-error @+1 {{dynamic-extent arrays require shared RC capability}}
  func.func private @rigid() -> !reussir.rc<!reussir.array<2 x ? x i32> rigid>
}

// -----

module {
  // expected-error @+1 {{dynamic-extent arrays do not support regional references}}
  func.func private @flex_ref(!reussir.ref<!reussir.array<? x i32> flex>)
}

// -----

module {
  // expected-error @+1 {{dynamic-extent arrays do not support regional references}}
  func.func private @rigid_ref(!reussir.ref<!reussir.array<? x i32> rigid>)
}

// -----

module {
  // expected-error @+1 {{dynamic-extent arrays do not support regional references}}
  func.func private @regional_ref(!reussir.ref<!reussir.array<? x i32> regional>)
}

// -----

module {
  // expected-error @+1 {{dynamic-extent arrays do not support regional boxes}}
  func.func private @box(!reussir.rc_box<regional !reussir.array<? x i32>>)
}
