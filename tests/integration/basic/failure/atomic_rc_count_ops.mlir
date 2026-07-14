// RUN: %reussir-opt %s -verify-diagnostics --split-input-file

// A blind count store would race concurrent increments on an atomic box.
module {
  func.func @set_atomic(%rc : !reussir.rc<i64 shared atomic>, %c : index) {
    // expected-error @+1 {{'reussir.rc.set' op cannot blind-store the count of an atomic RC pointer}}
    reussir.rc.set(%rc : !reussir.rc<i64 shared atomic>, %c : index)
    return
  }
}

// -----

// The atomic decrement primitive has no meaning on a nonatomic box.
module {
  func.func @fetch_sub_nonatomic(%rc : !reussir.rc<i64>) -> index {
    // expected-error @+1 {{'reussir.rc.fetch_sub' op fetch_sub requires an atomic RC pointer; a nonatomic box decrements through `rc.fetch` and `rc.set`}}
    %c = reussir.rc.fetch_sub(%rc : !reussir.rc<i64>) : index
    return %c : index
  }
}
