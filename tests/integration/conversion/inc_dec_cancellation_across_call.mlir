// RUN: %reussir-opt %s --reussir-inc-dec-cancellation | %FileCheck %s

// Regression (#290): `reussir-inc-dec-cancellation` must not cancel an
// `rc.inc` against a later aliased `rc.dec` when a *call* sits between them.
// A call site (`func.call`) is opaque and may consume the incremented
// reference — a closure application lowers to a `func.call` that takes
// ownership of its closure — so erasing the retain across it frees a value that
// is still live afterward. This is the closure-in-`[shared]`-record double-free:
// the closure was loaded twice from the same field (so the loads MustAlias) and
// the first borrow-extract retain was cancelled across the consuming `apply1`.

!clo = !reussir.closure<(i32) -> i32>

module {
  func.func private @apply1(%f: !reussir.rc<!clo>, %x: i32) -> i32

  func.func @retain_across_call(%f: !reussir.rc<!clo>) -> i32 {
    %c4 = arith.constant 4 : i32
    // The incremented reference is handed to (and consumed by) the call, so
    // this retain must survive: cancelling it against the decrement below would
    // release the value while the call still owns it.
    reussir.rc.inc (%f : !reussir.rc<!clo>)
    %r = func.call @apply1(%f, %c4) : (!reussir.rc<!clo>, i32) -> i32
    reussir.rc.dec (%f : !reussir.rc<!clo>)
    return %r : i32
  }
}

// The retain, the consuming call, and the release all survive — the call blocks
// the cancellation (without the fix the inc/dec pair is erased across it).
// CHECK-LABEL: func.func @retain_across_call
// CHECK:       reussir.rc.inc
// CHECK:       call @apply1
// CHECK:       reussir.rc.dec
