// RUN: %reussir-opt %s -reussir-rc-dispatch-fusion | %FileCheck %s --check-prefix=FUSE
// RUN: %reussir-opt %s -reussir-rc-dispatch-fusion -reussir-inc-dec-cancellation | \
// RUN:   %FileCheck %s --check-prefix=CANCEL
// RUN: %reussir-opt %s -reussir-rc-dispatch-fusion -reussir-rc-decrement-expansion | \
// RUN:   %FileCheck %s --check-prefix=EXPAND

// Destructuring-decrement fusion (Koka's `dropn_reuse` shape). `take` binds
// the cons cell's members and releases the box; `probe` borrows the box via
// an increment that every arm balances with a release.

!nilty = !reussir.record<compound "list.nil" [value] {}>
!consty = !reussir.record<compound "list.cons" [value] {i64, !reussir.record<variant "list" incomplete>}>
!listty = !reussir.record<variant "list" {!consty, !nilty}>
!rclist = !reussir.rc<!listty>
!tk = !reussir.token<align: 8, size: 32>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i64:64-n8:16:32:64-S128"} {
  // The bound retain is fused away and the release learns the arm: on the
  // unique path the tail transfers; on the shared path it is retained.
  // FUSE-LABEL: func.func @take
  // FUSE-NOT: reussir.rc.inc
  // FUSE: reussir.rc.dec(%{{.+}} : {{.+}}) : {{.+}}boundMembers = array<i64: 1>, destructureTag = 0 : index}
  //
  // The expansion releases nothing on the unique path (the only rc member is
  // bound and transfers) and retains the bound tail on the shared path.
  // EXPAND-LABEL: func.func @take
  // EXPAND: scf.if
  // EXPAND-NOT: reussir.ref.drop
  // EXPAND: reussir.rc.reinterpret
  // EXPAND: } else {
  // EXPAND: reussir.rc.set
  // EXPAND: reussir.rc.inc
  // EXPAND: } {reussir.expanded_decrement}
  func.func @take(%l: !rclist) -> !rclist {
    %ref = reussir.rc.borrow (%l : !rclist) : !reussir.ref<!listty>
    %r = reussir.record.dispatch (%ref : !reussir.ref<!listty>) -> !rclist {
      [0] -> {
        ^bb0(%cons: !reussir.ref<!consty>):
        %slot = reussir.ref.project (%cons : !reussir.ref<!consty>) [1] : !reussir.ref<!rclist>
        %tail = reussir.ref.load (%slot : !reussir.ref<!rclist>) : !rclist
        reussir.rc.inc (%tail : !rclist)
        %t = reussir.rc.dec (%l : !rclist) : !reussir.nullable<!tk>
        reussir.scf.yield %tail : !rclist
      }
      [1] -> {
        ^bb1(%nil: !reussir.ref<!nilty>):
        %t = reussir.rc.dec (%l : !rclist) : !reussir.nullable<!tk>
        reussir.scf.yield %l : !rclist
      }
    }
    return %r : !rclist
  }

  // A borrowing probe: the increment cancels against the per-arm releases —
  // including one nested in an inner single-shot branch — and the fused-away
  // bound retain rematerializes so the shared arm still owns its binding.
  // CANCEL-LABEL: func.func @probe
  // CANCEL-NOT: reussir.rc.dec
  // CANCEL: reussir.record.dispatch
  // CANCEL: reussir.rc.inc
  // CANCEL-NOT: reussir.rc.dec
  func.func @probe(%l: !rclist, %c: i1) -> !rclist {
    reussir.rc.inc (%l : !rclist)
    %ref = reussir.rc.borrow (%l : !rclist) : !reussir.ref<!listty>
    %r = reussir.record.dispatch (%ref : !reussir.ref<!listty>) -> !rclist {
      [0] -> {
        ^bb0(%cons: !reussir.ref<!consty>):
        %slot = reussir.ref.project (%cons : !reussir.ref<!consty>) [1] : !reussir.ref<!rclist>
        %tail = reussir.ref.load (%slot : !reussir.ref<!rclist>) : !rclist
        reussir.rc.inc (%tail : !rclist)
        %t = reussir.rc.dec (%l : !rclist) : !reussir.nullable<!tk>
        reussir.scf.yield %tail : !rclist
      }
      [1] -> {
        ^bb1(%nil: !reussir.ref<!nilty>):
        %b = scf.if %c -> i1 {
          %t = reussir.rc.dec (%l : !rclist) : !reussir.nullable<!tk>
          scf.yield %c : i1
        } else {
          %t = reussir.rc.dec (%l : !rclist) : !reussir.nullable<!tk>
          scf.yield %c : i1
        }
        reussir.scf.yield %l : !rclist
      }
    }
    return %r : !rclist
  }
}
