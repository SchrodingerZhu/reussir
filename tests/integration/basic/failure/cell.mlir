// RUN: %reussir-opt %s -verify-diagnostics

!cell_i64 = !reussir.cell<i64>
!rc_cell_i64 = !reussir.rc<!cell_i64>
!excl_cell_i64 = !reussir.cell<i64 exclusive>
!rc_excl_cell_i64 = !reussir.rc<!excl_cell_i64>
!atomic_cell_i64 = !reussir.cell<i64 atomic>
!rc_atomic_cell_i64 = !reussir.rc<!atomic_cell_i64>

module {
  func.func @create_type_mismatch(%value: i32) {
    // expected-error @+1 {{initial value type must match cell element type, got 'i32' and 'i64'}}
    %cell = reussir.cell.create value(%value : i32) : !rc_cell_i64
    return
  }

  func.func @get_type_mismatch(%cell: !rc_cell_i64) {
    // expected-error @+1 {{result type must match cell element type, expected 'i64', got 'i32'}}
    %value = reussir.cell.get(%cell : !rc_cell_i64) : i32
    return
  }

  func.func @set_type_mismatch(%cell: !rc_cell_i64, %value: i32) {
    // expected-error @+1 {{replacement value type must match cell element type, expected 'i64', got 'i32'}}
    reussir.cell.set(%value : i32, %cell : !rc_cell_i64)
    return
  }

  func.func @get_ordering_requires_atomic(%cell: !rc_cell_i64) {
    // expected-error @+1 {{atomic ordering is only valid for an atomic cell}}
    %value = reussir.cell.get(%cell : !rc_cell_i64) ordering(acquire) : i64
    return
  }

  func.func @set_ordering_requires_atomic(%cell: !rc_excl_cell_i64,
                                           %value: i64) {
    // expected-error @+1 {{atomic ordering is only valid for an atomic cell}}
    reussir.cell.set(%value : i64, %cell : !rc_excl_cell_i64) ordering(release)
    return
  }

  func.func @rmw_ordering_requires_atomic(%cell: !rc_excl_cell_i64) {
    // expected-error @+1 {{atomic ordering is only valid for an atomic cell}}
    reussir.cell.rmw(%cell : !rc_excl_cell_i64) ordering(acq_rel) {
      ^bb0(%old: i64):
        reussir.cell.yield(%old : i64)
    }
    return
  }

  func.func @get_rejects_release(%cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{atomic ordering 'release' is invalid for an atomic load}}
    %value = reussir.cell.get(%cell : !rc_atomic_cell_i64) ordering(release) : i64
    return
  }

  func.func @set_rejects_acquire(%cell: !rc_atomic_cell_i64, %value: i64) {
    // expected-error @+1 {{atomic ordering 'acquire' is invalid for an atomic store}}
    reussir.cell.set(%value : i64, %cell : !rc_atomic_cell_i64) ordering(acquire)
    return
  }

  func.func @rmw_rejects_unordered(%value: i64,
                                    %cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{atomic ordering 'unordered' is invalid for an atomic read-modify-write}}
    %old = reussir.cell.rmw addi(%value : i64, %cell : !rc_atomic_cell_i64) ordering(unordered) -> i64
    return
  }

  func.func @cell_must_be_shared(%value: i64) {
    // expected-error @+1 {{cell RC pointer must have shared capability, got rigid}}
    %cell = reussir.cell.create value(%value : i64)
      : !reussir.rc<!cell_i64 rigid>
    return
  }

  func.func @rmw_requires_exclusive(%cell: !rc_cell_i64) {
    // expected-error @+1 {{read-modify-write requires an exclusive or atomic cell, got a plain cell}}
    reussir.cell.rmw(%cell : !rc_cell_i64) {
      ^bb0(%old: i64):
        reussir.cell.yield(%old : i64)
    }
    return
  }

  func.func @rmw_argument_mismatch(%cell: !rc_excl_cell_i64) {
    // expected-error @+1 {{body argument type must match cell element type, expected 'i64', got 'i32'}}
    reussir.cell.rmw(%cell : !rc_excl_cell_i64) {
      ^bb0(%old: i32):
        %zero = arith.constant 0 : i64
        reussir.cell.yield(%zero : i64)
    }
    return
  }

  func.func @direct_rmw_requires_atomic(%value: i64,
                                        %cell: !rc_excl_cell_i64) {
    // expected-error @+1 {{direct atomic RMW form requires an atomic cell, got an exclusive cell}}
    %old = reussir.cell.rmw addi(%value : i64, %cell : !rc_excl_cell_i64) -> i64
    return
  }

  func.func @direct_rmw_kind_must_match(%value: i64,
                                        %cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{floating-point atomic RMW kind requires a floating-point cell element}}
    %old = reussir.cell.rmw addf(%value : i64, %cell : !rc_atomic_cell_i64) -> i64
    return
  }

  func.func @direct_rmw_requires_result(%value: i64,
                                        %cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{direct atomic RMW form must return the old value}}
    reussir.cell.rmw addi(%value : i64, %cell : !rc_atomic_cell_i64)
    return
  }

  func.func private @effectful(i64) -> i64

  func.func @atomic_rmw_body_must_be_pure(%cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{atomic RMW body must be memory-effect-free because it may be retried; operation has effects: func.call}}
    reussir.cell.rmw(%cell : !rc_atomic_cell_i64) {
      ^bb0(%current: i64):
        %next = func.call @effectful(%current) : (i64) -> i64
        reussir.cell.yield(%next : i64)
    }
    return
  }

  func.func @in_use_requires_exclusive(%cell: !rc_cell_i64) {
    // expected-error @+1 {{in-use flag only exists on an exclusive cell, got a plain cell}}
    %flag = reussir.cell.in_use(%cell : !rc_cell_i64) : i1
    return
  }

  func.func @atomic_cell_has_no_in_use_flag(%cell: !rc_atomic_cell_i64) {
    // expected-error @+1 {{in-use flag only exists on an exclusive cell, got an atomic cell}}
    %flag = reussir.cell.in_use(%cell : !rc_atomic_cell_i64) : i1
    return
  }

  func.func @plain_cell_has_no_flag_slot(%ref: !reussir.ref<!cell_i64>) {
    // expected-error @+1 {{cell projection index must be zero, got 1}}
    %flag = reussir.ref.project(%ref : !reussir.ref<!cell_i64>) [1]
      : !reussir.ref<i1 field>
    return
  }

  func.func @flag_slot_type_mismatch(%ref: !reussir.ref<!excl_cell_i64>) {
    // expected-error @+1 {{projected cell slot type mismatch: expected 'i1', got 'i64'}}
    %flag = reussir.ref.project(%ref : !reussir.ref<!excl_cell_i64>) [1]
      : !reussir.ref<i64 field>
    return
  }

  func.func @exclusive_flag_out_of_range(%ref: !reussir.ref<!excl_cell_i64>) {
    // expected-error @+1 {{cell projection index must be zero (element) or one (in-use flag), got 2}}
    %flag = reussir.ref.project(%ref : !reussir.ref<!excl_cell_i64>) [2]
      : !reussir.ref<i1 field>
    return
  }
}
