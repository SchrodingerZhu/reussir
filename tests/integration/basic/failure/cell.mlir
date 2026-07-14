// RUN: %reussir-opt %s -verify-diagnostics

!cell_i64 = !reussir.cell<i64>
!rc_cell_i64 = !reussir.rc<!cell_i64>

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

  func.func @cell_must_be_shared(%value: i64) {
    // expected-error @+1 {{cell RC pointer must have shared capability, got rigid}}
    %cell = reussir.cell.create value(%value : i64)
      : !reussir.rc<!cell_i64 rigid>
    return
  }

  func.func @rmw_argument_mismatch(%cell: !rc_cell_i64) {
    // expected-error @+1 {{body argument type must match cell element type, expected 'i64', got 'i32'}}
    reussir.cell.rmw(%cell : !rc_cell_i64) {
      ^bb0(%old: i32):
        %zero = arith.constant 0 : i64
        reussir.cell.yield(%zero : i64)
    }
    return
  }
}
