// RUN: %reussir-opt %s --reussir-self-tail-call-elimination | %FileCheck %s

// A direct self tail call returning a [value] record — the shape TRMC skips
// (no rc-boxed result). The function body becomes the before-region of an
// scf.while carrying the arguments; the recursive arm yields
// (true, next args, poison) and the finished arm (false, poisons, result).

!out = !reussir.record<compound "Out" [value] {i64, i64}>

// CHECK-LABEL: func.func @spin(
// CHECK-SAME:    %[[N:[a-z0-9]+]]: i64, %[[ACC:[a-z0-9]+]]: i64
// CHECK:  %[[RES:.+]]:3 = scf.while (%[[CN:.+]] = %[[N]], %[[CACC:.+]] = %[[ACC]])
// CHECK:    %[[COND:.+]] = arith.cmpi eq, %[[CN]]
// CHECK:    %[[W:.+]]:4 = scf.if %[[COND]] -> (i1, i64, i64, !reussir.record
// CHECK:      %[[DONE:.+]] = reussir.record.compound
// CHECK:      %[[FALSE:.+]] = arith.constant false
// CHECK:      %[[PN:.+]] = ub.poison : i64
// CHECK:      %[[PACC:.+]] = ub.poison : i64
// CHECK:      scf.yield %[[FALSE]], %[[PN]], %[[PACC]], %[[DONE]]
// CHECK:    } else {
// CHECK:      %[[TRUE:.+]] = arith.constant true
// CHECK:      %[[PR:.+]] = ub.poison : !reussir.record
// CHECK:      scf.yield %[[TRUE]], %{{.+}}, %{{.+}}, %[[PR]]
// CHECK:    }
// CHECK-NOT: func.call @spin
// CHECK:    scf.condition(%[[W]]#0) %[[W]]#1, %[[W]]#2, %[[W]]#3
// CHECK:  } do {
// CHECK:  ^bb0(%[[AN:.+]]: i64, %[[AACC:.+]]: i64, %{{.+}}: !reussir.record
// CHECK:    scf.yield %[[AN]], %[[AACC]]
// CHECK:  }
// CHECK:  return %[[RES]]#2
func.func @spin(%n: i64, %acc: i64) -> !out {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %cond = arith.cmpi eq, %n, %zero : i64
  %result = scf.if %cond -> (!out) {
    %done = reussir.record.compound(%acc, %acc : i64, i64) : !out
    scf.yield %done : !out
  } else {
    %n1 = arith.subi %n, %one : i64
    %acc1 = arith.addi %acc, %n : i64
    %next = func.call @spin(%n1, %acc1) : (i64, i64) -> !out
    scf.yield %next : !out
  }
  return %result : !out
}

// A function whose recursion is not in tail position must be left alone.
// CHECK-LABEL: func.func @not_tail(
// CHECK-NOT: scf.while
// CHECK: %[[R:.+]] = func.call @not_tail
// CHECK: arith.addi %[[R]],
func.func @not_tail(%n: i64) -> i64 {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %cond = arith.cmpi eq, %n, %zero : i64
  %result = scf.if %cond -> (i64) {
    scf.yield %zero : i64
  } else {
    %n1 = arith.subi %n, %one : i64
    %r = func.call @not_tail(%n1) : (i64) -> i64
    %sum = arith.addi %r, %one : i64
    scf.yield %sum : i64
  }
  return %result : i64
}
