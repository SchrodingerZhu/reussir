// RUN: %reussir-opt %s --reussir-unique-carrying-recursion-analysis | %FileCheck %s

module {
  // CHECK-LABEL: func.func @driver(
  // CHECK: attributes {reussir.carrying_uniqueness}
  func.func @driver(%rc: !reussir.rc<i64>) -> !reussir.rc<i64> {
    %c2 = arith.constant 2 : i64
    %0 = func.call @loop(%rc, %c2) : (!reussir.rc<i64>, i64) -> !reussir.rc<i64>
    return %0 : !reussir.rc<i64>
  }

  // CHECK-LABEL: func.func @loop(
  // CHECK: attributes {reussir.carrying_uniqueness}
  // CHECK: %[[NEXT:.+]] = reussir.rc.create value(%arg1 : i64) : !reussir.rc<i64>
  // CHECK: %[[RECURSE:.+]] = func.call @loop.unique(%[[NEXT]], %{{.+}}) : (!reussir.rc<i64>, i64) -> !reussir.rc<i64>
  func.func @loop(%rc: !reussir.rc<i64>, %n: i64) -> !reussir.rc<i64> {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %cond = arith.cmpi eq, %n, %c0 : i64
    %result = scf.if %cond -> (!reussir.rc<i64>) {
      scf.yield %rc : !reussir.rc<i64>
    } else {
      %next = reussir.rc.create value(%n : i64) : !reussir.rc<i64>
      %n1 = arith.subi %n, %c1 : i64
      %rec = func.call @loop(%next, %n1) : (!reussir.rc<i64>, i64) -> !reussir.rc<i64>
      scf.yield %rec : !reussir.rc<i64>
    }
    return %result : !reussir.rc<i64>
  }

  // CHECK-LABEL: func.func @branch_loop(
  // CHECK: attributes {reussir.carrying_uniqueness}
  // CHECK: func.call @branch_loop.unique_0(
  // CHECK: func.call @branch_loop.unique_1(
  func.func @branch_loop(%left: !reussir.rc<i64>, %right: !reussir.rc<i64>, %n: i64) -> (!reussir.rc<i64>, !reussir.rc<i64>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %n, %c0 : i64
    %out0, %out1 = scf.if %is_zero -> (!reussir.rc<i64>, !reussir.rc<i64>) {
      scf.yield %left, %right : !reussir.rc<i64>, !reussir.rc<i64>
    } else {
      %is_one = arith.cmpi eq, %n, %c1 : i64
      %branch0, %branch1 = scf.if %is_one -> (!reussir.rc<i64>, !reussir.rc<i64>) {
        %fresh_left = reussir.rc.create value(%n : i64) : !reussir.rc<i64>
        %n1 = arith.subi %n, %c1 : i64
        %rec0, %rec1 = func.call @branch_loop(%fresh_left, %right, %n1) : (!reussir.rc<i64>, !reussir.rc<i64>, i64) -> (!reussir.rc<i64>, !reussir.rc<i64>)
        scf.yield %rec0, %rec1 : !reussir.rc<i64>, !reussir.rc<i64>
      } else {
        %fresh_right = reussir.rc.create value(%n : i64) : !reussir.rc<i64>
        %n1 = arith.subi %n, %c1 : i64
        %rec0, %rec1 = func.call @branch_loop(%left, %fresh_right, %n1) : (!reussir.rc<i64>, !reussir.rc<i64>, i64) -> (!reussir.rc<i64>, !reussir.rc<i64>)
        scf.yield %rec0, %rec1 : !reussir.rc<i64>, !reussir.rc<i64>
      }
      scf.yield %branch0, %branch1 : !reussir.rc<i64>, !reussir.rc<i64>
    }
    return %out0, %out1 : !reussir.rc<i64>, !reussir.rc<i64>
  }

  // CHECK-LABEL: func.func @pair_loop(
  // CHECK: func.call @pair_loop.unique(
  // CHECK-NOT: @pair_loop.unique_
  func.func @pair_loop(%left: !reussir.rc<i64>, %right: !reussir.rc<i64>, %n: i64) -> (!reussir.rc<i64>, !reussir.rc<i64>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %is_zero = arith.cmpi eq, %n, %c0 : i64
    %out0, %out1 = scf.if %is_zero -> (!reussir.rc<i64>, !reussir.rc<i64>) {
      scf.yield %left, %right : !reussir.rc<i64>, !reussir.rc<i64>
    } else {
      %fresh_left = reussir.rc.create value(%n : i64) : !reussir.rc<i64>
      %fresh_right = reussir.rc.create value(%n : i64) : !reussir.rc<i64>
      %n1 = arith.subi %n, %c1 : i64
      %rec0, %rec1 = func.call @pair_loop(%fresh_left, %fresh_right, %n1) : (!reussir.rc<i64>, !reussir.rc<i64>, i64) -> (!reussir.rc<i64>, !reussir.rc<i64>)
      scf.yield %rec0, %rec1 : !reussir.rc<i64>, !reussir.rc<i64>
    }
    return %out0, %out1 : !reussir.rc<i64>, !reussir.rc<i64>
  }

  // CHECK-LABEL: func.func private @loop.unique(
  // CHECK: attributes {llvm.linkage = #llvm.linkage<internal>, reussir.carrying_uniqueness, reussir.unique_clone}
  // CHECK: reussir.rc.assume_unique(%arg0 : !reussir.rc<i64>)
  // CHECK: %[[NEXT2:.+]] = reussir.rc.create value(%arg1 : i64) : !reussir.rc<i64>
  // CHECK: %[[RECURSE2:.+]] = func.call @loop.unique(%[[NEXT2]], %{{.+}}) : (!reussir.rc<i64>, i64) -> !reussir.rc<i64>

  // CHECK-LABEL: func.func private @branch_loop.unique_0(
  // CHECK: reussir.rc.assume_unique(%arg0 : !reussir.rc<i64>)
  // CHECK-NOT: reussir.rc.assume_unique(%arg1 : !reussir.rc<i64>)
  // CHECK: func.call @branch_loop.unique_0(
  // CHECK: func.call @branch_loop.unique_0_1(

  // CHECK-LABEL: func.func private @branch_loop.unique_1(
  // CHECK: reussir.rc.assume_unique(%arg1 : !reussir.rc<i64>)
  // CHECK-NOT: reussir.rc.assume_unique(%arg0 : !reussir.rc<i64>)
  // CHECK: func.call @branch_loop.unique_0_1(
  // CHECK: func.call @branch_loop.unique_1(

  // CHECK-LABEL: func.func private @branch_loop.unique_0_1(
  // CHECK: reussir.rc.assume_unique(%arg0 : !reussir.rc<i64>)
  // CHECK: reussir.rc.assume_unique(%arg1 : !reussir.rc<i64>)
  // CHECK: func.call @branch_loop.unique_0_1(
  // CHECK: func.call @branch_loop.unique_0_1(

  // CHECK-LABEL: func.func private @pair_loop.unique(
  // CHECK: reussir.rc.assume_unique(%arg0 : !reussir.rc<i64>)
  // CHECK: reussir.rc.assume_unique(%arg1 : !reussir.rc<i64>)
  // CHECK: func.call @pair_loop.unique(
}
