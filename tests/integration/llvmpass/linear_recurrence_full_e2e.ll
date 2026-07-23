; RUN: %reussir-llvm-opt --linear-recurrence-pipeline=O2 %s -o %t.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %lli %t.ll

; The full chain with the exact production pipeline: recursion linearization
; at pipeline start, Kitamasa exponentiation at the vectorizer-start
; extension point. The naive doubly-recursive fibonacci is asked for
; fib(10^15) mod 2^64 — unreachable both for the exponential recursion
; (~2^(10^15) calls) and for a linear-time loop (~10^15 iterations); only
; the O(log n) exponentiation path can answer within the test timeout.

; CHECK-LABEL: define{{.*}} i64 @fib(i64 %n)
; CHECK-NOT:     call
; CHECK:         lshr
define i64 @fib(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 2
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %n1 = add i64 %n, -1
  %f1 = call i64 @fib(i64 %n1)
  %n2 = add i64 %n, -2
  %f2 = call i64 @fib(i64 %n2)
  %s = add i64 %f1, %f2
  ret i64 %s
}

define i32 @main() {
  %r = call i64 @fib(i64 1000000000000000)
  %ok = icmp eq i64 %r, 17845758328335168059
  %ret = select i1 %ok, i32 0, i32 1
  ret i32 %ret
}
