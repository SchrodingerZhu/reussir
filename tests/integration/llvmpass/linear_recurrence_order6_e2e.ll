; RUN: %reussir-llvm-opt --linear-recurrence-pipeline=O2 %s -o %t.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %lli %t.ll

; Order-6 recurrence with dense coefficients and an inhomogeneous constant:
; state size 6 is exactly the Kitamasa cap (kMaxStateSize), and the
; augmentation row brings the companion dimension to the maximal D = 7.
;   f(n) = n+1                                            for n < 6
;   f(n) = 2f(n-1) + 3f(n-2) + 5f(n-3) + 7f(n-4)
;        + 11f(n-5) + 13f(n-6) + 17                       for n >= 6
; sext(10^13) is unreachable for the exponential recursion and out of
; timeout range for a linear loop; only the O(log n) path answers.

; CHECK-LABEL: define{{.*}} i64 @sext(i64 %n)
; CHECK-NOT:     call
; CHECK:         reussir.matexp
; CHECK:         lshr
define i64 @sext(i64 %n) {
entry:
  %small = icmp ult i64 %n, 6
  br i1 %small, label %base, label %rec
base:
  %succ = add i64 %n, 1
  ret i64 %succ
rec:
  %n1 = add i64 %n, -1
  %f1 = call i64 @sext(i64 %n1)
  %t1 = shl i64 %f1, 1
  %n2 = add i64 %n, -2
  %f2 = call i64 @sext(i64 %n2)
  %t2 = mul i64 %f2, 3
  %n3 = add i64 %n, -3
  %f3 = call i64 @sext(i64 %n3)
  %t3 = mul i64 %f3, 5
  %n4 = add i64 %n, -4
  %f4 = call i64 @sext(i64 %n4)
  %t4 = mul i64 %f4, 7
  %n5 = add i64 %n, -5
  %f5 = call i64 @sext(i64 %n5)
  %t5 = mul i64 %f5, 11
  %n6 = add i64 %n, -6
  %f6 = call i64 @sext(i64 %n6)
  %t6 = mul i64 %f6, 13
  %s1 = add i64 %t1, %t2
  %s2 = add i64 %s1, %t3
  %s3 = add i64 %s2, %t4
  %s4 = add i64 %s3, %t5
  %s5 = add i64 %s4, %t6
  %s = add i64 %s5, 17
  ret i64 %s
}

define i32 @check(i64 %got, i64 %want, i32 %code) {
  %ok = icmp eq i64 %got, %want
  %r = select i1 %ok, i32 0, i32 %code
  ret i32 %r
}

define i32 @main() {
  %f0 = call i64 @sext(i64 0)
  %c0 = call i32 @check(i64 %f0, i64 1, i32 1)
  %f5 = call i64 @sext(i64 5)
  %c1 = call i32 @check(i64 %f5, i64 6, i32 2)
  %f6 = call i64 @sext(i64 6)
  %c2 = call i32 @check(i64 %f6, i64 120, i32 3)
  %f7 = call i64 @sext(i64 7)
  %c3 = call i32 @check(i64 %f7, i64 387, i32 4)
  %f8 = call i64 @sext(i64 8)
  %c4 = call i32 @check(i64 %f8, i64 1299, i32 5)
  %f2000 = call i64 @sext(i64 2000)
  %c5 = call i32 @check(i64 %f2000, i64 10227571276236485390, i32 6)
  %fhuge = call i64 @sext(i64 10000000000000)
  %c6 = call i32 @check(i64 %fhuge, i64 5994342898092240063, i32 7)
  %a0 = or i32 %c0, %c1
  %a1 = or i32 %a0, %c2
  %a2 = or i32 %a1, %c3
  %a3 = or i32 %a2, %c4
  %a4 = or i32 %a3, %c5
  %a5 = or i32 %a4, %c6
  ret i32 %a5
}
