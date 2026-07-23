; RUN: %reussir-llvm-opt --linear-recurrence-pipeline=O2 %s -o %t.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %lli %t.ll

; Wider-window regression test (order 4, dense coefficients, inhomogeneous
; constant; companion dimension D = 5 with the augmentation row). Placement
; matters here: run *after* the whole default pipeline, the rewrite never
; fires because SLP vectorizes the k=4 sliding window into vector PHIs the
; affine matcher cannot see. At the vectorizer-start extension point the
; loop is still scalar, and EarlyCSE right after the rewrite folds the
; schoolbook squaring's commutative duplicate products.
;
; quart(10^14) mod 2^64 is unreachable for both the exponential recursion
; and the linear loop; only the O(log n) path answers within the timeout.

; CHECK-LABEL: define{{.*}} i64 @quart(i64 %n)
; CHECK-NOT:     call
; CHECK:         reussir.matexp
; CHECK:         lshr
define i64 @quart(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 4
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %n1 = add i64 %n, -1
  %f1 = call i64 @quart(i64 %n1)
  %t1 = mul i64 %f1, 3
  %n2 = add i64 %n, -2
  %f2 = call i64 @quart(i64 %n2)
  %t2 = mul i64 %f2, 5
  %n3 = add i64 %n, -3
  %f3 = call i64 @quart(i64 %n3)
  %t3 = mul i64 %f3, 7
  %n4 = add i64 %n, -4
  %f4 = call i64 @quart(i64 %n4)
  %t4 = mul i64 %f4, 11
  %s1 = add i64 %t1, %t2
  %s2 = add i64 %s1, %t3
  %s3 = add i64 %s2, %t4
  %s = add i64 %s3, 13
  ret i64 %s
}

define i32 @check(i64 %got, i64 %want, i32 %code) {
  %ok = icmp eq i64 %got, %want
  %r = select i1 %ok, i32 0, i32 %code
  ret i32 %r
}

define i32 @main() {
  %q3 = call i64 @quart(i64 3)
  %c0 = call i32 @check(i64 %q3, i64 3, i32 1)
  %q4 = call i64 @quart(i64 4)
  %c1 = call i32 @check(i64 %q4, i64 39, i32 2)
  %q5 = call i64 @quart(i64 5)
  %c2 = call i32 @check(i64 %q5, i64 170, i32 3)
  %q100k = call i64 @quart(i64 100000)
  %c3 = call i32 @check(i64 %q100k, i64 14715427454508611264, i32 4)
  %qhuge = call i64 @quart(i64 100000000000000)
  %c4 = call i32 @check(i64 %qhuge, i64 735566407347175424, i32 5)
  %a = or i32 %c0, %c1
  %b = or i32 %a, %c2
  %c = or i32 %b, %c3
  %d = or i32 %c, %c4
  ret i32 %d
}
