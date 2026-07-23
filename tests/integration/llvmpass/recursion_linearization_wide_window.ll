; RUN: %reussir-llvm-opt --passes=reussir-recursion-linearization %s -o %t.lin.ll
; RUN: %FileCheck %s --check-prefix=LIN < %t.lin.ll
; RUN: %lli %t.lin.ll
; RUN: %reussir-llvm-opt --linear-recurrence-pipeline=O2 %s -o %t.prod.ll
; RUN: %FileCheck %s --check-prefix=PROD < %t.prod.ll
; RUN: %lli %t.prod.ll

; Medium-width window (order 8) with three heterogeneous base-case branches
; behind mixed eq/ult guards, each returning a different expression:
;   f(0) = 7 ; f(1) = 3 ; f(n) = 2n+1 for n < 8
;   f(n) = f(n-1) + 2f(n-2) + 3f(n-3) + f(n-4) + 2f(n-5) + 3f(n-6)
;        + f(n-7) + 2f(n-8) + 5                 for n >= 8
;
; The recursion floor analysis must land on C = 8 from the guard chain, and
; linearization must carry the full 8-wide sliding window. Through the
; production pipeline the state (8 PHIs) exceeds the Kitamasa cap
; (kMaxStateSize = 6), so the loop must stay a linear scalar (or
; SLP-vectorized) loop — no exponentiation kernel may be emitted for it.

; The linearized form must carry the full 8-wide window.
; LIN-LABEL: define i64 @wide(i64 %n)
; LIN-NOT:     call i64 @wide(
; LIN:         %reussir.w8 = phi i64
; LIN:         call i64 @wide.reussir.step(

; PROD-LABEL: define{{.*}} i64 @wide(i64 %n)
; PROD-NOT:     call
; PROD-NOT:     reussir.matexp
define i64 @wide(i64 %n) {
entry:
  %is0 = icmp eq i64 %n, 0
  br i1 %is0, label %ret0, label %chk1
ret0:
  ret i64 7
chk1:
  %is1 = icmp eq i64 %n, 1
  br i1 %is1, label %ret1, label %chksmall
ret1:
  ret i64 3
chksmall:
  %small = icmp ult i64 %n, 8
  br i1 %small, label %retsmall, label %rec
retsmall:
  %dbl = shl i64 %n, 1
  %odd = add i64 %dbl, 1
  ret i64 %odd
rec:
  %n1 = add i64 %n, -1
  %f1 = call i64 @wide(i64 %n1)
  %n2 = add i64 %n, -2
  %f2 = call i64 @wide(i64 %n2)
  %t2 = shl i64 %f2, 1
  %n3 = add i64 %n, -3
  %f3 = call i64 @wide(i64 %n3)
  %t3 = mul i64 %f3, 3
  %n4 = add i64 %n, -4
  %f4 = call i64 @wide(i64 %n4)
  %n5 = add i64 %n, -5
  %f5 = call i64 @wide(i64 %n5)
  %t5 = shl i64 %f5, 1
  %n6 = add i64 %n, -6
  %f6 = call i64 @wide(i64 %n6)
  %t6 = mul i64 %f6, 3
  %n7 = add i64 %n, -7
  %f7 = call i64 @wide(i64 %n7)
  %n8 = add i64 %n, -8
  %f8 = call i64 @wide(i64 %n8)
  %t8 = shl i64 %f8, 1
  %s1 = add i64 %f1, %t2
  %s2 = add i64 %s1, %t3
  %s3 = add i64 %s2, %f4
  %s4 = add i64 %s3, %t5
  %s5 = add i64 %s4, %t6
  %s6 = add i64 %s5, %f7
  %s7 = add i64 %s6, %t8
  %s = add i64 %s7, 5
  ret i64 %s
}

; Descending by more than the window cap (kMaxWindow = 16) must not be
; linearized by the pass. (The production pipeline's own tail-recursion
; elimination may still turn this accumulator-free single call into a loop —
; that is plain LLVM behavior, so the check is on the pass-only output.)

; LIN-LABEL: define i64 @bad_beyond_window(i64 %n)
; LIN:         call i64 @bad_beyond_window(
define i64 @bad_beyond_window(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 17
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %n17 = add i64 %n, -17
  %f = call i64 @bad_beyond_window(i64 %n17)
  %s = add i64 %f, 1
  ret i64 %s
}

define i32 @check(i64 %got, i64 %want, i32 %code) {
  %ok = icmp eq i64 %got, %want
  %r = select i1 %ok, i32 0, i32 %code
  ret i32 %r
}

define i32 @main() {
  %w0 = call i64 @wide(i64 0)
  %c0 = call i32 @check(i64 %w0, i64 7, i32 1)
  %w1 = call i64 @wide(i64 1)
  %c1 = call i32 @check(i64 %w1, i64 3, i32 2)
  %w5 = call i64 @wide(i64 5)
  %c2 = call i32 @check(i64 %w5, i64 11, i32 3)
  %w7 = call i64 @wide(i64 7)
  %c3 = call i32 @check(i64 %w7, i64 15, i32 4)
  %w8 = call i64 @wide(i64 8)
  %c4 = call i32 @check(i64 %w8, i64 134, i32 5)
  %w9 = call i64 @wide(i64 9)
  %c5 = call i32 @check(i64 %w9, i64 269, i32 6)
  %wbig = call i64 @wide(i64 100000)
  %c6 = call i32 @check(i64 %wbig, i64 16362317766162041378, i32 7)
  %b = call i64 @bad_beyond_window(i64 40)
  %c7 = call i32 @check(i64 %b, i64 8, i32 8)
  %a0 = or i32 %c0, %c1
  %a1 = or i32 %a0, %c2
  %a2 = or i32 %a1, %c3
  %a3 = or i32 %a2, %c4
  %a4 = or i32 %a3, %c5
  %a5 = or i32 %a4, %c6
  %a6 = or i32 %a5, %c7
  ret i32 %a6
}
