; RUN: %reussir-llvm-opt --passes='function(loop-simplify,reussir-linear-recurrence-matexp)' %s -o - | %FileCheck %s

; A rotated fib loop (i32 counter, i64 data) must be strength-reduced into a
; square-and-multiply exponentiation loop over the exponent's bits. The i32
; exit counter is not part of the affine state and simply dies with the loop.

; CHECK-LABEL: define i64 @fib_loop(i64 %n)
; CHECK:       reussir.matexp.check:
; CHECK:         %reussir.matexp.e = phi i32
; CHECK:         phi i64
; CHECK:       reussir.matexp.body:
; CHECK:         trunc i32 %reussir.matexp.e to i1
; CHECK:         mul i64
; CHECK:         lshr i32 %reussir.matexp.e, 1
; CHECK:       reussir.matexp.done:
define i64 @fib_loop(i64 %n) {
entry:
  %small = icmp ult i64 %n, 2
  br i1 %small, label %ret_n, label %ph
ret_n:
  ret i64 %n
ph:
  %count = trunc i64 %n to i32
  br label %loop
loop:
  %i = phi i32 [ 2, %ph ], [ %inext, %loop ]
  %a = phi i64 [ 0, %ph ], [ %b, %loop ]
  %b = phi i64 [ 1, %ph ], [ %next, %loop ]
  %next = add i64 %a, %b
  %inext = add i32 %i, 1
  %done = icmp eq i32 %i, %count
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %next
}

; A linear congruential generator: one state cell plus the augmentation row
; carrying the additive constant.

; CHECK-LABEL: define i64 @lcg(i64 %seed, i64 %rounds)
; CHECK:       reussir.matexp.check:
; CHECK:       reussir.matexp.done:
define i64 @lcg(i64 %seed, i64 %rounds) {
entry:
  %zero = icmp eq i64 %rounds, 0
  br i1 %zero, label %ret0, label %ph
ret0:
  ret i64 %seed
ph:
  br label %loop
loop:
  %i = phi i64 [ 0, %ph ], [ %inext, %loop ]
  %x = phi i64 [ %seed, %ph ], [ %xnext, %loop ]
  %scaled = mul i64 %x, 6364136223846793005
  %xnext = add i64 %scaled, 1442695040888963407
  %inext = add i64 %i, 1
  %done = icmp eq i64 %inext, %rounds
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %xnext
}

; A quadratic update is not an affine map; the loop must remain untouched.

; CHECK-LABEL: define i64 @quad(i64 %n)
; CHECK-NOT:   reussir.matexp
; CHECK:         %xnext = mul i64 %x, %x
define i64 @quad(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %x = phi i64 [ 3, %entry ], [ %xnext, %loop ]
  %xnext = mul i64 %x, %x
  %inext = add i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %xnext
}
