; RUN: %reussir-llvm-opt --passes=reussir-recursion-linearization %s -o - | %FileCheck %s

; Naive doubly-recursive fibonacci must be rewritten into the guarded
; dynamic-programming loop calling the outlined step helper, with no
; self-recursive call left behind.

; CHECK-LABEL: define i64 @fib(i64 %n)
; CHECK:         icmp ult i64 %n, 2
; CHECK:         call i64 @fib.reussir.step(i64 %n, i64 poison, i64 poison)
; CHECK:         call i64 @fib.reussir.step(i64 1, i64 poison, i64 poison)
; CHECK:         call i64 @fib.reussir.step(i64 0, i64 poison, i64 poison)
; CHECK:       reussir.dp.loop:
; CHECK:         %[[IV:.*]] = phi i64 [ 2, %reussir.seed ]
; CHECK:         call i64 @fib.reussir.step(i64 %[[IV]], i64 %reussir.w1, i64 %reussir.w2)
; CHECK-NOT:     call i64 @fib(
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

@sink = global i64 0

define void @escape(i64 %v) {
  store volatile i64 %v, ptr @sink
  ret void
}

; A side-effecting body must stay recursive.

; CHECK-LABEL: define i64 @bad_effect(i64 %n)
; CHECK:         call i64 @bad_effect(
define i64 @bad_effect(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 2
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  call void @escape(i64 %n)
  %n1 = add i64 %n, -1
  %f1 = call i64 @bad_effect(i64 %n1)
  ret i64 %f1
}

; Non-constant descent (divide and conquer) is out of scope and must stay
; recursive.

; CHECK-LABEL: define i64 @bad_halving(i64 %n)
; CHECK:         call i64 @bad_halving(
define i64 @bad_halving(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 2
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %h = lshr i64 %n, 1
  %f1 = call i64 @bad_halving(i64 %h)
  %s = add i64 %f1, 1
  ret i64 %s
}

; Without a dominating comparison bounding the induction argument there is no
; provable recursion floor; the function must stay recursive.

; CHECK-LABEL: define i64 @bad_unguarded(i64 %n)
; CHECK:         call i64 @bad_unguarded(
define i64 @bad_unguarded(i64 %n) {
entry:
  %n1 = add i64 %n, -1
  %f1 = call i64 @bad_unguarded(i64 %n1)
  ret i64 %f1
}

; The outlined helper keeps the base-case dispatch but replaces the recursive
; calls with the window parameters.

; CHECK-LABEL: define private i64 @fib.reussir.step(i64 %n, i64 %prev1, i64 %prev2)
; CHECK-NOT:     call
; CHECK:         add i64 %prev1, %prev2
