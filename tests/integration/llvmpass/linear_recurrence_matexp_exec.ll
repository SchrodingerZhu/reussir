; RUN: %lli %s
; RUN: %reussir-llvm-opt --passes='function(loop-simplify,reussir-linear-recurrence-matexp)' %s -o %t.ll
; RUN: %lli %t.ll

; The same module must compute identical results before and after the
; matrix-exponentiation rewrite (values fixed mod 2^64; distinct exit codes
; identify the failing case). `main`'s own control flow contains calls, so it
; is never a candidate itself.

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

define i32 @check(i64 %got, i64 %want, i32 %code) {
  %ok = icmp eq i64 %got, %want
  %r = select i1 %ok, i32 0, i32 %code
  ret i32 %r
}

define i32 @main() {
  %f2 = call i64 @fib_loop(i64 2)
  %c0 = call i32 @check(i64 %f2, i64 1, i32 1)
  %f3 = call i64 @fib_loop(i64 3)
  %c1 = call i32 @check(i64 %f3, i64 2, i32 2)
  %f90 = call i64 @fib_loop(i64 90)
  %c2 = call i32 @check(i64 %f90, i64 2880067194370816120, i32 3)
  %f1000 = call i64 @fib_loop(i64 1000)
  %c3 = call i32 @check(i64 %f1000, i64 817770325994397771, i32 4)
  %l = call i64 @lcg(i64 42, i64 1000000)
  %c4 = call i32 @check(i64 %l, i64 16854984035281278314, i32 5)
  %l0 = call i64 @lcg(i64 7, i64 0)
  %c5 = call i32 @check(i64 %l0, i64 7, i32 6)
  %q = call i64 @quad(i64 3)
  %c6 = call i32 @check(i64 %q, i64 6561, i32 7)
  %a = or i32 %c0, %c1
  %b = or i32 %a, %c2
  %c = or i32 %b, %c3
  %d = or i32 %c, %c4
  %e = or i32 %d, %c5
  %f = or i32 %e, %c6
  ret i32 %f
}
