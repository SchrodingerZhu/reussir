; RUN: %reussir-llvm-opt --passes=reussir-recursion-linearization %s -o %t.ll
; RUN: %lli %t.ll

; End-to-end execution of linearized recurrences. Every `check` returns a
; distinct nonzero exit code on mismatch, so a failure identifies the broken
; case. Large arguments (fib(200), gen(40)) would be unreachable for the
; original exponential recursions; expected values are computed mod 2^64.

; Naive doubly-recursive fib (unsigned guard).
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

; Signed-guard variant: f(n) = n for n < 2, including negative inputs.
define i64 @fib_signed(i64 %n) {
entry:
  %cmp = icmp slt i64 %n, 2
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %n1 = sub i64 %n, 1
  %f1 = call i64 @fib_signed(i64 %n1)
  %n2 = sub i64 %n, 2
  %f2 = call i64 @fib_signed(i64 %n2)
  %s = add i64 %f1, %f2
  ret i64 %s
}

; Order-3 recurrence with coefficients and an inhomogeneous constant:
; g(n) = 3 g(n-1) + 2 g(n-3) + 7, g(0)=1, g(1)=2, g(2)=3.
define i64 @gen(i64 %n) {
entry:
  %c0 = icmp eq i64 %n, 0
  br i1 %c0, label %r0, label %n0
r0:
  ret i64 1
n0:
  %c1 = icmp eq i64 %n, 1
  br i1 %c1, label %r1, label %n1b
r1:
  ret i64 2
n1b:
  %c2 = icmp eq i64 %n, 2
  br i1 %c2, label %r2, label %rec
r2:
  ret i64 3
rec:
  %m1 = add i64 %n, -1
  %g1 = call i64 @gen(i64 %m1)
  %t1 = mul i64 %g1, 3
  %m3 = add i64 %n, -3
  %g3 = call i64 @gen(i64 %m3)
  %t3 = shl i64 %g3, 1
  %s0 = add i64 %t1, %t3
  %s = add i64 %s0, 7
  ret i64 %s
}

; Generalized (nonlinear) combine — a max-plus recurrence:
; h(n) = max(h(n-1)+2, h(n-2)+3), h(0)=0, h(1)=1.
define i64 @maxplus(i64 %n) {
entry:
  %cmp = icmp ult i64 %n, 2
  br i1 %cmp, label %base, label %rec
base:
  ret i64 %n
rec:
  %n1 = add i64 %n, -1
  %h1 = call i64 @maxplus(i64 %n1)
  %a = add i64 %h1, 2
  %n2 = add i64 %n, -2
  %h2 = call i64 @maxplus(i64 %n2)
  %b = add i64 %h2, 3
  %cmp2 = icmp sgt i64 %a, %b
  %m = select i1 %cmp2, i64 %a, i64 %b
  ret i64 %m
}

define i32 @check(i64 %got, i64 %want, i32 %code) {
  %ok = icmp eq i64 %got, %want
  %r = select i1 %ok, i32 0, i32 %code
  ret i32 %r
}

define i32 @main() {
  %f0 = call i64 @fib(i64 0)
  %c0 = call i32 @check(i64 %f0, i64 0, i32 1)
  %f1 = call i64 @fib(i64 1)
  %c1 = call i32 @check(i64 %f1, i64 1, i32 2)
  %f2 = call i64 @fib(i64 2)
  %c2 = call i32 @check(i64 %f2, i64 1, i32 3)
  %f3 = call i64 @fib(i64 3)
  %c3 = call i32 @check(i64 %f3, i64 2, i32 4)
  %f90 = call i64 @fib(i64 90)
  %c4 = call i32 @check(i64 %f90, i64 2880067194370816120, i32 5)
  %f100 = call i64 @fib(i64 100)
  %c5 = call i32 @check(i64 %f100, i64 3736710778780434371, i32 6)
  %f200 = call i64 @fib(i64 200)
  %c6 = call i32 @check(i64 %f200, i64 17323038258947941269, i32 7)
  %fs = call i64 @fib_signed(i64 -5)
  %c7 = call i32 @check(i64 %fs, i64 -5, i32 8)
  %fs30 = call i64 @fib_signed(i64 30)
  %c8 = call i32 @check(i64 %fs30, i64 832040, i32 9)
  %g2 = call i64 @gen(i64 2)
  %c9 = call i32 @check(i64 %g2, i64 3, i32 10)
  %g3 = call i64 @gen(i64 3)
  %c10 = call i32 @check(i64 %g3, i64 18, i32 11)
  %g40 = call i64 @gen(i64 40)
  %c11 = call i32 @check(i64 %g40, i64 3616619083927076757, i32 12)
  %h1 = call i64 @maxplus(i64 1)
  %c12 = call i32 @check(i64 %h1, i64 1, i32 13)
  %h10 = call i64 @maxplus(i64 10)
  %c13 = call i32 @check(i64 %h10, i64 19, i32 14)
  %a = or i32 %c0, %c1
  %b = or i32 %a, %c2
  %c = or i32 %b, %c3
  %d = or i32 %c, %c4
  %e = or i32 %d, %c5
  %f = or i32 %e, %c6
  %g = or i32 %f, %c7
  %h = or i32 %g, %c8
  %i = or i32 %h, %c9
  %j = or i32 %i, %c10
  %k = or i32 %j, %c11
  %l = or i32 %k, %c12
  %m = or i32 %l, %c13
  ret i32 %m
}
