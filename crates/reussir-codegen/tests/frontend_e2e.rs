//! End-to-end frontend execution: Reussir source → elaborate → monomorphize →
//! lower to MLIR → run through the lowering pipeline → JIT-execute.
//!
//! This exercises the whole spine for the scalar/control-flow subset: the
//! result of calling the compiled function through the C ABI must match the
//! program's meaning.

use reussir_backend::pipeline::{LoweringOptions, run_lowering_pipeline};
use reussir_codegen::lower::lower_program;
use reussir_core::full::mono::monomorphize;
use reussir_core::{in_arena, semi::elaborate, surface};
use reussir_jit::{OptLevel, OrcJit};

// Share the crate's test helpers (a tracing-enabled backend context) by
// including the same source the unit tests use.
#[path = "../src/testing.rs"]
mod testing;

/// Compile `source` to a lowered LLVM-dialect module, then JIT it and hand the
/// engine to `run` to look up and call entry points.
fn jit_run<R>(source: &str, run: impl FnOnce(&OrcJit) -> R) -> R {
    let context = testing::context();
    // Frontend + lowering, inside the arena scope; the module outlives it.
    let mut module = in_arena(|tcx| {
        let parse = reussir_syntax::parse(source);
        assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
        let prog = surface::program(&parse.root);
        let elab = elaborate(tcx, &prog, parse.resolver());
        assert!(
            !elab.has_errors(),
            "elaboration errors: {:#?}",
            elab.reports
        );
        let (full, reports) = monomorphize(&elab.mono_input());
        assert!(reports.is_empty(), "mono reports: {reports:#?}");
        lower_program(&context, tcx, &full, None, None).expect("scalar lowering succeeds")
    });

    run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
        .expect("pipeline lowers to LLVM");

    let jit = OrcJit::with_runtime().expect("create JIT");
    jit.add_module(&module, OptLevel::Default)
        .expect("add lowered module");
    run(&jit)
}

#[test]
fn runs_recursive_fibonacci() {
    let src = r#"
        pub fn fibonacci(n: u64) -> u64 {
            if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
        }
        extern "C" trampoline "fibonacci_ffi" = fibonacci;
    "#;
    jit_run(src, |jit| {
        // Call through the exported C-ABI trampoline, not the internal symbol.
        let addr = jit.lookup("fibonacci_ffi").expect("lookup fibonacci_ffi");
        let fib: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(addr as usize) };
        // 0,1,1,2,3,5,8,13,21,34,55
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(10), 55);
        assert_eq!(fib(20), 6765);
    });
}

#[test]
fn runs_value_record_construction_and_projection() {
    // `[value]` records are by-value aggregates: no heap, no rc. Build a point,
    // pass it by value across internal calls, and project its fields —
    // exercising `reussir.record.compound` / `record.extract` and the
    // record-by-value function ABI end-to-end. The records stay internal; only
    // two scalars cross the trampoline (the Reussir C ABI packs ≥4 args behind a
    // pointer, an orthogonal concern, so we keep the entry point at two args).
    let src = r#"
        struct [value] Point { x: i64, y: i64 }
        fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }
        fn dot(a: Point, b: Point) -> i64 { a.x * b.x + a.y * b.y }
        pub fn sq_norm(x: i64, y: i64) -> i64 { dot(mk(x, y), mk(x, y)) }
        extern "C" trampoline "sq_norm_ffi" = sq_norm;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("sq_norm_ffi").expect("lookup sq_norm_ffi");
        let sq_norm: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // |(3,4)|² = 3*3 + 4*4 = 25
        assert_eq!(sq_norm(3, 4), 25);
        // |(5,6)|² = 25 + 36 = 61
        assert_eq!(sq_norm(5, 6), 61);
    });
}

#[test]
fn runs_shared_record_construction_and_projection() {
    // `[shared]` records are heap-allocated and reference-counted. `Outer` nests a
    // shared `Inner`, so the whole spine — allocate (`rc.create`), borrow + project
    // + load a field, retain a borrowed `rc` field (`rc.inc`), and release consumed
    // boxes (`rc.dec`) — runs through the runtime allocator. The records stay
    // internal; only a scalar crosses the trampoline.
    let src = r#"
        struct [shared] Inner { n: i64 }
        struct [shared] Outer { inner: Inner }
        fn mk_inner(n: i64) -> Inner { Inner { n: n } }
        fn mk_outer(i: Inner) -> Outer { Outer { inner: i } }
        fn inner_of(o: Outer) -> Inner { o.inner }
        fn n_of(i: Inner) -> i64 { i.n }
        pub fn build_and_read(n: i64) -> i64 { n_of(inner_of(mk_outer(mk_inner(n)))) }
        extern "C" trampoline "build_and_read_ffi" = build_and_read;
    "#;
    jit_run(src, |jit| {
        let a = jit
            .lookup("build_and_read_ffi")
            .expect("lookup build_and_read_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(f(5), 5);
        assert_eq!(f(42), 42);
    });
}

#[test]
fn runs_shared_record_deep_projection() {
    // A single chained field access `o.inner.n` crosses two shared records in one
    // projection: borrow the outer box, project + load the inner `rc` link, borrow
    // *that*, project + load the scalar. Exercises the multi-step projection walk
    // (path length > 1) and that the transient inner borrow is neither retained nor
    // double-freed (the consumed outer box is released once).
    let src = r#"
        struct [shared] Inner { n: i64 }
        struct [shared] Outer { inner: Inner }
        fn mk_inner(n: i64) -> Inner { Inner { n: n } }
        fn mk_outer(n: i64) -> Outer { Outer { inner: mk_inner(n) } }
        fn deep(o: Outer) -> i64 { o.inner.n }
        pub fn run(n: i64) -> i64 { deep(mk_outer(n)) }
        extern "C" trampoline "deep_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("deep_ffi").expect("lookup deep_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(f(9), 9);
        assert_eq!(f(-3), -3);
    });
}

#[test]
fn runs_shared_record_with_aliasing_and_release() {
    // Passing the same shared box to both parameters of `sum` aliases it, so the
    // ownership analysis inserts one `rc.inc` before the call; `sum` then releases
    // each parameter, so the box is freed exactly once. A wrong reference count
    // would double-free (crash) or leak — the arithmetic result pins correctness.
    let src = r#"
        struct [shared] Box { v: i64 }
        fn mk(v: i64) -> Box { Box { v: v } }
        fn sum(a: Box, b: Box) -> i64 { a.v + b.v }
        fn use_twice(b: Box) -> i64 { sum(b, b) }
        pub fn run(n: i64) -> i64 { use_twice(mk(n)) }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // use_twice(mk(7)) = 7 + 7 = 14
        assert_eq!(f(7), 14);
        assert_eq!(f(21), 42);
    });
}

#[test]
fn runs_shared_variant_construction_and_drop() {
    // Build a two-node shared (managed) list, then drop it unused and return the
    // scalar. This drives the whole variant-construction spine through the runtime:
    // each case packs its payload (`record.compound`), tags it (`record.variant`),
    // and boxes it (`rc.create`); the recursive `Cons` field is an `rc` link back
    // to the enum; and the unused list is released (`rc.dec`), so the tag-dispatched
    // drop runs over the heap-allocated nodes. A wrong refcount or a bad variant
    // layout would crash or leak — returning `n` intact pins it down.
    let src = r#"
        enum IntList { Nil, Cons(i64, IntList) }
        fn build(n: i64) -> i64 { let l = IntList::Cons{n, IntList::Cons{n, IntList::Nil}}; n }
        pub fn run(n: i64) -> i64 { build(n) }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(f(7), 7);
        assert_eq!(f(-13), -13);
    });
}

#[test]
fn lowers_variant_debug_info_through_the_pipeline() {
    // Lower a value enum and a recursive shared enum *with debug info on*, then
    // run the whole lowering pipeline and JIT-execute. This drives the backend's
    // debug-info conversion over variant types — the tag-plus-union composite and
    // the boxed-member pointer past the rc header — which the no-debug `jit_run`
    // path never reaches; a malformed variant `DICompositeType` would fail the
    // pipeline or the JIT here.
    let src = r#"
        enum [value] Shape { Dot, Circle(i64), Rect(i64, i64) }
        enum List { Nil, Cons(i64, List) }
        fn describe(s: Shape) -> i64 { let k = 0; k }
        fn walk(l: List) -> i64 { let z = 0; z }
        pub fn run(n: i64) -> i64 {
            let r = describe(Shape::Rect{n, n + 1});
            let w = walk(List::Cons{n, List::Nil});
            n + r + w
        }
        extern "C" trampoline "run_dbg_ffi" = run;
    "#;
    let context = testing::context();
    let mut module = in_arena(|tcx| {
        let parse = reussir_syntax::parse(src);
        assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
        let prog = surface::program(&parse.root);
        let elab = elaborate(tcx, &prog, parse.resolver());
        assert!(
            !elab.has_errors(),
            "elaboration errors: {:#?}",
            elab.reports
        );
        let (full, reports) = monomorphize(&elab.mono_input());
        assert!(reports.is_empty(), "mono reports: {reports:#?}");
        let path = std::path::Path::new("variant.rr");
        let map = reussir_codegen::source::SourceMap::new(path, src);
        lower_program(&context, tcx, &full, Some(&map), Some(parse.resolver()))
            .expect("variant lowering with debug info succeeds")
    });

    run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
        .expect("pipeline lowers variant debug info to LLVM");

    // The recursive `List` must lower to a DWARF type *cycle*: a `rec-self`
    // placeholder plus the named composite that carries the matching recursion
    // id. That cycle is what lets a debugger descend a list node by node;
    // without it the recursive field's pointee would be an empty forward
    // declaration. Tie the placeholder to its anchor by the shared recId.
    let text = module.as_operation().to_string();
    let self_at = text.find("isRecSelf = true").unwrap_or_else(|| {
        panic!("recursive variant must emit a rec-self DI placeholder:\n{text}")
    });
    let id_at = text[..self_at]
        .rfind("distinct[")
        .expect("rec-self carries a recId");
    let id_end = id_at + text[id_at..].find("]<>").expect("malformed recId") + "]<>".len();
    let rec_id = &text[id_at..id_end];
    let anchor = format!("di_composite_type<recId = {rec_id}, tag = DW_TAG_structure_type, name =");
    assert!(
        text.contains(&anchor),
        "rec-self id {rec_id} must anchor a named recursive composite:\n{text}"
    );

    let jit = OrcJit::with_runtime().expect("create JIT");
    jit.add_module(&module, OptLevel::Default)
        .expect("add lowered module");
    let a = jit.lookup("run_dbg_ffi").expect("lookup run_dbg_ffi");
    let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
    assert_eq!(f(5), 5);
}

#[test]
fn runs_a_closure_with_a_captured_value() {
    // A closure captures the enclosing `n` and takes one parameter. It lowers to
    // an inline `closure.create` over `(n, x) -> i64`; the capture is supplied
    // with `closure.apply`, and the call uniqifies, applies the argument, and
    // evaluates. If the whole create/apply/eval protocol and the closure-box
    // layout are right, the arithmetic comes back exact.
    let src = r#"
        pub fn adder(n: i64) -> i64 { (|x: i64| x + n)(1) }
        extern "C" trampoline "adder_ffi" = adder;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("adder_ffi").expect("lookup adder_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // (|x| x + n)(1) = 1 + n
        assert_eq!(f(41), 42);
        assert_eq!(f(-1), 0);
    });
}

#[test]
fn runs_a_closure_capturing_a_shared_record() {
    // The captured value is a heap-allocated `[shared]` record, so the closure
    // box holds an `rc` pointer in its payload. Evaluating the closure hands the
    // payload to the body as owned and the body consumes it, so the box is freed
    // exactly once (the capture is a last use — moved in, not dup'd). A wrong
    // refcount would double-free or leak the `Box`; the returned scalar pins it.
    let src = r#"
        struct [shared] Box { v: i64 }
        fn mk(v: i64) -> Box { Box { v: v } }
        pub fn run(n: i64) -> i64 {
            let b = mk(n);
            (|x: i64| x + b.v)(1)
        }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // (|x| x + b.v)(1) with b.v = n  ⇒  1 + n
        assert_eq!(f(41), 42);
        assert_eq!(f(99), 100);
    });
}

#[test]
fn runs_a_partially_then_fully_applied_closure() {
    // The inner `(…)(6)` supplies one of two parameters — a partial application
    // that uniqifies, applies, and stops at the residual `(y) -> i64` closure.
    // The outer `(…)(7)` uniqifies that residual, applies the last argument, and
    // evaluates. Exercises the multi-argument apply chain and both the
    // partial-application and full-application paths in one call.
    let src = r#"
        pub fn run(n: i64) -> i64 { ((|x: i64, y: i64| x * y + n)(6))(7) }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // 6 * 7 + n = 42 + n
        assert_eq!(f(0), 42);
        assert_eq!(f(8), 50);
    });
}

#[test]
fn runs_regional_record_construction_and_projection() {
    // A `[regional]` record is region-allocated as a `flex` box inside a
    // `region-run` scope; projecting a scalar field out of it leaves the region
    // as a plain `i64`. Drives the region lifecycle — allocate into the arena,
    // read the field, tear the region down — through the runtime.
    let src = r#"
        struct [regional] Cell { v: i64 }
        regional fn make(x: i64) -> [flex] Cell { Cell { v: x } }
        pub fn run(n: i64) -> i64 { regional { make(n).v } }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(f(7), 7);
        assert_eq!(f(-3), -3);
    });
}

#[test]
fn runs_a_reused_frozen_regional_record() {
    // A regional record that escapes its region is frozen to a `rigid` `rc` box —
    // a managed, reference-counted value. Reusing `c` across two `take` calls is
    // not a last use at the first, so the ownership analysis `dup`s it (a direct
    // `rc.inc` on the `rc` pointer), and each callee consumes and drops its
    // reference (`rc.dec`), freeing the box exactly once. A wrong count would
    // double-free (crash) or leak; the arithmetic result pins it down. This is
    // the runtime check for the managed-rc inc/dec path on a frozen record.
    let src = r#"
        struct [regional] Cell { v: i64 }
        regional fn make(x: i64) -> [flex] Cell { Cell { v: x } }
        fn take(c: Cell) -> i64 { c.v }
        pub fn run(n: i64) -> i64 {
            let c = regional { make(n) };
            take(c) + take(c)
        }
        extern "C" trampoline "run_ffi" = run;
    "#;
    jit_run(src, |jit| {
        let a = jit.lookup("run_ffi").expect("lookup run_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        // take(c) + take(c) = n + n
        assert_eq!(f(21), 42);
        assert_eq!(f(-5), -10);
    });
}

#[test]
fn runs_iterative_helper_and_signed_arithmetic() {
    let src = r#"
        fn iter_impl(n: u64, a: u64, b: u64) -> u64 {
            if n == 0 { a } else { iter_impl(n - 1, b, a + b) }
        }
        pub fn fib_iter(n: u64) -> u64 { iter_impl(n, 0, 1) }
        pub fn signed_mix(a: i32, b: i32) -> i32 { (a * b - 7) / 2 }
        extern "C" trampoline "fib_iter_ffi" = fib_iter;
        extern "C" trampoline "signed_mix_ffi" = signed_mix;
    "#;
    jit_run(src, |jit| {
        // Call through the exported C-ABI trampolines, not the internal symbols.
        let a = jit.lookup("fib_iter_ffi").expect("lookup fib_iter_ffi");
        let fib_iter: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(fib_iter(10), 55);
        assert_eq!(fib_iter(20), 6765);

        let s = jit.lookup("signed_mix_ffi").expect("lookup signed_mix_ffi");
        let signed_mix: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(s as usize) };
        // (6 * -5 - 7) / 2 = (-30 - 7) / 2 = -37 / 2 = -18 (signed truncation toward zero)
        assert_eq!(signed_mix(6, -5), -18);
    });
}
