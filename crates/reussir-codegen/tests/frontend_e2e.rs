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
        lower_program(&context, tcx, &full).expect("scalar lowering succeeds")
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
        let a = jit.lookup("build_and_read_ffi").expect("lookup build_and_read_ffi");
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(a as usize) };
        assert_eq!(f(5), 5);
        assert_eq!(f(42), 42);
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
