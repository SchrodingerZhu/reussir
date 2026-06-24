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

/// Compile `source` to a lowered LLVM-dialect module, then JIT it and hand the
/// engine to `run` to look up and call entry points.
fn jit_run<R>(source: &str, run: impl FnOnce(&OrcJit) -> R) -> R {
    let context = reussir_backend::context();
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
        lower_program(&context, &full).expect("scalar lowering succeeds")
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
