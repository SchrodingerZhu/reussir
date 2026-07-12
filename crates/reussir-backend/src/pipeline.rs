//! The Reussir lowering pipeline.
//!
//! Assembles the sequence of Reussir and upstream passes that lowers a Reussir
//! module down to the LLVM dialect, mirroring the backend's C++ pipeline. Each
//! pass is created through the Reussir C API and added to a melior pass manager,
//! with `func.func`-nested passes placed in their original order so the
//! interleaving with module-level passes is preserved.

use std::path::PathBuf;

use melior::Context;
use melior::ir::Module;
use melior::pass::{Pass, PassManager};

use reussir_backend_sys as sys;

/// Temporary module marker set by codegen when source contains inline transforms.
pub const INLINE_TRANSFORM_ATTR: &str = "reussir.inline_transform";
/// Embedded transform entry point invoked at the kernel anchor.
pub const INLINE_TRANSFORM_ENTRY_POINT: &str = "__reussir_inline_transform";

/// Optimization level for the lowering pipeline, matching the backend's
/// `ReussirOptOption` C enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization; the optimization-only prologue passes are skipped.
    None,
    /// Default optimization.
    Default,
    /// Aggressive optimization; also runs the unique-carrying recursion analysis.
    Aggressive,
    /// Optimize for size.
    Size,
    /// Lower for the TPDE fast back end.
    Tpde,
}

impl OptLevel {
    /// Whether the optimization-only prologue (inliner and friends) runs.
    fn runs_optimization(self) -> bool {
        !matches!(self, OptLevel::None)
    }

    /// This level as its matching `ReussirOptOption` C enum value — the contract
    /// the LLVM-IR-level backend pipeline (`reussirRunBackendLLVMPipeline`)
    /// expects. Owned here so callers (the JIT, the AOT compiler) need not
    /// re-encode it.
    pub fn as_reussir_opt_option(self) -> core::ffi::c_int {
        match self {
            OptLevel::None => 0,
            OptLevel::Default => 1,
            OptLevel::Aggressive => 2,
            OptLevel::Size => 3,
            OptLevel::Tpde => 4,
        }
    }
}

/// How nullary variants of shared rc-boxed enums are represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullaryVariantEncoding {
    /// Legacy layout: every construction heap-allocates a refcounted box.
    Boxed,
    /// Unboxed immediate whose top byte is `tag + 1` and whose low bits
    /// point at a per-tag dummy box. Hard-depends on the hardware ignoring
    /// the pointer's top byte on data accesses (aarch64 TBI); the top byte
    /// doubles as a dereference-free FFI tag decode.
    Tbi,
    /// Unboxed immediate that *is* the per-tag dummy box address, with an
    /// immortal refcount recognized by magnitude. No architectural
    /// dependency (works on any target, including wasm32); foreign code
    /// sees a layout-compatible box.
    Immortal,
}

/// A named interception point in the lowering pipeline where user-supplied
/// transform-dialect scripts run (issue #349). Each anchor is a documented,
/// stable IR contract: scripts see the module at that abstraction level and
/// must leave it consumable by the passes that follow.
///
/// A script targeting an anchor must define the anchor's entry point,
/// `transform.named_sequence @__reussir_anchor_<name>(%payload)`, in a module
/// carrying the `transform.with_named_sequence` attribute. Distinct anchors
/// have distinct entry points, so scripts for several anchors can coexist in
/// the context-wide transform library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Anchor {
    /// Before any Reussir pass: pure `reussir`-dialect IR straight from
    /// codegen, no tokens instantiated yet. For whole-program structural
    /// rewrites and custom analyses.
    Entry,
    /// After the Reussir-level passes (both `scf-ops-lowering` runs and the
    /// polymorphic-FFI compile are done), before the LLVM descent begins:
    /// array kernels are plain memref/scf/arith loop nests. The main
    /// scheduling surface — tiling, unrolling, fusion.
    Kernel,
}

impl Anchor {
    /// The anchor's user-facing name, as written in `--transform-script
    /// <file>@<name>`.
    pub fn name(self) -> &'static str {
        match self {
            Anchor::Entry => "entry",
            Anchor::Kernel => "kernel",
        }
    }

    /// The transform entry point the interpreter runs at this anchor:
    /// `__reussir_anchor_<name>`.
    pub fn entry_point(self) -> &'static str {
        match self {
            Anchor::Entry => "__reussir_anchor_entry",
            Anchor::Kernel => "__reussir_anchor_kernel",
        }
    }

    /// Parses a user-facing anchor name (`entry`/`kernel`).
    pub fn parse(name: &str) -> Option<Anchor> {
        match name {
            "entry" => Some(Anchor::Entry),
            "kernel" => Some(Anchor::Kernel),
            _ => None,
        }
    }
}

/// Knobs for [`run_lowering_pipeline`].
#[derive(Clone, Debug)]
pub struct LoweringOptions {
    /// Optimization level.
    pub opt: OptLevel,
    /// Allow the token-reuse pass to reuse tokens across function calls.
    pub reuse_token_across_call: bool,
    /// How nullary variants of shared rc-boxed enums are encoded. `rrc`
    /// exposes this as `--nullary-variant-encoding`; embedders (REPL/JIT,
    /// tests) default to [`NullaryVariantEncoding::Boxed`].
    pub nullary_variant_encoding: NullaryVariantEncoding,
    /// Run the invariant-group analysis pass.
    pub enable_invariant_analysis: bool,
    /// Lay out compound record members in packed physical order (descending
    /// storage alignment) rather than declaration order. On by default — the
    /// layout contract external consumers compile against; `rrc` exposes the
    /// off switch as `--no-pack-record-members`.
    pub pack_record_members: bool,
    /// Transform-dialect scripts to run, each at its [`Anchor`]. Empty means
    /// zero overhead: no transform pass is added and the pipeline is exactly
    /// the fixed pass list. `rrc` exposes this as `--transform-script`.
    pub transform_scripts: Vec<(Anchor, PathBuf)>,
    /// Emit whole-program devirtualization artifacts for closures (a `!type`
    /// id per vtable, keyed by return type, plus a call-site type test at
    /// every indirect eval/clone/drop). The backend pipeline's SPECULATIVE
    /// `WholeProgramDevirt` then folds single-implementation families into
    /// guarded direct calls — sound on any world (a vtable the module has
    /// never seen takes the indirect fallback arm), so this is a pure
    /// compile-time/perf knob, not a soundness gate. Off by default; `rrc`
    /// enables it at the opt levels that ask for whole-program effort.
    pub closure_wpd: bool,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            opt: OptLevel::Default,
            reuse_token_across_call: false,
            // Boxed by default: only target-aware drivers (rrc) pick an
            // immediate encoding, so embedders (REPL/JIT, tests) keep the
            // boxed layout untouched.
            nullary_variant_encoding: NullaryVariantEncoding::Boxed,
            // Off by default, mirroring the C++ `createLoweringPipeline`
            // (`enableInvariantAnalysis = false`).
            enable_invariant_analysis: false,
            // On by default: packed layout is the shipped default and the
            // layout contract; only an explicit driver flag turns it off.
            pack_record_members: true,
            transform_scripts: Vec::new(),
            // Off by default: embedders opt in explicitly; `rrc` turns it on
            // at `-O aggressive`/`-O size`.
            closure_wpd: false,
        }
    }
}

/// Stamps `module` with the target's layout facts — the `llvm.data_layout`
/// string, the `llvm.target_triple`, and the translated `dlti.dl_spec` that
/// MLIR `DataLayout` queries read. Call it before [`run_lowering_pipeline`]:
/// without the spec MLIR falls back to conservative defaults (e.g. `i64` at
/// ABI alignment 4), so every size and alignment the pipeline computes —
/// allocation sizes, spill alignments, load/store alignment annotations —
/// understates the target.
pub fn attach_target_spec(module: &Module, data_layout: &str, triple: &str) -> Result<(), String> {
    let data_layout = std::ffi::CString::new(data_layout)
        .map_err(|_| "data layout contains a NUL byte".to_string())?;
    let triple = std::ffi::CString::new(triple)
        .map_err(|_| "target triple contains a NUL byte".to_string())?;
    // SAFETY: `module` is live for the call, and both strings are
    // NUL-terminated; the CAPI only reads them to build attributes.
    let ok = unsafe {
        sys::reussirModuleAttachTargetSpec(module.to_raw(), data_layout.as_ptr(), triple.as_ptr())
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid data layout string `{}`",
            data_layout.to_string_lossy()
        ))
    }
}

// Wraps a Reussir C API pass factory result as a melior pass.
fn pass(raw: sys::mlir_sys::MlirPass) -> Pass {
    // SAFETY: the C API returns a freshly created, owned MlirPass.
    unsafe { Pass::from_raw(raw) }
}

// Views a string as the borrowed MlirStringRef the C API takes; `s` must
// outlive every use of the result.
fn string_ref(s: &str) -> sys::mlir_sys::MlirStringRef {
    sys::mlir_sys::MlirStringRef {
        data: s.as_ptr().cast(),
        length: s.len(),
    }
}

/// A small DSL describing the lowering pipeline as a declarative list of steps,
/// so the layout reads top-to-bottom instead of being buried in builder calls.
///
/// Each step is one of:
/// * `module: <factory>;` — a module-level pass;
/// * `func: <factory>;` — a pass nested under `func.func`;
/// * `if <cond> => { <steps> }` — a group run only when `<cond>` holds.
///
/// `<factory>` is a Reussir C API pass factory (an `unsafe` call); the macro
/// wraps each one and hands the resulting [`Pass`] to the pass manager. The
/// first argument names the [`PassManager`] to build into.
macro_rules! lowering_pipeline {
    // Done.
    ($pm:ident) => {};
    ($pm:ident,) => {};
    // Module-level pass.
    ($pm:ident, module: $factory:expr; $($rest:tt)*) => {{
        // SAFETY: each factory returns a freshly created, owned MlirPass.
        $pm.add_pass(pass(unsafe { $factory }));
        lowering_pipeline!($pm, $($rest)*);
    }};
    // Pass nested under `func.func`.
    ($pm:ident, func: $factory:expr; $($rest:tt)*) => {{
        // SAFETY: each factory returns a freshly created, owned MlirPass.
        $pm.nested_under("func.func").add_pass(pass(unsafe { $factory }));
        lowering_pipeline!($pm, $($rest)*);
    }};
    // Group of steps gated on a condition.
    ($pm:ident, if $cond:expr => { $($body:tt)* } $($rest:tt)*) => {{
        if $cond {
            lowering_pipeline!($pm, $($body)*);
        }
        lowering_pipeline!($pm, $($rest)*);
    }};
}

/// Runs the full Reussir lowering pipeline on `module`, leaving it in the LLVM
/// dialect ready for translation to LLVM IR.
pub fn run_lowering_pipeline(
    context: &Context,
    module: &mut Module,
    options: &LoweringOptions,
) -> Result<(), melior::Error> {
    let _span = tracing::debug_span!(
        "reussir_lowering",
        opt = ?options.opt,
        reuse_token_across_call = options.reuse_token_across_call,
        enable_invariant_analysis = options.enable_invariant_analysis,
        nullary_variant_encoding = ?options.nullary_variant_encoding,
        pack_record_members = options.pack_record_members,
    )
    .entered();

    // Record-member packing is a whole-compilation layout contract held on the
    // context-loaded dialect; set it once before any pass computes a layout.
    // SAFETY: `context` is live for the call; the CAPI only loads the Reussir
    // dialect and sets a bool on it.
    unsafe {
        sys::reussirContextSetPackRecordMembers(context.to_raw(), options.pack_record_members);
    }

    let manager = PassManager::new(context);

    // Transform scripts, marshalled for the C API: every script file is
    // preloaded into the context's transform library up front (one pass, all
    // paths), then each anchor with at least one script runs the interpreter
    // with that anchor's `__reussir_anchor_<name>` entry point. The `String`s
    // must stay alive until the factories below have copied them.
    let script_paths: Vec<String> = options
        .transform_scripts
        .iter()
        .map(|(_, path)| path.to_string_lossy().into_owned())
        .collect();
    let script_refs: Vec<sys::mlir_sys::MlirStringRef> =
        script_paths.iter().map(|s| string_ref(s)).collect();
    let at_anchor = |anchor: Anchor| options.transform_scripts.iter().any(|(a, _)| *a == anchor);

    lowering_pipeline!(manager,
        if !script_refs.is_empty() => {
            module: sys::reussirCreateTransformPreloadLibraryPass(
                script_refs.as_ptr(),
                script_refs.len() as isize,
            );
        }

        // `entry` anchor: pure reussir-dialect IR straight from codegen,
        // before any Reussir pass.
        if at_anchor(Anchor::Entry) => {
            module: sys::reussirCreateTransformInterpreterPass(
                string_ref(Anchor::Entry.entry_point()),
            );
        }

        // Nullary variants of shared rc-boxed enums become tagged pointer
        // immediates. Must run before any uniqueness/token pass so no
        // provenance or allocation token is ever attached to a rewritten
        // construction.
        if options.nullary_variant_encoding != NullaryVariantEncoding::Boxed => {
            module: sys::reussirCreateSpecialPointerTagPass(
                options.nullary_variant_encoding == NullaryVariantEncoding::Immortal,
            );
        }

        // Optimization-only prologue.
        if options.opt.runs_optimization() => {
            if options.opt == OptLevel::Aggressive => {
                module: sys::reussirCreateUniqueCarryingRecursionAnalysisPass();
            }
            module: sys::reussirCreateDefaultInlinerPass();
            // Beta-reduce the closure chains the inliner just made visible:
            // create→apply→eval collapses to the spliced body, and chained
            // uniqueness checks collapse into one (fused eval). Must run
            // before token instantiation / closure outlining, while create
            // bodies are still inline regions. Aggressive-only: inlining
            // closure bodies detaches the code from the closure abstraction
            // the programmer wrote, which hurts debuggability.
            if options.opt == OptLevel::Aggressive => {
                func: sys::reussirCreateClosureBetaReductionPass();
            }
        }

        // Reussir-level transformation and analysis.
        func:   sys::reussirCreateTokenInstantiationPass();
        module: sys::reussirCreateClosureOutliningPass();
        module: sys::reussirCreateRegionPatternsPass();
        // Fuse pattern-match consumption into destructuring decrements before
        // the cancellation pass (which cancels `inc; destructuring dec` into
        // borrow semantics) and the decrement expansion (which expands the
        // tagged decs shallowly).
        func:   sys::reussirCreateRcDispatchFusionPass();
        func:   sys::reussirCreateIncDecCancellationPass();
        module: sys::reussirCreateRcDecrementExpansionPass();
        func:   sys::reussirCreateInferVariantTagPass();
        module: sys::reussirCreateAcquireDropExpansionPass(false, false);
        module: sys::reussirCreateSCFOpsLoweringPass();
        func:   sys::reussirCreateIncDecCancellationPass();

        // Second acquire/drop expansion phase: expand decrements and outline
        // record drops.
        module: sys::reussirCreateAcquireDropExpansionPass(true, true);
        func:   sys::reussirCreateTokenReusePass(options.reuse_token_across_call);
        module: sys::reussirCreateSCFOpsLoweringPass();
        func:   sys::reussirCreateRcCreateSinkPass();
        func:   sys::reussirCreateRcCreateFusionPass();
        module: sys::reussirCreateTRMCRecursionAnalysisPass();
        module: sys::reussirCreateCompilePolymorphicFFIPass(false);

        // `kernel` anchor: the Reussir-level work is done (both
        // scf-ops-lowering runs, FFI compiled) and the LLVM descent has not
        // begun — array kernels are plain memref/scf/arith loop nests, the
        // transform dialect's home turf. Runs before the invariant analysis
        // so that pass sees the scheduled IR.
        if at_anchor(Anchor::Kernel) => {
            module: sys::reussirCreateTransformInterpreterPass(
                string_ref(Anchor::Kernel.entry_point()),
            );
        }

        if options.enable_invariant_analysis => {
            func: sys::reussirCreateInvariantGroupAnalysisPass();
        }

        // Lower to the LLVM dialect.
        module: sys::reussirCreateCanonicalizerPass();
        module: sys::reussirCreateControlFlowSinkPass();
        module: sys::reussirCreateSCFToControlFlowPass();
        module: sys::reussirCreateBasicOpsLoweringPass(options.closure_wpd);
        module: sys::reussirCreateConvertToLLVMPass();
        module: sys::reussirCreateReconcileUnrealizedCastsPass();
        // Convert fused Reussir debug-info attributes to LLVM DI now that
        // functions are `llvm.func`; a function's DI only survives translation
        // once its `llvm.func` carries a `DISubprogram`.
        module: sys::reussirCreateDebugInfoConversionPass();
        module: sys::reussirCreateCSEPass();
        module: sys::reussirCreateCanonicalizerPass();
    );

    tracing::trace!("running Reussir lowering pipeline");
    let result = manager.run(module);
    match &result {
        Ok(()) => tracing::debug!("lowered module to the LLVM dialect"),
        Err(error) => tracing::error!(?error, "lowering pipeline failed"),
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::operation::OperationLike;

    #[test]
    fn lowers_simple_module_to_llvm_dialect() {
        let context = crate::context();

        let source = r#"
            module {
              func.func @answer() -> i32 {
                %0 = arith.constant 42 : i32
                func.return %0 : i32
              }
            }
        "#;
        let mut module = Module::parse(&context, source).expect("module should parse");

        run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
            .expect("pipeline should succeed");

        assert!(module.as_operation().verify());
        // After lowering, the function is an llvm.func rather than a func.func.
        let rendered = module.as_operation().to_string();
        assert!(rendered.contains("llvm.func @answer"), "got:\n{rendered}");
    }

    // A loop-nest kernel of the shape the `kernel` anchor guarantees: by the
    // time the anchor runs, this function is untouched memref/scf/arith IR.
    const SUM_KERNEL: &str = r#"
        module {
          func.func @sum(%a: memref<4xi64>) -> i64 {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            %zero = arith.constant 0 : i64
            %r = scf.for %i = %c0 to %c4 step %c1
                iter_args(%acc = %zero) -> (i64) {
              %v = memref.load %a[%i] : memref<4xi64>
              %s = arith.addi %acc, %v : i64
              scf.yield %s : i64
            }
            func.return %r : i64
          }
        }
    "#;

    // Writes `script` to an auto-cleaned file and returns the handle; the
    // preload pass reads the script from disk by path.
    fn script_file(script: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::Builder::new()
            .suffix(".transform.mlir")
            .tempfile()
            .expect("temp script file");
        file.write_all(script.as_bytes()).expect("write script");
        file
    }

    #[test]
    fn kernel_anchor_script_schedules_the_loop() {
        let context = crate::context();
        let mut module = Module::parse(&context, SUM_KERNEL).expect("module should parse");

        // Fully unroll the 4-trip loop at the kernel anchor.
        let script = script_file(
            r#"
            module attributes {transform.with_named_sequence} {
              transform.named_sequence @__reussir_anchor_kernel(
                  %m: !transform.any_op {transform.readonly}) {
                %loops = transform.structured.match ops{["scf.for"]} in %m
                    : (!transform.any_op) -> !transform.op<"scf.for">
                transform.loop.unroll %loops { factor = 4 } : !transform.op<"scf.for">
                transform.yield
              }
            }
            "#,
        );
        let options = LoweringOptions {
            transform_scripts: vec![(Anchor::Kernel, script.path().to_path_buf())],
            ..LoweringOptions::default()
        };
        run_lowering_pipeline(&context, &mut module, &options).expect("pipeline should succeed");

        assert!(module.as_operation().verify());
        // The unrolled (and then canonicalized) loop is straight-line code:
        // four loads and no back edge.
        let rendered = module.as_operation().to_string();
        assert_eq!(
            rendered.matches("llvm.load").count(),
            4,
            "expected the loop fully unrolled, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("llvm.br"),
            "expected no loop back edge after full unroll, got:\n{rendered}"
        );
    }

    #[test]
    fn scriptless_options_add_no_transform_passes() {
        // The baseline pipeline must not observe the transform machinery at
        // all when no script is given — same behaviour as before the anchors
        // existed, kernel loop intact.
        let context = crate::context();
        let mut module = Module::parse(&context, SUM_KERNEL).expect("module should parse");
        run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
            .expect("pipeline should succeed");
        let rendered = module.as_operation().to_string();
        assert_eq!(
            rendered.matches("llvm.load").count(),
            1,
            "expected the loop untouched, got:\n{rendered}"
        );
    }

    #[test]
    fn script_missing_anchor_entry_point_fails_cleanly() {
        // A script that never defines the anchor's entry point is a pipeline
        // error (the interpreter cannot resolve `__reussir_anchor_kernel`),
        // not a silent no-op and not a crash.
        let context = crate::context();
        let mut module = Module::parse(&context, SUM_KERNEL).expect("module should parse");
        let script = script_file(
            r#"
            module attributes {transform.with_named_sequence} {
              transform.named_sequence @some_other_sequence(
                  %m: !transform.any_op {transform.readonly}) {
                transform.yield
              }
            }
            "#,
        );
        let options = LoweringOptions {
            transform_scripts: vec![(Anchor::Kernel, script.path().to_path_buf())],
            ..LoweringOptions::default()
        };
        run_lowering_pipeline(&context, &mut module, &options)
            .expect_err("missing entry point should fail the pipeline");
    }
}
