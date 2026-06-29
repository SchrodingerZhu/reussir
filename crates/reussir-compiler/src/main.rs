//! `reussir-compiler`: compile a Reussir source file to native code.
//!
//! Runs the whole pipeline — parse, elaborate, monomorphize, lower to MLIR, run
//! the Reussir lowering pipeline, then emit an object file, assembly, or LLVM IR
//! for the selected target (the native host by default, or `--target-triple`).
//! Exit 0 on success, 1 on a frontend (syntax/elaboration) or lowering error, 2
//! on usage or I/O error.

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use reussir_backend::llvm::LlvmLowering;
use reussir_backend::pipeline::{self, LoweringOptions, OptLevel};
use reussir_codegen::lower::lower_program;
use reussir_codegen::source::SourceMap;
use reussir_compiler::{OutputKind, RelocMode, TargetMachine, TargetSpec, emit_to_file};
use reussir_core::full::mono::monomorphize;
use reussir_core::semi::{Report, Severity, elaborate};
use reussir_core::{in_arena, surface};

/// Reussir ahead-of-time compiler.
#[derive(Parser)]
#[command(name = "reussir-compiler", version)]
struct Cli {
    /// Input source file (`-` reads from stdin).
    input: PathBuf,

    /// Output file. The artifact kind is inferred from its extension unless
    /// `--emit` is given.
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Artifact to emit: `obj`, `asm`, or `llvm-ir`. Defaults to the output
    /// extension (`.o`/`.s`/`.ll`), else `obj`.
    #[arg(short = 't', long = "emit")]
    emit: Option<String>,

    /// Optimization level: `none`, `default`, `aggressive`, or `size`.
    #[arg(short = 'O', long = "opt", default_value = "default")]
    opt: String,

    /// Relocation model: `default`, `pic`, or `static`.
    #[arg(long = "relocation-mode", default_value = "default")]
    relocation_mode: String,

    /// Target triple to compile for. Defaults to the native host.
    #[arg(long = "target-triple")]
    target_triple: Option<String>,

    /// Target CPU. Defaults to the native host CPU (generic for a custom triple).
    #[arg(long = "target-cpu")]
    target_cpu: Option<String>,

    /// Target features. Defaults to the native host features (none for a custom triple).
    #[arg(long = "target-features")]
    target_features: Option<String>,

    /// Let the token-reuse pass reuse tokens across function calls.
    #[arg(long = "reuse-across-call")]
    reuse_across_call: bool,

    /// Run the MLIR backend single-threaded (disable its thread pool). Useful for
    /// deterministic diagnostics and for debugging under tools that dislike the
    /// backend's worker threads (e.g. some sanitizers/debuggers).
    #[arg(long = "disable-backend-multithreading")]
    disable_backend_multithreading: bool,

    /// Emit DWARF debug info (source locations, function/variable debug types).
    #[arg(short = 'g', long = "debug")]
    debug: bool,
}

fn parse_emit(cli: &Cli) -> Result<OutputKind, String> {
    if let Some(e) = &cli.emit {
        return match e.as_str() {
            "obj" | "object" => Ok(OutputKind::Object),
            "asm" | "assembly" => Ok(OutputKind::Assembly),
            "llvm-ir" | "llvmir" | "ll" => Ok(OutputKind::LlvmIr),
            other => Err(format!("unknown --emit `{other}`")),
        };
    }
    Ok(match cli.output.extension().and_then(|e| e.to_str()) {
        Some("s") => OutputKind::Assembly,
        Some("ll") => OutputKind::LlvmIr,
        _ => OutputKind::Object,
    })
}

fn parse_opt(s: &str) -> Result<OptLevel, String> {
    match s {
        "none" => Ok(OptLevel::None),
        "default" => Ok(OptLevel::Default),
        "aggressive" => Ok(OptLevel::Aggressive),
        "size" => Ok(OptLevel::Size),
        other => Err(format!("unknown -O level `{other}`")),
    }
}

fn parse_reloc(s: &str) -> Result<RelocMode, String> {
    match s {
        "default" => Ok(RelocMode::Default),
        "pic" => Ok(RelocMode::Pic),
        "static" => Ok(RelocMode::Static),
        other => Err(format!("unknown --relocation-mode `{other}`")),
    }
}

fn read_input(path: &PathBuf) -> Result<(String, String), String> {
    use std::io::Read;
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        Ok(("<stdin>".to_owned(), buf))
    } else {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Ok((path.display().to_string(), text))
    }
}

/// Prints `reports` to stderr, each labelled with its severity, and returns
/// whether any was an error. Warnings alone do not fail the compile.
fn report_diagnostics(name: &str, reports: &[Report]) -> bool {
    let mut had_error = false;
    for report in reports {
        let label = match report.severity {
            Severity::Error => {
                had_error = true;
                "error"
            }
            Severity::Warning => "warning",
        };
        eprintln!("{name}: {label}: {}", report.message);
    }
    had_error
}

fn run(cli: &Cli) -> Result<bool, String> {
    let kind = parse_emit(cli)?;
    let opt = parse_opt(&cli.opt)?;
    let reloc = parse_reloc(&cli.relocation_mode)?;
    let (name, source) = read_input(&cli.input)?;

    let parse = reussir_syntax::parse(&source);
    if !parse.ok() {
        for err in &parse.errors {
            eprintln!("{name}: error: {err:?}");
        }
        return Ok(false);
    }
    let prog = surface::program(&parse.root);

    // The target machine is built first: its data layout feeds the polymorphic
    // FFI gather, which must run before the lowering pipeline erases those ops.
    let spec = TargetSpec {
        triple: cli.target_triple.clone(),
        cpu: cli.target_cpu.clone(),
        features: cli.target_features.clone(),
    };
    let machine = TargetMachine::new(&spec, opt, reloc)?;

    // Resolve the MIR's byte spans to source positions so lowered ops carry
    // `FileLineColLoc`s (and, with debug info, DWARF line tables). With `-g`, the
    // parser's name resolver feeds variable/function names into the debug info.
    let source_map = SourceMap::new(&cli.input, &source);
    let names = cli.debug.then(|| parse.resolver());

    let options = LoweringOptions {
        opt,
        reuse_token_across_call: cli.reuse_across_call,
        ..LoweringOptions::default()
    };
    let optimize_ffi = !matches!(opt, OptLevel::None);

    let context = reussir_backend::context();
    if cli.disable_backend_multithreading {
        context.enable_multi_threading(false);
    }
    // Frontend + lowering inside the arena scope. Polymorphic FFI is compiled and
    // gathered before the pipeline (which erases the polyffi ops) and linked in
    // after translation; the finalized LLVM module borrows neither the arena nor
    // the MLIR context, so it outlives `tcx`.
    let finalized = in_arena(|tcx| {
        let elab = elaborate(tcx, &prog, parse.resolver());
        if report_diagnostics(&name, &elab.reports) {
            return Err(String::new());
        }
        let (full, reports) = monomorphize(&elab.mono_input());
        if report_diagnostics(&name, &reports) {
            return Err(String::new());
        }
        let mut module = lower_program(&context, tcx, &full, Some(&source_map), names)
            .map_err(|e| format!("{name}: {e}"))?;
        let prepared = LlvmLowering::prepare(&module, machine.data_layout(), optimize_ffi)
            .map_err(|e| format!("{name}: {e}"))?;
        pipeline::run_lowering_pipeline(&context, &mut module, &options)
            .map_err(|e| format!("lowering pipeline failed: {e:?}"))?;
        prepared.finish(&module).map_err(|e| format!("{name}: {e}"))
    });
    let finalized = match finalized {
        Ok(finalized) => finalized,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}");
            }
            return Ok(false);
        }
    };

    emit_to_file(finalized, &machine, opt, kind, &cli.output)?;
    Ok(true)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
