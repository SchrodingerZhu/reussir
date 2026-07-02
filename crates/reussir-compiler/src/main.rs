//! `rrc`: the Reussir compiler driver.
//!
//! A clang-style pipeline driver over one totally-ordered chain of stages:
//!
//! ```text
//! .rr ──elaborate──▶ .hir ──monomorphize──▶ .mir ──lower──▶ .mlir ──pipeline──▶ mlir-llvm ──translate──▶ .ll ──▶ .s/.o
//! ```
//!
//! The input's extension (or `--from`) picks where to enter the chain; `-o`'s
//! extension (or `--emit`) picks where to leave it; the driver runs exactly the
//! transforms in between. Every intermediate is both a dump target and — except
//! the two derived MLIR forms — a re-entry point, so a stage can be inspected or
//! fed back in isolation. Exit 0 on success, 1 on a compile error, 2 on a usage
//! or I/O error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use palc::Parser;

use reussir_backend::llvm::LlvmLowering;
use reussir_backend::melior::ir::Module;
use reussir_backend::pipeline::{self, LoweringOptions, OptLevel};
use reussir_codegen::lower::lower_program;
use reussir_codegen::source::SourceMap;
use reussir_compiler::{
    OutputKind, RelocMode, TargetMachine, TargetSpec, emit_to_file, parse_llvm_ir,
};
use reussir_core::full::mir;
use reussir_core::full::mono::{MonoInput, monomorphize};
use reussir_core::semi::resolve::DefTable;
use reussir_core::semi::hir;
use reussir_core::semi::ty::TyCtxt;
use reussir_core::semi::{Report, Severity, elaborate};
use reussir_core::{in_arena, surface};
use reussir_syntax::kind::{Resolver, TokenKey};

/// A stage on the compilation chain, ordered from source to object so a target
/// can be compared against the input (`>=` means "reachable going forward").
///
/// Deriving [`palc::ValueEnum`] gives `--emit`/`--from` their value parsing and a
/// kebab-case [`Display`] (`MlirLlvm` → `mlir-llvm`, `LlvmIr` → `llvm-ir`, …).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, palc::ValueEnum)]
enum Stage {
    /// Reussir source (`.rr`).
    Rr,
    /// Elaborated, still-polymorphic HIR (`.hir`).
    Hir,
    /// Monomorphized ground MIR (`.mir`).
    Mir,
    /// The Reussir MLIR dialect, before the lowering pipeline (`.mlir`).
    Mlir,
    /// MLIR after the lowering pipeline — the LLVM dialect (`--emit mlir-llvm`).
    MlirLlvm,
    /// LLVM IR text (`.ll`).
    LlvmIr,
    /// Target assembly (`.s`).
    Asm,
    /// A relocatable object file (`.o`).
    Obj,
}

impl Stage {
    /// Whether this stage can be *read* as an input (it has a parser). The two
    /// derived MLIR/assembly forms and the object file are outputs only.
    fn is_input(self) -> bool {
        matches!(
            self,
            Stage::Rr | Stage::Hir | Stage::Mir | Stage::Mlir | Stage::LlvmIr
        )
    }

    /// The stage a file extension denotes, if any.
    fn from_extension(path: &Path) -> Option<Stage> {
        Some(match path.extension().and_then(|e| e.to_str())? {
            "rr" => Stage::Rr,
            "hir" => Stage::Hir,
            "mir" => Stage::Mir,
            "mlir" => Stage::Mlir,
            "mlir-llvm" => Stage::MlirLlvm,
            "ll" => Stage::LlvmIr,
            "s" => Stage::Asm,
            "o" => Stage::Obj,
            _ => return None,
        })
    }

    /// The `LLVM`-emitting output kind, for the three terminal stages.
    fn output_kind(self) -> Option<OutputKind> {
        match self {
            Stage::LlvmIr => Some(OutputKind::LlvmIr),
            Stage::Asm => Some(OutputKind::Assembly),
            Stage::Obj => Some(OutputKind::Object),
            _ => None,
        }
    }
}

/// Reussir compiler driver.
#[derive(Parser)]
#[command(name = "rrc", version)]
struct Cli {
    /// Input file (`-` reads Reussir source from stdin).
    input: PathBuf,

    /// Output file (`-` writes a text dump — `hir`/`mir`/`mlir`/`mlir-llvm` — to
    /// stdout). The target stage is inferred from its extension unless `--emit`
    /// is given.
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Stage to emit: `hir`, `mir`, `mlir`, `mlir-llvm`, `llvm-ir`, `asm`, or
    /// `obj`. Defaults to the output extension (else `obj`).
    #[arg(short = 't', long = "emit")]
    emit: Option<Stage>,

    /// Treat the input as this stage instead of inferring from its extension:
    /// `rr`, `hir`, `mir`, `mlir`, or `llvm-ir`.
    #[arg(short = 'x', long = "from")]
    from: Option<Stage>,

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

    /// Log the lowering/backend `tracing` events (to stderr) at DEBUG level.
    /// `RUST_LOG`, if set, takes precedence over this.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

/// The stage the input enters at: `--from` if given, else the extension. `-`
/// (stdin) has no extension, so it defaults to source.
fn resolve_input_stage(cli: &Cli) -> Result<Stage, String> {
    let stage = if let Some(from) = cli.from {
        from
    } else if cli.input.as_os_str() == "-" {
        Stage::Rr
    } else {
        Stage::from_extension(&cli.input).ok_or_else(|| {
            format!(
                "cannot infer the input stage of `{}`; pass --from",
                cli.input.display()
            )
        })?
    };
    if !stage.is_input() {
        return Err(format!("`{stage}` is an output-only stage; it cannot be read as input"));
    }
    Ok(stage)
}

/// The stage the driver stops at: `--emit` if given, else the output extension,
/// else an object file.
fn resolve_target(cli: &Cli) -> Result<Stage, String> {
    let stage = cli
        .emit
        .or_else(|| Stage::from_extension(&cli.output))
        .unwrap_or(Stage::Obj);
    if stage == Stage::Rr {
        return Err("`rr` is the source input, not an emittable stage".into());
    }
    Ok(stage)
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

/// Writes a text stage to `path`, or to stdout when it is `-`.
fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if path.as_os_str() == "-" {
        use std::io::Write;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(|e| format!("failed to write to stdout: {e}"))
    } else {
        std::fs::write(path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
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

/// What the front (arena-scoped) leg produces: a text dump for `hir`/`mir`, or an
/// MLIR module for `mlir` and anything past it. The module borrows the MLIR
/// context, not the type arena, so it outlives the [`in_arena`] scope.
enum Produced<'c> {
    Text(String),
    Module(Module<'c>),
}

/// Install a `tracing` subscriber writing to **stderr** (so it never corrupts a
/// `-o -` stdout dump). `RUST_LOG` wins if set; otherwise `-v` selects DEBUG and
/// the default is quiet (WARN).
fn init_tracing(verbose: bool) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(if verbose { "debug" } else { "warn" }));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    match run(&cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("error: {message}");
            }
            ExitCode::from(2)
        }
    }
}

fn run(cli: &Cli) -> Result<bool, String> {
    let input_stage = resolve_input_stage(cli)?;
    let target = resolve_target(cli)?;
    if target < input_stage {
        return Err(format!(
            "cannot emit `{target}` from a `{input_stage}` input: the pipeline only runs forward"
        ));
    }
    // Only the text dumps stream to stdout; `llvm-ir`/`asm`/`obj` go through the
    // file-based LLVM emitter, so `-o -` would write a file literally named `-`.
    if cli.output.as_os_str() == "-" && target.output_kind().is_some() {
        return Err(format!(
            "cannot write `{target}` to stdout; give a file path with -o"
        ));
    }
    let opt = parse_opt(&cli.opt)?;
    let reloc = parse_reloc(&cli.relocation_mode)?;
    let (name, source) = read_input(&cli.input)?;

    let spec = TargetSpec {
        triple: cli.target_triple.clone(),
        cpu: cli.target_cpu.clone(),
        features: cli.target_features.clone(),
    };

    // A `.ll` input skips the whole MLIR front: parse the IR and run the LLVM
    // backend straight to the requested artifact.
    if input_stage == Stage::LlvmIr {
        let kind = target
            .output_kind()
            .ok_or_else(|| format!("cannot emit `{target}` from LLVM IR"))?;
        let machine = TargetMachine::new(&spec, opt, reloc)?;
        let finalized = parse_llvm_ir(&name, &source)?;
        emit_to_file(finalized, &machine, opt, kind, &cli.output)?;
        return Ok(true);
    }

    let context = reussir_backend::context();
    if cli.disable_backend_multithreading {
        context.enable_multi_threading(false);
    }
    let source_map = SourceMap::new(&cli.input, &source);

    // Front leg: reach an MLIR module (or a text dump for `hir`/`mir`). A `.mlir`
    // input parses directly; source/`.hir`/`.mir` run the frontend in the arena.
    let produced = match input_stage {
        Stage::Mlir => Module::parse(&context, &source)
            .map(Produced::Module)
            .ok_or_else(|| format!("{name}: failed to parse MLIR module"))?,
        _ => match in_arena(|tcx| {
            frontend(&context, tcx, input_stage, target, &name, &source, &source_map, cli)
        }) {
            Ok(produced) => produced,
            Err(msg) => {
                if !msg.is_empty() {
                    eprintln!("{msg}");
                }
                return Ok(false);
            }
        },
    };

    let mut module = match produced {
        Produced::Text(text) => {
            write_text(&cli.output, &text)?;
            return Ok(true);
        }
        Produced::Module(module) => module,
    };

    // A `.mlir` dump is the module as it stands, before the lowering pipeline.
    if target == Stage::Mlir {
        write_text(&cli.output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    // MLIR → LLVM leg: the target machine's data layout feeds the polymorphic-FFI
    // gather, which must run before the pipeline erases those ops.
    let machine = TargetMachine::new(&spec, opt, reloc)?;
    let options = LoweringOptions {
        opt,
        reuse_token_across_call: cli.reuse_across_call,
        ..LoweringOptions::default()
    };
    let optimize_ffi = !matches!(opt, OptLevel::None);
    let prepared = LlvmLowering::prepare(&module, machine.data_layout(), optimize_ffi)
        .map_err(|e| format!("{name}: {e}"))?;
    pipeline::run_lowering_pipeline(&context, &mut module, &options)
        .map_err(|e| format!("lowering pipeline failed: {e:?}"))?;

    // After the pipeline the module is the LLVM dialect; dump it before it is
    // translated out of MLIR. (`prepared`'s `Drop` releases the gathered FFI.)
    if target == Stage::MlirLlvm {
        drop(prepared);
        write_text(&cli.output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    let finalized = prepared.finish(&module).map_err(|e| format!("{name}: {e}"))?;
    let kind = target
        .output_kind()
        .expect("llvm-ir/asm/obj target past the mlir-llvm stage");
    emit_to_file(finalized, &machine, opt, kind, &cli.output)?;
    Ok(true)
}

/// The arena-scoped front leg for a source/`.hir`/`.mir` input: run the frontend
/// from the entry stage to `target`, yielding a text dump or a lowered module.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
#[allow(clippy::too_many_arguments)]
fn frontend<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    input: Stage,
    target: Stage,
    name: &str,
    source: &str,
    source_map: &SourceMap<'_>,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    match input {
        Stage::Rr => {
            let parse = reussir_syntax::parse(source);
            if !parse.ok() {
                for err in &parse.errors {
                    eprintln!("{name}: error: {err:?}");
                }
                return Err(String::new());
            }
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            if report_diagnostics(name, &elab.reports) {
                return Err(String::new());
            }
            if target == Stage::Hir {
                let text = hir::print::Printer::new(&elab.defs, elab.resolver).program(
                    &elab.elaborated,
                    &elab.records,
                    &elab.trampolines,
                );
                return Ok(Produced::Text(text));
            }
            let (full, reports) = monomorphize(&elab.mono_input());
            if report_diagnostics(name, &reports) {
                return Err(String::new());
            }
            finish_mir(
                context,
                tcx,
                target,
                name,
                Some(source_map),
                cli.debug,
                &full,
                &elab.defs,
                elab.resolver,
            )
        }
        Stage::Hir => {
            let parsed = hir::build::parse_program(tcx, source)
                .map_err(|e| format!("{name}: {e}"))?;
            if target == Stage::Hir {
                let text = hir::print::Printer::new(&parsed.defs, &parsed.names).program(
                    &parsed.funcs,
                    &parsed.records,
                    &parsed.trampolines,
                );
                return Ok(Produced::Text(text));
            }
            let input = MonoInput {
                tcx,
                defs: &parsed.defs,
                resolver: &parsed.names,
                elaborated: &parsed.funcs,
                records: &parsed.records,
                trampolines: &parsed.trampolines,
            };
            let (full, reports) = monomorphize(&input);
            if report_diagnostics(name, &reports) {
                return Err(String::new());
            }
            // A re-parsed IR carries no original source, so debug locations would
            // be meaningless: lower without a source map.
            finish_mir(
                context, tcx, target, name, None, cli.debug, &full, &parsed.defs, &parsed.names,
            )
        }
        Stage::Mir => {
            let parsed = mir::build::parse_program(tcx, source)
                .map_err(|e| format!("{name}: {e}"))?;
            finish_mir(
                context,
                tcx,
                target,
                name,
                None,
                cli.debug,
                &parsed.program,
                &parsed.defs,
                &parsed.names,
            )
        }
        _ => unreachable!("frontend only handles rr/hir/mir inputs"),
    }
}

/// Finish the front leg from a ground MIR program: a `mir` dump, or lowering to
/// an MLIR module for `mlir` and beyond.
#[allow(clippy::too_many_arguments)]
fn finish_mir<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    target: Stage,
    name: &str,
    source_map: Option<&SourceMap<'_>>,
    debug: bool,
    program: &mir::Program<'tcx>,
    defs: &DefTable,
    resolver: &dyn Resolver<TokenKey>,
) -> Result<Produced<'c>, String> {
    if target == Stage::Mir {
        return Ok(Produced::Text(
            mir::print::Printer::new(defs, resolver).program(program),
        ));
    }
    // Names feed variable/function debug info; only meaningful with `-g`.
    let names = debug.then_some(resolver);
    let module = lower_program(context, tcx, program, source_map, names)
        .map_err(|e| format!("{name}: {e}"))?;
    Ok(Produced::Module(module))
}
