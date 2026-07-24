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

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use palc::Parser;

use reussir_backend::llvm::{LlvmLowering, PolyffiPaths};
use reussir_backend::melior::ir::Module;
use reussir_backend::pipeline::{self, Anchor, LoweringOptions, NullaryVariantEncoding, OptLevel};
use reussir_codegen::lower::{
    CodegenUnit, LinkagePolicy, LoweringError, Sanitizer, lower_program, lower_unit,
};
use reussir_codegen::source::{FileId, SourceCache};
use reussir_compiler::package;
use reussir_compiler::{
    OutputKind, RelocMode, TargetMachine, TargetSpec, emit_to_file, parse_llvm_ir,
};
use reussir_core::full::mir;
use reussir_core::full::mono::{MonoInput, monomorphize};
use reussir_core::semi::hir;
use reussir_core::semi::resolve::DefTable;
use reussir_core::semi::ty::TyCtxt;
use reussir_core::semi::{elaborate, render_reports};
use reussir_core::{in_arena, surface};
use reussir_syntax::diagnostics;
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

/// CLI surface for [`Sanitizer`]: the kinds a build may declare via
/// `--sanitizer` (matching `REUSSIR_RUNTIME_SANITIZERS` plus `undefined`).
#[derive(Clone, Copy, PartialEq, Eq, palc::ValueEnum)]
enum SanitizerCli {
    /// AddressSanitizer: annotates functions `sanitize_address`.
    Address,
    /// LeakSanitizer: no function attribute (pure runtime mechanism).
    Leak,
    /// MemorySanitizer: annotates functions `sanitize_memory`.
    Memory,
    /// ThreadSanitizer: annotates functions `sanitize_thread`.
    Thread,
    /// UndefinedBehaviorSanitizer: no function attribute (UBSan checks are
    /// emitted by the C/C++ frontend, not an attribute-driven LLVM pass).
    Undefined,
}

impl From<SanitizerCli> for Sanitizer {
    fn from(kind: SanitizerCli) -> Self {
        match kind {
            SanitizerCli::Address => Sanitizer::Address,
            SanitizerCli::Leak => Sanitizer::Leak,
            SanitizerCli::Memory => Sanitizer::Memory,
            SanitizerCli::Thread => Sanitizer::Thread,
            SanitizerCli::Undefined => Sanitizer::Undefined,
        }
    }
}

/// CLI surface for [`NullaryVariantEncoding`], with the extra
/// `arch-dependent` value that resolves per target triple.
#[derive(Clone, Copy, PartialEq, Eq, palc::ValueEnum)]
enum VariantEncoding {
    /// Best encoding the target supports: TBI on aarch64 (LAM and friends
    /// may follow), the arch-independent form elsewhere.
    ArchDependent,
    /// The immortal dummy-box encoding — no TBI/LAM pointer tricks, works
    /// on any target (including wasm32).
    ArchIndependent,
    /// Legacy heap-boxed layout; no immediates.
    Boxed,
}

impl VariantEncoding {
    /// Resolves the CLI choice against the target `triple`.
    fn resolve(self, triple: &str) -> NullaryVariantEncoding {
        match self {
            VariantEncoding::ArchDependent => {
                if triple.starts_with("aarch64") {
                    NullaryVariantEncoding::Tbi
                } else {
                    NullaryVariantEncoding::Immortal
                }
            }
            VariantEncoding::ArchIndependent => NullaryVariantEncoding::Immortal,
            VariantEncoding::Boxed => NullaryVariantEncoding::Boxed,
        }
    }
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
    /// Input file (`-` reads Reussir source from stdin). With
    /// `--package-name`, the input file is the package's crate root; omitted
    /// with `--package-root`.
    input: Option<PathBuf>,

    /// Compile a whole package rooted at this directory: `lib.rr` is the
    /// crate root and `mod` declarations discover the other files. Requires
    /// `--package-name`; replaces the input file.
    #[arg(long = "package-root")]
    package_root: Option<PathBuf>,

    /// The package's name — the first segment of every item's module path
    /// (and of its mangled symbols). With an input file, that file becomes
    /// the package's crate root and its `mod` declarations discover sibling
    /// files; without one, `--package-root` names the root directory. `core`
    /// is reserved for the built-in core package.
    #[arg(long = "package-name")]
    package_name: Option<String>,

    /// Instead of compiling, describe the package's source graph — `lib.rr`
    /// plus everything reachable through `mod` declarations — as JSON:
    /// `{"package": …, "files": [{"path": …, "module": […]}]}`, with files in
    /// discovery order and paths canonical. Requires package mode; the JSON
    /// goes to `-o` (stdout by default).
    #[arg(long = "scan-deps")]
    scan_deps: bool,

    /// Output file (`-` writes a text dump — `hir`/`mir`/`mlir`/`mlir-llvm` — to
    /// stdout). The target stage is inferred from its extension unless `--emit`
    /// is given. Required except with `--scan-deps`.
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

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

    /// Relocation model: `default`, `pic`, `static`, or `dynamic-no-pic`.
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

    /// `rustc` used to compile polymorphic-FFI textures. A bare name resolves
    /// through `PATH` (the host-toolchain workflow: `--polyffi-rust-path
    /// rustc`); takes precedence over the `REUSSIR_RUSTC` environment
    /// variable and the built-in probe list.
    #[arg(long = "polyffi-rust-path", value_name = "PATH")]
    polyffi_rust_path: Option<PathBuf>,

    /// Directory searched for the Rust packages polymorphic-FFI textures
    /// link against (`rustc -L`) — `libreussir_rt` and friends. Repeatable;
    /// each directory becomes its own `-L`. Takes precedence over the
    /// `REUSSIR_RUSTC_DEPS` environment variable (itself a path-separated
    /// list) and the built-in probe list.
    #[arg(long = "polyffi-libdir", value_name = "DIR")]
    polyffi_libdir: Vec<PathBuf>,

    /// Let the token-reuse pass reuse tokens across function calls.
    #[arg(long = "reuse-across-call")]
    reuse_across_call: bool,

    /// Disable whole-program devirtualization of closures. WPD tags every
    /// closure vtable with its return-type family id, asserts it at the
    /// indirect eval/clone/drop call sites, and lets the backend's
    /// speculative WholeProgramDevirt fold single-implementation dispatches
    /// into guarded direct calls (direct call that inlines on the expected
    /// vtable, indirect fallback otherwise — sound on any world). Enabled by
    /// default at `-O aggressive` and `-O size`.
    #[arg(long = "no-closure-wpd")]
    no_closure_wpd: bool,

    /// Run a transform-dialect script at a named pipeline anchor:
    /// `<file.mlir>[@<anchor>]`, where `<anchor>` is `entry` (pure Reussir IR,
    /// before any pipeline pass) or `kernel` (loop nests as memref/scf/arith,
    /// before the LLVM descent; the default). Repeatable. The script must
    /// define `transform.named_sequence @__reussir_anchor_<anchor>` in a
    /// module with the `transform.with_named_sequence` attribute.
    #[arg(long = "transform-script")]
    transform_script: Vec<String>,

    /// Lay out compound record members in declaration order instead of the
    /// default packed order (members sorted by descending storage alignment,
    /// eliminating inter-member padding). Packing is the shipped default and
    /// the layout contract; this restores the unpacked layout.
    #[arg(long = "no-pack-record-members")]
    no_pack_record_members: bool,

    /// Run the MLIR backend single-threaded (disable its thread pool). Useful for
    /// deterministic diagnostics and for debugging under tools that dislike the
    /// backend's worker threads (e.g. some sanitizers/debuggers).
    #[arg(long = "disable-backend-multithreading")]
    disable_backend_multithreading: bool,

    /// Declare the sanitizers this build targets (repeatable). Every emitted
    /// function definition — including backend-outlined closure bodies and
    /// drop/acquire glue — is annotated with the matching LLVM function
    /// attribute (`sanitize_address`/`sanitize_memory`/`sanitize_thread`), so
    /// a downstream `clang -fsanitize=… -x ir` compile actually instruments
    /// the generated code: LLVM's sanitizer passes skip plain memory accesses
    /// in unannotated functions, while atomics and allocator interceptors
    /// stay visible either way — deceptively so. `leak` and `undefined` are
    /// accepted for uniformity but carry no function attribute (leak checking
    /// is a pure runtime mechanism; UBSan checks are emitted by the C/C++
    /// frontend, not an attribute-driven LLVM pass). The shadow-based kinds
    /// (`address`/`memory`/`thread`) require an explicit untagged
    /// `--nullary-variant-encoding` (`arch-independent` or `boxed`): a
    /// TBI-tagged immediate's top byte breaks shadow-address arithmetic.
    #[arg(long = "sanitizer", value_enum)]
    sanitizer: Vec<SanitizerCli>,

    /// Emit DWARF debug info (source locations, function/variable debug types).
    #[arg(short = 'g', long = "debug")]
    debug: bool,

    /// How nullary enum variants are represented. `arch-dependent` picks
    /// the best encoding the target supports: on aarch64 the TBI form (top
    /// byte = tag + 1, low bits = a per-tag dummy box; hardware top-byte
    /// ignore masks stray dereferences, and foreign code may decode the tag
    /// from the pointer without dereferencing); elsewhere it falls back to
    /// the arch-independent form. `arch-independent` uses no TBI/LAM-style
    /// pointer tricks on any target: the immediate is the dummy box address
    /// itself, with an immortal refcount — foreign code sees a
    /// layout-compatible box. `boxed` keeps the legacy heap-boxed layout.
    #[arg(long = "nullary-variant-encoding", value_enum,
          default_value_t = VariantEncoding::ArchDependent)]
    nullary_variant_encoding: VariantEncoding,

    /// Split codegen into this many units. Each function's body is emitted in
    /// exactly one unit (a stable hash of its mangled symbol picks which);
    /// with N > 1 the outputs are `<stem>.<i>.<ext>` siblings of `-o` and
    /// private functions get external visibility (cross-unit calls resolve at
    /// link time).
    #[arg(long = "codegen-units", default_value = "1")]
    codegen_units: u32,

    /// Omit source locations (the file table and `[start..end]` spans) from
    /// `hir`/`mir` text dumps. The default dump is lossless — it round-trips
    /// spans and file attribution — but structural readers (FileCheck tests,
    /// quick inspection) may prefer the bare program.
    #[arg(long = "no-source-locations")]
    no_source_locations: bool,

    /// Log the lowering/backend `tracing` events (to stderr) at DEBUG level.
    /// `RUST_LOG`, if set, takes precedence over this.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

/// The stage the input enters at: `--from` if given, else the extension. `-`
/// (stdin) has no extension, so it defaults to source.
fn resolve_input_stage(cli: &Cli, input: &Path) -> Result<Stage, String> {
    let stage = if let Some(from) = cli.from {
        from
    } else if input.as_os_str() == "-" {
        Stage::Rr
    } else {
        Stage::from_extension(input).ok_or_else(|| {
            format!(
                "cannot infer the input stage of `{}`; pass --from",
                input.display()
            )
        })?
    };
    if !stage.is_input() {
        return Err(format!(
            "`{stage}` is an output-only stage; it cannot be read as input"
        ));
    }
    Ok(stage)
}

/// The stage the driver stops at: `--emit` if given, else the output extension,
/// else an object file.
fn resolve_target(cli: &Cli) -> Result<Stage, String> {
    let stage = cli
        .emit
        .or_else(|| cli.output.as_deref().and_then(Stage::from_extension))
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

/// Parses the `--transform-script` specs: `<file>[@<anchor>]`, anchor
/// defaulting to `kernel`. The anchor is whatever follows the *last* `@`, so
/// a path may itself contain `@` as long as an explicit anchor is given.
fn parse_transform_scripts(specs: &[String]) -> Result<Vec<(Anchor, PathBuf)>, String> {
    specs
        .iter()
        .map(|spec| match spec.rsplit_once('@') {
            Some((path, anchor)) => match Anchor::parse(anchor) {
                Some(anchor) => Ok((anchor, PathBuf::from(path))),
                None => Err(format!(
                    "unknown anchor `{anchor}` in `--transform-script {spec}`: \
                     expected `entry` or `kernel` (append `@kernel` explicitly \
                     if the file name itself contains `@`)"
                )),
            },
            None => Ok((Anchor::Kernel, PathBuf::from(spec))),
        })
        .collect()
}

fn parse_reloc(s: &str) -> Result<RelocMode, String> {
    match s {
        "default" => Ok(RelocMode::Default),
        "pic" => Ok(RelocMode::Pic),
        "static" => Ok(RelocMode::Static),
        "dynamic-no-pic" => Ok(RelocMode::DynamicNoPic),
        other => Err(format!("unknown --relocation-mode `{other}`")),
    }
}

/// Read the input into a fresh source cache; the input becomes
/// [`FileId::ROOT`].
fn read_input(path: &PathBuf) -> Result<SourceCache, String> {
    use std::io::Read;
    let mut cache = SourceCache::new();
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        cache.add_virtual("<stdin>", buf);
    } else {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        cache.add_file(path, text);
    }
    Ok(cache)
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

/// What the front (arena-scoped) leg produces: a text dump for `hir`/`mir`, or an
/// MLIR module for `mlir` and anything past it. The module borrows the MLIR
/// context, not the type arena, so it outlives the [`in_arena`] scope.
enum Produced<'c> {
    Text(String),
    Module(Module<'c>),
    /// One module per codegen unit (`--codegen-units` > 1), in unit order.
    Units(Vec<Module<'c>>),
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
    // `palc` surfaces `--help` as a parse *error* rendered to stderr with a
    // non-zero exit. Restore the usual CLI convention: help goes to stdout and
    // exits 0 (a `--help | grep …` pipeline must not fail). Other parse errors
    // stay on stderr with the usage exit code (2).
    let cli = match Cli::try_parse_from(std::env::args_os()) {
        Ok(cli) => cli,
        Err(err) => match err.try_into_help() {
            Ok(help) => {
                println!("{help}");
                return ExitCode::SUCCESS;
            }
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        },
    };
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

/// Where a package compilation's crate root comes from: `lib.rr` under a
/// `--package-root` directory, or the input file itself when `--package-name`
/// accompanies a positional input.
enum PackageRoot {
    Dir(PathBuf),
    File(PathBuf),
}

fn run(cli: &Cli) -> Result<bool, String> {
    let package = match (&cli.package_root, &cli.package_name, &cli.input) {
        (Some(_), None, _) => return Err("--package-root requires --package-name".into()),
        (Some(_), Some(_), Some(_)) => {
            return Err("give either an input file or --package-root, not both".into());
        }
        (Some(root), Some(name), None) => Some((PackageRoot::Dir(root.clone()), name.clone())),
        (None, Some(name), Some(input)) => {
            // The crate root anchors `mod` discovery on disk, so it must be a
            // real file.
            if input.as_os_str() == "-" {
                return Err(
                    "--package-name roots the package at the input file; stdin has no \
                     location for `mod` discovery"
                        .into(),
                );
            }
            Some((PackageRoot::File(input.clone()), name.clone()))
        }
        (None, Some(_), None) => {
            return Err(
                "--package-name needs a crate root: an input file or --package-root".into(),
            );
        }
        (None, None, _) => None,
    };

    // `--scan-deps` stops after discovery: list the package's source graph
    // instead of compiling.
    if cli.scan_deps {
        return scan_deps(cli, package.as_ref());
    }

    let Some(output) = cli.output.as_deref() else {
        return Err("no output file given; pass -o".into());
    };
    let target = resolve_target(cli)?;
    // Only the text dumps stream to stdout; `llvm-ir`/`asm`/`obj` go through the
    // file-based LLVM emitter, so `-o -` would write a file literally named `-`.
    if output.as_os_str() == "-" && target.output_kind().is_some() {
        return Err(format!(
            "cannot write `{target}` to stdout; give a file path with -o"
        ));
    }
    if cli.codegen_units == 0 {
        return Err("--codegen-units must be at least 1".into());
    }
    // Shadow-based sanitizers (ASan/MSan/TSan) compute a shadow address
    // arithmetically from the pointer; a TBI-tagged nullary immediate's top
    // byte — which the hardware ignores on the data access — bleeds into
    // that arithmetic and lands the shadow lookup in unmapped space. The
    // `arch-dependent` default may resolve to the TBI encoding, so require
    // an explicit untagged choice rather than silently downgrading.
    if cli.sanitizer.iter().any(|s| {
        matches!(
            s,
            SanitizerCli::Address | SanitizerCli::Memory | SanitizerCli::Thread
        )
    }) && cli.nullary_variant_encoding == VariantEncoding::ArchDependent
    {
        return Err(
            "--sanitizer address/memory/thread cannot be combined with the (default) \
             `arch-dependent` nullary-variant encoding: on aarch64 it resolves to \
             TBI-tagged immediates, whose tag byte breaks the sanitizers' shadow-address \
             arithmetic; pass --nullary-variant-encoding arch-independent (or boxed)"
                .into(),
        );
    }
    if cli.codegen_units > 1 && output.as_os_str() == "-" {
        return Err(
            "--codegen-units > 1 writes one file per unit; give a file path with -o".into(),
        );
    }
    let opt = parse_opt(&cli.opt)?;
    let reloc = parse_reloc(&cli.relocation_mode)?;
    let spec = TargetSpec {
        triple: cli.target_triple.clone(),
        cpu: cli.target_cpu.clone(),
        features: cli.target_features.clone(),
    };

    // Package mode: discover the files from the crate root's `mod`
    // declarations and elaborate the whole package as one translation unit.
    if let Some((root, pkg_name)) = package {
        if cli.from.is_some_and(|f| f != Stage::Rr) {
            return Err("a package is always Reussir source; --from does not apply".into());
        }
        if let PackageRoot::File(input) = &root {
            // A rooted input also carries an extension; make a non-source one
            // (`.hir`, `.mir`, …) fail like `--from` instead of parsing as source.
            if resolve_input_stage(cli, input)? != Stage::Rr {
                return Err(
                    "a package is always Reussir source; the crate root must be a `.rr` file"
                        .into(),
                );
            }
        }
        let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
        let pkg = match load_package_or_render(&root, &pkg_name, &interner) {
            Ok(pkg) => pkg,
            Err(msg) if msg.is_empty() => return Ok(false),
            Err(msg) => return Err(msg),
        };
        let name = pkg.cache.name(FileId::ROOT).to_owned();
        let context = reussir_backend::context();
        if cli.disable_backend_multithreading {
            context.enable_multi_threading(false);
        }
        let produced =
            match in_arena(|tcx| frontend_package(&context, tcx, target, &pkg, &interner, cli)) {
                Ok(produced) => produced,
                Err(msg) => {
                    if !msg.is_empty() {
                        eprintln!("{msg}");
                    }
                    return Ok(false);
                }
            };
        return backend(cli, &context, produced, target, opt, reloc, &spec, &name, output);
    }

    let Some(input) = &cli.input else {
        return Err("no input file (or --package-root/--package-name) given".into());
    };
    let input_stage = resolve_input_stage(cli, input)?;
    if target < input_stage {
        return Err(format!(
            "cannot emit `{target}` from a `{input_stage}` input: the pipeline only runs forward"
        ));
    }
    let sources = read_input(input)?;
    let name = sources.name(FileId::ROOT).to_owned();
    let source = sources.source(FileId::ROOT);

    if cli.codegen_units > 1 && matches!(input_stage, Stage::Mlir | Stage::LlvmIr) {
        return Err(format!(
            "--codegen-units applies while lowering to MLIR; a `{input_stage}` input is already a single unit"
        ));
    }
    // A `.ll` input skips the whole MLIR front: parse the IR and run the LLVM
    // backend straight to the requested artifact.
    if input_stage == Stage::LlvmIr {
        let kind = target
            .output_kind()
            .ok_or_else(|| format!("cannot emit `{target}` from LLVM IR"))?;
        let machine = TargetMachine::new(&spec, opt, reloc)?;
        let finalized = parse_llvm_ir(&name, source)?;
        emit_to_file(finalized, &machine, opt, kind, output)?;
        return Ok(true);
    }

    let context = reussir_backend::context();
    if cli.disable_backend_multithreading {
        context.enable_multi_threading(false);
    }
    // Front leg: reach an MLIR module (or a text dump for `hir`/`mir`). A `.mlir`
    // input parses directly; source/`.hir`/`.mir` run the frontend in the arena.
    let produced = match input_stage {
        Stage::Mlir => Module::parse(&context, source)
            .map(Produced::Module)
            .ok_or_else(|| format!("{name}: failed to parse MLIR module"))?,
        _ => match in_arena(|tcx| frontend(&context, tcx, input_stage, target, &sources, cli)) {
            Ok(produced) => produced,
            Err(msg) => {
                if !msg.is_empty() {
                    eprintln!("{msg}");
                }
                return Ok(false);
            }
        },
    };
    backend(cli, &context, produced, target, opt, reloc, &spec, &name, output)
}

/// Discover and parse the package from its crate root — `lib.rr` under a
/// root directory, or the rooting input file itself — rendering syntax errors
/// through the same ariadne path as compile diagnostics.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn load_package_or_render(
    root: &PackageRoot,
    name: &str,
    interner: &std::sync::Arc<reussir_syntax::MultiThreadedTokenInterner>,
) -> Result<package::PackageSource, String> {
    let loaded = match root {
        PackageRoot::Dir(dir) => package::load_package(dir, name, interner),
        PackageRoot::File(file) => package::load_package_rooted(file, name, interner),
    };
    match loaded {
        Ok(pkg) => Ok(pkg),
        Err(package::PackageError::Message(msg)) => Err(msg),
        Err(package::PackageError::ParseErrors {
            cache,
            file,
            errors,
        }) => {
            let color = std::io::stderr().is_terminal();
            let _ = diagnostics::render_errors(&cache, file, &errors, color, std::io::stderr().lock());
            Err(String::new())
        }
    }
}

/// `--scan-deps`: run package discovery — which parses every file to follow
/// its `mod` declarations — and describe the source graph as JSON instead of
/// compiling: the package name and, in discovery order, every file's
/// canonical path and module path.
fn scan_deps(cli: &Cli, package: Option<&(PackageRoot, String)>) -> Result<bool, String> {
    let Some((root, pkg_name)) = package else {
        return Err(
            "--scan-deps requires package mode: --package-root, or an input file rooted \
             by --package-name"
                .into(),
        );
    };
    let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
    let pkg = match load_package_or_render(root, pkg_name, &interner) {
        Ok(pkg) => pkg,
        Err(msg) if msg.is_empty() => return Ok(false),
        Err(msg) => return Err(msg),
    };
    let files: Vec<serde_json::Value> = pkg
        .files
        .iter()
        .map(|file| {
            serde_json::json!({
                "path": pkg.cache.name(file.file),
                "module": file.module,
            })
        })
        .collect();
    let graph = serde_json::json!({ "package": pkg_name, "files": files });
    let mut text = serde_json::to_string_pretty(&graph)
        .map_err(|e| format!("failed to encode the source graph: {e}"))?;
    text.push('\n');
    write_text(cli.output.as_deref().unwrap_or(Path::new("-")), &text)?;
    Ok(true)
}

/// The shared back leg from a produced text/module: write a text dump, or run
/// the MLIR lowering pipeline and emit the requested artifact.
#[allow(clippy::too_many_arguments)]
fn backend(
    cli: &Cli,
    context: &reussir_backend::melior::Context,
    produced: Produced<'_>,
    target: Stage,
    opt: OptLevel,
    reloc: RelocMode,
    spec: &TargetSpec,
    name: &str,
    output: &Path,
) -> Result<bool, String> {
    let module = match produced {
        Produced::Text(text) => {
            write_text(output, &text)?;
            return Ok(true);
        }
        // A partitioned build runs the whole back leg once per unit, writing
        // `<stem>.<i>.<ext>` siblings of `-o`.
        Produced::Units(units) => {
            for (index, module) in units.into_iter().enumerate() {
                let out = unit_output(output, index);
                backend_module(cli, context, module, target, opt, reloc, spec, name, &out)?;
            }
            return Ok(true);
        }
        Produced::Module(module) => module,
    };

    backend_module(
        cli, context, module, target, opt, reloc, spec, name, output,
    )
}

/// The back leg for one module, writing to `output` (per-unit paths in a
/// partitioned build; `-o` itself otherwise).
#[allow(clippy::too_many_arguments)]
/// Resolve the explicit polyffi toolchain locations from the command line.
/// Both are validated here so a typo fails with a clear driver diagnostic
/// instead of a texture-compile error deep in the pipeline.
fn polyffi_paths(cli: &Cli) -> Result<PolyffiPaths, String> {
    let rust_path = cli
        .polyffi_rust_path
        .as_deref()
        .map(resolve_rustc)
        .transpose()?;
    let libdirs = cli
        .polyffi_libdir
        .iter()
        .map(|dir| {
            if !dir.is_dir() {
                return Err(format!(
                    "--polyffi-libdir `{}` is not a directory",
                    dir.display()
                ));
            }
            Ok(dir.to_string_lossy().into_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PolyffiPaths { rust_path, libdirs })
}

/// A bare `--polyffi-rust-path` name (no separator) searches `PATH`; anything
/// else is used as given. Either way the executable must exist.
fn resolve_rustc(path: &Path) -> Result<String, String> {
    let is_bare = path.components().count() == 1 && !path.is_absolute();
    if is_bare && !path.exists() {
        let name = path.as_os_str().to_owned();
        let found = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths).find_map(|dir| {
                let candidate = dir.join(&name);
                if candidate.is_file() {
                    return Some(candidate);
                }
                if cfg!(windows) {
                    let with_exe = dir.join(format!("{}.exe", name.to_string_lossy()));
                    if with_exe.is_file() {
                        return Some(with_exe);
                    }
                }
                None
            })
        });
        if let Some(found) = found {
            return Ok(found.to_string_lossy().into_owned());
        }
    }
    if !path.is_file() {
        return Err(format!(
            "--polyffi-rust-path `{}` does not exist",
            path.display()
        ));
    }
    Ok(path.to_string_lossy().into_owned())
}

fn backend_module(
    cli: &Cli,
    context: &reussir_backend::melior::Context,
    mut module: Module<'_>,
    target: Stage,
    opt: OptLevel,
    reloc: RelocMode,
    spec: &TargetSpec,
    name: &str,
    output: &Path,
) -> Result<bool, String> {
    // A `.mlir` dump is the module as it stands, before the lowering pipeline.
    if target == Stage::Mlir {
        write_text(output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    // MLIR → LLVM leg: the target machine's data layout feeds the polymorphic-FFI
    // gather, which must run before the pipeline erases those ops, and is
    // stamped on the module so the pipeline's MLIR `DataLayout` queries compute
    // real target sizes and alignments.
    let machine = TargetMachine::new(spec, opt, reloc)?;
    pipeline::attach_target_spec(&module, machine.data_layout(), machine.triple())?;
    // Variant box sizing is a per-type property now: an enum carries
    // `#[repr(fixed)]` (uniform max-arm) or defaults to per-constructor
    // sizing, threaded into the dialect variant record type's `fixed` flag —
    // no module-wide switch.
    // Closure WPD is guarded (speculative WholeProgramDevirt), hence sound
    // regardless of what other modules exist — a vtable this module has never
    // seen just takes the indirect fallback arm. Reserve it for the opt
    // levels that ask for whole-program effort anyway: the artifacts cost
    // compile time, and the type-test intrinsics only lower inside the
    // optimizing backend pipeline. Under multiple codegen units each unit
    // still only devirtualizes the families it can see whole, so the win
    // shrinks but nothing breaks.
    let closure_wpd = !cli.no_closure_wpd && matches!(opt, OptLevel::Aggressive | OptLevel::Size);
    let options = LoweringOptions {
        opt,
        reuse_token_across_call: cli.reuse_across_call,
        nullary_variant_encoding: cli.nullary_variant_encoding.resolve(machine.triple()),
        pack_record_members: !cli.no_pack_record_members,
        closure_wpd,
        transform_scripts: parse_transform_scripts(&cli.transform_script)?,
        ..LoweringOptions::default()
    };
    let optimize_ffi = !matches!(opt, OptLevel::None);
    let prepared = LlvmLowering::prepare(
        &module,
        machine.data_layout(),
        optimize_ffi,
        &polyffi_paths(cli)?,
    )
    .map_err(|e| format!("{name}: {e}"))?;
    pipeline::run_lowering_pipeline(context, &mut module, &options)
        .map_err(|e| format!("lowering pipeline failed: {e:?}"))?;

    // After the pipeline the module is the LLVM dialect; dump it before it is
    // translated out of MLIR. (`prepared`'s `Drop` releases the gathered FFI.)
    if target == Stage::MlirLlvm {
        drop(prepared);
        write_text(output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    let finalized = prepared
        .finish(&module)
        .map_err(|e| format!("{name}: {e}"))?;
    let kind = target
        .output_kind()
        .expect("llvm-ir/asm/obj target past the mlir-llvm stage");
    emit_to_file(finalized, &machine, opt, kind, output)?;
    Ok(true)
}

/// The output path for codegen unit `index`: the unit index slots in before
/// the extension (`out/foo.o` → `out/foo.0.o`; no extension → `foo.0`).
fn unit_output(output: &Path, index: usize) -> PathBuf {
    match output.extension().and_then(|e| e.to_str()) {
        Some(ext) => output.with_extension(format!("{index}.{ext}")),
        None => output.with_extension(index.to_string()),
    }
}

/// The arena-scoped front leg for package mode: elaborate every discovered
/// file as one translation unit, then continue exactly as the single-file
/// source path does.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn frontend_package<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    target: Stage,
    pkg: &package::PackageSource,
    interner: &std::sync::Arc<reussir_syntax::MultiThreadedTokenInterner>,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    use reussir_core::semi::{PackageFile, elaborate_package};

    let name = pkg.cache.name(FileId::ROOT);
    let programs: Vec<surface::Program> = pkg
        .files
        .iter()
        .map(|f| surface::program(&f.parse.root))
        .collect();
    let mut keys = interner.clone();
    let files: Vec<PackageFile> = pkg
        .files
        .iter()
        .zip(&programs)
        .map(|(f, program)| PackageFile {
            file: f.file,
            module: f
                .module
                .iter()
                .map(|seg| reussir_syntax::Interner::get_or_intern(&mut keys, seg))
                .collect(),
            program,
        })
        .collect();
    let elab = elaborate_package(tcx, &files, interner);
    if render_reports(&pkg.cache, &elab.reports) {
        return Err(String::new());
    }
    if target == Stage::Hir {
        let printer = if cli.no_source_locations {
            hir::print::Printer::new(&elab.defs, elab.resolver)
        } else {
            hir::print::Printer::with_sources(&elab.defs, elab.resolver, &pkg.cache)
        }
        .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
        .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports);
        let strings = elab.strings.entries();
        let text = printer.program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
        return Ok(Produced::Text(text));
    }
    let (full, reports) = monomorphize(&elab.mono_input());
    if render_reports(&pkg.cache, &reports) {
        return Err(String::new());
    }
    finish_mir(
        context,
        tcx,
        target,
        name,
        Some(&pkg.cache),
        cli,
        &full,
        &elab.defs,
        elab.resolver,
    )
}

/// The arena-scoped front leg for a source/`.hir`/`.mir` input: run the frontend
/// from the entry stage to `target`, yielding a text dump or a lowered module.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn frontend<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    input: Stage,
    target: Stage,
    sources: &SourceCache,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    let name = sources.name(FileId::ROOT);
    let source = sources.source(FileId::ROOT);
    match input {
        Stage::Rr => {
            let parse = reussir_syntax::parse(source);
            if !parse.ok() {
                let color = std::io::stderr().is_terminal();
                let _ = diagnostics::render_errors(
                    sources,
                    FileId::ROOT,
                    &parse.errors,
                    color,
                    std::io::stderr().lock(),
                );
                return Err(String::new());
            }
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            if render_reports(sources, &elab.reports) {
                return Err(String::new());
            }
            if target == Stage::Hir {
                let printer = if cli.no_source_locations {
                    hir::print::Printer::new(&elab.defs, elab.resolver)
                } else {
                    hir::print::Printer::with_sources(&elab.defs, elab.resolver, sources)
                }
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports);
                let strings = elab.strings.entries();
                let text =
                    printer.program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
                return Ok(Produced::Text(text));
            }
            let (full, reports) = monomorphize(&elab.mono_input());
            if render_reports(sources, &reports) {
                return Err(String::new());
            }
            finish_mir(
                context,
                tcx,
                target,
                name,
                Some(sources),
                cli,
                &full,
                &elab.defs,
                elab.resolver,
            )
        }
        Stage::Hir => {
            let parsed =
                hir::build::parse_program(tcx, source).map_err(|e| format!("{name}: {e}"))?;
            // The dump's file table names the original sources. Rebuild the
            // same dense ids, but delay opening paths until diagnostics or
            // debug locations actually need source text. An old, table-less
            // dump has no locations at all.
            let dump_sources = refetch_sources(&parsed.files);
            if target == Stage::Hir {
                let printer = match &dump_sources {
                    Some(cache) if !cli.no_source_locations => {
                        hir::print::Printer::with_sources(&parsed.defs, &parsed.names, cache)
                    }
                    _ => hir::print::Printer::new(&parsed.defs, &parsed.names),
                }
                .with_transform_metadata(&parsed.transform_anchors, &parsed.transform_scripts)
                .with_ffi_metadata(&parsed.ffi_preludes, &parsed.ffi_imports);
                let text = printer.program(
                    &parsed.funcs,
                    &parsed.strings,
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
                transform_anchors: &parsed.transform_anchors,
                transform_scripts: &parsed.transform_scripts,
                ffi_imports: &parsed.ffi_imports,
                ffi_preludes: &parsed.ffi_preludes,
                strings: parsed.strings.clone(),
            };
            let (full, reports) = monomorphize(&input);
            match &dump_sources {
                Some(cache) => {
                    if render_reports(cache, &reports) {
                        return Err(String::new());
                    }
                }
                None => {
                    if render_reports(sources, &reports) {
                        return Err(String::new());
                    }
                }
            }
            finish_mir(
                context,
                tcx,
                target,
                name,
                dump_sources.as_ref(),
                cli,
                &full,
                &parsed.defs,
                &parsed.names,
            )
        }
        Stage::Mir => {
            let parsed =
                mir::build::parse_program(tcx, source).map_err(|e| format!("{name}: {e}"))?;
            let dump_sources = refetch_sources(&parsed.files);
            finish_mir(
                context,
                tcx,
                target,
                name,
                dump_sources.as_ref(),
                cli,
                &parsed.program,
                &parsed.defs,
                &parsed.names,
            )
        }
        _ => unreachable!("frontend only handles rr/hir/mir inputs"),
    }
}

/// Rebuild a source cache from an IR dump's file table so its spans keep
/// resolving: real paths are registered for lazy loading, while virtual
/// (`<bracketed>`) entries become name-only placeholders. `None` means the dump
/// was printed without locations.
fn refetch_sources(files: &[String]) -> Option<SourceCache> {
    if files.is_empty() {
        return None;
    }
    let mut cache = SourceCache::new();
    for name in files {
        if name.starts_with('<') {
            cache.add_unavailable(name);
        } else {
            cache.add_lazy_file(name);
        }
    }
    Some(cache)
}

/// Return a lowered value, rendering a source-owned lowering failure through
/// the same ariadne path as parser and elaboration diagnostics. A lowering
/// failure without usable source metadata retains the traditional plain-text
/// driver error.
fn lower_or_render<T>(
    result: std::result::Result<T, LoweringError>,
    sources: Option<&SourceCache>,
    name: &str,
) -> Result<T, String> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            if let (Some(sources), Some((file, span))) = (sources, error.source_location())
                && file.index() < sources.len()
            {
                let diagnostic = diagnostics::Diagnostic {
                    file,
                    span: span.map(|span| (span.start, span.end)),
                    severity: diagnostics::Severity::Error,
                    message: error.message(),
                };
                let color = std::io::stderr().is_terminal();
                let _ = diagnostics::render(
                    sources,
                    std::slice::from_ref(&diagnostic),
                    color,
                    std::io::stderr().lock(),
                );
                Err(String::new())
            } else {
                Err(format!("{name}: {error}"))
            }
        }
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
    sources: Option<&SourceCache>,
    cli: &Cli,
    program: &mir::Program<'tcx>,
    defs: &DefTable,
    resolver: &dyn Resolver<TokenKey>,
) -> Result<Produced<'c>, String> {
    if target == Stage::Mir {
        let printer = match sources {
            Some(cache) if !cli.no_source_locations => {
                mir::print::Printer::with_sources(defs, resolver, cache)
            }
            _ => mir::print::Printer::new(defs, resolver),
        };
        return Ok(Produced::Text(printer.program(program)));
    }
    // Names feed variable/function debug info; only meaningful with `-g`.
    let names = cli.debug.then_some(resolver);
    // AOT linkage: internal / linkonce_odr definitions per the policy; the
    // trampolines and pub functions stay the object's external ABI surface.
    let linkage = LinkagePolicy::aot_for_triple(cli.target_triple.as_deref());
    let sanitizers: Vec<Sanitizer> = cli.sanitizer.iter().map(|&s| s.into()).collect();
    if cli.codegen_units > 1 {
        let mut units = Vec::with_capacity(cli.codegen_units as usize);
        for index in 0..cli.codegen_units {
            let unit = CodegenUnit {
                index,
                count: cli.codegen_units,
            };
            let module = lower_or_render(
                lower_unit(
                    context,
                    tcx,
                    program,
                    sources,
                    names,
                    unit,
                    linkage,
                    &sanitizers,
                ),
                sources,
                name,
            )?;
            units.push(module);
        }
        return Ok(Produced::Units(units));
    }
    let module = lower_or_render(
        lower_program(context, tcx, program, sources, names, linkage, &sanitizers),
        sources,
        name,
    )?;
    Ok(Produced::Module(module))
}
