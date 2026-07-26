//! The `rrc` command line: the [`Cli`] surface, the resolution of entry and
//! exit stages from flags and extensions, the small flag parsers, and the
//! input/output plumbing shared by every text stage.

use std::path::{Path, PathBuf};

use palc::Parser;
use reussir_backend::pipeline::{Anchor, OptLevel};
use reussir_codegen::source::SourceCache;

use super::stage::{Lto, RuntimeLinkage, SanitizerCli, Stage, VariantEncoding};
use crate::RelocMode;

/// Reussir compiler driver.
#[derive(Parser)]
#[command(name = "rrc", version)]
pub(crate) struct Cli {
    /// Input file (`-` reads Reussir source from stdin). With
    /// `--package-name`, the input file is the package's crate root; omitted
    /// with `--package-root`.
    pub(crate) input: Option<PathBuf>,

    /// Compile a whole package rooted at this directory: `lib.rr` is the
    /// crate root and `mod` declarations discover the other files. Requires
    /// `--package-name`; replaces the input file.
    #[arg(long = "package-root")]
    pub(crate) package_root: Option<PathBuf>,

    /// The package's name — the first segment of every item's module path
    /// (and of its mangled symbols). With an input file, that file becomes
    /// the package's crate root and its `mod` declarations discover sibling
    /// files; without one, `--package-root` names the root directory. `core`
    /// is reserved for the built-in core package.
    #[arg(long = "package-name")]
    pub(crate) package_name: Option<String>,

    /// Instead of compiling, describe the package's source graph — `lib.rr`
    /// plus everything reachable through `mod` declarations — as JSON:
    /// `{"package": …, "files": [{"path": …, "module": […]}]}`, with files in
    /// discovery order and paths canonical. Requires package mode; the JSON
    /// goes to `-o` (stdout by default).
    #[arg(long = "scan-deps")]
    pub(crate) scan_deps: bool,

    /// Output file (`-` writes a text dump — `hir`/`mir`/`mlir`/`mlir-llvm` — to
    /// stdout). The target stage is inferred from its extension unless `--emit`
    /// is given. Required except with `--scan-deps`.
    #[arg(short = 'o', long)]
    pub(crate) output: Option<PathBuf>,

    /// Stage to emit: `hir`, `mir`, `mlir`, `mlir-llvm`, `llvm-ir`, `asm`,
    /// `obj`, `staticlib`, `dynlib`, or `executable`. Defaults to the output
    /// extension (else `obj`).
    #[arg(short = 't', long = "emit")]
    pub(crate) emit: Option<Stage>,

    /// Build with link-time optimization: `none` (the default) emits finished
    /// native objects, while `thin` and `fat` emit LLVM bitcode built by the
    /// matching LTO pre-link pipeline, so the linker optimizes across every
    /// codegen unit. For `staticlib` the bitcode is archived for the
    /// consumer's linker; for `dynlib` and `executable` the driver's own link
    /// step hands it to the linker, which must carry the LTO plugin.
    #[arg(long = "lto", value_enum, default_value_t = Lto::None)]
    pub(crate) lto: Lto,

    /// How a linked target carries the Reussir runtime: `static` (the
    /// default) bundles the runtime archive into the product; `dynamic` links
    /// the shared runtime library, which must then be found at load time.
    /// Applies to `--emit executable` and `--emit dynlib`.
    #[arg(long = "runtime-linkage", value_enum)]
    pub(crate) runtime_linkage: Option<RuntimeLinkage>,

    /// The linker rustc drives for the link step, passed as `-C linker=`.
    /// Useful where rustc's own discovery resolves the wrong tool — a vcvars
    /// shell whose PATH carries a coreutils `link` ahead of MSVC's. Applies
    /// to `--emit executable` and `--emit dynlib`.
    #[arg(long = "linker", value_name = "PATH")]
    pub(crate) linker: Option<PathBuf>,

    /// Extra argument for the link step, passed through as `-C link-arg=`
    /// (repeatable; appended after the driver's own link inputs). Applies to
    /// `--emit executable` and `--emit dynlib`.
    #[arg(long = "link-arg", value_name = "ARG")]
    pub(crate) link_arg: Vec<String>,

    /// Treat the input as this stage instead of inferring from its extension:
    /// `rr`, `hir`, `mir`, `mlir`, or `llvm-ir`.
    #[arg(short = 'x', long = "from")]
    pub(crate) from: Option<Stage>,

    /// Optimization level: `none`, `default`, `aggressive`, or `size`.
    #[arg(short = 'O', long = "opt", default_value = "default")]
    pub(crate) opt: String,

    /// Relocation model: `default`, `pic`, `static`, or `dynamic-no-pic`.
    #[arg(long = "relocation-mode", default_value = "default")]
    pub(crate) relocation_mode: String,

    /// Target triple to compile for. Defaults to the native host.
    #[arg(long = "target-triple")]
    pub(crate) target_triple: Option<String>,

    /// Target CPU. Defaults to the native host CPU (generic for a custom triple).
    #[arg(long = "target-cpu")]
    pub(crate) target_cpu: Option<String>,

    /// Target features. Defaults to the native host features (none for a custom triple).
    #[arg(long = "target-features")]
    pub(crate) target_features: Option<String>,

    /// A dependency package interface: `NAME=PATH` loads the `.rri` at PATH
    /// under the package name NAME (which must match the interface's own
    /// header). Repeatable; names must be distinct, and neither `core` nor
    /// this compilation's package name. See `docs/design/rri.md`.
    #[arg(long = "extern", value_name = "NAME=PATH")]
    pub(crate) externs: Vec<crate::externs::ExternPair>,

    /// The source root a loaded interface's package-root-relative file table
    /// re-anchors onto: `NAME=DIR` applies to the `--extern` of the same
    /// NAME. Without it that interface's files load as unfetchable virtual
    /// files (locations degrade to name-only).
    #[arg(long = "extern-src", value_name = "NAME=DIR")]
    pub(crate) extern_srcs: Vec<crate::externs::ExternPair>,

    /// `rustc` used to compile polymorphic-FFI textures. A bare name resolves
    /// through `PATH` (the host-toolchain workflow: `--polyffi-rust-path
    /// rustc`); takes precedence over the `REUSSIR_RUSTC` environment
    /// variable and the built-in probe list.
    #[arg(long = "polyffi-rust-path", value_name = "PATH")]
    pub(crate) polyffi_rust_path: Option<PathBuf>,

    /// Directory searched for the Rust packages polymorphic-FFI textures
    /// link against (`rustc -L`) — `libreussir_rt` and friends. Repeatable;
    /// each directory becomes its own `-L`. Takes precedence over the
    /// `REUSSIR_RUSTC_DEPS` environment variable (itself a path-separated
    /// list) and the built-in probe list.
    #[arg(long = "polyffi-libdir", value_name = "DIR")]
    pub(crate) polyffi_libdir: Vec<PathBuf>,

    /// Let the token-reuse pass reuse tokens across function calls.
    #[arg(long = "reuse-across-call")]
    pub(crate) reuse_across_call: bool,

    /// Disable whole-program devirtualization of closures. WPD tags every
    /// closure vtable with its return-type family id, asserts it at the
    /// indirect eval/clone/drop call sites, and lets the backend's
    /// speculative WholeProgramDevirt fold single-implementation dispatches
    /// into guarded direct calls (direct call that inlines on the expected
    /// vtable, indirect fallback otherwise — sound on any world). Enabled by
    /// default at `-O aggressive` and `-O size`.
    #[arg(long = "no-closure-wpd")]
    pub(crate) no_closure_wpd: bool,

    /// Run a transform-dialect script at a named pipeline anchor:
    /// `<file.mlir>[@<anchor>]`, where `<anchor>` is `entry` (pure Reussir IR,
    /// before any pipeline pass) or `kernel` (loop nests as memref/scf/arith,
    /// before the LLVM descent; the default). Repeatable. The script must
    /// define `transform.named_sequence @__reussir_anchor_<anchor>` in a
    /// module with the `transform.with_named_sequence` attribute.
    #[arg(long = "transform-script")]
    pub(crate) transform_script: Vec<String>,

    /// Lay out compound record members in declaration order instead of the
    /// default packed order (members sorted by descending storage alignment,
    /// eliminating inter-member padding). Packing is the shipped default and
    /// the layout contract; this restores the unpacked layout.
    #[arg(long = "no-pack-record-members")]
    pub(crate) no_pack_record_members: bool,

    /// Run the MLIR backend single-threaded (disable its thread pool). Useful for
    /// deterministic diagnostics and for debugging under tools that dislike the
    /// backend's worker threads (e.g. some sanitizers/debuggers).
    #[arg(long = "disable-backend-multithreading")]
    pub(crate) disable_backend_multithreading: bool,

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
    pub(crate) sanitizer: Vec<SanitizerCli>,

    /// Emit DWARF debug info (source locations, function/variable debug types).
    #[arg(short = 'g', long = "debug")]
    pub(crate) debug: bool,

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
    pub(crate) nullary_variant_encoding: VariantEncoding,

    /// Split codegen into this many units. Each function's body is emitted in
    /// exactly one unit (a stable hash of its mangled symbol picks which);
    /// with N > 1 the outputs are `<stem>.<i>.<ext>` siblings of `-o` and
    /// private functions get external visibility (cross-unit calls resolve at
    /// link time).
    #[arg(long = "codegen-units", default_value = "1")]
    pub(crate) codegen_units: u32,

    /// Omit source locations (the file table and `[start..end]` spans) from
    /// `hir`/`mir` text dumps. The default dump is lossless — it round-trips
    /// spans and file attribution — but structural readers (FileCheck tests,
    /// quick inspection) may prefer the bare program.
    #[arg(long = "no-source-locations")]
    pub(crate) no_source_locations: bool,

    /// Log the lowering/backend `tracing` events (to stderr) at DEBUG level.
    /// `RUST_LOG`, if set, takes precedence over this.
    #[arg(short = 'v', long = "verbose")]
    pub(crate) verbose: bool,
}

/// The stage the input enters at: `--from` if given, else the extension. `-`
/// (stdin) has no extension, so it defaults to source.
pub(crate) fn resolve_input_stage(cli: &Cli, input: &Path) -> Result<Stage, String> {
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
pub(crate) fn resolve_target(cli: &Cli) -> Result<Stage, String> {
    let stage = cli
        .emit
        .or_else(|| cli.output.as_deref().and_then(Stage::from_extension))
        .unwrap_or(Stage::Obj);
    if stage == Stage::Rr {
        return Err("`rr` is the source input, not an emittable stage".into());
    }
    Ok(stage)
}

pub(crate) fn parse_opt(s: &str) -> Result<OptLevel, String> {
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
pub(crate) fn parse_transform_scripts(specs: &[String]) -> Result<Vec<(Anchor, PathBuf)>, String> {
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

pub(crate) fn parse_reloc(s: &str) -> Result<RelocMode, String> {
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
pub(crate) fn read_input(path: &PathBuf) -> Result<SourceCache, String> {
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
pub(crate) fn write_text(path: &Path, text: &str) -> Result<(), String> {
    if path.as_os_str() == "-" {
        use std::io::Write;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(|e| format!("failed to write to stdout: {e}"))
    } else {
        std::fs::write(path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }
}

/// Install a `tracing` subscriber writing to **stderr** (so it never corrupts a
/// `-o -` stdout dump). `RUST_LOG` wins if set; otherwise `-v` selects DEBUG and
/// the default is quiet (WARN).
pub(crate) fn init_tracing(verbose: bool) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(if verbose { "debug" } else { "warn" }));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
