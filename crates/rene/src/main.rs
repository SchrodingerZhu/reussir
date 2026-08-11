//! `rene`: the Reussir package manager driver.
//!
//! `rene new` scaffolds a fresh package: a directory with a commented
//! `rene.ncl`, a `src/lib.rr` crate root for the requested target kinds
//! (`--bin`, `--lib`, `--staticlib`, freely combined; `--target` pins a
//! cross triple), and a git repository with the build directory ignored;
//! `--interactive` asks about each choice instead.
//! `rene build` locates the package manifest (`rene.ncl`), waits for and takes
//! the build directory's lock (the status database), records the package's
//! source graph
//! (re-scanning it through `rrc --scan-deps` only when something moved), bakes
//! the bundled `reussir-rt` runtime with the user's Rust toolchain if needed,
//! and compiles the manifest's declared `targets` under the selected
//! `--profile`, printing the artifact paths. A package declaring no targets
//! stops after the bake and prints the library directories to pass to
//! `rrc --polyffi-libdir` by hand. `rene inspect` reports the source graph as
//! JSON; `rene clean` deletes the build directory unless another instance
//! holds it.
//!
//! Progress goes to stderr via `tracing`; stdout carries only the final
//! machine-readable listing. Exit 0 on success, 1 on a failed command, 2 on
//! a usage error.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use rene::db::{self, BuildDir, CleanOutcome};
use rene::{compile, core_src, deps, exec, fresh, manifest, new, plan, pool, resolve, rt};

/// The default build directory name, next to the manifest.
const BUILD_DIR: &str = "reussir-build";

#[derive(Parser)]
#[command(name = "rene", version)]
struct Cli {
    /// Verbose logging (DEBUG level; `RUST_LOG` takes precedence).
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Output colors: `auto` (when stderr is a terminal), `always`, or
    /// `never`. The resolved policy is forwarded to rrc diagnostics.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto, global = true)]
    color: ColorChoice,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, PartialEq, Eq, palc::ValueEnum)]
enum ColorChoice {
    Auto,
    Always,
    Never,
}

impl ColorChoice {
    fn resolve(self, terminal: bool) -> bool {
        match self {
            ColorChoice::Auto => terminal,
            ColorChoice::Always => true,
            ColorChoice::Never => false,
        }
    }
}

/// Arguments shared by every command that touches the build directory.
#[derive(palc::Args)]
struct Location {
    /// Path to the package manifest. Defaults to the nearest `rene.ncl`
    /// at or above the current directory.
    #[arg(long = "manifest-path")]
    manifest_path: Option<PathBuf>,

    /// The build directory. Defaults to `reussir-build` next to the
    /// manifest.
    #[arg(long = "build-dir")]
    build_dir: Option<PathBuf>,
}

#[derive(palc::Subcommand)]
enum Command {
    Build(BuildArgs),
    Clean(CleanArgs),
    Inspect(InspectArgs),
    New(NewArgs),
}

/// Create a new package: a directory with a commented `rene.ncl` manifest, a
/// `src/lib.rr` crate root, and (by default) a git repository ignoring the
/// build directory.
#[derive(palc::Args)]
struct NewArgs {
    /// Directory to create the package in; its final component names the
    /// package unless `--name` says otherwise.
    path: PathBuf,

    /// The package's name (`rrc --package-name`, the first segment of every
    /// item's module path). Defaults to the directory's name.
    #[arg(long)]
    name: Option<String>,

    /// Declare an executable target — the default when no kind is requested.
    /// Kinds combine: `--bin --lib` declares both products over one source
    /// tree.
    #[arg(long)]
    bin: bool,

    /// Declare a dynamic library target.
    #[arg(long)]
    lib: bool,

    /// Declare a static library target.
    #[arg(long)]
    staticlib: bool,

    /// Record TRIPLE as every profile's default machine target, so a plain
    /// `rene build` cross-compiles (`rene build --target` still overrides).
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// Version control to set up in the new package: `git` (the default)
    /// initializes a repository — unless already inside one — and writes a
    /// `.gitignore`; `none` does neither.
    #[arg(long, value_name = "VCS")]
    vcs: Option<new::Vcs>,

    /// Ask about each choice instead (the flags above become the defaults
    /// an empty answer keeps).
    #[arg(short = 'i', long)]
    interactive: bool,
}

/// Build the current package: bake the bundled reussir-rt runtime if needed,
/// then compile the manifest's declared targets under the selected profile.
/// A package declaring no targets stops after the bake and prints the
/// library directories to pass to rrc by hand.
#[derive(palc::Args)]
struct BuildArgs {
    #[command(flatten)]
    location: Location,

    /// The build profile: a built-in (`dev`, `release`) or one the manifest
    /// declares under `profiles`. Artifacts land in
    /// `<build-dir>/<profile>/`.
    #[arg(long, default_value = "dev")]
    profile: String,

    /// Build only this declared executable target (repeatable). If neither
    /// `--bin` nor `--lib` is given, build every declared target.
    #[arg(long = "bin", value_name = "NAME")]
    bins: Vec<String>,

    /// Build only this declared library target (repeatable). Both `dynlib`
    /// and `staticlib` targets are libraries.
    #[arg(long = "lib", value_name = "NAME")]
    libs: Vec<String>,

    /// Build for this machine target triple. Overrides the selected
    /// profile's `default_target_triple`; otherwise rustc's host is used.
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// The linker rrc's link step hands to rustc (`rrc --linker`),
    /// overriding the profile's. Useful where rustc's own discovery resolves
    /// the wrong tool.
    #[arg(long)]
    linker: Option<PathBuf>,

    /// How many compile processes may run at once (default: the machine's
    /// available parallelism).
    #[arg(short = 'j', long = "jobs")]
    jobs: Option<std::num::NonZeroUsize>,
}

/// Delete the build directory, unless another rene is using it.
#[derive(palc::Args)]
struct CleanArgs {
    #[command(flatten)]
    location: Location,
}

/// Print the package's recorded source graph as JSON: the configuration
/// checksum, whether the record still stands, and every file with its module
/// path, modification time, size, and Blake3 digest.
#[derive(palc::Args)]
struct InspectArgs {
    #[command(flatten)]
    location: Location,

    /// Report the recorded graph as it stands instead of re-scanning a stale
    /// one: no `rrc`, no writes, and no build directory created if there is
    /// none. `state` still says whether the record can be trusted.
    #[arg(long)]
    frozen: bool,

    /// Add a `resolution` section: every package of the transitive
    /// dependency graph pinned to its solved version (the pubgrub
    /// feasibility check must pass).
    #[arg(long)]
    solved: bool,

    /// Add a `graph` section: the transitive dependency graph — each
    /// package's directory, declared version, constraints, and edges.
    #[arg(long)]
    graph: bool,

    /// Add a `plan` section: for every package in dependency-first order,
    /// the `rrc` commands the cross-package build will run (interfaces and
    /// archives for dependencies, the declared targets for the root).
    /// Bake-decided paths appear as `<placeholders>`.
    #[arg(long)]
    commands: bool,

    /// The profile the `plan` section renders against.
    #[arg(long, default_value = "dev")]
    profile: String,

    /// Render commands for this machine target triple. Overrides the
    /// selected profile's `default_target_triple`; otherwise rustc's host
    /// is used.
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// The linker override the compared build ran with (`rene build
    /// --linker`): part of the root products' fingerprints, so freshness
    /// must judge against the same value.
    #[arg(long)]
    linker: Option<PathBuf>,
}

fn main() -> ExitCode {
    // Same convention as `rrc`: `palc` renders `--help` as a parse error, so
    // route help back to stdout with exit 0 and keep real usage errors on
    // stderr with exit 2.
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
    let color = cli.color.resolve(std::io::stderr().is_terminal());
    init_tracing(cli.verbose, color);
    // One completion-based runtime (io_uring/IOCP) drives the whole command:
    // child processes and file reads await on it, and independent work —
    // digests, target compiles — runs concurrently on the single thread.
    let runtime = match compio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("error: cannot start the async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    let result = runtime.block_on(async {
        match cli.command {
            Command::Build(args) => build(&args, color).await,
            Command::Clean(args) => clean(&args.location),
            Command::Inspect(args) => inspect(&args, color).await,
            Command::New(args) => new::run(&new::Options {
                path: args.path,
                name: args.name,
                bin: args.bin,
                lib: args.lib,
                staticlib: args.staticlib,
                target: args.target,
                vcs: args.vcs,
                interactive: args.interactive,
            }),
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Install a `tracing` subscriber writing to **stderr** (stdout carries the
/// machine-readable output). `RUST_LOG` wins if set; otherwise `-v` selects
/// DEBUG and the default shows build progress (INFO).
fn init_tracing(verbose: bool, color: bool) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        // pubgrub narrates every solver step at INFO; that is solver
        // debugging, not build progress, so it stays behind RUST_LOG.
        .unwrap_or_else(|_| {
            EnvFilter::new(if verbose {
                "debug,pubgrub=warn"
            } else {
                "info,pubgrub=warn"
            })
        });
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        // Plain text off-terminal by default; `--color` can explicitly
        // override that for tools which capture and re-display the stream.
        .with_ansi(color)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Resolve the build directory: the flag as given, else `reussir-build` next
/// to the (located) manifest. The manifest is only searched for when needed,
/// so `clean --build-dir …` works outside any package.
fn resolve_build_dir(location: &Location) -> Result<PathBuf, String> {
    if let Some(dir) = &location.build_dir {
        return Ok(dir.clone());
    }
    let manifest = locate_manifest(&location.manifest_path)?;
    Ok(manifest
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(BUILD_DIR))
}

fn locate_manifest(flag: &Option<PathBuf>) -> Result<PathBuf, String> {
    match flag {
        Some(path) => Ok(path.clone()),
        None => {
            let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
            manifest::locate(&cwd).ok_or_else(|| {
                format!(
                    "no `{}` found in `{}` or any parent directory",
                    manifest::MANIFEST_FILE,
                    cwd.display()
                )
            })
        }
    }
}

async fn build(args: &BuildArgs, color: bool) -> Result<(), String> {
    let location = &args.location;
    let manifest_path = locate_manifest(&location.manifest_path)?;
    let loaded = manifest::load(&manifest_path).map_err(|e| e.to_string())?;
    tracing::debug!(
        package = %loaded.manifest.package.name,
        manifest = %loaded.path.display(),
        profile = %args.profile,
        "building"
    );
    // Resolve the profile before any work: an unknown name is a usage
    // problem, not something to discover after a runtime bake.
    let profile = manifest::resolve_profile(&loaded.manifest, &args.profile)?;
    let root = resolve_build_dir(location)?;
    // Builds sharing a directory queue on its database lock. Take it before
    // materializing bundled sources so concurrent invocations serialize all
    // writes, then hold it (`dir`) for the rest of the build.
    let dir = BuildDir::open_wait(&root)
        .await
        .map_err(|e| e.to_string())?;
    if dir.is_cleaning().map_err(|e| e.to_string())? {
        return Err(pending_clean(&root));
    }

    // Dependency resolution, before any expensive compile work: load the
    // transitive path graph and let pubgrub verify the constraints hold
    // together (each package exists in exactly one version today, so this is
    // a feasibility check; see `resolve`). A dependency-less package is a
    // one-node graph through the same machinery.
    let core_dir = core_src::unpack(&root)?;
    let graph = resolve::load_graph_with(&loaded, Some(&core_dir))?;
    if graph.nodes.len() > 1 {
        let solution = resolve::check(&graph)?;
        for (name, version) in &solution.pinned {
            tracing::debug!(package = %name, %version, "resolved");
        }
    }
    // The source graph first: it is cheap, it fails fast on a broken package,
    // and a build that finds nothing moved skips straight to the freshness
    // checks below.
    let sources = deps::prepare(&dir, &loaded, color).await?;
    // The bake links the runtime dylib with the same linker pinning the
    // driver-level links get: the CLI override first, then the profile's.
    // The progress surface exists from here on: the bake's spinner first,
    // the pipeline's bars after.
    let progress = indicatif::MultiProgress::new();
    let bake_linker = args.linker.clone().or_else(|| profile.linker.clone());
    let requested_target = args
        .target
        .as_deref()
        .or(profile.default_target_triple.as_deref());
    let artifacts = rt::prepare(&dir, requested_target, bake_linker.as_deref(), &progress).await?;
    let target = artifacts.target.clone();

    // No declared targets: stop after the bake, reporting the libdirs for
    // whoever drives rrc by hand — the pre-target workflow, kept working.
    if loaded.manifest.targets.is_empty() {
        if !sources.rescanned() {
            tracing::debug!(files = sources.files.len(), "nothing to do");
        }
        tracing::info!("no targets declared; pass these directories to `rrc --polyffi-libdir`:");
        for libdir in artifacts.libdirs() {
            println!("{}", libdir.display());
        }
        return Ok(());
    }

    // One pool for the whole build: `-j` admission over the main event
    // loop, which monitors every child process.
    let pool = pool::Pool::new(args.jobs);

    // The dependency pipeline: every node of the graph but the root, each
    // dispatched the moment its last dependency finishes, each freshness-
    // checked, compiled (interface + archive), and recorded.
    let built = exec::build_deps(
        &dir,
        &graph,
        &artifacts,
        &exec::Options {
            profile_name: &args.profile,
            profile: &profile,
            target: &target,
            linker: args.linker.as_deref(),
            jobs: args.jobs,
            build_dir: &root,
            color,
        },
        &pool,
        &progress,
    )
    .await?;
    if graph.nodes.len() > 1 {
        tracing::debug!(
            built,
            fresh = graph.nodes.len() - 1 - built,
            "dependencies ready"
        );
    }

    let products = compile::build(
        &dir,
        &loaded,
        &sources,
        &artifacts,
        &compile::Options {
            profile_name: args.profile.clone(),
            profile,
            target: target.clone(),
            bins: args.bins.clone(),
            libs: args.libs.clone(),
            linker: args.linker.clone(),
            // The cone's artifact digests enter the fingerprint, so a
            // product consuming a changed upstream artifact re-fingerprints.
            // Computed *after* the dependency pipeline: it hashes the final
            // artifacts the targets will consume.
            upstream: rene::fresh::root_upstream_digests(
                &graph,
                &root.join(&args.profile).join("deps"),
                &target,
            ),
            jobs: args.jobs,
            color,
        },
        &graph,
        &compile::Execution {
            pool: &pool,
            progress: &progress,
        },
    )
    .await?;
    // Stdout carries the artifact listing, one path per target, in the
    // order they were declared (or selected).
    for product in &products {
        println!("{}", product.path.display());
    }
    Ok(())
}

/// `rene inspect`: report the recorded source graph as JSON on stdout. By
/// default a stale record is refreshed first (the same scan `build` runs, but
/// without the runtime bake), so the report describes the package as it is
/// now; `--frozen` reports what is on record and never writes.
async fn inspect(args: &InspectArgs, color: bool) -> Result<(), String> {
    let location = &args.location;
    let frozen = args.frozen;
    let manifest_path = locate_manifest(&location.manifest_path)?;
    let loaded = manifest::load(&manifest_path).map_err(|e| e.to_string())?;
    let root = resolve_build_dir(location)?;
    let hash = deps::config_hash(&loaded.dump);

    // The handle is held for the section rendering below: freshness reads
    // the same records a build would.
    let (state, files, held) = if frozen {
        match BuildDir::open_existing(&root).map_err(|e| e.to_string())? {
            Some(dir) => {
                let state = deps::staleness(&dir, &hash)?;
                let files = dir.sources().map_err(|e| e.to_string())?;
                (state, files, Some(dir))
            }
            // No build directory at all: nothing has been recorded, and
            // `--frozen` must not create one to say so.
            None => (deps::Staleness::Uninitialized, Vec::new(), None),
        }
    } else {
        let dir = BuildDir::open(&root).map_err(|e| e.to_string())?;
        if dir.is_cleaning().map_err(|e| e.to_string())? {
            return Err(pending_clean(&root));
        }
        let prepared = deps::prepare(&dir, &loaded, color).await?;
        // After a refresh the record is by construction current; report that
        // rather than the reason it was rebuilt (which the log already
        // carries).
        (deps::Staleness::UpToDate, prepared.files, Some(dir))
    };

    let mut report = serde_json::json!({
        "package": loaded.manifest.package.name,
        "manifest": loaded.path.display().to_string(),
        "build_dir": root.display().to_string(),
        "config_hash": hash,
        "state": state.tag(),
        "reason": state.to_string(),
        "files": files.iter().map(deps::SourceFile::to_json).collect::<Vec<_>>(),
    });
    // The dependency-graph sections, on demand. One load serves all three;
    // `--solved` additionally requires the feasibility check to hold.
    if args.solved || args.graph || args.commands {
        let core_dir = core_src::unpack(&root)?;
        let graph = resolve::load_graph_with(&loaded, Some(&core_dir))?;
        if args.solved {
            let solution = resolve::check(&graph)?;
            report["resolution"] = solution
                .pinned
                .iter()
                .map(|(name, version)| (name.clone(), version.to_string().into()))
                .collect::<serde_json::Map<String, serde_json::Value>>()
                .into();
        }
        if args.graph {
            report["graph"] = serde_json::json!({
                "root": graph.root,
                "nodes": graph
                    .nodes
                    .iter()
                    .map(|(name, node)| {
                        (name.clone(), serde_json::json!({
                            "dir": node.dir.display().to_string(),
                            "version": node.version.to_string(),
                            "dependencies": node
                                .dependencies
                                .iter()
                                .map(|dep| serde_json::json!({
                                    "package": dep,
                                    "constraint": node
                                        .loaded
                                        .manifest
                                        .dependencies
                                        .get(dep)
                                        .and_then(|d| d.version.as_deref())
                                        .unwrap_or("*"),
                                }))
                                .collect::<Vec<_>>(),
                        }))
                    })
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
            });
        }
        if args.commands {
            let profile = manifest::resolve_profile(&loaded.manifest, &args.profile)?;
            let requested_target = args
                .target
                .as_deref()
                .or(profile.default_target_triple.as_deref());
            let target = rt::resolve_target(requested_target).await?;
            let bake = fresh::recorded_bake(held.as_ref(), &target)?;
            let states = fresh::states(
                &graph,
                &fresh::Context {
                    dir: held.as_ref(),
                    profile_name: &args.profile,
                    profile: &profile,
                    target: &target,
                    linker: args.linker.as_deref(),
                    build_dir: &root,
                },
            )?;
            report["plan"] = plan::render(
                &graph,
                &plan::Options {
                    profile_name: &args.profile,
                    profile: &profile,
                    target: &target,
                    linker: args.linker.as_deref(),
                    build_dir: &root,
                },
                bake.as_ref(),
                Some(&states),
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn pending_clean(root: &std::path::Path) -> String {
    format!(
        "build directory `{}` has a pending or interrupted clean; \
         run `rene clean` first",
        root.display()
    )
}

fn clean(location: &Location) -> Result<(), String> {
    let root = resolve_build_dir(location)?;
    match db::clean(&root).map_err(|e| e.to_string())? {
        CleanOutcome::Missing => tracing::info!(dir = %root.display(), "nothing to clean"),
        CleanOutcome::Removed => tracing::info!(dir = %root.display(), "removed"),
    }
    Ok(())
}
