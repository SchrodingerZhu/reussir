//! `rene`: the Reussir package manager driver.
//!
//! `rene build` locates the package manifest (`rene.ncl`), takes the build
//! directory's lock (the status database), records the package's source graph
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

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use rene::db::{self, BuildDir, CleanOutcome};
use rene::{compile, deps, fresh, manifest, plan, resolve, rt};

/// The default build directory name, next to the manifest.
const BUILD_DIR: &str = "reussir-build";

#[derive(Parser)]
#[command(name = "rene", version)]
struct Cli {
    /// Verbose logging (DEBUG level; `RUST_LOG` takes precedence).
    #[arg(short = 'v', long)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
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

    /// Build only this declared target (repeatable). Default: all of them.
    #[arg(long = "target")]
    targets: Vec<String>,

    /// The linker rrc's link step hands to rustc (`rrc --linker`),
    /// overriding the profile's. Useful where rustc's own discovery resolves
    /// the wrong tool.
    #[arg(long)]
    linker: Option<PathBuf>,
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
    init_tracing(cli.verbose);
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
            Command::Build(args) => build(&args).await,
            Command::Clean(args) => clean(&args.location),
            Command::Inspect(args) => inspect(&args).await,
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
fn init_tracing(verbose: bool) {
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

async fn build(args: &BuildArgs) -> Result<(), String> {
    let location = &args.location;
    let manifest_path = locate_manifest(&location.manifest_path)?;
    let loaded = manifest::load(&manifest_path).map_err(|e| e.to_string())?;
    tracing::info!(
        package = %loaded.manifest.package.name,
        manifest = %loaded.path.display(),
        profile = %args.profile,
        "building"
    );
    // Resolve the profile before any work: an unknown name is a usage
    // problem, not something to discover after a runtime bake.
    let profile = manifest::resolve_profile(&loaded.manifest, &args.profile)?;
    // Dependency resolution, before any expensive work too: load the
    // transitive path graph and let pubgrub verify the constraints hold
    // together (each package exists in exactly one version today, so this is
    // a feasibility check; see `resolve`).
    let graph = if loaded.manifest.dependencies.is_empty() {
        None
    } else {
        let graph = resolve::load_graph(&loaded)?;
        let solution = resolve::check(&graph)?;
        for (name, version) in &solution.pinned {
            tracing::info!(package = %name, %version, "resolved");
        }
        Some(graph)
    };

    let root = resolve_build_dir(location)?;
    // Opening the status database takes the build directory's lock; hold it
    // (`dir`) for the rest of the build.
    let dir = BuildDir::open(&root).map_err(|e| e.to_string())?;
    if dir.is_cleaning().map_err(|e| e.to_string())? {
        return Err(pending_clean(&root));
    }
    // The source graph first: it is cheap, it fails fast on a broken package,
    // and a build that finds nothing moved skips straight to the freshness
    // checks below.
    let sources = deps::prepare(&dir, &loaded).await?;
    // The bake links the runtime dylib with the same linker pinning the
    // driver-level links get: the CLI override first, then the profile's.
    let bake_linker = args.linker.clone().or_else(|| profile.linker.clone());
    let artifacts = rt::prepare(&dir, bake_linker.as_deref()).await?;

    // No declared targets: stop after the bake, reporting the libdirs for
    // whoever drives rrc by hand — the pre-target workflow, kept working.
    if loaded.manifest.targets.is_empty() {
        if !sources.rescanned() {
            tracing::info!(files = sources.files.len(), "nothing to do");
        }
        tracing::info!("no targets declared; pass these directories to `rrc --polyffi-libdir`:");
        for libdir in artifacts.libdirs() {
            println!("{}", libdir.display());
        }
        return Ok(());
    }

    let products = compile::build(
        &dir,
        &loaded,
        &sources,
        &artifacts,
        &compile::Options {
            profile_name: args.profile.clone(),
            profile,
            targets: args.targets.clone(),
            linker: args.linker.clone(),
            // The cone's artifact digests enter the fingerprint, so a
            // product consuming a changed upstream artifact re-fingerprints
            // (an absent one digests to a sentinel — see `fresh`).
            upstream: graph
                .as_ref()
                .map(|graph| {
                    rene::fresh::root_upstream_digests(
                        graph,
                        &root.join(&args.profile).join("deps"),
                    )
                })
                .unwrap_or_default(),
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
async fn inspect(args: &InspectArgs) -> Result<(), String> {
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
        let prepared = deps::prepare(&dir, &loaded).await?;
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
        let graph = resolve::load_graph(&loaded)?;
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
                                    "constraint": node.loaded.manifest.dependencies[dep]
                                        .version
                                        .as_deref()
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
            let states = fresh::states(
                &graph,
                &fresh::Context {
                    dir: held.as_ref(),
                    profile_name: &args.profile,
                    profile: &profile,
                    build_dir: &root,
                },
            )?;
            report["plan"] = plan::render(
                &graph,
                &plan::Options {
                    profile_name: &args.profile,
                    profile: &profile,
                    linker: None,
                    build_dir: &root,
                },
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
