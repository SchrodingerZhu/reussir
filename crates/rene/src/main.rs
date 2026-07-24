//! `rene`: the Reussir package manager driver.
//!
//! `rene build` locates the package manifest (`rene.ncl`), takes the build
//! directory's lock (the status database), bakes the bundled `reussir-rt`
//! runtime with the user's Rust toolchain if needed, and prints the library
//! directories to pass to `rrc --polyffi-libdir`. `rene clean` deletes the
//! build directory unless another instance holds it. Compiling the package
//! itself comes in a later stage.
//!
//! Progress goes to stderr via `tracing`; stdout carries only the final
//! machine-readable listing. Exit 0 on success, 1 on a failed command, 2 on
//! a usage error.

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use rene::db::{self, BuildDir, CleanOutcome};
use rene::{manifest, rt};

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
}

/// Build the current package: bake the bundled reussir-rt runtime if needed,
/// then print the library directories to pass to rrc.
#[derive(palc::Args)]
struct BuildArgs {
    #[command(flatten)]
    location: Location,
}

/// Delete the build directory, unless another rene is using it.
#[derive(palc::Args)]
struct CleanArgs {
    #[command(flatten)]
    location: Location,
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
    let result = match cli.command {
        Command::Build(args) => build(&args.location),
        Command::Clean(args) => clean(&args.location),
    };
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
        .unwrap_or_else(|_| EnvFilter::new(if verbose { "debug" } else { "info" }));
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

fn build(location: &Location) -> Result<(), String> {
    let manifest_path = locate_manifest(&location.manifest_path)?;
    let loaded = manifest::load(&manifest_path).map_err(|e| e.to_string())?;
    tracing::info!(
        package = %loaded.manifest.package.name,
        manifest = %loaded.path.display(),
        "building"
    );

    let root = resolve_build_dir(location)?;
    // Opening the status database takes the build directory's lock; hold it
    // (`dir`) for the rest of the build.
    let dir = BuildDir::open(&root).map_err(|e| e.to_string())?;
    if dir.is_cleaning().map_err(|e| e.to_string())? {
        return Err(format!(
            "build directory `{}` has a pending or interrupted clean; \
             run `rene clean` first",
            root.display()
        ));
    }
    let artifacts = rt::prepare(&dir)?;

    tracing::info!("TODO: compiling the package is not implemented yet");
    tracing::info!("pass these directories to `rrc --polyffi-libdir`:");
    for libdir in artifacts.libdirs() {
        println!("{}", libdir.display());
    }
    Ok(())
}

fn clean(location: &Location) -> Result<(), String> {
    let root = resolve_build_dir(location)?;
    match db::clean(&root).map_err(|e| e.to_string())? {
        CleanOutcome::Missing => tracing::info!(dir = %root.display(), "nothing to clean"),
        CleanOutcome::Removed => tracing::info!(dir = %root.display(), "removed"),
    }
    Ok(())
}
