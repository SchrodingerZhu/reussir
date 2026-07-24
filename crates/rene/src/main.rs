//! `rene`: the Reussir package manager driver.
//!
//! First stage: `rene build` locates the package manifest (`rene.ncl`),
//! evaluates it, and dumps the resulting configuration. The actual build —
//! resolving dependencies, detecting the Rust sysroot for polymorphic FFI,
//! and driving `rrc` — comes in later stages.
//!
//! Exit 0 on success, 1 on a failed command, 2 on a usage error.

use std::path::PathBuf;
use std::process::ExitCode;

use palc::Parser;

use rene::manifest;

#[derive(Parser)]
#[command(name = "rene", version)]
struct Cli {
    /// Verbose logging (DEBUG level; `RUST_LOG` takes precedence).
    #[arg(short = 'v', long)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(palc::Subcommand)]
enum Command {
    /// Build the current package. (Stub: evaluates and dumps the manifest.)
    Build {
        /// Path to the package manifest. Defaults to the nearest `rene.ncl`
        /// at or above the current directory.
        #[arg(long = "manifest-path")]
        manifest_path: Option<PathBuf>,
    },
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
        Command::Build { manifest_path } => build(manifest_path),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Install a `tracing` subscriber writing to **stderr** (stdout belongs to the
/// config dump). `RUST_LOG` wins if set; otherwise `-v` selects DEBUG and the
/// default is quiet (WARN).
fn init_tracing(verbose: bool) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(if verbose { "debug" } else { "warn" }));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn build(manifest_path: Option<PathBuf>) -> Result<(), String> {
    let path = match manifest_path {
        Some(path) => path,
        None => {
            let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
            manifest::locate(&cwd).ok_or_else(|| {
                format!(
                    "no `{}` found in `{}` or any parent directory",
                    manifest::MANIFEST_FILE,
                    cwd.display()
                )
            })?
        }
    };
    let loaded = manifest::load(&path).map_err(|e| e.to_string())?;
    tracing::debug!(
        manifest = %loaded.path.display(),
        package = %loaded.manifest.package.name,
        dependencies = loaded.manifest.dependencies.len(),
        "manifest loaded"
    );
    println!("TODO");
    println!("{}", loaded.dump);
    Ok(())
}
