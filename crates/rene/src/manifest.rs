//! Locating and evaluating the package manifest.
//!
//! The manifest is a Nickel *program*, not a data file: it may import other
//! files, apply contracts, and merge records; `rene` sees only the fully
//! evaluated result. Of that result only the fields in [`Manifest`] are
//! interpreted today — everything else passes through to the dump untouched,
//! so a manifest can carry fields for future `rene` stages (or for other
//! tools) without being rejected.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nickel_lang::{Context, ErrorFormat};
use serde::Deserialize;

/// The manifest file name a package directory is identified by.
pub const MANIFEST_FILE: &str = "rene.ncl";

/// The interpreted portion of an evaluated manifest.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub package: Package,
    /// Dependencies by name. First stage: file-system paths only.
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
}

#[derive(Debug, Deserialize)]
pub struct Package {
    /// The package's name — becomes `rrc --package-name`, the first segment
    /// of every item's module path.
    pub name: String,
    pub version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Dependency {
    /// Directory of the dependency package, relative to the manifest's
    /// directory unless absolute.
    pub path: PathBuf,
}

/// A successfully loaded manifest.
#[derive(Debug)]
pub struct Loaded {
    /// The manifest file the package was loaded from.
    pub path: PathBuf,
    pub manifest: Manifest,
    /// The full evaluated configuration as pretty-printed JSON — including
    /// fields [`Manifest`] does not interpret.
    pub dump: String,
}

/// Why a manifest could not be loaded.
#[derive(Debug)]
pub enum ManifestError {
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    /// Nickel evaluation failed; carries the rendered diagnostics.
    Eval(String),
    /// The evaluated configuration does not match the [`Manifest`] shape.
    Schema(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io { path, error } => {
                write!(f, "cannot read `{}`: {error}", path.display())
            }
            ManifestError::Eval(diagnostics) => {
                write!(f, "failed to evaluate the manifest\n{diagnostics}")
            }
            ManifestError::Schema(msg) => write!(f, "invalid manifest: {msg}"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Find the nearest `rene.ncl` at or above `start`, cargo-style, so `rene`
/// works from any subdirectory of a package.
pub fn locate(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|dir| dir.join(MANIFEST_FILE))
        .find(|candidate| candidate.is_file())
}

/// Evaluate the manifest at `path`.
pub fn load(path: &Path) -> Result<Loaded, ManifestError> {
    tracing::debug!(manifest = %path.display(), "evaluating manifest");
    let source = std::fs::read_to_string(path).map_err(|error| ManifestError::Io {
        path: path.to_owned(),
        error,
    })?;

    // The source name is the real path, so `import` resolves relative to the
    // manifest and diagnostics point at it; the parent directory is also an
    // explicit import root as a fallback.
    let mut ctx = Context::new().with_source_name(path.display().to_string());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        ctx = ctx.with_added_import_paths(vec![dir.as_os_str().to_owned()]);
    }

    // `for_export` honors `not_exported`, letting a manifest keep private
    // helper fields out of the configuration it presents.
    let expr = ctx
        .eval_deep_for_export(&source)
        .map_err(|e| ManifestError::Eval(render(&e)))?;
    let json = ctx
        .expr_to_json(&expr)
        .map_err(|e| ManifestError::Eval(render(&e)))?;
    let manifest: Manifest = expr
        .to_serde()
        .map_err(|e| ManifestError::Schema(e.to_string()))?;

    // Nickel exports compact JSON; re-render pretty for the dump.
    let dump = serde_json::from_str::<serde_json::Value>(&json)
        .and_then(|v| serde_json::to_string_pretty(&v))
        .unwrap_or(json);

    Ok(Loaded {
        path: path.to_owned(),
        manifest,
        dump,
    })
}

/// Render a Nickel error's diagnostics as plain text (no ANSI: the result is
/// embedded in a [`ManifestError`] and may end up in logs or test output).
fn render(error: &nickel_lang::Error) -> String {
    let mut out = Vec::new();
    match error.format(&mut out, ErrorFormat::Text) {
        Ok(()) => String::from_utf8_lossy(&out).trim_end().to_owned(),
        Err(e) => format!("(diagnostics unavailable: {e})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join(MANIFEST_FILE);
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn loads_a_minimal_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(
            tmp.path(),
            r#"{ package = { name = "hello", version = "0.1.0" } }"#,
        );

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.manifest.package.name, "hello");
        assert_eq!(loaded.manifest.package.version.as_deref(), Some("0.1.0"));
        assert!(loaded.manifest.dependencies.is_empty());
        // The dump carries the evaluated config.
        assert!(loaded.dump.contains("\"hello\""));
    }

    #[test]
    fn manifest_is_a_program_with_path_dependencies() {
        let tmp = tempfile::tempdir().unwrap();
        // Nickel evaluation (interpolation, merging) happens before `rene`
        // reads the result, and unknown fields pass through untouched.
        let path = write_manifest(
            tmp.path(),
            r#"
            let base = "my" in
            {
              package = { name = "%{base}lib" },
              dependencies.utils = { path = "../utils" },
              future-field = { anything = [1, 2, 3] },
            }
            "#,
        );

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.manifest.package.name, "mylib");
        assert_eq!(loaded.manifest.package.version, None);
        assert_eq!(
            loaded.manifest.dependencies["utils"].path,
            PathBuf::from("../utils")
        );
        assert!(loaded.dump.contains("future-field"));
    }

    #[test]
    fn locate_walks_up_from_a_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(tmp.path(), r#"{ package = { name = "up" } }"#);
        let sub = tmp.path().join("src").join("nested");
        std::fs::create_dir_all(&sub).unwrap();

        assert_eq!(locate(&sub).as_deref(), Some(&*path));
        assert_eq!(locate(tmp.path()).as_deref(), Some(&*path));
    }

    #[test]
    fn evaluation_errors_carry_diagnostics() {
        let tmp = tempfile::tempdir().unwrap();
        // A contract violation: `name` must be a string.
        let path = write_manifest(
            tmp.path(),
            r#"{ package = { name | String = 42 } }"#,
        );

        match load(&path) {
            Err(ManifestError::Eval(diag)) => assert!(diag.contains("contract")),
            other => panic!("expected an evaluation error, got {other:?}"),
        }
    }

    #[test]
    fn schema_errors_name_the_problem() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(tmp.path(), r#"{ package = { version = "1" } }"#);

        match load(&path) {
            Err(ManifestError::Schema(_)) => {}
            other => panic!("expected a schema error, got {other:?}"),
        }
    }
}
