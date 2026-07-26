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
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub package: Package,
    /// Dependencies by name. First stage: file-system paths only.
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
    /// The products `rene build` compiles, by name — the name becomes the
    /// artifact's file stem (`demo` → `demo`/`demo.exe`, `libdemo.so`, …). A
    /// package with no targets builds nothing; `build` still scans and bakes.
    #[serde(default)]
    pub targets: BTreeMap<String, Target>,
    /// Named sets of build knobs, selected with `rene build --profile`. A
    /// profile named like a built-in (`dev`, `release`) *refines* it: fields
    /// it does not set keep the built-in's values.
    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Package {
    /// The package's name — becomes `rrc --package-name`, the first segment
    /// of every item's module path.
    pub name: String,
    pub version: Option<String>,
}

/// One declared dependency. Unknown fields are rejected — a typo'd
/// constraint that silently vanished would change what resolution accepts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dependency {
    /// Directory of the dependency package, relative to the manifest's
    /// directory unless absolute. The only *source* supported today; a
    /// dependency giving no path is reserved for remote registries and is
    /// rejected at load with that explanation.
    pub path: Option<PathBuf>,
    /// The version constraint the resolved dependency must satisfy:
    /// `"1.2.3"` (exact), `"^1.2.3"` (compatible), `"1.2"`/`"1"` (prefix),
    /// or `"*"` (any, the default when absent). Checked by resolution
    /// against the dependency package's declared `package.version`.
    pub version: Option<String>,
}

impl Dependency {
    /// The dependency's source directory. The load-time validation
    /// guarantees this for every manifest that got past [`load`].
    pub fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("validated at load: only path dependencies exist today")
    }
}

/// What a target builds — the three link products `rrc` emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetKind {
    Executable,
    Dynlib,
    Staticlib,
}

impl TargetKind {
    /// The `rrc --emit` value this kind compiles as.
    pub fn emit(self) -> &'static str {
        match self {
            TargetKind::Executable => "executable",
            TargetKind::Dynlib => "dynlib",
            TargetKind::Staticlib => "staticlib",
        }
    }

    /// Whether `rrc` finishes this kind with a link step — where the
    /// link-only knobs (`runtime_linkage`, `linker`, `link_args`) apply.
    pub fn is_linked(self) -> bool {
        matches!(self, TargetKind::Executable | TargetKind::Dynlib)
    }
}

/// A declared build product. Unknown fields are rejected: a typo here would
/// otherwise silently change what gets built.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub kind: TargetKind,
}

/// A named set of build knobs, mirroring `rrc`'s flags one for one — the
/// values are passed through verbatim, so `rrc --help` is the reference for
/// what each accepts, and an invalid one fails there with its diagnostic.
/// Every field is optional; what a profile leaves unset falls back to the
/// built-in it refines (see [`resolve_profile`]), then to `rrc`'s defaults.
/// Unknown fields are rejected: a typo'd knob that silently vanished would
/// change codegen without a trace.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    /// `rrc -O`: `none`, `default`, `aggressive`, or `size`.
    pub opt: Option<String>,
    /// `rrc -g`: emit DWARF debug info.
    pub debug: Option<bool>,
    /// `rrc --lto`: `none`, `thin`, or `fat`.
    pub lto: Option<String>,
    /// `rrc --relocation-mode`. Unset, `rene` passes `pic`: every declared
    /// kind is a link product (or archived into one), and shared objects and
    /// position-independent executables need it.
    pub relocation_mode: Option<String>,
    /// `rrc --codegen-units`.
    pub codegen_units: Option<u32>,
    /// `rrc --runtime-linkage` (linked targets): `static` or `dynamic`.
    pub runtime_linkage: Option<String>,
    /// `rrc --linker` (linked targets); `rene build --linker` overrides it.
    pub linker: Option<PathBuf>,
    /// `rrc --link-arg`, one per element (linked targets).
    #[serde(default)]
    pub link_args: Vec<String>,
    /// `rrc --sanitizer`, one per element.
    #[serde(default)]
    pub sanitizers: Vec<String>,
    /// `rrc --nullary-variant-encoding`.
    pub nullary_variant_encoding: Option<String>,
    /// `rrc --target-triple`.
    pub target_triple: Option<String>,
    /// `rrc --target-cpu`.
    pub target_cpu: Option<String>,
    /// `rrc --target-features`.
    pub target_features: Option<String>,
    /// `rrc --reuse-across-call` when true.
    pub reuse_across_call: Option<bool>,
    /// `rrc --no-closure-wpd` when false (the knob is spelled positively
    /// here; unset leaves rrc's own default).
    pub closure_wpd: Option<bool>,
    /// `rrc --no-pack-record-members` when false (spelled positively here).
    pub pack_record_members: Option<bool>,
    /// Escape hatch: extra `rrc` arguments appended verbatim, for the knobs
    /// this schema does not name (`--transform-script`, …).
    #[serde(default)]
    pub extra_flags: Vec<String>,
}

impl Profile {
    /// This profile with `over`'s set fields winning; list fields
    /// concatenate, this profile's elements first.
    fn refined_by(mut self, over: &Profile) -> Profile {
        macro_rules! take {
            ($($field:ident),*) => {
                $(if over.$field.is_some() {
                    self.$field = over.$field.clone();
                })*
            };
        }
        take!(
            opt,
            debug,
            lto,
            relocation_mode,
            codegen_units,
            runtime_linkage,
            linker,
            nullary_variant_encoding,
            target_triple,
            target_cpu,
            target_features,
            reuse_across_call,
            closure_wpd,
            pack_record_members
        );
        self.link_args.extend(over.link_args.iter().cloned());
        self.sanitizers.extend(over.sanitizers.iter().cloned());
        self.extra_flags.extend(over.extra_flags.iter().cloned());
        self
    }
}

/// The built-in profile of that name, if any: `dev` (unoptimized, debug
/// info — the default profile) and `release` (aggressive optimization).
fn builtin_profile(name: &str) -> Option<Profile> {
    match name {
        "dev" => Some(Profile {
            opt: Some("none".to_owned()),
            debug: Some(true),
            ..Profile::default()
        }),
        "release" => Some(Profile {
            opt: Some("aggressive".to_owned()),
            ..Profile::default()
        }),
        _ => None,
    }
}

/// The profile `name` resolves to: the built-in of that name refined by the
/// manifest's, either alone, and an error when the name is neither — an
/// unknown profile silently meaning "all defaults" would be a typo trap.
pub fn resolve_profile(manifest: &Manifest, name: &str) -> Result<Profile, String> {
    let builtin = builtin_profile(name);
    let declared = manifest.profiles.get(name);
    match (builtin, declared) {
        (Some(base), Some(over)) => Ok(base.refined_by(over)),
        (Some(base), None) => Ok(base),
        (None, Some(profile)) => Ok(profile.clone()),
        (None, None) => Err(format!(
            "no profile named `{name}`: the manifest declares none by that \
             name and it is not a built-in (`dev`, `release`)"
        )),
    }
}

/// A successfully loaded manifest.
#[derive(Debug, Clone)]
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
    for (name, dep) in &manifest.dependencies {
        if dep.path.is_none() {
            return Err(ManifestError::Schema(format!(
                "dependency `{name}` gives no `path`; only local path \
                 dependencies are supported today (a bare version constraint \
                 will later select a registry release)"
            )));
        }
    }

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
              dependencies.utils = { path = "../utils", version = "^0.2" },
              future-field = { anything = [1, 2, 3] },
            }
            "#,
        );

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.manifest.package.name, "mylib");
        assert_eq!(loaded.manifest.package.version, None);
        assert_eq!(
            loaded.manifest.dependencies["utils"].path(),
            Path::new("../utils")
        );
        assert_eq!(
            loaded.manifest.dependencies["utils"].version.as_deref(),
            Some("^0.2")
        );
        assert!(loaded.dump.contains("future-field"));
    }

    /// A bare version constraint names a registry release; until remote
    /// sources exist that is refused with the reason, not a panic later.
    #[test]
    fn a_dependency_without_a_path_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(
            tmp.path(),
            r#"{ package.name = "app", dependencies.utils = { version = "^1.0" } }"#,
        );
        let err = load(&path).unwrap_err().to_string();
        assert!(err.contains("dependency `utils` gives no `path`"), "{err}");
        assert!(err.contains("registry release"), "{err}");
    }

    /// A typo'd constraint field must not silently vanish.
    #[test]
    fn a_dependency_with_unknown_fields_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(
            tmp.path(),
            r#"{ package.name = "app", dependencies.utils = { path = "../u", verison = "1" } }"#,
        );
        let err = load(&path).unwrap_err().to_string();
        assert!(err.contains("verison"), "{err}");
    }

    #[test]
    fn targets_and_profiles_parse_from_the_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(
            tmp.path(),
            r#"
            {
              package = { name = "demo" },
              targets = {
                demo = { kind = 'executable },
                shared = { kind = 'dynlib },
                archive = { kind = 'staticlib },
              },
              profiles = {
                release = { lto = "thin", codegen_units = 4, link_args = ["-lm"] },
                asan = { sanitizers = ["address"], nullary_variant_encoding = "arch-independent" },
              },
            }
            "#,
        );

        let loaded = load(&path).unwrap();
        let m = &loaded.manifest;
        assert_eq!(m.targets["demo"].kind, TargetKind::Executable);
        assert_eq!(m.targets["shared"].kind, TargetKind::Dynlib);
        assert_eq!(m.targets["archive"].kind, TargetKind::Staticlib);
        assert_eq!(m.profiles["release"].lto.as_deref(), Some("thin"));
        assert_eq!(m.profiles["release"].codegen_units, Some(4));
        assert_eq!(m.profiles["release"].link_args, ["-lm"]);
        assert_eq!(m.profiles["asan"].sanitizers, ["address"]);
    }

    /// A profile named like a built-in refines it: what it does not set keeps
    /// the built-in's values. An undeclared, non-built-in name is an error.
    #[test]
    fn profiles_resolve_against_the_builtins() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_manifest(
            tmp.path(),
            r#"
            {
              package = { name = "demo" },
              profiles = {
                release = { lto = "thin" },
                custom = { opt = "size" },
              },
            }
            "#,
        );
        let manifest = load(&path).unwrap().manifest;

        // Pure built-ins.
        let dev = resolve_profile(&manifest, "dev").unwrap();
        assert_eq!(dev.opt.as_deref(), Some("none"));
        assert_eq!(dev.debug, Some(true));

        // Refinement: the manifest's lto lands, the built-in's opt stays.
        let release = resolve_profile(&manifest, "release").unwrap();
        assert_eq!(release.opt.as_deref(), Some("aggressive"));
        assert_eq!(release.lto.as_deref(), Some("thin"));

        // A declared profile with no built-in stands alone.
        let custom = resolve_profile(&manifest, "custom").unwrap();
        assert_eq!(custom.opt.as_deref(), Some("size"));
        assert_eq!(custom.debug, None);

        let err = resolve_profile(&manifest, "bogus").unwrap_err();
        assert!(err.contains("no profile named `bogus`"), "{err}");
    }

    /// A typo'd knob must be a schema error, not a silently ignored field —
    /// it would change codegen without a trace.
    #[test]
    fn unknown_profile_and_target_fields_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        for manifest in [
            r#"{ package = { name = "p" }, profiles = { dev = { optt = "none" } } }"#,
            r#"{ package = { name = "p" }, targets = { t = { kind = 'executable, knid = 1 } } }"#,
            r#"{ package = { name = "p" }, targets = { t = { kind = "exe" } } }"#,
        ] {
            let path = write_manifest(tmp.path(), manifest);
            match load(&path) {
                Err(ManifestError::Schema(_)) => {}
                other => panic!("expected a schema error for {manifest}, got {other:?}"),
            }
        }
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
        let path = write_manifest(tmp.path(), r#"{ package = { name | String = 42 } }"#);

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
