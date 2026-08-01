//! `rene new`: scaffolding a fresh package.
//!
//! The scaffold is a directory holding a commented `rene.ncl` manifest, a
//! `src/lib.rr` crate root matching the requested target kinds, and — unless
//! `--vcs none` — a git repository with the build directory ignored. Target
//! kinds combine freely (`--bin --lib --staticlib` is one package with three
//! products over the same source tree), and `--target` records a machine
//! triple as every built-in profile's default so a plain `rene build`
//! cross-compiles.
//!
//! `--interactive` asks about each choice instead, with the flags providing
//! the defaults: a stepped wizard ([`interactive`]) on a real terminal,
//! plain line prompts when stdin or stderr is piped. The generated files are
//! rendered from `templates/*.stpl` (sailfish, compiled in at build time);
//! prompts and progress go to stderr, per the crate's convention that stdout
//! carries only machine-readable output (`new` prints nothing there).

mod interactive;

use std::io::{BufRead, IsTerminal, Write as _};
use std::path::{Path, PathBuf};

use sailfish::TemplateSimple;

use crate::manifest::MANIFEST_FILE;

/// The version-control choice for `rene new --vcs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, palc::ValueEnum)]
pub enum Vcs {
    /// Initialize a git repository (unless already inside one) and write a
    /// `.gitignore` covering the build directory. The default.
    Git,
    /// No version control, no ignore file.
    None,
}

/// What the CLI collected for `new`; [`run`] resolves it into a [`Plan`],
/// interactively when asked.
pub struct Options {
    /// Directory to create the package in.
    pub path: PathBuf,
    /// Explicit package name; defaults to `path`'s final component.
    pub name: Option<String>,
    /// Declare an executable target (the default when no kind is chosen).
    pub bin: bool,
    /// Declare a dynamic library target.
    pub lib: bool,
    /// Declare a static library target.
    pub staticlib: bool,
    /// Machine triple recorded as every profile's default target.
    pub target: Option<String>,
    /// Version control; `None` means the flag was not given (git).
    pub vcs: Option<Vcs>,
    /// Ask about each choice, the flags above as defaults.
    pub interactive: bool,
}

/// The resolved scaffold: every choice pinned, ready to render.
struct Plan {
    name: String,
    version: String,
    bin: bool,
    lib: bool,
    staticlib: bool,
    target: Option<String>,
    vcs: Vcs,
}

/// `templates/rene.ncl.stpl`: the commented manifest.
#[derive(TemplateSimple)]
#[template(path = "rene.ncl.stpl", escape = false)]
struct ManifestTemplate<'a> {
    name: &'a str,
    version: &'a str,
    /// The cross triple, defaulted into both built-in profiles when given.
    target: Option<&'a str>,
    /// The declared targets as `(key, kind)`; empty renders the commented
    /// example instead.
    targets: Vec<(String, &'static str)>,
}

/// `templates/lib.rr.stpl`: the crate root, with a `#[main]` entry when the
/// package declares an executable.
#[derive(TemplateSimple)]
#[template(path = "lib.rr.stpl", escape = false)]
struct RootTemplate<'a> {
    name: &'a str,
    bin: bool,
}

/// `templates/gitignore`: fully static, so no template engine involved.
const GITIGNORE: &str = include_str!("../../templates/gitignore");

/// Create the package described by `opts`.
pub fn run(opts: &Options) -> Result<(), String> {
    let dir = &opts.path;

    // Refuse to write into anything that already has content: scaffolding
    // over an existing package (or any stray files) would silently mix the
    // template into it. An existing *empty* directory is fine. Checked
    // before anything else — an occupied destination is the more
    // fundamental problem than a bad name, and interactive mode must not
    // ask its questions only to fail here.
    if dir.exists() {
        if !dir.is_dir() {
            return Err(format!("`{}` exists and is not a directory", dir.display()));
        }
        let occupied = dir
            .read_dir()
            .map_err(|e| format!("cannot read `{}`: {e}", dir.display()))?
            .next()
            .is_some();
        if occupied {
            return Err(format!(
                "destination `{}` already exists and is not empty",
                dir.display()
            ));
        }
    }

    let plan = resolve(opts)?;
    let manifest = render(
        ManifestTemplate {
            name: &plan.name,
            version: &plan.version,
            target: plan.target.as_deref(),
            targets: targets(&plan),
        },
        "manifest",
    )?;
    let root = render(
        RootTemplate {
            name: &plan.name,
            bin: plan.bin,
        },
        "crate-root",
    )?;

    let src = dir.join("src");
    std::fs::create_dir_all(&src).map_err(|e| format!("cannot create `{}`: {e}", src.display()))?;
    write(&dir.join(MANIFEST_FILE), &manifest)?;
    write(&src.join("lib.rr"), &root)?;
    if plan.vcs == Vcs::Git {
        init_git(dir)?;
    }

    tracing::info!(
        package = %plan.name,
        manifest = %dir.join(MANIFEST_FILE).display(),
        "created; `rene build` from inside the directory builds it"
    );
    Ok(())
}

/// Pin every choice: the flags as given, or — interactively — the answers,
/// with the flags as the offered defaults.
fn resolve(opts: &Options) -> Result<Plan, String> {
    // The directory's own name is the natural default; `--name` overrides
    // it, and interactive mode offers whichever of the two applies.
    let default_name = match &opts.name {
        Some(name) => name.clone(),
        None => opts
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };

    if !opts.interactive {
        check_name(&default_name)?;
        if let Some(triple) = &opts.target {
            check_literal(triple, "target triple")?;
        }
        return Ok(Plan {
            name: default_name,
            version: "0.1.0".to_owned(),
            // No kind requested means an executable, cargo-style.
            bin: opts.bin || !(opts.lib || opts.staticlib),
            lib: opts.lib,
            staticlib: opts.staticlib,
            target: opts.target.clone(),
            vcs: opts.vcs.unwrap_or(Vcs::Git),
        });
    }

    // A real terminal gets the wizard; piped stdin or stderr (scripts,
    // tests) falls back to one plain line prompt per choice.
    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        interactive::wizard(opts, &default_name)
    } else {
        prompt_lines(opts, &default_name)
    }
}

/// The line-prompt fallback: one question per choice on stderr, answers on
/// stdin, the flags (or their defaults) shown as the value an empty answer
/// keeps. Invalid names re-prompt rather than abort: the user is right
/// there.
fn prompt_lines(opts: &Options, default_name: &str) -> Result<Plan, String> {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let name = loop {
        let (answer, eof) = ask(&mut input, "package name", default_name)?;
        match check_name(&answer) {
            Ok(()) => break answer,
            // At EOF the same default would come back forever; fail instead.
            Err(reason) if eof => return Err(reason),
            Err(reason) => eprintln!("{reason}"),
        }
    };
    let version = loop {
        let (answer, eof) = ask(&mut input, "version", "0.1.0")?;
        match check_literal(&answer, "version") {
            Ok(()) => break answer,
            Err(reason) if eof => return Err(reason),
            Err(reason) => eprintln!("{reason}"),
        }
    };
    let no_kind_flag = !(opts.bin || opts.lib || opts.staticlib);
    let bin = confirm(
        &mut input,
        "add an executable target?",
        opts.bin || no_kind_flag,
    )?;
    let lib = confirm(&mut input, "add a dynamic library target?", opts.lib)?;
    let staticlib = confirm(&mut input, "add a static library target?", opts.staticlib)?;
    let target = loop {
        let (answer, eof) = ask(
            &mut input,
            "cross-compile machine target (empty = the host)",
            opts.target.as_deref().unwrap_or(""),
        )?;
        if answer.is_empty() {
            break None;
        }
        match check_literal(&answer, "target triple") {
            Ok(()) => break Some(answer),
            Err(reason) if eof => return Err(reason),
            Err(reason) => eprintln!("{reason}"),
        }
    };
    let git = confirm(
        &mut input,
        "initialize a git repository?",
        opts.vcs.unwrap_or(Vcs::Git) == Vcs::Git,
    )?;
    Ok(Plan {
        name,
        version,
        // All three kinds declined still scaffolds a package: one with no
        // targets, the libdir-printing workflow `build` supports.
        bin,
        lib,
        staticlib,
        target,
        vcs: if git { Vcs::Git } else { Vcs::None },
    })
}

/// Prompt for a line; an empty answer (or EOF) keeps `default`. The flag
/// reports EOF so re-prompting loops can stop instead of spinning on a
/// default that keeps failing validation.
fn ask(input: &mut impl BufRead, question: &str, default: &str) -> Result<(String, bool), String> {
    eprint!("{question} [{default}]: ");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let read = input
        .read_line(&mut line)
        .map_err(|e| format!("cannot read the answer: {e}"))?;
    let answer = line.trim();
    let answer = if answer.is_empty() {
        default.to_owned()
    } else {
        answer.to_owned()
    };
    Ok((answer, read == 0))
}

/// Prompt for yes/no; anything but a y/n answer (including EOF) keeps
/// `default`.
fn confirm(input: &mut impl BufRead, question: &str, default: bool) -> Result<bool, String> {
    let hint = if default { "Y/n" } else { "y/N" };
    let (answer, _) = ask(input, question, hint)?;
    Ok(match answer.to_ascii_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    })
}

/// The package name becomes `rrc --package-name` — the first segment of
/// every item's module path — and is embedded in the manifest as a string
/// literal, so it is held to identifier shape.
fn check_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let lead_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if lead_ok && chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Ok(());
    }
    Err(format!(
        "`{name}` cannot name a package: the name becomes the first segment \
         of every item's module path, so it must be an identifier \
         (`[A-Za-z_][A-Za-z0-9_]*`); pick one with `--name`"
    ))
}

/// Versions and triples are embedded in the manifest as string literals;
/// keep them to characters that cannot escape the quotes (or confuse
/// Nickel's `%{` interpolation).
fn check_literal(value: &str, what: &str) -> Result<(), String> {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '+'))
    {
        return Ok(());
    }
    Err(format!(
        "`{value}` is not a plausible {what}: expected only ASCII letters, \
         digits, `.`, `-`, `_`, or `+`"
    ))
}

/// The declared targets as `(key, kind)` pairs. A single kind claims the
/// package name for its artifact stem; with several kinds over the same
/// sources the executable keeps the bare name and each library key is
/// suffixed, keeping the map keys (and Windows' suffix-less import
/// libraries) apart. The keys are only defaults — the comment in the
/// rendered manifest says they are free to edit.
fn targets(plan: &Plan) -> Vec<(String, &'static str)> {
    let mut targets = Vec::new();
    let single = [plan.bin, plan.lib, plan.staticlib]
        .into_iter()
        .filter(|&kind| kind)
        .count()
        == 1;
    let key = |suffix: &str| {
        if single {
            plan.name.clone()
        } else {
            format!("{}{suffix}", plan.name)
        }
    };
    if plan.bin {
        targets.push((plan.name.clone(), "'executable"));
    }
    if plan.lib {
        targets.push((key("_dylib"), "'dynlib"));
    }
    if plan.staticlib {
        targets.push((key("_static"), "'staticlib"));
    }
    targets
}

/// Set up git: the ignore file always, `git init` only when the scaffold is
/// not already inside a work tree (a package inside an existing repository
/// must not become a nested one). A missing `git` binary degrades to a
/// warning — the scaffold itself is complete without it.
fn init_git(dir: &Path) -> Result<(), String> {
    write(&dir.join(".gitignore"), GITIGNORE)?;
    if dir
        .ancestors()
        .any(|ancestor| ancestor.join(".git").exists())
    {
        tracing::debug!(
            dir = %dir.display(),
            "already inside a git work tree; not initializing another"
        );
        return Ok(());
    }
    match std::process::Command::new("git")
        .arg("init")
        .current_dir(dir)
        .output()
    {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "`git init` failed in `{}`: {}",
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => {
            tracing::warn!(%error, "cannot run `git init`; skipping (`--vcs none` silences this)");
            Ok(())
        }
    }
}

fn write(path: &Path, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents).map_err(|e| format!("cannot write `{}`: {e}", path.display()))
}

/// Render a template, restoring the trailing newline sailfish trims from
/// the template file — the scaffold's files must end in one.
fn render(template: impl TemplateSimple, what: &str) -> Result<String, String> {
    let mut text = template
        .render_once()
        .map_err(|e| format!("cannot render the {what} template: {e}"))?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(dir: &Path) -> Options {
        Options {
            path: dir.to_owned(),
            name: None,
            bin: false,
            lib: false,
            staticlib: false,
            target: None,
            vcs: Some(Vcs::None),
            interactive: false,
        }
    }

    /// The rendered manifest must actually evaluate, and to the shape the
    /// flags asked for — the template and the schema drift apart otherwise.
    #[test]
    fn the_default_scaffold_is_a_loadable_executable_package() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("hello");
        run(&options(&dir)).unwrap();

        let loaded = crate::manifest::load(&dir.join(MANIFEST_FILE)).unwrap();
        assert_eq!(loaded.manifest.package.name, "hello");
        assert_eq!(loaded.manifest.package.version.as_deref(), Some("0.1.0"));
        assert_eq!(
            loaded.manifest.targets["hello"].kind,
            crate::manifest::TargetKind::Executable
        );
        let root = std::fs::read_to_string(dir.join("src/lib.rr")).unwrap();
        assert!(root.contains("#[main]"), "{root}");
        assert!(!dir.join(".gitignore").exists());
    }

    /// Kinds combine; the library keys are suffixed so the map keys stay
    /// unique, and the cross triple lands in both built-in profiles.
    #[test]
    fn combined_kinds_and_a_cross_target_render_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("multi");
        let opts = Options {
            bin: true,
            lib: true,
            staticlib: true,
            target: Some("wasm32-wasip1".to_owned()),
            ..options(&dir)
        };
        run(&opts).unwrap();

        let manifest = crate::manifest::load(&dir.join(MANIFEST_FILE))
            .unwrap()
            .manifest;
        use crate::manifest::TargetKind::*;
        assert_eq!(manifest.targets["multi"].kind, Executable);
        assert_eq!(manifest.targets["multi_dylib"].kind, Dynlib);
        assert_eq!(manifest.targets["multi_static"].kind, Staticlib);
        for profile in ["dev", "release"] {
            let resolved = crate::manifest::resolve_profile(&manifest, profile).unwrap();
            assert_eq!(
                resolved.default_target_triple.as_deref(),
                Some("wasm32-wasip1"),
                "{profile}"
            );
        }
    }

    /// A lone library claims the bare package name for its artifact stem.
    #[test]
    fn a_single_library_keeps_the_package_name() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("only");
        run(&Options {
            lib: true,
            ..options(&dir)
        })
        .unwrap();
        let manifest = crate::manifest::load(&dir.join(MANIFEST_FILE))
            .unwrap()
            .manifest;
        assert_eq!(
            manifest.targets["only"].kind,
            crate::manifest::TargetKind::Dynlib
        );
        // No executable requested, no entry point scaffolded.
        let root = std::fs::read_to_string(dir.join("src/lib.rr")).unwrap();
        assert!(!root.contains("#[main]"), "{root}");
    }

    #[test]
    fn occupied_destinations_and_bad_names_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let occupied = tmp.path().join("taken");
        std::fs::create_dir_all(&occupied).unwrap();
        std::fs::write(occupied.join("keep.txt"), "").unwrap();
        let err = run(&options(&occupied)).unwrap_err();
        assert!(err.contains("not empty"), "{err}");

        let bad = tmp.path().join("my-pkg");
        let err = run(&options(&bad)).unwrap_err();
        assert!(err.contains("--name"), "{err}");
        assert!(!bad.exists(), "a refused scaffold must leave nothing");

        // `--name` rescues a directory whose own name cannot be a package's.
        run(&Options {
            name: Some("my_pkg".to_owned()),
            ..options(&bad)
        })
        .unwrap();
        assert_eq!(
            crate::manifest::load(&bad.join(MANIFEST_FILE))
                .unwrap()
                .manifest
                .package
                .name,
            "my_pkg"
        );
    }
}
