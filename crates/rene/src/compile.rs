//! Compiling the package's declared targets through `rrc`.
//!
//! One `rrc` invocation per (target, profile): the package rooted at
//! `src/lib.rr`, the profile's knobs as flags, `--emit` from the target's
//! kind, and the baked runtime's libdirs for polyffi and the link step.
//! Artifacts land in `<build-dir>/<profile>/`, named per platform
//! (`demo`/`demo.exe`/`demo.wasm`,
//! `libdemo.so`/`libdemo.dylib`/`demo.dll`,
//! `libdemo.a`/`demo.lib`).
//!
//! Each product's build is recorded in the status database under
//! [`tables::product_key`] with the fingerprint it was built from — the
//! evaluated-config hash (which covers targets and profiles), the source
//! graph's content digests, the toolchain, and the CLI overrides. A build
//! whose fingerprint matches and whose artifact still exists reruns nothing.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};

use serde::{Deserialize, Serialize};

use std::ffi::OsString;

use crate::db::BuildDir;
use crate::deps::{self, Prepared};
use crate::manifest::{Loaded, Profile, TargetKind};
use crate::plan;
use crate::pool::Pool;
use crate::resolve::Graph;
use crate::rt::RtArtifacts;
use crate::tables;

/// What the CLI resolved for this build: the profile in force and the
/// selection/override flags.
pub struct Options {
    /// The profile's name — the artifacts' directory under the build dir,
    /// and half of each product's status key.
    pub profile_name: String,
    /// The resolved profile (built-in refined by the manifest's).
    pub profile: Profile,
    /// The resolved machine target: CLI, profile default, then rustc host.
    pub target: String,
    /// `--bin` filters; empty together with [`Self::libs`] means every
    /// declared target.
    pub bins: Vec<String>,
    /// `--lib` filters; both `dynlib` and `staticlib` are libraries.
    pub libs: Vec<String>,
    /// `rene build --linker`, overriding the profile's.
    pub linker: Option<PathBuf>,
    /// The dependency cone's artifact digests
    /// ([`crate::fresh::upstream_digests`]) — part of the fingerprint, so a
    /// changed upstream artifact re-fingerprints every product consuming it.
    pub upstream: std::collections::BTreeMap<String, String>,
    /// `rene build -j`: the bound on concurrent compile processes.
    pub jobs: Option<std::num::NonZeroUsize>,
    /// Whether compiler diagnostics should retain ANSI styling while they
    /// are buffered away from the terminal.
    pub color: bool,
}

/// A built (or reused) product.
pub struct Product {
    pub name: String,
    pub path: PathBuf,
}

/// One product's recorded build (JSON under [`tables::product_key`]).
#[derive(Serialize, Deserialize)]
pub struct ProductRecord {
    pub fingerprint: String,
    pub path: PathBuf,
}

/// Build the selected targets, reusing whatever the record proves current.
/// The stale ones compile on the process pool — independent `rrc` processes
/// over the same baked runtime, at most `-j` of them at once — and are
/// recorded in declared order once all of them succeed.
pub async fn build(
    dir: &BuildDir,
    loaded: &Loaded,
    sources: &Prepared,
    rt: &RtArtifacts,
    opts: &Options,
    graph: &Graph,
    pool: &Pool,
) -> Result<Vec<Product>, String> {
    let manifest = &loaded.manifest;
    let selected: Vec<&str> = if opts.bins.is_empty() && opts.libs.is_empty() {
        manifest.targets.keys().map(String::as_str).collect()
    } else {
        for name in &opts.bins {
            check_selected_bin(manifest, name)?;
        }
        for name in &opts.libs {
            check_selected_library(manifest, name)?;
        }
        opts.bins
            .iter()
            .chain(&opts.libs)
            .map(String::as_str)
            .collect()
    };

    let out_dir = dir.root().join(&opts.profile_name);
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("cannot create `{}`: {e}", out_dir.display()))?;
    let fingerprint = fingerprint(loaded, &sources.files, rt, opts);

    // Partition into current and stale first, so the stale ones can run
    // concurrently while the listing keeps the declared order.
    let mut plan = Vec::with_capacity(selected.len());
    for name in selected {
        let kind = manifest.targets[name].kind;
        let path = out_dir.join(artifact_file(name, kind, Some(&opts.target)));
        let key = tables::product_key(&opts.profile_name, name);
        let current = product_is_current(dir, &key, &fingerprint, &path)?;
        if current {
            tracing::debug!(target = name, "up to date");
        }
        plan.push((name, kind, path, key, current));
    }
    // The root's commands come from the same synthesis the dump prints
    // (and the dependency pipeline runs): what `inspect --commands` shows
    // is what happens here.
    let mut planned: std::collections::BTreeMap<String, plan::PlannedCommand> =
        plan::node_commands(
            graph,
            &graph.root,
            &plan::Options {
                profile_name: &opts.profile_name,
                profile: &opts.profile,
                target: &opts.target,
                linker: opts.linker.as_deref(),
                build_dir: dir.root(),
            },
            Some(rt),
        )
        .into_iter()
        .filter_map(|command| Some((command.target.clone()?, command)))
        .collect();
    let mut invocations = Vec::new();
    for (name, kind, path, _, current) in &plan {
        if *current {
            continue;
        }
        tracing::debug!(target = name, kind = kind.emit(), out = %path.display(), "compiling");
        let command = planned
            .remove(*name)
            .ok_or_else(|| format!("target `{name}` missing from the plan"))?;
        invocations.push(Invocation::from_planned(command, &rt.rustc, opts.color));
    }
    // Every child owns a private stderr pipe. Wait for all jobs, then replay
    // each complete diagnostic block in declaration order: target compiles
    // remain concurrent, but their ANSI escape sequences can never interleave.
    let completed = futures_util::future::join_all(
        invocations
            .into_iter()
            .map(|invocation| pool.run(move || invocation.run())),
    )
    .await;
    let mut failure = None;
    for result in completed {
        let result = match result {
            Ok(result) => result,
            Err(error) => Err(error),
        };
        if let Err(error) = result.and_then(CompletedInvocation::report)
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let mut products = Vec::with_capacity(plan.len());
    for (name, _, path, key, current) in plan {
        if !current {
            let record = ProductRecord {
                fingerprint: fingerprint.clone(),
                path: path.clone(),
            };
            let record = serde_json::to_string(&record).map_err(|e| e.to_string())?;
            dir.set_status(&[(key.as_str(), record.as_str())])
                .map_err(|e| e.to_string())?;
        }
        products.push(Product {
            name: name.to_owned(),
            path,
        });
    }
    Ok(products)
}

fn check_selected_bin(manifest: &crate::manifest::Manifest, name: &str) -> Result<(), String> {
    let Some(target) = manifest.targets.get(name) else {
        return Err(unknown_selection(manifest, name, "binary", |kind| {
            kind == TargetKind::Executable
        }));
    };
    if target.kind != TargetKind::Executable {
        return Err(format!(
            "target `{name}` is {}, not a binary; select it with `--lib {name}`",
            target.kind.emit()
        ));
    }
    Ok(())
}

fn check_selected_library(manifest: &crate::manifest::Manifest, name: &str) -> Result<(), String> {
    let Some(target) = manifest.targets.get(name) else {
        return Err(unknown_selection(
            manifest,
            name,
            "library",
            TargetKind::is_library,
        ));
    };
    if !target.kind.is_library() {
        return Err(format!(
            "target `{name}` is {}, not a library; select it with `--bin {name}`",
            target.kind.emit()
        ));
    }
    Ok(())
}

fn unknown_selection(
    manifest: &crate::manifest::Manifest,
    name: &str,
    group: &str,
    accepts: impl Fn(TargetKind) -> bool,
) -> String {
    let declared = manifest
        .targets
        .iter()
        .filter(|(_, target)| accepts(target.kind))
        .map(|(name, _)| format!("`{name}`"))
        .collect::<Vec<_>>();
    let suffix = if declared.is_empty() {
        format!("the manifest declares no {group} targets")
    } else {
        format!("the manifest declares {}", declared.join(", "))
    };
    format!("no {group} target named `{name}`: {suffix}")
}

/// One `rrc` compile as owned data — everything a pool worker needs to
/// spawn it, detached from the borrows of the planning pass.
pub(crate) struct Invocation {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Vec<(&'static str, OsString)>,
    /// What the failure message names.
    out: PathBuf,
    /// The parent terminal's resolved color policy. The child writes to a
    /// pipe, so `auto` inside rrc would otherwise discard styling.
    color: bool,
}

impl Invocation {
    /// An `rrc` invocation from a planned command: the real program
    /// resolved, the baking toolchain pinned through `REUSSIR_RUSTC`.
    pub(crate) fn from_planned(
        command: plan::PlannedCommand,
        rustc: &Path,
        color: bool,
    ) -> Invocation {
        Invocation {
            program: deps::resolve_rrc(),
            args: command.args.into_iter().map(OsString::from).collect(),
            // The toolchain that baked the runtime is the one the link step
            // and the polyffi texture compiles must agree with.
            envs: vec![("REUSSIR_RUSTC", rustc.into())],
            out: command.out,
            color,
        }
    }

    /// Spawn and await the process (on whichever runtime polls this), keeping
    /// its diagnostics private until the scheduler can replay them as one
    /// indivisible block.
    pub(crate) async fn run(self) -> Result<CompletedInvocation, String> {
        let mut cmd = compio::process::Command::new(&self.program);
        cmd.args(&self.args);
        // A verbose `rene` runs a verbose `rrc`; its DEBUG phase events join
        // the same private diagnostic block and are replayed intact.
        if tracing::enabled!(tracing::Level::DEBUG) {
            cmd.arg("-v");
        }
        for (key, value) in &self.envs {
            cmd.env(key, value);
        }
        // rrc sees a pipe here, so carry the already-resolved parent policy
        // explicitly. This preserves terminal color without leaking escapes
        // into redirected logs.
        cmd.arg("--color")
            .arg(if self.color { "always" } else { "never" });
        let output = cmd
            .stderr(Stdio::piped())
            .expect("pipe rrc stderr")
            .output()
            .await
            .map_err(|e| format!("cannot run `{}`: {e}", self.program.display()))?;
        Ok(CompletedInvocation {
            status: output.status,
            diagnostics: output.stderr,
            out: self.out,
        })
    }
}

/// One finished compiler process. Diagnostics stay as raw bytes so ANSI
/// sequences and any non-UTF-8 tool output survive the buffering unchanged.
pub(crate) struct CompletedInvocation {
    status: ExitStatus,
    diagnostics: Vec<u8>,
    out: PathBuf,
}

impl CompletedInvocation {
    /// Replay the complete diagnostic block under stderr's process-wide lock,
    /// then turn a failing exit status into rene's target-level error.
    pub(crate) fn report(self) -> Result<(), String> {
        if !self.diagnostics.is_empty() {
            let mut stderr = std::io::stderr().lock();
            stderr
                .write_all(&self.diagnostics)
                .and_then(|()| stderr.flush())
                .map_err(|e| format!("cannot write rrc diagnostics: {e}"))?;
        }
        if !self.status.success() {
            return Err(format!(
                "compiling `{}` failed (see rrc's diagnostics above)",
                self.out.display()
            ));
        }
        Ok(())
    }
}

/// The `rrc` flags a profile expands to, as plain strings — shared by the
/// real invocation above and the planned-command dump ([`crate::plan`]).
pub(crate) fn profile_flags(
    profile: &Profile,
    target: &str,
    kind: TargetKind,
    linker: Option<&Path>,
) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    fn push(args: &mut Vec<String>, flag: &str, value: &str) {
        args.push(flag.to_owned());
        args.push(value.to_owned());
    }
    if let Some(opt) = &profile.opt {
        push(&mut args, "-O", opt);
    }
    if profile.debug == Some(true) {
        args.push("-g".to_owned());
    }
    if let Some(lto) = &profile.lto
        && lto != "none"
    {
        push(&mut args, "--lto", lto);
    }
    // Every declared kind is a link product or archived into one; PIC is the
    // working default, with the profile the override.
    push(
        &mut args,
        "--relocation-mode",
        profile.relocation_mode.as_deref().unwrap_or("pic"),
    );
    if let Some(units) = profile.codegen_units {
        push(&mut args, "--codegen-units", &units.to_string());
    }
    for sanitizer in &profile.sanitizers {
        push(&mut args, "--sanitizer", sanitizer);
    }
    if let Some(encoding) = &profile.nullary_variant_encoding {
        push(&mut args, "--nullary-variant-encoding", encoding);
    }
    push(&mut args, "--target-triple", target);
    if let Some(cpu) = &profile.target_cpu {
        push(&mut args, "--target-cpu", cpu);
    }
    if let Some(features) = &profile.target_features {
        push(&mut args, "--target-features", features);
    }
    if profile.reuse_across_call == Some(true) {
        args.push("--reuse-across-call".to_owned());
    }
    if profile.closure_wpd == Some(false) {
        args.push("--no-closure-wpd".to_owned());
    }
    if profile.pack_record_members == Some(false) {
        args.push("--no-pack-record-members".to_owned());
    }
    // The link-only knobs go to the linked kinds alone, so one profile can
    // serve targets of every kind without tripping rrc's strictness.
    if kind.is_linked() {
        if let Some(linkage) = &profile.runtime_linkage {
            push(&mut args, "--runtime-linkage", linkage);
        }
        if let Some(linker) = linker.or(profile.linker.as_deref()) {
            push(&mut args, "--linker", &linker.display().to_string());
        }
        for arg in &profile.link_args {
            args.push(format!("--link-arg={arg}"));
        }
    }
    args.extend(profile.extra_flags.iter().cloned());
    args
}

/// Everything a product's freshness hangs on, digested: the evaluated
/// manifest (targets and profiles included), the profile *name* (two names
/// may resolve to equal knobs but build into different directories), every
/// source file's content digest, the toolchain, and the CLI linker override.
/// The source-graph paths are covered through the config hash + digests; the
/// artifact's own existence is checked separately.
pub(crate) fn fingerprint(
    loaded: &Loaded,
    files: &[deps::SourceFile],
    rt: &RtArtifacts,
    opts: &Options,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(deps::config_hash(&loaded.dump).as_bytes());
    hasher.update(opts.profile_name.as_bytes());
    hasher.update(opts.target.as_bytes());
    for file in files {
        hasher.update(file.path.as_bytes());
        hasher.update(&file.record.hash);
    }
    for (name, digest) in &opts.upstream {
        hasher.update(name.as_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.update(rt.rustc_version.as_bytes());
    hasher.update(rt.staticlib.display().to_string().as_bytes());
    if let Some(linker) = &opts.linker {
        hasher.update(linker.display().to_string().as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Is the recorded build of this product still the one this build would do?
pub(crate) fn product_is_current(
    dir: &BuildDir,
    key: &str,
    fingerprint: &str,
    path: &Path,
) -> Result<bool, String> {
    let Some(record) = dir.status(key).map_err(|e| e.to_string())? else {
        return Ok(false);
    };
    let Ok(record) = serde_json::from_str::<ProductRecord>(&record) else {
        // An older rene may have written a different shape; just rebuild.
        return Ok(false);
    };
    Ok(record.fingerprint == fingerprint && record.path == path && path.is_file())
}

/// The artifact's file name for `name`, per the target platform — the
/// resolved machine target.
pub(crate) fn artifact_file(name: &str, kind: TargetKind, triple: Option<&str>) -> String {
    let windows = triple.map_or(cfg!(windows), |t| t.contains("windows"));
    let apple = triple.map_or(cfg!(target_vendor = "apple"), |t| t.contains("apple"));
    let wasm = triple.is_some_and(|t| t.starts_with("wasm"));
    match kind {
        TargetKind::Executable if windows => format!("{name}.exe"),
        TargetKind::Executable if wasm => format!("{name}.wasm"),
        TargetKind::Executable => name.to_owned(),
        TargetKind::Dynlib if windows => format!("{name}.dll"),
        TargetKind::Dynlib if apple => format!("lib{name}.dylib"),
        TargetKind::Dynlib if wasm => format!("{name}.wasm"),
        TargetKind::Dynlib => format!("lib{name}.so"),
        TargetKind::Staticlib if windows => format!("{name}.lib"),
        TargetKind::Staticlib => format!("lib{name}.a"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_names_follow_the_target_platform() {
        let linux = Some("x86_64-unknown-linux-gnu");
        let mac = Some("aarch64-apple-darwin");
        let win = Some("x86_64-pc-windows-msvc");
        let wasm = Some("wasm32-wasip1");
        for (kind, expect) in [
            (
                TargetKind::Executable,
                ["demo", "demo", "demo.exe", "demo.wasm"],
            ),
            (
                TargetKind::Dynlib,
                ["libdemo.so", "libdemo.dylib", "demo.dll", "demo.wasm"],
            ),
            (
                TargetKind::Staticlib,
                ["libdemo.a", "libdemo.a", "demo.lib", "libdemo.a"],
            ),
        ] {
            assert_eq!(artifact_file("demo", kind, linux), expect[0]);
            assert_eq!(artifact_file("demo", kind, mac), expect[1]);
            assert_eq!(artifact_file("demo", kind, win), expect[2]);
            assert_eq!(artifact_file("demo", kind, wasm), expect[3]);
        }
    }
}
