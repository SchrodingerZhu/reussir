//! The build plan: what the coming cross-package build will run, as data.
//!
//! `rene inspect --commands` dumps, for every package of the resolved graph
//! in dependency-first order, the `rrc` invocations that build it: each
//! dependency emits its interface (`--emit rri`) and its archive
//! (`--emit staticlib`) into `<build-dir>/<profile>/deps/`, and the root's
//! declared targets compile against them (`--extern` / `--extern-src` for
//! elaboration and diagnostics re-anchoring, the archives joined into the
//! link with `--link-lib`).
//! Nothing here executes; the dump is the orchestration contract — lit pins
//! it, and the cross-package build implements it.
//!
//! Paths the runtime bake decides (the pinned rustc, the polyffi library
//! directories) expand from the recorded bake when one exists — the dump is
//! then directly executable — and appear as `<placeholders>` in a build
//! directory nothing was ever baked in, keeping the plan renderable and
//! deterministic without baking.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::compile;
use crate::fresh;
use crate::manifest::{Profile, TargetKind};
use crate::resolve::Graph;
use crate::rt::RtArtifacts;

/// What the plan is rendered against.
pub struct Options<'a> {
    pub profile_name: &'a str,
    pub profile: &'a Profile,
    pub target: &'a str,
    /// `rene build --linker`, overriding the profile's.
    pub linker: Option<&'a Path>,
    pub build_dir: &'a Path,
}

/// The placeholder argv entries for bake-decided paths.
const RUST_LIBDIR: &str = "<rust-libdir>";
const RT_DEPS_DIR: &str = "<rt-deps-dir>";
const BAKED_RUSTC: &str = "<baked-rustc>";

/// One planned `rrc` invocation for a node — the shared synthesis behind
/// both the dump ([`render`]) and the executed build ([`crate::exec`]), so
/// the two cannot drift: what `inspect --commands` prints is what runs.
pub(crate) struct PlannedCommand {
    /// The `--emit` stage (`rri`, `staticlib`, or a root target's kind).
    pub(crate) emit: &'static str,
    /// The root target this builds, if it is a root command.
    pub(crate) target: Option<String>,
    /// The artifact the command writes.
    pub(crate) out: PathBuf,
    /// Full argv minus the leading program.
    pub(crate) args: Vec<String>,
}

/// The commands that build one node of the graph, in execution order.
pub(crate) fn node_commands(
    graph: &Graph,
    name: &str,
    opts: &Options,
    bake: Option<&RtArtifacts>,
) -> Vec<PlannedCommand> {
    let profile_dir = opts.build_dir.join(opts.profile_name);
    let deps_dir = profile_dir.join("deps");
    let node = &graph.nodes[name];
    let transitive = transitive_deps(graph);
    let externs = extern_args(graph, &transitive[name], &deps_dir);
    if name == graph.root {
        // Archives joined in dependency order: dependents before
        // dependencies (the reverse of the build order, root excluded).
        let mut link_order = topological(graph);
        link_order.reverse();
        link_order.retain(|n| *n != graph.root);
        node.loaded
            .manifest
            .targets
            .iter()
            .map(|(target, decl)| {
                let out =
                    profile_dir.join(compile::artifact_file(target, decl.kind, Some(opts.target)));
                let mut args = package_args(node, &out, decl.kind.emit());
                args.extend(compile::profile_flags(
                    opts.profile,
                    opts.target,
                    decl.kind,
                    opts.linker,
                ));
                args.extend(polyffi_args(bake));
                args.extend(externs.iter().cloned());
                if decl.kind.is_linked() {
                    for dep in &link_order {
                        args.push("--link-lib".to_owned());
                        args.push(
                            archive_path(&deps_dir, dep, opts.target)
                                .display()
                                .to_string(),
                        );
                    }
                }
                PlannedCommand {
                    emit: decl.kind.emit(),
                    target: Some(target.clone()),
                    out,
                    args,
                }
            })
            .collect()
    } else {
        let rri = deps_dir.join(format!("{name}.rri"));
        let mut interface = package_args(node, &rri, "rri");
        interface.extend(externs.iter().cloned());
        let archive = archive_path(&deps_dir, name, opts.target);
        let mut staticlib = package_args(node, &archive, "staticlib");
        staticlib.extend(compile::profile_flags(
            opts.profile,
            opts.target,
            TargetKind::Staticlib,
            opts.linker,
        ));
        staticlib.extend(polyffi_args(bake));
        staticlib.extend(externs.iter().cloned());
        vec![
            PlannedCommand {
                emit: "rri",
                target: None,
                out: rri,
                args: interface,
            },
            PlannedCommand {
                emit: "staticlib",
                target: None,
                out: archive,
                args: staticlib,
            },
        ]
    }
}

/// Render the plan as JSON: `nodes` in dependency-first order (each node's
/// dependencies precede it), each with the commands that build it and —
/// when the caller ran the freshness check — its `state`/`reason`.
pub fn render(
    graph: &Graph,
    opts: &Options,
    bake: Option<&RtArtifacts>,
    states: Option<&BTreeMap<String, fresh::State>>,
) -> serde_json::Value {
    let nodes: Vec<serde_json::Value> = topological(graph)
        .iter()
        .map(|name| {
            let node = &graph.nodes[name.as_str()];
            let commands: Vec<serde_json::Value> = node_commands(graph, name, opts, bake)
                .into_iter()
                .map(|command| {
                    let mut argv = vec!["rrc".to_owned()];
                    argv.extend(command.args);
                    match command.target {
                        Some(target) => serde_json::json!({
                            "target": target,
                            "emit": command.emit,
                            "argv": argv,
                        }),
                        None => serde_json::json!({ "emit": command.emit, "argv": argv }),
                    }
                })
                .collect();
            let mut rendered = serde_json::json!({
                "package": name,
                "version": node.version.to_string(),
                "dir": node.dir.display().to_string(),
                "commands": commands,
            });
            if let Some(state) = states.and_then(|states| states.get(name.as_str())) {
                rendered["state"] = state.tag().into();
                rendered["reason"] = state.to_string().into();
            }
            rendered
        })
        .collect();

    let rustc = bake.map_or(BAKED_RUSTC.to_owned(), |bake| {
        bake.rustc.display().to_string()
    });
    serde_json::json!({
        "profile": opts.profile_name,
        "target": opts.target,
        "env": { "REUSSIR_RUSTC": rustc },
        "nodes": nodes,
    })
}

/// The invocation prefix shared by every command of a package.
fn package_args(node: &crate::resolve::Node, out: &Path, emit: &str) -> Vec<String> {
    vec![
        "--package-root".to_owned(),
        node.dir.join("src").display().to_string(),
        "--package-name".to_owned(),
        node.loaded.manifest.package.name.clone(),
        "--emit".to_owned(),
        emit.to_owned(),
        "-o".to_owned(),
        out.display().to_string(),
    ]
}

/// `--extern`/`--extern-src` pairs for a package's (transitive) dependency
/// set: the interface to elaborate against, and the source root diagnostics
/// re-anchor onto.
fn extern_args(graph: &Graph, deps: &[String], deps_dir: &Path) -> Vec<String> {
    let mut args = Vec::new();
    for dep in deps {
        let node = &graph.nodes[dep.as_str()];
        args.push("--extern".to_owned());
        args.push(format!(
            "{dep}={}",
            deps_dir.join(format!("{dep}.rri")).display()
        ));
        args.push("--extern-src".to_owned());
        args.push(format!("{dep}={}", node.dir.join("src").display()));
    }
    args
}

fn polyffi_args(bake: Option<&RtArtifacts>) -> Vec<String> {
    let libdirs = match bake {
        Some(bake) => bake
            .libdirs()
            .map(|path| path.display().to_string())
            .collect(),
        None => vec![RUST_LIBDIR.to_owned(), RT_DEPS_DIR.to_owned()],
    };
    libdirs
        .into_iter()
        .flat_map(|path| ["--polyffi-libdir".to_owned(), path])
        .collect()
}

pub(crate) fn archive_path(deps_dir: &Path, name: &str, target: &str) -> PathBuf {
    deps_dir.join(compile::artifact_file(
        name,
        TargetKind::Staticlib,
        Some(target),
    ))
}

/// Dependency-first order: every node appears after all of its dependencies
/// (post-order DFS from the root, name-sorted siblings from the BTreeMap).
pub(crate) fn topological(graph: &Graph) -> Vec<String> {
    let mut order = Vec::new();
    let mut visited = std::collections::BTreeSet::new();
    fn visit(
        graph: &Graph,
        name: &str,
        visited: &mut std::collections::BTreeSet<String>,
        order: &mut Vec<String>,
    ) {
        if !visited.insert(name.to_owned()) {
            return;
        }
        for dep in &graph.nodes[name].dependencies {
            visit(graph, dep, visited, order);
        }
        order.push(name.to_owned());
    }
    visit(graph, &graph.root, &mut visited, &mut order);
    order
}

/// Every package's transitive dependency set, name-sorted. Passed whole to
/// each compile: a dependency's interface may ship generic bodies that reach
/// its own dependencies, so the consumer needs the full downward cone in
/// scope.
pub(crate) fn transitive_deps(graph: &Graph) -> BTreeMap<String, Vec<String>> {
    let mut memo: BTreeMap<String, Vec<String>> = BTreeMap::new();
    fn cone(graph: &Graph, name: &str, memo: &mut BTreeMap<String, Vec<String>>) -> Vec<String> {
        if let Some(done) = memo.get(name) {
            return done.clone();
        }
        // Seed with the empty set first so a dependency cycle terminates
        // (the cycle member's cone is then an underapproximation inside the
        // cycle, completed when the outermost call finishes).
        memo.insert(name.to_owned(), Vec::new());
        let mut set = std::collections::BTreeSet::new();
        for dep in &graph.nodes[name].dependencies {
            set.insert(dep.clone());
            set.extend(cone(graph, dep, memo));
        }
        let cone: Vec<String> = set.into_iter().collect();
        memo.insert(name.to_owned(), cone.clone());
        cone
    }
    let names: Vec<String> = graph.nodes.keys().cloned().collect();
    for name in names {
        cone(graph, &name, &mut memo);
    }
    memo
}
