//! Executing the plan: the ready-queue dependency pipeline.
//!
//! Every dependency node runs the commands [`crate::plan::node_commands`]
//! synthesizes — the same ones `inspect --commands` dumps — as soon as
//! **all** of its dependencies have finished: no wave barriers, a node
//! whose cone completes early starts early. The processes go through the
//! [`Pool`], so `-j` bounds how many run at once while the scheduler
//! tracks readiness above it.
//!
//! Per node: freshness is checked at dispatch time (its upstream artifacts
//! are final by then — a current node is skipped whole), its sources are
//! scanned, its interface and archive compile, and [`fresh::record_dep`]
//! writes what the next build proves currency against. Progress renders on
//! an indicatif multi-bar (hidden when stderr is not a terminal).

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::compile;
use crate::db::BuildDir;
use crate::deps;
use crate::fresh;
use crate::manifest::Profile;
use crate::plan;
use crate::pool::Pool;
use crate::resolve::Graph;
use crate::rt::RtArtifacts;

/// What the pipeline runs under.
pub struct Options<'a> {
    pub profile_name: &'a str,
    pub profile: &'a Profile,
    pub target: &'a str,
    /// `rene build --linker`, overriding the profile's.
    pub linker: Option<&'a Path>,
    /// `rene build -j`: the process admission bound (held by the [`Pool`]).
    pub jobs: Option<NonZeroUsize>,
    pub build_dir: &'a Path,
    /// Whether buffered rrc diagnostics retain ANSI styling.
    pub color: bool,
}

/// Build every dependency of the graph through the pool, each dispatched
/// the moment its last dependency finishes. Returns how many nodes actually
/// compiled (the rest were current).
pub async fn build_deps(
    dir: &BuildDir,
    graph: &Graph,
    rt: &RtArtifacts,
    opts: &Options<'_>,
    pool: &Pool,
    progress: &MultiProgress,
) -> Result<usize, String> {
    let dep_names: Vec<&str> = graph
        .nodes
        .keys()
        .map(String::as_str)
        .filter(|name| *name != graph.root)
        .collect();
    if dep_names.is_empty() {
        return Ok(0);
    }
    let deps_dir = opts.build_dir.join(opts.profile_name).join("deps");
    std::fs::create_dir_all(&deps_dir)
        .map_err(|e| format!("cannot create `{}`: {e}", deps_dir.display()))?;

    let overall = progress.add(ProgressBar::new(dep_names.len() as u64).with_style(
        ProgressStyle::with_template("{bar:24} {pos}/{len} dependencies").expect("static template"),
    ));

    // Readiness over direct edges: a node dispatches when its last
    // unfinished dependency completes.
    let mut waiting_on: BTreeMap<&str, usize> = dep_names
        .iter()
        .map(|name| (*name, graph.nodes[*name].dependencies.len()))
        .collect();
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for name in &dep_names {
        for dep in &graph.nodes[*name].dependencies {
            dependents.entry(dep.as_str()).or_default().push(name);
        }
    }

    let mut in_flight = FuturesUnordered::new();
    for (name, waiting) in &waiting_on {
        if *waiting == 0 {
            in_flight.push(run_node(name, dir, graph, rt, opts, pool, progress));
        }
    }
    let mut built = 0;
    while let Some((finished, result)) = in_flight.next().await {
        built += result? as usize;
        overall.inc(1);
        for dependent in dependents.get(finished).into_iter().flatten() {
            let waiting = waiting_on
                .get_mut(dependent)
                .expect("every dependent is a known node");
            *waiting -= 1;
            if *waiting == 0 {
                in_flight.push(run_node(dependent, dir, graph, rt, opts, pool, progress));
            }
        }
    }
    overall.finish_and_clear();
    progress.remove(&overall);
    Ok(built)
}

/// Build one node (or prove it current), tagging the result with its name
/// so the scheduler can release its dependents.
async fn run_node<'g>(
    name: &'g str,
    dir: &BuildDir,
    graph: &'g Graph,
    rt: &RtArtifacts,
    opts: &Options<'_>,
    pool: &Pool,
    progress: &MultiProgress,
) -> (&'g str, Result<bool, String>) {
    let result = build_node(name, dir, graph, rt, opts, pool, progress).await;
    (name, result)
}

async fn build_node(
    name: &str,
    dir: &BuildDir,
    graph: &Graph,
    rt: &RtArtifacts,
    opts: &Options<'_>,
    pool: &Pool,
    progress: &MultiProgress,
) -> Result<bool, String> {
    let ctx = fresh::Context {
        dir: Some(dir),
        profile_name: opts.profile_name,
        profile: opts.profile,
        target: opts.target,
        linker: opts.linker,
        build_dir: opts.build_dir,
    };
    // At this point the node's upstream artifacts are final, so the local
    // check is exact: current means nothing to do at all.
    let state = fresh::node_state(graph, name, &ctx)?;
    if state.is_current() {
        tracing::debug!(package = name, "up to date");
        return Ok(false);
    }
    tracing::debug!(package = name, reason = %state, "building");

    let bar =
        progress.add(ProgressBar::new_spinner().with_message(format!("{name}: scanning sources")));
    bar.enable_steady_tick(Duration::from_millis(120));

    // The dependency's own source graph — recorded with the build so the
    // next run's freshness walk has something to compare against.
    let src_root = graph.nodes[name].dir.join("src");
    let package = name.to_owned();
    let color = opts.color;
    let mut sources = pool
        .run(move || async move { deps::scan(&src_root, &package, color).await })
        .await??;
    sources.sort_by(|a, b| a.path.cmp(&b.path));

    let plan_opts = plan::Options {
        profile_name: opts.profile_name,
        profile: opts.profile,
        target: opts.target,
        linker: opts.linker,
        build_dir: opts.build_dir,
    };
    for command in plan::node_commands(graph, name, &plan_opts, Some(rt)) {
        bar.set_message(format!("{name}: {}", command.emit));
        let invocation = compile::Invocation::from_planned(command, &rt.rustc, opts.color);
        let completed = pool.run(move || invocation.run()).await??;
        // Clear/redraw every active bar around the indivisible diagnostic
        // block. `suspend` also serializes these callbacks on the scheduler.
        progress.suspend(|| completed.report())?;
    }
    fresh::record_dep(graph, name, &ctx, rt, &sources)?;
    bar.finish_and_clear();
    progress.remove(&bar);
    tracing::debug!(package = name, "built");
    Ok(true)
}
