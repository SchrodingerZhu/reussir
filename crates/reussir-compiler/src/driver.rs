//! The `rrc` driver: the whole compiler pipeline behind the binary, which
//! is a shim over [`main`].
//!
//! A clang-style pipeline driver over one totally-ordered chain of stages:
//!
//! ```text
//! .rr ──elaborate──▶ .hir ──monomorphize──▶ .mir ──lower──▶ .mlir ──pipeline──▶ mlir-llvm ──translate──▶ .ll ──▶ .s/.o
//! ```
//!
//! The input's extension (or `--from`) picks where to enter the chain; `-o`'s
//! extension (or `--emit`) picks where to leave it; the driver runs exactly the
//! transforms in between. Every intermediate is both a dump target and — except
//! the two derived MLIR forms — a re-entry point, so a stage can be inspected or
//! fed back in isolation. Exit 0 on success, 1 on a compile error, 2 on a usage
//! or I/O error.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use palc::Parser;

use crate::package;
use crate::{TargetMachine, TargetSpec, emit_to_file, parse_llvm_ir};
use reussir_backend::melior::ir::Module;
use reussir_codegen::source::FileId;
use reussir_core::in_arena;
use reussir_syntax::diagnostics;

mod backend;
mod cli;
mod frontend;
mod link;
mod stage;

use backend::backend;
use cli::{
    Cli, init_tracing, parse_opt, parse_reloc, read_input, resolve_input_stage, resolve_target,
    write_text,
};
use frontend::{frontend, frontend_package};
use link::{ScratchMembers, link_product};
use stage::{Lto, SanitizerCli, Stage, VariantEncoding};

/// What the front (arena-scoped) leg produces: a text dump for `hir`/`mir`, or an
/// MLIR module for `mlir` and anything past it. The module borrows the MLIR
/// context, not the type arena, so it outlives the [`in_arena`] scope.
/// The C-ABI export surface the front leg collected: the program's exported
/// (non-import) trampoline symbols, `#[main]`'s `__reussir_main` among them.
/// `None` when the input entered past MIR (`.mlir`/`.ll`), where the surface
/// is no longer recorded — the link step then cannot name a dynamic library's
/// exports, and skips the executable's entry-point pre-check.
type ExportSurface = Option<Vec<String>>;

enum Produced<'c> {
    Text(String),
    Module(Module<'c>, ExportSurface),
    /// One module per codegen unit (`--codegen-units` > 1), in unit order.
    Units(Vec<Module<'c>>, ExportSurface),
}

pub fn main() -> ExitCode {
    // `palc` surfaces `--help` as a parse *error* rendered to stderr with a
    // non-zero exit. Restore the usual CLI convention: help goes to stdout and
    // exits 0 (a `--help | grep …` pipeline must not fail). Other parse errors
    // stay on stderr with the usage exit code (2).
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
    match run(&cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("error: {message}");
            }
            ExitCode::from(2)
        }
    }
}

/// Where a package compilation's crate root comes from: `lib.rr` under a
/// `--package-root` directory, or the input file itself when `--package-name`
/// accompanies a positional input.
enum PackageRoot {
    Dir(PathBuf),
    File(PathBuf),
}

fn run(cli: &Cli) -> Result<bool, String> {
    let package = match (&cli.package_root, &cli.package_name, &cli.input) {
        (Some(_), None, _) => return Err("--package-root requires --package-name".into()),
        (Some(_), Some(_), Some(_)) => {
            return Err("give either an input file or --package-root, not both".into());
        }
        (Some(root), Some(name), None) => Some((PackageRoot::Dir(root.clone()), name.clone())),
        (None, Some(name), Some(input)) => {
            // The crate root anchors `mod` discovery on disk, so it must be a
            // real file.
            if input.as_os_str() == "-" {
                return Err(
                    "--package-name roots the package at the input file; stdin has no \
                     location for `mod` discovery"
                        .into(),
                );
            }
            Some((PackageRoot::File(input.clone()), name.clone()))
        }
        (None, Some(_), None) => {
            return Err(
                "--package-name needs a crate root: an input file or --package-root".into(),
            );
        }
        (None, None, _) => None,
    };

    // Extern package paths root at package names, so only a package can
    // consume them (mirrors --emit rri, which only a package can produce).
    if package.is_none() && !(cli.externs.is_empty() && cli.extern_srcs.is_empty()) {
        return Err(
            "--extern requires package mode: --package-root, or an input file rooted \
             by --package-name"
                .into(),
        );
    }

    // `--scan-deps` stops after discovery: list the package's source graph
    // instead of compiling.
    if cli.scan_deps {
        return scan_deps(cli, package.as_ref());
    }

    let Some(output) = cli.output.as_deref() else {
        return Err("no output file given; pass -o".into());
    };
    let target = resolve_target(cli)?;
    // The interface header names the package it describes; only package mode
    // has that name authoritatively (a plain file or dump does not).
    if target == Stage::Rri && package.is_none() {
        return Err(
            "--emit rri describes a package interface; compile a package \
             (--package-root, or an input file with --package-name)"
                .into(),
        );
    }
    if cli.lto != Lto::None && target != Stage::Staticlib && !target.is_linked() {
        return Err(format!(
            "--lto applies to `staticlib`, `dynlib`, and `executable`, not \
             `{target}`: only a packaged or linked target consumes the \
             bitcode it produces"
        ));
    }
    // The link-step knobs mean nothing without a link step; refuse them up
    // front rather than silently ignoring them.
    if !target.is_linked() {
        if cli.runtime_linkage.is_some() {
            return Err(format!(
                "--runtime-linkage applies to `executable` and `dynlib`, not `{target}`"
            ));
        }
        if cli.linker.is_some() {
            return Err(format!(
                "--linker applies to `executable` and `dynlib`, not `{target}`"
            ));
        }
        if !cli.link_arg.is_empty() {
            return Err(format!(
                "--link-arg applies to `executable` and `dynlib`, not `{target}`"
            ));
        }
    }
    // Only the text dumps stream to stdout; the file-emitting and linked
    // targets would write a file literally named `-`.
    if output.as_os_str() == "-" && target.output_kind(cli.lto).is_some() {
        return Err(format!(
            "cannot write `{target}` to stdout; give a file path with -o"
        ));
    }
    if cli.codegen_units == 0 {
        return Err("--codegen-units must be at least 1".into());
    }
    // Shadow-based sanitizers (ASan/MSan/TSan) compute a shadow address
    // arithmetically from the pointer; a TBI-tagged nullary immediate's top
    // byte — which the hardware ignores on the data access — bleeds into
    // that arithmetic and lands the shadow lookup in unmapped space. The
    // `arch-dependent` default may resolve to the TBI encoding, so require
    // an explicit untagged choice rather than silently downgrading.
    if cli.sanitizer.iter().any(|s| {
        matches!(
            s,
            SanitizerCli::Address | SanitizerCli::Memory | SanitizerCli::Thread
        )
    }) && cli.nullary_variant_encoding == VariantEncoding::ArchDependent
    {
        return Err(
            "--sanitizer address/memory/thread cannot be combined with the (default) \
             `arch-dependent` nullary-variant encoding: on aarch64 it resolves to \
             TBI-tagged immediates, whose tag byte breaks the sanitizers' shadow-address \
             arithmetic; pass --nullary-variant-encoding arch-independent (or boxed)"
                .into(),
        );
    }
    if cli.codegen_units > 1 && output.as_os_str() == "-" {
        return Err(
            "--codegen-units > 1 writes one file per unit; give a file path with -o".into(),
        );
    }
    let opt = parse_opt(&cli.opt)?;
    let reloc = parse_reloc(&cli.relocation_mode)?;
    let spec = TargetSpec {
        triple: cli.target_triple.clone(),
        cpu: cli.target_cpu.clone(),
        features: cli.target_features.clone(),
    };

    // Package mode: discover the files from the crate root's `mod`
    // declarations and elaborate the whole package as one translation unit.
    if let Some((root, pkg_name)) = package {
        if cli.from.is_some_and(|f| f != Stage::Rr) {
            return Err("a package is always Reussir source; --from does not apply".into());
        }
        if let PackageRoot::File(input) = &root {
            // A rooted input also carries an extension; make a non-source one
            // (`.hir`, `.mir`, …) fail like `--from` instead of parsing as source.
            if resolve_input_stage(cli, input)? != Stage::Rr {
                return Err(
                    "a package is always Reussir source; the crate root must be a `.rr` file"
                        .into(),
                );
            }
        }
        let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
        let mut pkg = match load_package_or_render(&root, &pkg_name, &interner) {
            Ok(pkg) => pkg,
            Err(msg) if msg.is_empty() => return Ok(false),
            Err(msg) => return Err(msg),
        };
        let name = pkg.cache.name(FileId::ROOT).to_owned();
        let context = reussir_backend::context();
        if cli.disable_backend_multithreading {
            context.enable_multi_threading(false);
        }
        let produced = match in_arena(|tcx| {
            frontend_package(&context, tcx, target, &mut pkg, &pkg_name, &interner, cli)
        }) {
            Ok(produced) => produced,
            Err(msg) => {
                if !msg.is_empty() {
                    eprintln!("{msg}");
                }
                return Ok(false);
            }
        };
        return backend(
            cli, &context, produced, target, opt, reloc, &spec, &name, output,
        );
    }

    let Some(input) = &cli.input else {
        return Err("no input file (or --package-root/--package-name) given".into());
    };
    let input_stage = resolve_input_stage(cli, input)?;
    if target < input_stage {
        return Err(format!(
            "cannot emit `{target}` from a `{input_stage}` input: the pipeline only runs forward"
        ));
    }
    let sources = read_input(input)?;
    let name = sources.name(FileId::ROOT).to_owned();
    let source = sources.source(FileId::ROOT);

    if cli.codegen_units > 1 && matches!(input_stage, Stage::Mlir | Stage::LlvmIr) {
        return Err(format!(
            "--codegen-units applies while lowering to MLIR; a `{input_stage}` input is already a single unit"
        ));
    }
    // A dynamic library's exports are the trampolines the *frontend* records;
    // a backend-stage input no longer carries them, and exporting every
    // external symbol instead would leak the cross-unit linkage surface.
    if target == Stage::Dynlib && matches!(input_stage, Stage::Mlir | Stage::LlvmIr) {
        return Err(format!(
            "cannot emit `dynlib` from a `{input_stage}` input: the export surface \
             (the program's trampolines) is only recorded up to `mir`; resume from \
             source, `hir`, or `mir`"
        ));
    }
    // A `.ll` input skips the whole MLIR front: parse the IR and run the LLVM
    // backend straight to the requested artifact.
    if input_stage == Stage::LlvmIr {
        let kind = target
            .output_kind(cli.lto)
            .ok_or_else(|| format!("cannot emit `{target}` from LLVM IR"))?;
        let machine = TargetMachine::new(&spec, opt, reloc)?;
        let finalized = parse_llvm_ir(&name, source)?;
        // One module, so a static library here archives a single member, and
        // an executable links a single object.
        if target == Stage::Staticlib || target.is_linked() {
            let mut lib = ScratchMembers::new(output, cli.lto)?;
            emit_to_file(finalized, &machine, opt, kind, &lib.next_member())?;
            if target == Stage::Staticlib {
                return lib.finish(output, machine.triple()).map(|()| true);
            }
            return link_product(cli, target, machine.triple(), None, &lib, output).map(|()| true);
        }
        emit_to_file(finalized, &machine, opt, kind, output)?;
        return Ok(true);
    }

    let context = reussir_backend::context();
    if cli.disable_backend_multithreading {
        context.enable_multi_threading(false);
    }
    // Front leg: reach an MLIR module (or a text dump for `hir`/`mir`). A `.mlir`
    // input parses directly; source/`.hir`/`.mir` run the frontend in the arena.
    let produced = match input_stage {
        Stage::Mlir => Module::parse(&context, source)
            .map(|module| Produced::Module(module, None))
            .ok_or_else(|| format!("{name}: failed to parse MLIR module"))?,
        _ => match in_arena(|tcx| frontend(&context, tcx, input_stage, target, &sources, cli)) {
            Ok(produced) => produced,
            Err(msg) => {
                if !msg.is_empty() {
                    eprintln!("{msg}");
                }
                return Ok(false);
            }
        },
    };
    backend(
        cli, &context, produced, target, opt, reloc, &spec, &name, output,
    )
}

/// Discover and parse the package from its crate root — `lib.rr` under a
/// root directory, or the rooting input file itself — rendering syntax errors
/// through the same ariadne path as compile diagnostics.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn load_package_or_render(
    root: &PackageRoot,
    name: &str,
    interner: &std::sync::Arc<reussir_syntax::MultiThreadedTokenInterner>,
) -> Result<package::PackageSource, String> {
    let loaded = match root {
        PackageRoot::Dir(dir) => package::load_package(dir, name, interner),
        PackageRoot::File(file) => package::load_package_rooted(file, name, interner),
    };
    match loaded {
        Ok(pkg) => Ok(pkg),
        Err(package::PackageError::Message(msg)) => Err(msg),
        Err(package::PackageError::ParseErrors {
            cache,
            file,
            errors,
        }) => {
            let color = std::io::stderr().is_terminal();
            let _ =
                diagnostics::render_errors(&cache, file, &errors, color, std::io::stderr().lock());
            Err(String::new())
        }
    }
}

/// `--scan-deps`: run package discovery — which parses every file to follow
/// its `mod` declarations — and describe the source graph as JSON instead of
/// compiling: the package name and, in discovery order, every file's
/// canonical path and module path.
fn scan_deps(cli: &Cli, package: Option<&(PackageRoot, String)>) -> Result<bool, String> {
    let Some((root, pkg_name)) = package else {
        return Err(
            "--scan-deps requires package mode: --package-root, or an input file rooted \
             by --package-name"
                .into(),
        );
    };
    let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
    let pkg = match load_package_or_render(root, pkg_name, &interner) {
        Ok(pkg) => pkg,
        Err(msg) if msg.is_empty() => return Ok(false),
        Err(msg) => return Err(msg),
    };
    let files: Vec<serde_json::Value> = pkg
        .files
        .iter()
        .map(|file| {
            serde_json::json!({
                "path": pkg.cache.name(file.file),
                "module": file.module,
            })
        })
        .collect();
    let graph = serde_json::json!({ "package": pkg_name, "files": files });
    let mut text = serde_json::to_string_pretty(&graph)
        .map_err(|e| format!("failed to encode the source graph: {e}"))?;
    text.push('\n');
    write_text(cli.output.as_deref().unwrap_or(Path::new("-")), &text)?;
    Ok(true)
}
