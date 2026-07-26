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

use reussir_backend::llvm::LlvmLowering;
use reussir_backend::melior::ir::Module;
use reussir_backend::pipeline::{self, LoweringOptions, OptLevel};
use reussir_codegen::lower::{
    CodegenUnit, LinkagePolicy, LoweringError, Sanitizer, lower_program, lower_unit,
};
use reussir_codegen::source::{FileId, SourceCache};
use crate::package;
use crate::{
    RelocMode, TargetMachine, TargetSpec, emit_to_file, parse_llvm_ir,
};
use reussir_core::full::interface;
use reussir_core::full::mir;
use reussir_core::full::mono::{MonoInput, monomorphize};
use reussir_core::semi::hir;
use reussir_core::semi::resolve::DefTable;
use reussir_core::semi::ty::TyCtxt;
use reussir_core::semi::{elaborate, render_reports};
use reussir_core::{in_arena, surface};
use reussir_syntax::diagnostics;
use reussir_syntax::kind::{Resolver, TokenKey};

mod cli;
mod link;
mod stage;

use cli::{
    Cli, init_tracing, parse_opt, parse_reloc, parse_transform_scripts, read_input,
    resolve_input_stage, resolve_target, write_text,
};
use link::{ScratchMembers, link_product,
    polyffi_paths,
};
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
        let pkg = match load_package_or_render(&root, &pkg_name, &interner) {
            Ok(pkg) => pkg,
            Err(msg) if msg.is_empty() => return Ok(false),
            Err(msg) => return Err(msg),
        };
        let name = pkg.cache.name(FileId::ROOT).to_owned();
        let context = reussir_backend::context();
        if cli.disable_backend_multithreading {
            context.enable_multi_threading(false);
        }
        let produced =
            match in_arena(|tcx| frontend_package(&context, tcx, target, &pkg, &interner, cli)) {
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

/// The shared back leg from a produced text/module: write a text dump, or run
/// the MLIR lowering pipeline and emit the requested artifact.
#[allow(clippy::too_many_arguments)]
fn backend(
    cli: &Cli,
    context: &reussir_backend::melior::Context,
    produced: Produced<'_>,
    target: Stage,
    opt: OptLevel,
    reloc: RelocMode,
    spec: &TargetSpec,
    name: &str,
    output: &Path,
) -> Result<bool, String> {
    let (modules, partitioned, exports) = match produced {
        Produced::Text(text) => {
            write_text(output, &text)?;
            return Ok(true);
        }
        Produced::Units(units, exports) => (units, true, exports),
        Produced::Module(module, exports) => (vec![module], false, exports),
    };

    // A static library and the linked targets are the ones whose units do not
    // each become a user-visible file: every unit is emitted into a scratch
    // directory, and the archive — or the linked product — is the
    // deliverable. This is also where `--codegen-units` stops being
    // observable to the consumer — N units or one, the result exposes the
    // same symbols.
    if target == Stage::Staticlib || target.is_linked() {
        let mut lib = ScratchMembers::new(output, cli.lto)?;
        for module in modules {
            let member = lib.next_member();
            backend_module(
                cli, context, module, target, opt, reloc, spec, name, &member,
            )?;
        }
        let machine = TargetMachine::new(spec, opt, reloc)?;
        if target == Stage::Staticlib {
            return lib.finish(output, machine.triple()).map(|()| true);
        }
        return link_product(
            cli,
            target,
            machine.triple(),
            exports.as_deref(),
            &lib,
            output,
        )
        .map(|()| true);
    }

    // Otherwise each unit writes its own artifact: `<stem>.<i>.<ext>`
    // siblings of `-o` when partitioned, `-o` itself for a single module.
    for (index, module) in modules.into_iter().enumerate() {
        let out = if partitioned {
            unit_output(output, index)
        } else {
            output.to_owned()
        };
        backend_module(cli, context, module, target, opt, reloc, spec, name, &out)?;
    }
    Ok(true)
}

fn backend_module(
    cli: &Cli,
    context: &reussir_backend::melior::Context,
    mut module: Module<'_>,
    target: Stage,
    opt: OptLevel,
    reloc: RelocMode,
    spec: &TargetSpec,
    name: &str,
    output: &Path,
) -> Result<bool, String> {
    // A `.mlir` dump is the module as it stands, before the lowering pipeline.
    if target == Stage::Mlir {
        write_text(output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    // MLIR → LLVM leg: the target machine's data layout feeds the polymorphic-FFI
    // gather, which must run before the pipeline erases those ops, and is
    // stamped on the module so the pipeline's MLIR `DataLayout` queries compute
    // real target sizes and alignments.
    let machine = TargetMachine::new(spec, opt, reloc)?;
    pipeline::attach_target_spec(&module, machine.data_layout(), machine.triple())?;
    // Variant box sizing is a per-type property now: an enum carries
    // `#[repr(fixed)]` (uniform max-arm) or defaults to per-constructor
    // sizing, threaded into the dialect variant record type's `fixed` flag —
    // no module-wide switch.
    // Closure WPD is guarded (speculative WholeProgramDevirt), hence sound
    // regardless of what other modules exist — a vtable this module has never
    // seen just takes the indirect fallback arm. Reserve it for the opt
    // levels that ask for whole-program effort anyway: the artifacts cost
    // compile time, and the type-test intrinsics only lower inside the
    // optimizing backend pipeline. Under multiple codegen units each unit
    // still only devirtualizes the families it can see whole, so the win
    // shrinks but nothing breaks.
    let closure_wpd = !cli.no_closure_wpd && matches!(opt, OptLevel::Aggressive | OptLevel::Size);
    let options = LoweringOptions {
        opt,
        reuse_token_across_call: cli.reuse_across_call,
        nullary_variant_encoding: cli.nullary_variant_encoding.resolve(machine.triple()),
        pack_record_members: !cli.no_pack_record_members,
        closure_wpd,
        transform_scripts: parse_transform_scripts(&cli.transform_script)?,
        ..LoweringOptions::default()
    };
    let optimize_ffi = !matches!(opt, OptLevel::None);
    let prepared = LlvmLowering::prepare(
        &module,
        machine.data_layout(),
        optimize_ffi,
        &polyffi_paths(cli)?,
    )
    .map_err(|e| format!("{name}: {e}"))?;
    pipeline::run_lowering_pipeline(context, &mut module, &options)
        .map_err(|e| format!("lowering pipeline failed: {e:?}"))?;

    // After the pipeline the module is the LLVM dialect; dump it before it is
    // translated out of MLIR. (`prepared`'s `Drop` releases the gathered FFI.)
    if target == Stage::MlirLlvm {
        drop(prepared);
        write_text(output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    let finalized = prepared
        .finish(&module)
        .map_err(|e| format!("{name}: {e}"))?;
    let kind = target
        .output_kind(cli.lto)
        .expect("file-emitting or linked target past the mlir-llvm stage");
    emit_to_file(finalized, &machine, opt, kind, output)?;
    Ok(true)
}

/// The output path for codegen unit `index`: the unit index slots in before
/// the extension (`out/foo.o` → `out/foo.0.o`; no extension → `foo.0`).
fn unit_output(output: &Path, index: usize) -> PathBuf {
    match output.extension().and_then(|e| e.to_str()) {
        Some(ext) => output.with_extension(format!("{index}.{ext}")),
        None => output.with_extension(index.to_string()),
    }
}

/// The arena-scoped front leg for package mode: elaborate every discovered
/// file as one translation unit, then continue exactly as the single-file
/// source path does.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn frontend_package<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    target: Stage,
    pkg: &package::PackageSource,
    interner: &std::sync::Arc<reussir_syntax::MultiThreadedTokenInterner>,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    use reussir_core::semi::{PackageFile, elaborate_package};

    let name = pkg.cache.name(FileId::ROOT);
    let programs: Vec<surface::Program> = pkg
        .files
        .iter()
        .map(|f| surface::program(&f.parse.root))
        .collect();
    let mut keys = interner.clone();
    let files: Vec<PackageFile> = pkg
        .files
        .iter()
        .zip(&programs)
        .map(|(f, program)| PackageFile {
            file: f.file,
            module: f
                .module
                .iter()
                .map(|seg| reussir_syntax::Interner::get_or_intern(&mut keys, seg))
                .collect(),
            program,
        })
        .collect();
    let elab = elaborate_package(tcx, &files, interner);
    if render_reports(&pkg.cache, &elab.reports) {
        return Err(String::new());
    }
    if target == Stage::Hir {
        let printer = if cli.no_source_locations {
            hir::print::Printer::new(&elab.defs, elab.resolver)
        } else {
            hir::print::Printer::with_sources(&elab.defs, elab.resolver, &pkg.cache)
        }
        .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
        .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports);
        let strings = elab.strings.entries();
        let text = printer.program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
        return Ok(Produced::Text(text));
    }
    if target == Stage::Rri {
        let closure = interface::export_closure(&elab.mono_input());
        // The ancillary tables travel only as far as the closure reaches:
        // strings the shipped bodies reference, ffi textures of shipped
        // functions (with the preludes of their files), transform scripts
        // only when a shipped body is anchored. Trampolines never ship —
        // they are this package's own link surface and mono roots.
        let strings: Vec<_> = elab
            .strings
            .entries()
            .into_iter()
            .filter(|(token, _)| closure.strings.contains(token))
            .collect();
        let ffi_imports: rustc_hash::FxHashMap<_, _> = elab
            .ffi_imports
            .iter()
            .filter(|(def, _)| closure.bodies.contains(def))
            .map(|(def, import)| (*def, import.clone()))
            .collect();
        let ffi_files: rustc_hash::FxHashSet<FileId> =
            ffi_imports.values().map(|import| import.file).collect();
        let ffi_preludes: Vec<_> = elab
            .ffi_preludes
            .iter()
            .filter(|prelude| ffi_files.contains(&prelude.file))
            .cloned()
            .collect();
        let anchors: Vec<_> = elab
            .transform_anchors
            .iter()
            .copied()
            .filter(|def| closure.bodies.contains(def))
            .collect();
        let scripts: Vec<_> = if anchors.is_empty() {
            Vec::new()
        } else {
            elab.transform_scripts.clone()
        };
        // The source cache stores canonical paths (package discovery dedups
        // through them), so the root must be canonicalized the same way for
        // the prefix strip to hold.
        let file_root = match (&cli.package_root, &cli.input) {
            (Some(root), _) => Some(root.clone()),
            (None, Some(input)) => input.parent().map(Path::to_path_buf),
            _ => None,
        }
        .map(|root| root.canonicalize().unwrap_or(root));
        let printer = if cli.no_source_locations {
            hir::print::Printer::new(&elab.defs, elab.resolver)
        } else {
            hir::print::Printer::with_sources(&elab.defs, elab.resolver, &pkg.cache)
        }
        .with_transform_metadata(&anchors, &scripts)
        .with_ffi_metadata(&ffi_preludes, &ffi_imports)
        .with_interface(hir::print::InterfaceEmit {
            format: interface::RRI_FORMAT,
            package: cli.package_name.as_deref().unwrap_or_default(),
            producer: concat!("rrc ", env!("CARGO_PKG_VERSION")),
            bodies: &closure.bodies,
            protos: &closure.protos,
            records: &closure.records,
            file_root: file_root.as_deref(),
        });
        let text = printer.program(&elab.elaborated, &strings, &elab.records, &[]);
        return Ok(Produced::Text(text));
    }
    let (full, reports) = monomorphize(&elab.mono_input());
    if render_reports(&pkg.cache, &reports) {
        return Err(String::new());
    }
    finish_mir(
        context,
        tcx,
        target,
        name,
        Some(&pkg.cache),
        cli,
        &full,
        &elab.defs,
        elab.resolver,
    )
}

/// The arena-scoped front leg for a source/`.hir`/`.mir` input: run the frontend
/// from the entry stage to `target`, yielding a text dump or a lowered module.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
fn frontend<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    input: Stage,
    target: Stage,
    sources: &SourceCache,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    let name = sources.name(FileId::ROOT);
    let source = sources.source(FileId::ROOT);
    match input {
        Stage::Rr => {
            let parse = reussir_syntax::parse(source);
            if !parse.ok() {
                let color = std::io::stderr().is_terminal();
                let _ = diagnostics::render_errors(
                    sources,
                    FileId::ROOT,
                    &parse.errors,
                    color,
                    std::io::stderr().lock(),
                );
                return Err(String::new());
            }
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            if render_reports(sources, &elab.reports) {
                return Err(String::new());
            }
            if target == Stage::Hir {
                let printer = if cli.no_source_locations {
                    hir::print::Printer::new(&elab.defs, elab.resolver)
                } else {
                    hir::print::Printer::with_sources(&elab.defs, elab.resolver, sources)
                }
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports);
                let strings = elab.strings.entries();
                let text =
                    printer.program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
                return Ok(Produced::Text(text));
            }
            let (full, reports) = monomorphize(&elab.mono_input());
            if render_reports(sources, &reports) {
                return Err(String::new());
            }
            finish_mir(
                context,
                tcx,
                target,
                name,
                Some(sources),
                cli,
                &full,
                &elab.defs,
                elab.resolver,
            )
        }
        Stage::Hir => {
            let parsed =
                hir::build::parse_program(tcx, source).map_err(|e| format!("{name}: {e}"))?;
            // An interface is not a resumable program: its prototypes have no
            // bodies to compile and its mono roots stayed home. Loading one
            // is `--extern`'s job (a later PR), not pipeline re-entry.
            if parsed.header.is_some() {
                return Err(format!(
                    "{name}: this is a package interface (`.rri`), not a plain HIR \
                     program; an interface describes another package's exports and \
                     cannot re-enter the pipeline"
                ));
            }
            // The dump's file table names the original sources. Rebuild the
            // same dense ids, but delay opening paths until diagnostics or
            // debug locations actually need source text. An old, table-less
            // dump has no locations at all.
            let dump_sources = refetch_sources(&parsed.files);
            if target == Stage::Hir {
                let printer = match &dump_sources {
                    Some(cache) if !cli.no_source_locations => {
                        hir::print::Printer::with_sources(&parsed.defs, &parsed.names, cache)
                    }
                    _ => hir::print::Printer::new(&parsed.defs, &parsed.names),
                }
                .with_transform_metadata(&parsed.transform_anchors, &parsed.transform_scripts)
                .with_ffi_metadata(&parsed.ffi_preludes, &parsed.ffi_imports);
                let text = printer.program(
                    &parsed.funcs,
                    &parsed.strings,
                    &parsed.records,
                    &parsed.trampolines,
                );
                return Ok(Produced::Text(text));
            }
            let input = MonoInput {
                tcx,
                defs: &parsed.defs,
                resolver: &parsed.names,
                elaborated: &parsed.funcs,
                records: &parsed.records,
                trampolines: &parsed.trampolines,
                transform_anchors: &parsed.transform_anchors,
                transform_scripts: &parsed.transform_scripts,
                ffi_imports: &parsed.ffi_imports,
                ffi_preludes: &parsed.ffi_preludes,
                strings: parsed.strings.clone(),
            };
            let (full, reports) = monomorphize(&input);
            match &dump_sources {
                Some(cache) => {
                    if render_reports(cache, &reports) {
                        return Err(String::new());
                    }
                }
                None => {
                    if render_reports(sources, &reports) {
                        return Err(String::new());
                    }
                }
            }
            finish_mir(
                context,
                tcx,
                target,
                name,
                dump_sources.as_ref(),
                cli,
                &full,
                &parsed.defs,
                &parsed.names,
            )
        }
        Stage::Mir => {
            let parsed =
                mir::build::parse_program(tcx, source).map_err(|e| format!("{name}: {e}"))?;
            let dump_sources = refetch_sources(&parsed.files);
            finish_mir(
                context,
                tcx,
                target,
                name,
                dump_sources.as_ref(),
                cli,
                &parsed.program,
                &parsed.defs,
                &parsed.names,
            )
        }
        _ => unreachable!("frontend only handles rr/hir/mir inputs"),
    }
}

/// Rebuild a source cache from an IR dump's file table so its spans keep
/// resolving: real paths are registered for lazy loading, while virtual
/// (`<bracketed>`) entries become name-only placeholders. `None` means the dump
/// was printed without locations.
fn refetch_sources(files: &[String]) -> Option<SourceCache> {
    if files.is_empty() {
        return None;
    }
    let mut cache = SourceCache::new();
    for name in files {
        if name.starts_with('<') {
            cache.add_unavailable(name);
        } else {
            cache.add_lazy_file(name);
        }
    }
    Some(cache)
}

/// Return a lowered value, rendering a source-owned lowering failure through
/// the same ariadne path as parser and elaboration diagnostics. A lowering
/// failure without usable source metadata retains the traditional plain-text
/// driver error.
fn lower_or_render<T>(
    result: std::result::Result<T, LoweringError>,
    sources: Option<&SourceCache>,
    name: &str,
) -> Result<T, String> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            if let (Some(sources), Some((file, span))) = (sources, error.source_location())
                && file.index() < sources.len()
            {
                let diagnostic = diagnostics::Diagnostic {
                    file,
                    span: span.map(|span| (span.start, span.end)),
                    severity: diagnostics::Severity::Error,
                    message: error.message(),
                };
                let color = std::io::stderr().is_terminal();
                let _ = diagnostics::render(
                    sources,
                    std::slice::from_ref(&diagnostic),
                    color,
                    std::io::stderr().lock(),
                );
                Err(String::new())
            } else {
                Err(format!("{name}: {error}"))
            }
        }
    }
}

/// Finish the front leg from a ground MIR program: a `mir` dump, or lowering to
/// an MLIR module for `mlir` and beyond.
#[allow(clippy::too_many_arguments)]
fn finish_mir<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    target: Stage,
    name: &str,
    sources: Option<&SourceCache>,
    cli: &Cli,
    program: &mir::Program<'tcx>,
    defs: &DefTable,
    resolver: &dyn Resolver<TokenKey>,
) -> Result<Produced<'c>, String> {
    if target == Stage::Mir {
        let printer = match sources {
            Some(cache) if !cli.no_source_locations => {
                mir::print::Printer::with_sources(defs, resolver, cache)
            }
            _ => mir::print::Printer::new(defs, resolver),
        };
        return Ok(Produced::Text(printer.program(program)));
    }
    // Names feed variable/function debug info; only meaningful with `-g`.
    let names = cli.debug.then_some(resolver);
    // The link step's view of the program: its exported trampolines — the
    // dynamic library's export surface, and where `#[main]`'s
    // `__reussir_main` shows up for the executable's entry-point check.
    let exports: Vec<String> = program
        .trampolines
        .iter()
        .filter(|t| !t.import)
        .map(|t| program.symbol(t.export).to_owned())
        .collect();
    // AOT linkage: internal / linkonce_odr definitions per the policy; the
    // trampolines and pub functions stay the object's external ABI surface.
    let linkage = LinkagePolicy::aot_for_triple(cli.target_triple.as_deref());
    let sanitizers: Vec<Sanitizer> = cli.sanitizer.iter().map(|&s| s.into()).collect();
    if cli.codegen_units > 1 {
        let mut units = Vec::with_capacity(cli.codegen_units as usize);
        for index in 0..cli.codegen_units {
            let unit = CodegenUnit {
                index,
                count: cli.codegen_units,
            };
            let module = lower_or_render(
                lower_unit(
                    context,
                    tcx,
                    program,
                    sources,
                    names,
                    unit,
                    linkage,
                    &sanitizers,
                ),
                sources,
                name,
            )?;
            units.push(module);
        }
        return Ok(Produced::Units(units, Some(exports)));
    }
    let module = lower_or_render(
        lower_program(context, tcx, program, sources, names, linkage, &sanitizers),
        sources,
        name,
    )?;
    Ok(Produced::Module(module, Some(exports)))
}
