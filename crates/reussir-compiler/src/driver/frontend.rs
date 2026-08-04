//! The front leg: parse, elaborate, and monomorphize an input — a source
//! file, a whole package, or a textual `.hir`/`.mir` re-entry — stopping at
//! whichever stage `--emit` requests: a text dump (HIR, the `.rri`
//! interface, MIR) or the lowered module(s) handed to the back leg.

use std::io::IsTerminal;
use std::path::Path;

use reussir_codegen::lower::{
    CodegenUnit, LinkagePolicy, LoweringError, Sanitizer, lower_program, lower_unit,
};
use reussir_codegen::source::{FileId, SourceCache};
use reussir_core::full::interface;
use reussir_core::full::mir;
use reussir_core::full::mono::{MonoInput, monomorphize};
use reussir_core::semi::hir;
use reussir_core::semi::resolve::DefTable;
use reussir_core::semi::ty::TyCtxt;
use reussir_core::semi::{elaborate, render_reports};
use reussir_core::surface;
use reussir_syntax::diagnostics;
use reussir_syntax::kind::{Resolver, TokenKey};

use crate::package;

use super::Produced;
use super::cli::Cli;
use super::stage::Stage;

/// The arena-scoped front leg for package mode: elaborate every discovered
/// file as one translation unit, then continue exactly as the single-file
/// source path does.
///
/// An empty `Err` string signals diagnostics were already printed (a compile
/// failure, exit 1) rather than a driver error (exit 2).
pub(crate) fn frontend_package<'c, 'tcx>(
    context: &'c reussir_backend::melior::Context,
    tcx: &TyCtxt<'tcx>,
    target: Stage,
    pkg: &mut package::PackageSource,
    pkg_name: &str,
    interner: &std::sync::Arc<reussir_syntax::MultiThreadedTokenInterner>,
    cli: &Cli,
) -> Result<Produced<'c>, String> {
    use reussir_core::semi::{ExternPackage, PackageFile, elaborate_package_with_externs};

    // Dependency interfaces load and gate first — their tables must exist
    // before any consumer resolution runs.
    let extern_specs = crate::externs::join_specs(&cli.externs, &cli.extern_srcs)?;
    let externs = crate::externs::load(tcx, &extern_specs, Some(pkg_name))?;

    // Extern sources join the consumer's cache only when the pipeline runs
    // into monomorphization: that is where reports and debug locations can
    // land inside imported bodies. The text-dump targets (`hir`, `rri`) print
    // the consumer's file table verbatim, so growing the cache there would
    // leak consumer-anchored dependency paths into the artifact (breaking the
    // rri's byte-stability across checkouts).
    let extern_files: Vec<Vec<FileId>> = if target > Stage::Rri {
        externs
            .iter()
            .map(|ext| ext.register_sources(&mut pkg.cache))
            .collect()
    } else {
        externs.iter().map(|_| Vec::new()).collect()
    };
    let pkg: &package::PackageSource = pkg;

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
    let extern_pkgs: Vec<ExternPackage<'_, 'tcx>> = externs
        .iter()
        .zip(&extern_files)
        .map(|(ext, files)| ExternPackage {
            name: &ext.name,
            parsed: &ext.parsed,
            files,
        })
        .collect();
    let elab = elaborate_package_with_externs(tcx, &files, interner, &extern_pkgs);
    if render_reports(&pkg.cache, &elab.reports) {
        return Err(String::new());
    }
    if target == Stage::Hir {
        let bounds = elab.bound_names();
        let printer = if cli.no_source_locations {
            hir::print::Printer::new(&elab.defs, elab.resolver)
        } else {
            hir::print::Printer::with_sources(&elab.defs, elab.resolver, &pkg.cache)
        }
        .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
        .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
        .with_bounds(&bounds);
        let strings = elab.strings.entries();
        // The dump describes the consumer package: extern-imported records
        // are filtered out (extern bodies never joined `elaborated`).
        let records = elab.local_records();
        let text = printer.program(&elab.elaborated, &strings, &records, &elab.trampolines);
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
        let bounds = elab.bound_names();
        let printer = if cli.no_source_locations {
            hir::print::Printer::new(&elab.defs, elab.resolver)
        } else {
            hir::print::Printer::with_sources(&elab.defs, elab.resolver, &pkg.cache)
        }
        .with_transform_metadata(&anchors, &scripts)
        .with_ffi_metadata(&ffi_preludes, &ffi_imports)
        .with_bounds(&bounds)
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
pub(crate) fn frontend<'c, 'tcx>(
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
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
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
            let elab = elaborate(tcx, &prog, &interner);
            if render_reports(sources, &elab.reports) {
                return Err(String::new());
            }
            if target == Stage::Hir {
                let bounds = elab.bound_names();
                let printer = if cli.no_source_locations {
                    hir::print::Printer::new(&elab.defs, elab.resolver)
                } else {
                    hir::print::Printer::with_sources(&elab.defs, elab.resolver, sources)
                }
                .with_transform_metadata(&elab.transform_anchors, &elab.transform_scripts)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .with_bounds(&bounds);
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
                .with_ffi_metadata(&parsed.ffi_preludes, &parsed.ffi_imports)
                .with_bounds(&parsed.generic_bounds);
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
                // A plain `.hir` re-entry is a closed world; interfaces load
                // via `--extern` (package mode) only.
                externs: Default::default(),
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
pub(crate) fn refetch_sources(files: &[String]) -> Option<SourceCache> {
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
pub(crate) fn lower_or_render<T>(
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
pub(crate) fn finish_mir<'c, 'tcx>(
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
