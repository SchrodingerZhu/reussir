//! The back leg: take what the front leg produced — a text dump, one lowered
//! module, or per-unit MIR — through the MLIR pipeline and the LLVM backend
//! to the requested artifact, including the packaged (staticlib) and linked
//! (executable/dynlib) finishes.

use std::path::{Path, PathBuf};

use reussir_backend::llvm::LlvmLowering;
use reussir_backend::melior::ir::Module;
use reussir_backend::pipeline::{self, LoweringOptions, OptLevel};
use reussir_backend_sys as sys;

use crate::{RelocMode, TargetMachine, TargetSpec, emit_to_file};

use super::Produced;
use super::cli::{Cli, parse_transform_scripts, write_text};
use super::link::{ScratchMembers, link_product, polyffi_paths};
use super::stage::Stage;

/// The shared back leg from a produced text/module: write a text dump, or run
/// the MLIR lowering pipeline and emit the requested artifact.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backend(
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

    // One context-wide streamer aggregates remarks from every function pass
    // and codegen unit, then finalizes the deterministic JSON report when the
    // context drops at the end of this compilation.
    if let Some(path) = &cli.token_reuse_remarks {
        let path = path.to_string_lossy();
        let path_ref = sys::mlir_sys::MlirStringRef {
            data: path.as_ptr().cast(),
            length: path.len(),
        };
        // SAFETY: `context` and `path` are live for the call; the C API copies
        // the path while constructing its owned output stream.
        if !unsafe { sys::reussirContextEnableTokenReuseRemarks(context.to_raw(), path_ref) } {
            return Err(format!(
                "failed to initialize token-reuse remark output `{}`",
                path
            ));
        }
    }

    // A static library and the linked targets are the ones whose units do not
    // each become a user-visible file: every unit is emitted into a scratch
    // directory, and the archive — or the linked product — is the
    // deliverable. This is also where `--codegen-units` stops being
    // observable to the consumer — N units or one, the result exposes the
    // same symbols.
    if target == Stage::Staticlib || target.is_linked() {
        let mut lib = ScratchMembers::new(output, cli.lto)?;
        for (index, module) in modules.into_iter().enumerate() {
            let member = lib.next_member();
            tracing::debug!(unit = index, "compiling codegen unit");
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn backend_module(
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
        instrument_nonlinear_ffi: cli.instrument_nonlinear_ffi,
        emit_token_reuse_remarks: cli.token_reuse_remarks.is_some(),
        nullary_variant_encoding: cli.nullary_variant_encoding.resolve(machine.triple()),
        pack_record_members: !cli.no_pack_record_members,
        closure_wpd,
        transform_scripts: parse_transform_scripts(&cli.transform_script)?,
        ..LoweringOptions::default()
    };
    let optimize_ffi = !matches!(opt, OptLevel::None);
    tracing::debug!("gathering polymorphic FFI");
    let prepared = LlvmLowering::prepare(
        &module,
        machine.data_layout(),
        machine.triple(),
        optimize_ffi,
        &polyffi_paths(cli)?,
    )
    .map_err(|e| format!("{name}: {e}"))?;
    tracing::debug!("running the MLIR lowering pipeline");
    pipeline::run_lowering_pipeline(context, &mut module, &options)
        .map_err(|e| format!("lowering pipeline failed: {e:?}"))?;

    // After the pipeline the module is the LLVM dialect; dump it before it is
    // translated out of MLIR. (`prepared`'s `Drop` releases the gathered FFI.)
    if target == Stage::MlirLlvm {
        drop(prepared);
        write_text(output, &module.as_operation().to_string())?;
        return Ok(true);
    }

    tracing::debug!("translating to LLVM IR");
    let finalized = prepared
        .finish(&module)
        .map_err(|e| format!("{name}: {e}"))?;
    let kind = target
        .output_kind(cli.lto)
        .expect("file-emitting or linked target past the mlir-llvm stage");
    tracing::debug!("optimizing and emitting the object");
    emit_to_file(finalized, &machine, opt, kind, output)?;
    tracing::debug!(output = %output.display(), "emitted");
    Ok(true)
}

/// The output path for codegen unit `index`: the unit index slots in before
/// the extension (`out/foo.o` → `out/foo.0.o`; no extension → `foo.0`).
pub(crate) fn unit_output(output: &Path, index: usize) -> PathBuf {
    match output.extension().and_then(|e| e.to_str()) {
        Some(ext) => output.with_extension(format!("{index}.{ext}")),
        None => output.with_extension(index.to_string()),
    }
}
