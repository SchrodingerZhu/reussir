//! Lowering the monomorphized Full MIR to MLIR, built in memory.
//!
//! The frontend produces a ground [`mir::Program`]; this module builds the
//! corresponding MLIR module through the `func`/`arith`/`scf` ODS builders
//! (melior) and the Reussir-dialect / hand-written builders re-exported by
//! [`reussir_backend`]. [`reussir_backend::pipeline`] then lowers the result to
//! the LLVM dialect.
//!
//! The work is split across submodules:
//! * [`ty`] — ground scalar Reussir types → MLIR types, and the numeric
//!   classification used to pick cast ops;
//! * [`expr`] — the per-function recursive tree-walk that emits the ops.
//!
//! # Scope
//!
//! Currently lowered:
//! * the **scalar / control-flow subset** — integer and floating-point values,
//!   arithmetic and comparison, `if`, `let`/sequencing, direct calls, exported
//!   trampolines;
//! * **by-value (`[value]`) records** — construction via
//!   `reussir.record.compound`, field projection via `reussir.record.extract`;
//! * **shared (`[shared]`) records** — heap-allocated and reference-counted
//!   (`reussir.rc.create`, borrow/project/load), with the ownership analysis
//!   placing the `dup`/`drop` reference-count ops (see [`expr`]);
//! * **regional (`[regional]`) records** — construction of region-allocated
//!   `flex` boxes (`reussir.rc.create … region`), `region-run` scopes
//!   (`reussir.region.run`/`region.yield`), calls into `regional` functions
//!   (threading the implicit region handle), projection of their scalar /
//!   by-value / `[shared]` members (capability-aware borrow/project/load), and
//!   mutation of a `[field]` link (`c->f := …` — project the `nullable<rc<…>>`
//!   slot to a writable `field` reference and store), with the `Nullable`
//!   constructor lowering to `reussir.nullable.create`.
//! * **strings** — global string literals and string-pattern dispatch through
//!   the Reussir string dialect;
//! * **closures** — a closure lowers to a shared `rc<closure<…>>` built inline
//!   (`reussir.closure.create` with a body region; the closure-outlining pass
//!   generates the evaluate/drop/clone functions), with captures supplied as the
//!   leading inputs via `reussir.closure.apply`. A call first makes its
//!   (possibly-shared) target unique with `reussir.closure.uniqify` — `apply`
//!   mutates the closure box in place — then applies its arguments the same way
//!   and, once fully applied, evaluates with `reussir.closure.eval`; partial
//!   application stops at the residual closure.
//!
//! Reading a `[field]` link back, a non-`[field]` member that is itself a
//! regional record, and some enum paths are not yet lowered and surface as an explicit [`LoweringError`]
//! rather than wrong code.
//!
//! Every callee/trampoline target in the MIR is already resolved to an interned
//! [`mir::Symbol`](reussir_core::full::mir::Symbol) (mono did the mangling), so
//! lowering only looks names up in the program's symbol table — it needs no
//! `DefTable`/resolver/`Mangler`.

mod debug;
mod expr;
mod math;
mod ty;

use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use xxhash_rust::xxh3::xxh3_64;

use reussir_backend::builders;
use reussir_backend::melior::Context;
use reussir_backend::melior::ir::attribute::{ArrayAttribute, Attribute, StringAttribute};
use reussir_backend::melior::ir::operation::{OperationLike, OperationMutLike};
use reussir_backend::melior::ir::{BlockLike, Location, Module};
use reussir_backend::pipeline::{INLINE_TRANSFORM_ATTR, INLINE_TRANSFORM_ENTRY_POINT};

use reussir_core::full::mir;
use reussir_core::semi::TransformScript;
use reussir_core::semi::ty::TyCtxt;
use reussir_core::surface::Span;
use reussir_syntax::kind::{Resolver, TokenKey};

use crate::source::{FileId, SourceCache};
use expr::Lowerer;

/// How lowered functions map to LLVM linkage.
///
/// The JIT and the AOT compiler need opposite defaults. The REPL adds modules
/// to one ORC session incrementally and re-declares previously emitted
/// functions, so every definition must stay externally visible for
/// cross-module resolution. An AOT object wants the opposite: symbols as
/// narrow as possible so LLVM can discard, inline, and devirtualize freely,
/// and so separately compiled units agree on which definitions may collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkagePolicy {
    /// Every definition keeps external linkage (the pre-policy behavior).
    Jit,
    /// Ahead-of-time linkage, mirroring rustc's model:
    /// - monomorphized generic instances (`_RI…` symbols) are `weak_odr`
    ///   when `dedup_instances` holds — any other compilation unit may emit
    ///   the same instance and the linker keeps one; `weak_odr` rather than
    ///   `linkonce_odr` because the instance's home unit may not reference
    ///   it itself. The mechanism carrying the merge is per-format: ELF and
    ///   Mach-O bind weak symbols natively, while on COFF the backend
    ///   attaches `comdat any` to every weak-for-linker definition once the
    ///   final `llvm::Module` exists (`reussirAttachCoffComdats`, called
    ///   from `LlvmLowering::finish`);
    /// - private non-generic functions are `internal`, unless marked
    ///   `mono_export` (reachable from a generic body a foreign package may
    ///   instantiate, so the symbol must survive for cross-package links);
    /// - `pub` non-generic functions and trampolines stay strong external —
    ///   they are the object's ABI surface.
    Aot { dedup_instances: bool },
}

impl LinkagePolicy {
    /// The AOT policy for a target triple (`None` = native host).
    ///
    /// Currently target-independent: instances dedup on every format, since
    /// the COFF gap (no weak symbol binding) is closed downstream by the
    /// backend's comdat attachment on the final `llvm::Module`. The triple
    /// stays in the signature as the seam any future per-target policy
    /// hangs off.
    pub fn aot_for_triple(_triple: Option<&str>) -> Self {
        LinkagePolicy::Aot {
            dedup_instances: true,
        }
    }
}

/// A construct the current lowering subset does not handle.
///
/// Most lowering failures are internal and therefore carry only a message. A
/// failure originating in source-owned metadata, such as an inline transform,
/// additionally carries its file and byte span so drivers can render it through
/// the same source-caret path as parser and elaboration diagnostics.
#[derive(Debug, Clone)]
pub struct LoweringError(LoweringErrorKind);

#[derive(Debug, Clone)]
enum LoweringErrorKind {
    Message(Cow<'static, str>),
    Source {
        message: Cow<'static, str>,
        file: FileId,
        span: Option<Span>,
    },
}

impl From<&'static str> for LoweringErrorKind {
    fn from(message: &'static str) -> Self {
        Self::Message(Cow::Borrowed(message))
    }
}

impl From<String> for LoweringErrorKind {
    fn from(message: String) -> Self {
        Self::Message(Cow::Owned(message))
    }
}

impl From<Cow<'static, str>> for LoweringErrorKind {
    fn from(message: Cow<'static, str>) -> Self {
        Self::Message(message)
    }
}

impl LoweringError {
    fn at_source(message: impl Into<Cow<'static, str>>, file: FileId, span: Option<Span>) -> Self {
        Self(LoweringErrorKind::Source {
            message: message.into(),
            file,
            span,
        })
    }

    /// The diagnostic text without a source-name prefix.
    pub fn message(&self) -> &str {
        match &self.0 {
            LoweringErrorKind::Message(message) | LoweringErrorKind::Source { message, .. } => {
                message
            }
        }
    }

    /// The source file and byte span, when lowering traced this failure back to
    /// source-owned metadata. A missing span still retains the file identity for
    /// a named plain-text fallback.
    pub fn source_location(&self) -> Option<(FileId, Option<Span>)> {
        match &self.0 {
            LoweringErrorKind::Message(_) => None,
            LoweringErrorKind::Source { file, span, .. } => Some((*file, *span)),
        }
    }
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lowering error: {}", self.message())?;
        if let Some((file, span)) = self.source_location() {
            match span {
                Some(span) => write!(
                    f,
                    " (file {}, bytes {}..{})",
                    file.index(),
                    span.start,
                    span.end
                ),
                None => write!(f, " (file {})", file.index()),
            }
        } else {
            Ok(())
        }
    }
}

impl std::error::Error for LoweringError {}

type Result<T> = std::result::Result<T, LoweringError>;

fn err<T>(msg: impl Into<Cow<'static, str>>) -> Result<T> {
    Err(LoweringError(LoweringErrorKind::Message(msg.into())))
}

/// A sanitizer the build is compiled for. Lowering annotates every function
/// it emits with the matching LLVM function attribute (via the `passthrough`
/// mechanism) so the sanitizer instrumentation passes act on the generated
/// code: LLVM's ASan/MSan/TSan passes instrument **plain memory accesses only
/// in functions carrying `sanitize_address`/`sanitize_memory`/
/// `sanitize_thread`** — atomics, function entries, and allocator
/// interceptors are visible regardless, which makes an unannotated binary
/// look deceptively instrumented. Clang attaches these attributes to every
/// function it frontends under `-fsanitize=…`, but never to `-x ir` input,
/// so the annotation must come from this compiler.
///
/// `Leak` and `Undefined` carry no function attribute: leak checking is
/// purely a runtime/allocator mechanism, and UBSan's checks are emitted by
/// the C/C++ frontend rather than an attribute-driven LLVM pass. They are
/// accepted so a driver can uniformly declare the sanitizers a build targets.
///
/// The module is also stamped with a `reussir.sanitize` attribute (the
/// attribute-string array) so backend passes that *create* functions after
/// codegen — outlined closure bodies, outlined drop/acquire glue,
/// trampolines — attach the same set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sanitizer {
    Address,
    Leak,
    Memory,
    Thread,
    Undefined,
}

impl Sanitizer {
    /// The LLVM function attribute this sanitizer's instrumentation pass keys
    /// on, or `None` when instrumentation is not attribute-driven.
    pub fn function_attribute(self) -> Option<&'static str> {
        match self {
            Sanitizer::Address => Some("sanitize_address"),
            Sanitizer::Memory => Some("sanitize_memory"),
            Sanitizer::Thread => Some("sanitize_thread"),
            Sanitizer::Leak | Sanitizer::Undefined => None,
        }
    }
}

/// The module attribute carrying the sanitizer attribute strings, consumed by
/// the backend passes that create functions after codegen (mirrored in
/// `include/Reussir/IR/ReussirOps.h`).
const SANITIZE_ATTR: &str = "reussir.sanitize";

/// The deduplicated attribute strings for a sanitizer set.
fn sanitizer_attr_strings(sanitizers: &[Sanitizer]) -> Vec<&'static str> {
    let mut attrs = Vec::new();
    for s in sanitizers {
        if let Some(a) = s.function_attribute()
            && !attrs.contains(&a)
        {
            attrs.push(a);
        }
    }
    attrs
}

/// One codegen unit of a partitioned compilation: this unit's index and the
/// total unit count.
///
/// Functions are assigned to units by a **stable** hash of their mangled
/// symbol (xxh3 — a portable, spec-defined hash, not a std hasher, so the
/// partition is deterministic across builds and compiler versions): a
/// function's body is emitted only in its home unit, every other unit sees a
/// body-less
/// declaration the linker resolves across objects. Symbols are mangled from
/// fully-qualified paths, so the assignment is reproducible for a given
/// program. With more than one unit, `private` functions are promoted to
/// external visibility — a cross-unit call must resolve at link time.
#[derive(Clone, Copy, Debug)]
pub struct CodegenUnit {
    pub index: u32,
    pub count: u32,
}

impl CodegenUnit {
    /// Whether `symbol`'s body belongs to this unit.
    pub fn is_home(&self, symbol: &str) -> bool {
        self.count <= 1
            || xxh3_64(symbol.as_bytes()) % u64::from(self.count) == u64::from(self.index)
    }

    /// Whether this partition splits the program at all (count > 1).
    pub fn is_split(&self) -> bool {
        self.count > 1
    }
}

/// Build the MLIR module for a whole program in `context`.
///
/// `tcx` is the type arena the program was monomorphized in; the ownership
/// analysis that places reference-count ops is run per function against it.
/// `source`, when given, labels each lowered op with a `FileLineColLoc` resolved
/// from the MIR's byte spans (and carries the file name into the module's debug
/// attributes); without it, ops get `unknown` locations.
///
/// `names`, when given *together with* `source`, additionally emits DWARF debug
/// info: it resolves the interned source names of functions, parameters, and
/// locals so the debug-info conversion pass can describe each variable and its
/// (precise) type. Without `names`, only line locations are emitted.
pub fn lower_program<'c, 'tcx>(
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    program: &mir::Program<'tcx>,
    source: Option<&SourceCache>,
    names: Option<&dyn Resolver<TokenKey>>,
    linkage: LinkagePolicy,
    sanitizers: &[Sanitizer],
) -> Result<Module<'c>> {
    lower_unit(
        context,
        tcx,
        program,
        source,
        names,
        CodegenUnit { index: 0, count: 1 },
        linkage,
        sanitizers,
    )
}

/// Build the MLIR module for one [`CodegenUnit`] of `program`: this unit's
/// functions with bodies, every other unit's as declarations, trampolines
/// co-located with their targets, and the full record set (layout metadata
/// carries no symbols, so each unit is self-contained for types).
pub fn lower_unit<'c, 'tcx>(
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    program: &mir::Program<'tcx>,
    source: Option<&SourceCache>,
    names: Option<&dyn Resolver<TokenKey>>,
    unit: CodegenUnit,
    linkage: LinkagePolicy,
    sanitizers: &[Sanitizer],
) -> Result<Module<'c>> {
    let mut module = Module::new(Location::unknown(context));
    let sanitize_attrs = sanitizer_attr_strings(sanitizers);
    let lowerer = Lowerer::new(context, tcx, program, source, names, linkage)
        .with_unit(unit)
        .with_sanitize_attrs(sanitize_attrs.clone());
    lowerer.set_module_debug_attrs(&mut module);
    // Backend passes that create functions after codegen (outlined closure
    // bodies, drop/acquire glue, trampolines) read this module attribute and
    // attach the same sanitizer set — see `inheritSanitizerPassthrough`.
    if !sanitize_attrs.is_empty() {
        let strings: Vec<Attribute> = sanitize_attrs
            .iter()
            .map(|a| StringAttribute::new(context, a).into())
            .collect();
        module
            .as_operation_mut()
            .set_attribute(SANITIZE_ATTR, ArrayAttribute::new(context, &strings).into());
    }
    let body = module.body();
    for (token, payload) in &program.string_literals {
        body.append_operation(builders::str_global(
            context,
            &token.mangle(),
            payload,
            Location::unknown(context),
        ));
    }
    for func in &program.functions {
        body.append_operation(lowerer.function(func)?);
    }
    for t in &program.trampolines {
        // A trampoline is a body-carrying wrapper: it belongs to its target's
        // home unit (the wrapper and the wrapped body co-locate; for an
        // import, the target *is* the materialized definition).
        if !unit.is_home(program.symbol(t.target)) {
            continue;
        }
        let build = if t.import {
            builders::trampoline_import
        } else {
            builders::trampoline_export
        };
        body.append_operation(build(
            context,
            &t.abi,
            program.symbol(t.export),
            program.symbol(t.target),
            Location::unknown(context),
        ));
    }
    // The boundary wrapper texture of each `#[ffi(import)]` instance rides
    // its native function's home unit, next to the import trampoline.
    for import in &program.ffi_imports {
        if !unit.is_home(program.symbol(import.symbol)) {
            continue;
        }
        body.append_operation(builders::polyffi_texture(
            context,
            &import.texture,
            Location::unknown(context),
        ));
    }
    // Standalone foreign textures (opaque drop hooks), anchored by symbol.
    for texture in &program.ffi_textures {
        if !unit.is_home(program.symbol(texture.anchor)) {
            continue;
        }
        body.append_operation(builders::polyffi_texture(
            context,
            &texture.texture,
            Location::unknown(context),
        ));
    }
    // Reussir-side rc glue the foreign wrappers link against; one pair per
    // shared-record instance that crosses the boundary, in its home unit.
    for glue in &program.ffi_rc_glue {
        if !unit.is_home(program.symbol(glue.acquire)) {
            continue;
        }
        for op in lowerer.ffi_rc_glue_pair(glue)? {
            body.append_operation(op);
        }
    }
    // Every opaque instance's drop hook is declared in every unit: the
    // `rc.dec` lowering calls it, and its verifier requires the symbol. The
    // definition arrives with the linked foreign bitcode.
    for record in &program.records {
        let mir::RecordLayout::Opaque { drop_hook, .. } = record.layout else {
            continue;
        };
        body.append_operation(lowerer.ffi_drop_hook_decl(record, drop_hook)?);
    }
    embed_inline_transforms(context, &mut module, &program.transform_scripts)?;
    tracing::debug!(mlir = %module.as_operation(), "lowered program to MLIR");
    Ok(module)
}

fn embed_inline_transforms<'c>(
    context: &'c Context,
    module: &mut Module<'c>,
    scripts: &[TransformScript],
) -> Result<()> {
    if scripts.is_empty() {
        return Ok(());
    }

    for script in scripts {
        let (validation, body_start, body_end) = validation_module(&script.body);
        parse_transform_module(
            context,
            &validation,
            "inline transform",
            Some(TransformSource {
                script,
                body_start,
                body_end,
            }),
        )?;
    }

    let parsed = parse_transform_module(
        context,
        &inline_transform_module(scripts),
        "compiler-generated inline transform dispatcher",
        None,
    )?;
    let target = module.body();
    let mut operation = parsed.body().first_operation();
    while let Some(current) = operation {
        operation = current.next_in_block();
        target.append_operation(unsafe { current.to_ref() }.clone());
    }

    let mut module = module.as_operation_mut();
    module.set_attribute("transform.with_named_sequence", Attribute::unit(context));
    module.set_attribute(INLINE_TRANSFORM_ATTR, Attribute::unit(context));
    Ok(())
}

fn inline_transform_module(scripts: &[TransformScript]) -> String {
    let mut module = String::from("module attributes {transform.with_named_sequence} {\n");
    for (index, script) in scripts.iter().enumerate() {
        let _ = writeln!(
            module,
            r#"transform.named_sequence @__reussir_inline_{index}(
%target: !transform.any_op {{transform.readonly}}) {}"#,
            script.body
        );
    }
    let _ = writeln!(
        module,
        r#"transform.named_sequence @{INLINE_TRANSFORM_ENTRY_POINT}(
%module: !transform.any_op {{transform.readonly}}) {{
%target = transform.structured.match ops{{["func.func"]}}
attributes{{reussir.transform_anchor}} in %module
: (!transform.any_op) -> !transform.any_op"#
    );
    for index in 0..scripts.len() {
        let _ = writeln!(
            module,
            r#"transform.include @__reussir_inline_{index} failures(propagate) (%target)
: (!transform.any_op) -> ()"#
        );
    }
    module.push_str("transform.yield\n}\n}\n");
    module
}

fn validation_module(body: &str) -> (String, usize, usize) {
    const PREFIX: &str = r#"module attributes {transform.with_named_sequence} {
transform.named_sequence @__reussir_validate(
%target: !transform.any_op {transform.readonly}) "#;
    let mut module = String::with_capacity(PREFIX.len() + body.len() + 2);
    module.push_str(PREFIX);
    let body_start = module.len();
    module.push_str(body);
    let body_end = module.len();
    module.push_str("\n}");
    (module, body_start, body_end)
}

#[derive(Clone, Copy)]
struct TransformSource<'a> {
    script: &'a TransformScript,
    /// Half-open byte range occupied by `script.body` in the generated wrapper.
    body_start: usize,
    body_end: usize,
}

impl TransformSource<'_> {
    fn source_span(self, wrapper_offset: usize) -> Option<Span> {
        let script_span = self.script.span?;
        if self.script.body.is_empty()
            || wrapper_offset < self.body_start
            || wrapper_offset > self.body_end
        {
            return None;
        }
        // A parser error at the body's closing boundary should still point at
        // the final source byte rather than escaping the transform span.
        let relative = (wrapper_offset - self.body_start).min(self.script.body.len() - 1);
        let tail = self.script.body.get(relative..)?;
        let width = tail.chars().next()?.len_utf8();
        let start = script_span
            .start
            .checked_add(u32::try_from(relative).ok()?)?;
        let end = start.checked_add(u32::try_from(width).ok()?)?;
        (end <= script_span.end).then_some(Span { start, end })
    }
}

#[derive(Debug)]
struct CapturedDiagnostic {
    message: String,
    /// One-based coordinates in the generated validation module.
    line_col: Option<(usize, usize)>,
}

fn parse_transform_module<'c>(
    context: &'c Context,
    text: &str,
    description: &str,
    source: Option<TransformSource<'_>>,
) -> Result<Module<'c>> {
    if let Some(offset) = text.find('\0') {
        return Err(transform_error(
            description,
            "transform contains a NUL byte".to_string(),
            source,
            source.and_then(|source| source.source_span(offset)),
        ));
    }
    let diagnostics = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&diagnostics);
    let handler = context.attach_diagnostic_handler(move |diagnostic| {
        let line_col = diagnostic_line_col(diagnostic.location());
        captured
            .lock()
            .expect("diagnostic lock")
            .push(CapturedDiagnostic {
                message: diagnostic.to_string(),
                line_col,
            });
        true
    });
    let module = Module::parse(context, text);
    let verified = module
        .as_ref()
        .is_some_and(|module| module.as_operation().verify());
    context.detach_diagnostic_handler(handler);
    if verified {
        return Ok(module.expect("verified module"));
    }
    let diagnostics = diagnostics.lock().expect("diagnostic lock");
    let detail = if diagnostics.is_empty() {
        "MLIR rejected the transform module".to_string()
    } else {
        diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    };
    let span = source.and_then(|source| {
        diagnostics.iter().find_map(|diagnostic| {
            diagnostic
                .line_col
                .and_then(|(line, column)| byte_offset_at(text, line, column))
                .and_then(|offset| source.source_span(offset))
        })
    });
    Err(transform_error(description, detail, source, span))
}

fn transform_error(
    description: &str,
    detail: String,
    source: Option<TransformSource<'_>>,
    span: Option<Span>,
) -> LoweringError {
    let message = format!("invalid {description}: {detail}");
    match source {
        Some(source) => {
            LoweringError::at_source(message, source.script.file, span.or(source.script.span))
        }
        None => LoweringError(LoweringErrorKind::Message(message.into())),
    }
}

fn diagnostic_line_col(location: Location<'_>) -> Option<(usize, usize)> {
    // MLIR's C API exposes construction but not decomposition of a plain
    // FileLineColLoc. Its canonical textual form is `loc("file":line:column)`;
    // split from the right so Windows drive-letter colons remain harmless.
    let location = location.to_string();
    let location = location.strip_prefix("loc(")?.strip_suffix(')')?;
    let (prefix, column) = location.rsplit_once(':')?;
    let (_, line) = prefix.rsplit_once(':')?;
    Some((line.parse().ok()?, column.parse().ok()?))
}

fn byte_offset_at(text: &str, line: usize, column: usize) -> Option<usize> {
    if line == 0 || column == 0 {
        return None;
    }
    let mut line_start = 0;
    for _ in 1..line {
        line_start += text.get(line_start..)?.find('\n')? + 1;
    }
    let line_end = text
        .get(line_start..)?
        .find('\n')
        .map_or(text.len(), |offset| line_start + offset);
    let offset = line_start.checked_add(column - 1)?;
    (offset <= line_end).then_some(offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reussir_backend::melior::ir::operation::OperationLike;
    use reussir_core::full::mono::monomorphize;
    use reussir_core::{in_arena, semi::elaborate, surface};

    /// Elaborate + monomorphize + lower `source`, checking the module verifies
    /// and returning its printed text for assertions.
    fn lower_source(source: &str) -> String {
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let module = lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                .expect("lowering succeeds");
            assert!(
                module.as_operation().verify(),
                "module verifies:\n{}",
                module.as_operation()
            );
            module.as_operation().to_string()
        })
    }

    #[test]
    fn embeds_inline_transform_dispatcher() {
        let printed = lower_source(
            r#"#[transform_anchor]
fn target(x: i32) -> i32 { x }
transform [{
  transform.annotate %target "test.first" : !transform.any_op
  transform.yield
}];
transform [{
  transform.annotate %target "test.second" : !transform.any_op
  transform.yield
}];"#,
        );
        for expected in [
            "reussir.inline_transform",
            "reussir.transform_anchor",
            "no_inline",
            "@__reussir_inline_0",
            "@__reussir_inline_1",
            "@__reussir_inline_transform",
        ] {
            assert!(printed.contains(expected), "missing {expected}:\n{printed}");
        }
        let first = printed.find("include @__reussir_inline_0").unwrap();
        let second = printed.find("include @__reussir_inline_1").unwrap();
        assert!(first < second, "scripts must dispatch in source order");
    }

    #[test]
    fn reports_invalid_inline_transform_at_source() {
        let source = r#"#[transform_anchor]
fn target(x: i32) -> i32 { x }
transform [{
  transform.not_a_real_op %target
  transform.yield
}];"#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(source);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let program = surface::program(&parse.root);
            let elab = elaborate(tcx, &program, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let error = lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                .expect_err("Melior must reject the transform body");
            assert!(
                error.message().contains("transform.not_a_real_op"),
                "{error}"
            );
            let (file, span) = error.source_location().expect("source-owned diagnostic");
            assert_eq!(file, crate::source::FileId::ROOT);
            let span = span.expect("exact malformed-operation span");
            let expected = u32::try_from(source.find("transform.not_a_real_op").unwrap()).unwrap();
            assert_eq!(span.start, expected);
            assert_eq!(span.end, expected + 1);
            // The structured fallback remains usable by embedders that do not
            // have a SourceCache to render against.
            assert!(error.to_string().contains("file 0"), "{error}");
        });
    }

    #[test]
    fn attaches_source_locations() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        let src = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n";
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut map = crate::source::SourceCache::new();
            map.add_file("add.rr", src);
            let module = lower_program(
                &context,
                tcx,
                &full,
                Some(&map),
                None,
                LinkagePolicy::Jit,
                &[],
            )
            .expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with locations");
            assert!(printed.contains("loc(\"add.rr\":1:"), "{printed}");
        });
    }

    #[test]
    fn emits_debug_info_for_a_function() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        // Exercises every kind of debug type emitted: scalars, a `[value]` record
        // (with named members), a `[shared]` record (boxed), and a `let` local.
        let src = r#"
            pub struct [value] Point { x: i64, y: f64 }
            pub struct [shared] Boxed { n: i64 }
            fn proj(p: Point) -> i64 { let q = p.x; q }
            fn unbox(b: Boxed) -> i64 { b.n }
            pub fn entry(a: i64, b: f64) -> i64 {
                let p = Point { x: a, y: b };
                let bx = Boxed { n: a };
                proj(p) + unbox(bx)
            }
            extern "C" trampoline "entry" = entry;
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut map = crate::source::SourceCache::new();
            map.add_file("pt.rr", src);
            let module = lower_program(
                &context,
                tcx,
                &full,
                Some(&map),
                Some(parse.resolver()),
                LinkagePolicy::Jit,
                &[],
            )
            .expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with debug info");
            // Module file attributes, the subprogram + parameter array, and every
            // debug-type form including precise member names and the boxed type.
            assert!(printed.contains("reussir.dbg.file_basename"), "{printed}");
            assert!(printed.contains("dbg_subprogram"), "{printed}");
            assert!(printed.contains("dbg_func_args"), "{printed}");
            assert!(printed.contains("dbg_recordtype"), "{printed}");
            assert!(printed.contains("dbg_inttype"), "{printed}");
            assert!(printed.contains("dbg_fptype"), "{printed}");
            assert!(printed.contains("dbg_boxedtype"), "{printed}");
            assert!(printed.contains("dbg_localvar"), "{printed}");
            // Field names are preserved (not positional).
            assert!(printed.contains("name : \"x\""), "{printed}");
            assert!(printed.contains("name : \"y\""), "{printed}");
        });
    }

    #[test]
    fn emits_debug_info_for_variants() {
        use reussir_backend::melior::ir::operation::OperationPrintingFlags;
        // A `[value]` enum (inline tagged union) and a managed (`[shared]`) enum
        // (a boxed variant). Each enum-typed parameter gets a variant debug type;
        // the shared one is additionally a boxed type. Case names ride along as
        // member names.
        let src = r#"
            pub enum [value] Tag { A, B(i64) }
            pub enum List { Nil, Cons(i64, List) }
            fn use_tag(t: Tag) -> i64 { 0 }
            fn use_list(l: List) -> i64 { 0 }
            pub fn entry(n: i64) -> i64 { use_tag(Tag::B{n}) + use_list(List::Nil) }
            extern "C" trampoline "entry" = entry;
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut map = crate::source::SourceCache::new();
            map.add_file("variant.rr", src);
            let module = lower_program(
                &context,
                tcx,
                &full,
                Some(&map),
                Some(parse.resolver()),
                LinkagePolicy::Jit,
                &[],
            )
            .expect("lowering succeeds");
            let printed = module
                .as_operation()
                .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
                .expect("print with debug info");
            // A variant debug type is emitted, and the shared variant is boxed.
            assert!(printed.contains("dbg_recordtype"), "{printed}");
            assert!(printed.contains("is_variant : true"), "{printed}");
            assert!(printed.contains("dbg_boxedtype"), "{printed}");
            // Case names ride along as the variant's member names.
            assert!(printed.contains("\"A\""), "{printed}");
            assert!(printed.contains("\"B\""), "{printed}");
            assert!(printed.contains("\"Nil\""), "{printed}");
            assert!(printed.contains("\"Cons\""), "{printed}");
        });
    }

    #[test]
    fn codegen_units_partition_every_symbol_exactly_once() {
        let symbols = ["_RNvC1p1a", "_RNvC1p1b", "_RNvNvC1p1m1c", "_RC4main"];
        for count in [1u32, 2, 3, 7] {
            for sym in symbols {
                let homes = (0..count)
                    .filter(|&index| CodegenUnit { index, count }.is_home(sym))
                    .count();
                assert_eq!(
                    homes, 1,
                    "symbol {sym} must be home in exactly one of {count} units"
                );
            }
        }
        // The assignment is a pure function of the symbol (deterministic).
        let unit = CodegenUnit { index: 1, count: 3 };
        assert_eq!(unit.is_home("_RNvC1p1a"), unit.is_home("_RNvC1p1a"));
    }

    #[test]
    fn a_partitioned_unit_declares_foreign_functions_and_defines_its_own() {
        let src = r#"
            fn helper(n: u64) -> u64 { n + 1 }
            pub fn entry(n: u64) -> u64 { helper(n) }
            extern "C" trampoline "entry" = entry;
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");

            let count = 4;
            let mut definitions = 0;
            let mut trampolines = 0;
            for index in 0..count {
                let unit = CodegenUnit { index, count };
                // AOT policy on purpose: a partitioned build must keep every
                // home definition externally linkable (no `internal`), which
                // the policy guarantees via the same `is_split` promotion.
                let module = lower_unit(
                    &context,
                    tcx,
                    &full,
                    None,
                    None,
                    unit,
                    LinkagePolicy::Aot {
                        dedup_instances: true,
                    },
                    &[],
                )
                .expect("unit lowering succeeds");
                assert!(
                    module.as_operation().verify(),
                    "unit {index} verifies:\n{}",
                    module.as_operation()
                );
                let text = module.as_operation().to_string();
                // Both symbols appear in every unit (definition or declaration)...
                for f in &full.functions {
                    assert!(text.contains(full.symbol(f.symbol)), "{text}");
                    if unit.is_home(full.symbol(f.symbol)) {
                        definitions += 1;
                        // ...a home definition is externally visible (no
                        // `private`, which multi-unit promotion strips)...
                        let head = text
                            .lines()
                            .find(|l| l.contains(&format!("func.func @{}(", full.symbol(f.symbol))))
                            .unwrap_or_else(|| {
                                panic!("missing head for {}", full.symbol(f.symbol))
                            });
                        assert!(!head.contains("private"), "{head}");
                    }
                }
                // ...and the trampoline rides with its target's home unit.
                if text.contains("reussir.trampoline") {
                    trampolines += 1;
                }
            }
            // Every function body lands in exactly one unit.
            assert_eq!(definitions, full.functions.len());
            assert_eq!(trampolines, 1, "the trampoline belongs to one unit");
        });
    }

    #[test]
    fn lowers_fibonacci_to_verifiable_mlir() {
        let src = r#"
            pub fn fibonacci(n: u64) -> u64 {
                if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
            }
            extern "C" trampoline "fibonacci_ffi" = fibonacci;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("func.func @_RC9fibonacci"), "{mlir}");
        assert!(mlir.contains("arith.cmpi ule"), "{mlir}");
        assert!(mlir.contains("scf.if"), "{mlir}");
        assert!(mlir.contains("call @_RC9fibonacci"), "{mlir}");
        assert!(mlir.contains("reussir.trampoline"), "{mlir}");
    }

    #[test]
    fn lowers_value_records_to_verifiable_mlir() {
        // A `[value]` record is a by-value aggregate: construction lowers to
        // `reussir.record.compound` and field access to `reussir.record.extract`.
        let src = r#"
            pub struct [value] Point { x: i64, y: i64 }
            pub fn dot(a: Point, b: Point) -> i64 { a.x * b.x + a.y * b.y }
            pub fn mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }
            extern "C" trampoline "dot_ffi" = dot;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.record.compound"), "{mlir}");
        assert!(mlir.contains("reussir.record.extract"), "{mlir}");
        assert!(mlir.contains("!reussir.record<"), "{mlir}");
    }

    #[test]
    fn lowers_shared_records_to_reference_counted_mlir() {
        // A `[shared]` record is heap-allocated and reference-counted. Building
        // one lowers to `record.compound` + `rc.create`; reading a field borrows
        // the box (`rc.borrow`) and navigates it (`ref.project` / `ref.load`); and
        // the ownership analysis surrounds the uses with `rc.inc` / `rc.dec`.
        //
        // `Outer` nests a shared `Inner`, so its field is stored as an `rc` link:
        // projecting it out loads the inner pointer and retains it (`rc.inc`),
        // while the consumed outer box is released (`rc.dec`).
        let src = r#"
            pub struct [shared] Inner { n: i64 }
            pub struct [shared] Outer { inner: Inner }
            fn mk_inner(n: i64) -> Inner { Inner { n: n } }
            fn mk_outer(i: Inner) -> Outer { Outer { inner: i } }
            fn inner_of(o: Outer) -> Inner { o.inner }
            fn n_of(i: Inner) -> i64 { i.n }
            pub fn build_and_read(n: i64) -> i64 { n_of(inner_of(mk_outer(mk_inner(n)))) }
            extern "C" trampoline "build_and_read_ffi" = build_and_read;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("reussir.rc.borrow"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.load"), "{mlir}");
        assert!(mlir.contains("reussir.rc.inc"), "{mlir}");
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
    }

    #[test]
    fn lowers_a_chained_projection_through_shared_records() {
        // `o.inner.n` is a single projection with a two-element path crossing two
        // shared records: borrow the outer box, project + load the inner `rc` link,
        // borrow that, then project the scalar — so the walk emits two `rc.borrow`s.
        let src = r#"
            pub struct [shared] Inner { n: i64 }
            pub struct [shared] Outer { inner: Inner }
            pub fn deep(o: Outer) -> i64 { o.inner.n }
        "#;
        let mlir = lower_source(src);
        assert_eq!(
            mlir.matches("reussir.rc.borrow").count(),
            2,
            "expected two borrows for a two-level shared chain:\n{mlir}"
        );
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
    }

    #[test]
    fn lowers_a_recursive_shared_record() {
        // A `[shared]` record whose field is the record itself is finite — the
        // field is an `rc` pointer, not an inline copy. Lowering its type must
        // terminate: the identified record is published on the construction stack
        // before its members, so the self-reference on the field resolves to the
        // in-progress handle instead of recursing forever.
        let src = r#"
            pub struct [shared] Node { next: Node }
            pub fn forward(n: Node) -> Node { n }
        "#;
        let mlir = lower_source(src);
        // The record's own field is an `rc` link back to itself.
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("Node"), "{mlir}");
    }

    #[test]
    fn drops_an_unused_shared_parameter() {
        // An owned `[shared]` parameter that the body never uses is dead on entry,
        // so the ownership analysis releases it immediately with `rc.dec`.
        let src = r#"
            pub struct [shared] Box { v: i64 }
            pub fn ignore(b: Box, fallback: i64) -> i64 { fallback }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
    }

    #[test]
    fn lowers_value_variant_construction() {
        // A `[value]` enum is an inline tagged union. Each case's fields are packed
        // into its `{enum}::{case}` payload compound with `record.compound`, which
        // `record.variant [tag]` then tags into the variant record. A fieldless
        // case builds an empty payload compound. No `rc` is involved.
        let src = r#"
            pub enum [value] Shape { Dot, Segment(i64, i64) }
            pub fn dot() -> Shape { Shape::Dot }
            pub fn segment(x: i64, y: i64) -> Shape { Shape::Segment{x, y} }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.record<variant"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[0]"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[1]"), "{mlir}");
        assert!(mlir.contains("reussir.record.compound"), "{mlir}");
        // No heap allocation for a by-value enum.
        assert!(!mlir.contains("reussir.rc.create"), "{mlir}");
    }

    #[test]
    fn lowers_shared_variant_construction() {
        // A managed (default `[shared]`) enum boxes the tagged value: build the
        // case payload, tag it with `record.variant`, then `rc.create` the box. The
        // recursive `Cons` field is itself an `rc` link back to the enum, so the
        // record type lowers finitely (the self-reference resolves to the
        // in-progress handle).
        let src = r#"
            pub enum IntList { Nil, Cons(i64, IntList) }
            pub fn one(x: i64) -> IntList { IntList::Cons{x, IntList::Nil} }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("!reussir.record<variant"), "{mlir}");
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[0]"), "{mlir}");
        assert!(mlir.contains("reussir.record.variant[1]"), "{mlir}");
        assert_eq!(
            mlir.matches("reussir.rc.create").count(),
            2,
            "one box per constructed case (Nil and Cons):\n{mlir}"
        );
    }

    #[test]
    fn lowers_regional_record_construction_and_region_run() {
        // A `[regional]` record is region-allocated with a `flex` (region-local,
        // mutable) box. A `regional` function takes the region it allocates into
        // as an implicit `!reussir.region` parameter and builds into it
        // (`rc.create … region`); a `region-run` scope (`reussir.region.run`)
        // establishes that region and threads it into the regional call.
        let src = r#"
            pub struct [regional] TestCell { v: i64 }
            regional fn make(x: i64) -> [flex] TestCell { TestCell { v: x } }
            pub fn run(n: i64) -> TestCell { regional { make(n) } }
        "#;
        let mlir = lower_source(src);
        // The regional function carries the implicit region parameter and builds
        // a `flex` box into it.
        assert!(mlir.contains("!reussir.region"), "{mlir}");
        assert!(mlir.contains("!reussir.rc<"), "{mlir}");
        assert!(mlir.contains("flex"), "{mlir}");
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("region("), "{mlir}");
        // The value escapes the region, so `run` yields it as a `rigid` box.
        assert!(mlir.contains("rigid"), "{mlir}");
        // The region scope and its terminator.
        assert!(mlir.contains("reussir.region.run"), "{mlir}");
        assert!(mlir.contains("reussir.region.yield"), "{mlir}");
    }

    #[test]
    fn lowers_regional_record_through_the_pipeline() {
        // The regional construction subset lowers all the way to the LLVM
        // dialect: token instantiation and the region patterns pass turn
        // `region.run` into an allocation scope, attach the box vtable, and
        // freeze the flex result on the way out.
        let src = r#"
            pub struct [regional] TestCell { v: i64 }
            regional fn make(x: i64) -> [flex] TestCell { TestCell { v: x } }
            pub fn run(n: i64) -> TestCell { regional { make(n) } }
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the regional module to LLVM");
        });
    }

    #[test]
    fn lowers_regional_record_field_projection() {
        // Projecting a field out of a `flex` regional record borrows it at its
        // `flex` capability (`rc.borrow`), navigates by reference
        // (`reussir.ref.project`), and loads the scalar field (`reussir.ref.load`)
        // — the result leaves the region as a plain `i64`, so nothing escapes.
        let src = r#"
            pub struct [regional] TestCell { v: i64 }
            regional fn make(x: i64) -> [flex] TestCell { TestCell { v: x } }
            pub fn run(n: i64) -> i64 { regional { make(n).v } }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.borrow"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.load"), "{mlir}");
        // The reference into the region-local record carries `flex` capability.
        assert!(mlir.contains("ref<") && mlir.contains("flex"), "{mlir}");
    }

    #[test]
    fn lowers_regional_field_link_assignment() {
        // Assigning through a mutable `[field]` link on a `flex` regional record:
        // the record's `[field]` member lowers to a `nullable<rc<…>>` slot; the
        // assignment borrows the record at `flex`, projects the slot to a writable
        // `field`-capability reference, builds the `Nullable::NonNull{..}` value
        // (`reussir.nullable.create`), and stores it (`reussir.ref.store`).
        let src = r#"
            pub struct [regional] TestCell { v: i64, next: [field] TestCell }
            regional fn set(c: [flex] TestCell, x: [flex] TestCell) -> i64 {
                c->next := Nullable::NonNull{x};
                c.v
            }
        "#;
        let mlir = lower_source(src);
        // The `[field]` member is a nullable rc link in the record's layout.
        assert!(mlir.contains("!reussir.nullable<"), "{mlir}");
        // Projecting the `[field]` slot out of a `flex` parent yields a writable
        // `field`-capability reference.
        assert!(mlir.contains("field>"), "{mlir}");
        assert!(mlir.contains("reussir.nullable.create"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.store"), "{mlir}");
    }

    #[test]
    fn lowers_explicit_arc_member_as_atomic_link() {
        // An `Arc` member is the written form of the whole-subtree rule: the
        // member slot in the record type spells `rc<…, atomic>`, and reading
        // it out of a *bare* (normal) container hands out the atomic rc with
        // no arc context involved.
        let src = r#"
            pub struct Inner { v: i64 }
            pub struct Holder { h: Arc<Inner> }
            pub fn get(x: Holder) -> Arc<Inner> { x.h }
        "#;
        let mlir = lower_source(src);
        // The member type is the explicit atomic link inside the record body…
        assert!(
            mlir.contains("{!reussir.rc<!reussir.record<compound \"_RC5Inner\" {i64}> atomic>}"),
            "{mlir}"
        );
        // …and the container itself stays a normal shared box.
        assert!(!mlir.contains("\"_RC6Holder\"{{.*}} atomic"), "{mlir}");
    }

    #[test]
    fn promoted_member_stays_bare_in_the_declaration() {
        // The §3.3 spine: one record serves both colorings — the recursive
        // member is declared bare (no `atomic` inside the record body), and
        // the box's own atomic kind derives the interior discipline (atomic
        // create, atomic borrow).
        let src = r#"
            pub enum List<T> { Nil, Cons(T, List<T>) }
            pub fn make() -> Arc<List<i64>> {
                Arc<List<i64>>::Cons{1, Arc<List<i64>>::Nil}
            }
        "#;
        let mlir = lower_source(src);
        // The Cons arm's tail member is the bare variant record, not an rc.
        assert!(
            mlir.contains("{i64, !reussir.record<variant \"_RIC4ListxE\">}"),
            "{mlir}"
        );
        // The boxes create atomically and borrows inherit the context.
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("atomic>"), "{mlir}");
    }

    #[test]
    fn lowers_arc_to_an_atomic_rc_box() {
        // An arc'd value is the same shared record behind an atomically
        // counted box: the constructor creates `!reussir.rc<…, atomic>` and a
        // read borrows it at a matching `shared atomic` reference.
        let src = r#"
            pub struct Data { value: i64 }
            pub fn make(v: i64) -> Arc<Data> { Arc<Data> { value: v } }
            pub fn read(a: Arc<Data>) -> i64 { a.value }
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.create"), "{mlir}");
        assert!(mlir.contains("atomic>"), "{mlir}");
        assert!(mlir.contains("shared atomic>"), "{mlir}");
    }

    #[test]
    fn rejects_arc_instantiated_at_a_non_shared_record() {
        // A non-box `Arc` inner is diagnosed at elaboration (concrete) and by
        // mono's instantiation check (generic); the lowering rejection is the
        // backstop for MIR that reaches codegen without those passes (e.g.
        // directly lowered textual MIR). Strip the reports and drive lowering
        // directly to exercise it.
        let src = r#"
            pub struct [value] Pair { a: i64 }
            fn id<T>(x: T) -> T { x }
            pub fn bad(x: Arc<Pair>) -> Arc<Pair> { id(x) }
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            let (full, _) = monomorphize(&elab.mono_input());
            let err = lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                .expect_err("an arc over a non-shared record must not lower");
            assert!(
                err.to_string()
                    .contains("`Arc` requires a shared rc box inner"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn arc_of_array_reports_unimplemented_lowering() {
        // An array (or closure) is a valid `Arc` inner at the type level, but
        // only the `[shared]` record lowering has landed — keep the gap an
        // explicit error rather than a malformed box.
        let src = r#"
            pub fn f(x: Arc<[f64; 8]>) -> Arc<[f64; 8]> { x }
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "{:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "{reports:#?}");
            let err = lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                .expect_err("an arc'd array must not lower yet");
            assert!(
                err.to_string()
                    .contains("arc'd arrays and closures do not lower yet"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn rejects_nullable_over_a_non_pointer_element() {
        // The frontend's pointer-like check for a `Nullable` inner is
        // conservative and passes a `[value]` record; that record lowers to an
        // inline (non-pointer) type, which cannot sit behind a `nullable`. Lower
        // it to an explicit error rather than a malformed `nullable<record<…>>`.
        let src = r#"
            pub struct [value] Pair { a: i64 }
            fn wrap(p: Pair) -> Nullable<Pair> { Nullable::NonNull{ p } }
        "#;
        let context = crate::testing::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let err = lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                .expect_err("a nullable over a non-pointer element must not lower");
            assert!(
                err.to_string().contains("does not lower to a pointer"),
                "unexpected error: {err}"
            );
        });
    }

    #[test]
    fn lowers_regional_field_construction_and_self_link() {
        // The canonical loop-back shape, now that a `[field]` slot's null is
        // colored `flex`: construct a `flex` `TestCell` whose `[field]` `next` is
        // `Nullable::Null` (a null nullable), then link it to itself
        // (`c->next := Nullable::NonNull{c}`) and read a scalar back. Exercises
        // `[field]` construction (null + record-with-nullable-member layout) and
        // assignment together, all the way through the lowering pipeline.
        let src = r#"
            pub struct [regional] TestCell { v: i64, next: [field] TestCell }
            regional fn loop_back(seed: i64) -> i64 {
                let c = TestCell { v: seed, next: Nullable::Null };
                c->next := Nullable::NonNull{c};
                c.v
            }
            pub fn run(n: i64) -> i64 { regional { loop_back(n) } }
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            let printed = module.as_operation().to_string();
            // The null field and the self-link both build nullable rc values.
            assert!(printed.contains("reussir.nullable.create"), "{printed}");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the [field] module to LLVM");
        });
    }

    #[test]
    fn lowers_and_runs_through_the_pipeline() {
        let src = r#"
            pub fn add3(a: i32, b: i32, c: i32) -> i32 { a + b + c }
            extern "C" trampoline "add3_ffi" = add3;
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the scalar module to LLVM");
        });
    }

    #[test]
    fn uniqifies_a_zero_argument_closure_call_before_eval() {
        // A zero-argument call evaluates the target with no `apply`. `eval` still
        // hands the closure's payload to the body to consume, so a shared target
        // (which may carry payload from earlier curried applications) must be made
        // unique first — evaluating it in place would let the body consume values
        // another holder still owns. So the target is uniqified even with no
        // arguments to apply.
        let src = r#"
            pub fn thunk() -> i64 { (|| 42)() }
        "#;
        let mlir = lower_source(src);
        assert!(
            mlir.contains("reussir.closure.uniqify"),
            "no uniqify:\n{mlir}"
        );
        assert!(mlir.contains("reussir.closure.eval"), "no eval:\n{mlir}");
        assert!(
            !mlir.contains("reussir.closure.apply"),
            "unexpected apply:\n{mlir}"
        );
    }

    #[test]
    fn drops_a_frozen_regional_record_by_reference_count() {
        // A regional record that escapes its region is frozen to a `rigid` `rc`
        // box — a managed, reference-counted value, not an inline aggregate. When
        // such a value is bound and then goes dead, the ownership analysis drops
        // it, which must lower to a direct `reussir.rc.dec` on the `rc` pointer
        // (not an inline `ref.drop`, which would mis-handle the pointer). Here `c`
        // is read by two projections — a projection *borrows* the parent and the
        // extracted `i64` field is not itself reference-counted, so neither use
        // transfers ownership (no `dup`): `c` is simply dropped once after the
        // last one.
        let src = r#"
            pub struct [regional] TestCell { v: i64 }
            regional fn make(x: i64) -> [flex] TestCell { TestCell { v: x } }
            pub fn run(n: i64) -> i64 {
                let c = regional { make(n) };
                c.v + c.v
            }
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            let printed = module.as_operation().to_string();
            // The frozen `rigid` box is dropped with a direct reference-count
            // decrement, not spilled and dropped as an inline record.
            assert!(printed.contains("reussir.rc.dec"), "{printed}");
            assert!(printed.contains("rigid"), "{printed}");
            // Borrowing projections of a non-owned (`i64`) field transfer no
            // ownership, so `c` is dropped exactly once with no matching `inc`.
            assert!(!printed.contains("reussir.rc.inc"), "{printed}");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the frozen-record drop to LLVM");
        });
    }

    #[test]
    fn dups_a_frozen_regional_record_reused_across_calls() {
        // The `inc` dual of the drop test: a frozen `rigid` regional record reused
        // in two calls is not a last use at the first, so the ownership analysis
        // `dup`s it — which must lower to a direct `reussir.rc.inc` on the `rc`
        // pointer (the managed-rc path), then each callee consumes its own
        // reference. Passing `c` to `take` twice forces exactly this.
        let src = r#"
            pub struct [regional] TestCell { v: i64 }
            regional fn make(x: i64) -> [flex] TestCell { TestCell { v: x } }
            fn take(c: TestCell) -> i64 { c.v }
            pub fn run(n: i64) -> i64 {
                let c = regional { make(n) };
                take(c) + take(c)
            }
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            let printed = module.as_operation().to_string();
            // The reused frozen box is retained with a direct reference-count
            // increment, and dropped (by the callees / at last use) with a
            // decrement — both on the `rigid` `rc` pointer.
            assert!(printed.contains("reussir.rc.inc"), "{printed}");
            assert!(printed.contains("reussir.rc.dec"), "{printed}");
            assert!(printed.contains("rigid"), "{printed}");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the frozen-record reuse to LLVM");
        });
    }

    #[test]
    fn lowers_a_closure_with_a_capture_to_verifiable_mlir() {
        // A closure captures the enclosing `n` and takes one parameter `x`. It
        // lowers to an inline `reussir.closure.create` over `(n, x) -> i64`; the
        // captured `n` is supplied with a `reussir.closure.apply`, leaving a
        // `(x) -> i64` closure. Calling it applies `x` and evaluates
        // (`reussir.closure.eval`) the fully-applied closure.
        let src = r#"
            pub fn adder(n: i64) -> i64 { (|x: i64| x + n)(1) }
            extern "C" trampoline "adder_ffi" = adder;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.closure.create"), "{mlir}");
        assert!(mlir.contains("reussir.closure.apply"), "{mlir}");
        assert!(mlir.contains("reussir.closure.eval"), "{mlir}");
        assert!(mlir.contains("reussir.closure.yield"), "{mlir}");
        // The call makes the (possibly-shared) target unique before the in-place
        // `apply`; the capture-application on the fresh `create` result does not.
        assert!(mlir.contains("reussir.closure.uniqify"), "{mlir}");
        // The closure value is a shared `rc` over a `closure` type.
        assert!(mlir.contains("!reussir.closure<"), "{mlir}");
    }

    #[test]
    fn lowers_a_partially_applied_closure() {
        // Applying fewer arguments than parameters stops at the residual closure:
        // the inner `(…)(1)` supplies one of two parameters, so it lowers to a
        // `reussir.closure.apply` whose result is still a closure (no
        // `reussir.closure.eval`); the outer `(…)(2)` fully applies and evaluates.
        let src = r#"
            pub fn sum(n: i64) -> i64 { ((|x: i64, y: i64| x + y + n)(1))(2) }
            extern "C" trampoline "sum_ffi" = sum;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.closure.create"), "{mlir}");
        assert!(mlir.contains("reussir.closure.apply"), "{mlir}");
        // The final application evaluates the fully-applied closure.
        assert!(mlir.contains("reussir.closure.eval"), "{mlir}");
    }

    #[test]
    fn lowers_a_closure_through_the_pipeline() {
        // The whole closure path lowers to the LLVM dialect: the closure-outlining
        // pass turns the inline `closure.create` body into an evaluate function
        // with a vtable and drop/clone functions, and apply/eval lower to the
        // runtime protocol.
        let src = r#"
            pub fn adder(n: i64) -> i64 { (|x: i64| x + n)(1) }
            extern "C" trampoline "adder_ffi" = adder;
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the closure module to LLVM");
        });
    }

    #[test]
    fn lowers_a_match_on_a_shared_enum() {
        // A consuming traversal over a recursive `[shared]` enum: the scrutinee is
        // borrowed to a reference, `reussir.record.dispatch` branches on the tag,
        // the `Cons` arm projects/loads its fields out of the case-payload
        // reference, and each arm terminates with `reussir.scf.yield`.
        let src = r#"
            pub enum List<T> { Nil, Cons(T, List<T>) }
            pub fn sum(list: List<i64>) -> i64 {
                match list {
                    List::Nil => 0,
                    List::Cons(x, xs) => x + sum(xs)
                }
            }
            extern "C" trampoline "sum_ffi" = sum;
        "#;
        let mlir = lower_source(src);
        assert!(mlir.contains("reussir.rc.borrow"), "{mlir}");
        assert!(mlir.contains("reussir.record.dispatch"), "{mlir}");
        assert!(mlir.contains("reussir.ref.project"), "{mlir}");
        assert!(mlir.contains("reussir.ref.load"), "{mlir}");
        assert!(mlir.contains("reussir.scf.yield"), "{mlir}");
        // The consumed scrutinee is dropped in each arm; the aliased-out `xs`
        // link is retained before the recursive call consumes it.
        assert!(mlir.contains("reussir.rc.dec"), "{mlir}");
        assert!(mlir.contains("reussir.rc.inc"), "{mlir}");
    }

    #[test]
    fn lowers_a_match_through_the_pipeline() {
        // The whole match path lowers to the LLVM dialect: `record.dispatch`
        // expands during ConvertToSTD to an `scf.index_switch` and thence to LLVM
        // control flow, and the reference projections lower to GEP/load.
        let src = r#"
            pub enum List<T> { Nil, Cons(T, List<T>) }
            pub fn sum(list: List<i64>) -> i64 {
                match list {
                    List::Nil => 0,
                    List::Cons(x, xs) => x + sum(xs)
                }
            }
            extern "C" trampoline "sum_ffi" = sum;
        "#;
        let context = reussir_backend::context();
        in_arena(|tcx| {
            let parse = reussir_syntax::parse(src);
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, parse.resolver());
            assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
            let (full, reports) = monomorphize(&elab.mono_input());
            assert!(reports.is_empty(), "mono reports: {reports:#?}");
            let mut module =
                lower_program(&context, tcx, &full, None, None, LinkagePolicy::Jit, &[])
                    .expect("lowering succeeds");
            reussir_backend::pipeline::run_lowering_pipeline(
                &context,
                &mut module,
                &reussir_backend::pipeline::LoweringOptions::default(),
            )
            .expect("pipeline lowers the match module to LLVM");
        });
    }
}
