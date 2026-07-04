//! The REPL session: persistent elaborator + persistent JIT, one input at a
//! time.

use std::sync::Arc;

use melior::Context;
// Brings `verify` into scope for the pre-JIT module check.
use melior::ir::operation::OperationLike as _;
use rustc_hash::FxHashSet;

use reussir_backend::pipeline::{self, LoweringOptions};
use reussir_codegen::lower::lower_program;
use reussir_core::full::mono::monomorphize;
use reussir_core::semi::ty::{DefId, Flexivity, Ty, TyCtxt, TyKind};
use reussir_core::semi::{Checkpoint, Elaborator, Report, Severity};
use reussir_core::{in_arena, surface};
use reussir_jit::{ModuleHandle, OptLevel, OrcJit};
use reussir_syntax::diagnostics::ParseError;
use reussir_syntax::source::{FileId, SourceCache};
use reussir_syntax::{
    Interner, MultiThreadedTokenInterner, ReplInputKind, new_threaded_interner, parse_repl,
};

/// Session-wide configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub opt: OptLevel,
}

/// What one input produced. Frontends render these; the session never prints.
pub enum Outcome {
    /// Blank input; nothing to do.
    Empty,
    /// `:q` / `:quit`.
    Quit,
    /// `:clear` — the driver should tear the session down and start fresh.
    ClearRequested,
    /// Informational command output (`:help`, `:dump …`, `:type …`, `:set …`).
    Text(String),
    /// Definitions were accepted and (where instantiable) compiled.
    Definitions { count: usize, warnings: Vec<Report> },
    /// An expression was evaluated. `ty` is the rendered type; `value` the
    /// rendered result.
    Value {
        value: String,
        ty: String,
        warnings: Vec<Report>,
    },
    /// A global `let` was accepted: the value is now bound for later inputs
    /// (rebinding the same name overwrites, releasing the old value).
    Binding {
        name: String,
        value: String,
        ty: String,
        warnings: Vec<Report>,
    },
    /// Syntax errors in the input; `file` is the input's entry in the
    /// session's [`SourceCache`] (render against it).
    ParseErrors {
        file: FileId,
        errors: Vec<ParseError>,
    },
    /// Elaboration/monomorphization diagnostics (input was rolled back).
    Reports(Vec<Report>),
    /// A backend (lowering/JIT) failure; the input was rolled back.
    Backend(String),
}

/// Why a frontend's drive loop ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Quit,
    /// Restart with a fresh session (`:clear`).
    Clear,
}

/// Build a session (interner, MLIR context, JIT, arena scope) and hand it to
/// `drive` — a frontend loop. Returns how the loop ended; `Exit::Clear`
/// callers loop around to build a fresh session.
pub fn run<F>(config: Config, drive: F) -> Result<Exit, String>
where
    F: for<'a, 'tcx> FnOnce(&mut ReplSession<'a, 'tcx>) -> Exit,
{
    let interner = Arc::new(new_threaded_interner());
    let context = reussir_backend::context();
    let jit = OrcJit::with_runtime()?;
    Ok(in_arena(|tcx| {
        let mut session = ReplSession::new(tcx, &interner, &context, &jit, config);
        drive(&mut session)
    }))
}

/// One global `let` binding: an owned `__ReplBox<T>` kept alive between
/// inputs. The box releases itself ([`crate::value::BoundBox`]'s `Drop`) on
/// rebinding and on session drop — while the JIT (borrowed by the session,
/// so it strictly outlives it) still has the dec companion materialized.
struct GlobalBinding<'jit, 'tcx> {
    name: reussir_syntax::kind::TokenKey,
    display: String,
    ty: Ty<'tcx>,
    boxed: crate::value::BoundBox<'jit>,
}

/// A module added to the JIT whose input is not yet permanent.
struct PendingModule<'jit> {
    handle: ModuleHandle<'jit>,
    symbols: Vec<String>,
}

pub struct ReplSession<'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    interner: Arc<MultiThreadedTokenInterner>,
    pub(crate) elab: Elaborator<'a, 'tcx>,
    context: &'a Context,
    /// The session's boxing prelude record (`__ReplBox<T>`); every
    /// expression result crosses the C ABI wrapped in one (see
    /// `reussir_core::semi::repl`).
    repl_box: DefId,
    /// Ground record shapes harvested from every input's monomorphized
    /// program; the value walker reflects results through these.
    shapes: crate::reflect::ShapeTable<'tcx>,
    /// Printer hooks consulted before the structural fallback.
    pub printers: crate::reflect::PrinterRegistry,
    /// Tracked JIT modules, one per successfully added input. Each handle
    /// borrows the [`OrcJit`] (owned by [`run`]'s scope, outliving the
    /// session), so releasing every tracker before the LLJIT is disposed —
    /// disposal with live tracker references deadlocks — is enforced by the
    /// borrow rather than by drop order.
    modules: Vec<ModuleHandle<'a>>,
    jit: &'a OrcJit,
    /// Mangled symbols (and trampoline exports) already materialized in the
    /// JIT. Later modules emit these as body-less declarations
    /// ([`crate::externalize`]); ORC resolves the calls across modules.
    pub(crate) emitted: FxHashSet<String>,
    /// Global `let` bindings, in binding order — the order of the extra
    /// box parameters every later wrapper takes. Rebinding overwrites in
    /// place (the old box released); the whole list is released on session
    /// drop (`:clear` included) while the JIT still holds the dec
    /// companions.
    bindings: Vec<GlobalBinding<'a, 'tcx>>,
    /// The `__repl_expr_N` counter (advanced only on success).
    counter: usize,
    pub(crate) opt: OptLevel,
    /// The JIT's data layout and triple, stamped on every module before its
    /// lowering pipeline so MLIR `DataLayout` queries compute real target
    /// sizes and alignments (cached once — they never change per engine).
    data_layout: String,
    triple: String,
    /// Every input of the session, as `<repl>` virtual files. The persistent
    /// elaborator can report against an *older* input's span (e.g. a generic
    /// defined earlier failing to monomorphize now), so all inputs stay
    /// renderable — not just the current one. Grows monotonically; failed
    /// inputs' entries are harmless (`:clear` starts a fresh session).
    sources: SourceCache,
}

impl<'a, 'tcx> ReplSession<'a, 'tcx> {
    pub fn new(
        tcx: &'a TyCtxt<'tcx>,
        interner: &'a Arc<MultiThreadedTokenInterner>,
        context: &'a Context,
        jit: &'a OrcJit,
        config: Config,
    ) -> Self {
        let mut interning = interner.clone();
        let box_name = Interner::get_or_intern(&mut interning, "__ReplBox");
        let box_param = Interner::get_or_intern(&mut interning, "T");
        let box_field = Interner::get_or_intern(&mut interning, "value");
        let mut elab = Elaborator::new(tcx, interner);
        // Before the first checkpoint, so no rollback can ever retract it.
        let repl_box = elab
            .install_repl_box(box_name, box_param, box_field)
            .expect("fresh session");
        ReplSession {
            tcx,
            interner: interner.clone(),
            elab,
            repl_box,
            context,
            jit,
            emitted: FxHashSet::default(),
            bindings: Vec::new(),
            shapes: crate::reflect::ShapeTable::default(),
            printers: crate::reflect::PrinterRegistry::default(),
            modules: Vec::new(),
            counter: 0,
            opt: config.opt,
            data_layout: jit.data_layout(),
            triple: jit.triple(),
            sources: SourceCache::new(),
        }
    }

    /// The session's source cache (for rendering [`Outcome`] reports).
    pub fn sources(&self) -> &SourceCache {
        &self.sources
    }

    /// Register one input (or `:type` snippet) as a `<repl>` virtual file and
    /// point the elaborator's report attribution at it.
    fn add_input(&mut self, text: &str) -> FileId {
        let file = self.sources.add_virtual("<repl>", text);
        self.elab.set_current_file(file);
        file
    }

    /// Evaluate one complete input (a command, definitions, or an
    /// expression). Multiline assembly (`:{ … }:`, smart Enter) is the
    /// frontend's job; by the time input reaches here it is one chunk.
    pub fn eval(&mut self, input: &str) -> Outcome {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Outcome::Empty;
        }
        if let Some(command) = trimmed.strip_prefix(':') {
            return crate::commands::dispatch(self, command);
        }
        self.eval_source(input)
    }

    fn eval_source(&mut self, source: &str) -> Outcome {
        // Comment-only (or otherwise trivia-only) input is a no-op, not an
        // empty expression sequence — script files are full of `// CHECK`
        // lines.
        let (tokens, _) = reussir_syntax::lexer::tokenize(source);
        if tokens.iter().all(|t| t.kind.is_trivia()) {
            return Outcome::Empty;
        }
        let file = self.add_input(source);
        let parsed = parse_repl(source, self.interner.clone());
        if !parsed.parse.ok() {
            return Outcome::ParseErrors {
                file,
                errors: parsed.parse.errors,
            };
        }
        let cp = self.elab.checkpoint();
        match parsed.kind {
            ReplInputKind::Items => {
                let program = surface::program(&parsed.parse.root);
                let count = program.len();
                match self.elab.try_extend(&program) {
                    Err(reports) => Outcome::Reports(reports),
                    Ok(warnings) => match self.compile_new(&cp) {
                        Err(outcome) => outcome,
                        Ok(pending) => {
                            self.commit(pending);
                            Outcome::Definitions { count, warnings }
                        }
                    },
                }
            }
            ReplInputKind::Expr => {
                let expr = surface::repl_expr(&parsed.parse.root);
                // A single top-level `let` is a *global binding*: its value
                // persists for later inputs. Multi-statement sequences keep
                // ordinary block semantics.
                let (rhs, binder) = match expr.kind() {
                    surface::ExprKind::ExprSeq(mut exprs) if exprs.len() == 1 => {
                        let only = exprs.pop().expect("length checked");
                        match only.kind() {
                            surface::ExprKind::Let(bname, ascription, value) => {
                                (value, Some((bname, ascription)))
                            }
                            _ => (only, None),
                        }
                    }
                    _ => (expr, None),
                };
                let export = format!("__repl_expr_{}", self.counter);
                let name = Interner::get_or_intern(&mut self.interner.clone(), &export);
                let dec_name =
                    Interner::get_or_intern(&mut self.interner.clone(), &format!("{export}_dec"));
                let in_scope: Vec<_> = self.bindings.iter().map(|b| (b.name, b.ty)).collect();
                match self.elab.try_repl_expr(
                    &reussir_core::semi::repl::ReplExprRequest {
                        name,
                        dec_name,
                        export: &export,
                        ascription: binder.as_ref().and_then(|(_, a)| a.as_ref()),
                        bindings: &in_scope,
                        repl_box: self.repl_box,
                    },
                    &rhs,
                ) {
                    Err(reports) => Outcome::Reports(reports),
                    Ok((ty, _)) if binder.is_some() && matches!(ty.kind(), TyKind::Unit) => {
                        self.elab.rollback(&cp);
                        Outcome::Reports(vec![Report {
                            severity: Severity::Error,
                            message: "cannot bind a unit value".to_string(),
                            span: None,
                            file,
                        }])
                    }
                    Ok((ty, warnings)) => match self.compile_new(&cp) {
                        Err(outcome) => outcome,
                        // The input becomes permanent only after the
                        // trampoline materialized and ran: a lookup failure
                        // evicts the module and rolls the elaborator back,
                        // leaving the session exactly as before the input.
                        Ok(pending) => {
                            let box_ty =
                                self.tcx
                                    .mk_record(self.repl_box, &[ty], Flexivity::Irrelevant);
                            let args: Vec<*const u8> =
                                self.bindings.iter().map(|b| b.boxed.as_ptr()).collect();
                            let result = if binder.is_some() {
                                crate::value::call_and_bind(
                                    self.jit,
                                    &export,
                                    ty,
                                    box_ty,
                                    &self.shapes,
                                    &self.printers,
                                    &args,
                                )
                                .map(|(value, bound)| (value, Some(bound)))
                            } else {
                                crate::value::call_and_render(
                                    self.jit,
                                    &export,
                                    ty,
                                    box_ty,
                                    &self.shapes,
                                    &self.printers,
                                    &args,
                                )
                                .map(|value| (value, None))
                            };
                            match result {
                                Ok((value, bound)) => {
                                    self.commit(pending);
                                    self.counter += 1;
                                    // The wrapper is unspellable (an identifier
                                    // cannot start with `_`), so no later input
                                    // can ever reference `__repl_expr_N` again:
                                    // retire its trampoline root so future
                                    // monomorphizations walk only the *new*
                                    // input's call graph instead of every
                                    // historical expression's. The wrapper's
                                    // HIR stays behind as dead code. (The dec
                                    // companion's *code* stays materialized in
                                    // the JIT — a binding's release calls it
                                    // long after this input.)
                                    let before = self.elab.trampolines.len();
                                    self.elab
                                        .trampolines
                                        .retain(|t| !t.name.starts_with(&export));
                                    debug_assert!(
                                        self.elab.trampolines.len() < before,
                                        "the wrapper's roots must be registered"
                                    );
                                    match (binder, bound) {
                                        (Some((bname, _)), Some(bound)) => {
                                            let display = self.elab.sym(bname.value).to_string();
                                            self.register_binding(
                                                bname.value,
                                                display.clone(),
                                                ty,
                                                bound,
                                            );
                                            Outcome::Binding {
                                                name: display,
                                                value,
                                                ty: self.render_ty(ty),
                                                warnings,
                                            }
                                        }
                                        _ => Outcome::Value {
                                            value,
                                            ty: self.render_ty(ty),
                                            warnings,
                                        },
                                    }
                                }
                                Err(message) => {
                                    if let Some(pending) = pending {
                                        let _ = self.jit.remove_module(&pending.handle);
                                    }
                                    self.elab.rollback(&cp);
                                    Outcome::Backend(message)
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    /// Install (or overwrite) a global binding. Rebinding releases the old
    /// box and keeps the binding's position, so the wrapper-parameter order
    /// stays stable.
    fn register_binding(
        &mut self,
        name: reussir_syntax::kind::TokenKey,
        display: String,
        ty: Ty<'tcx>,
        boxed: crate::value::BoundBox<'a>,
    ) {
        let entry = GlobalBinding {
            name,
            display,
            ty,
            boxed,
        };
        if let Some(existing) = self.bindings.iter_mut().find(|b| b.name == name) {
            // The overwrite drops the old entry's box, releasing it.
            *existing = entry;
        } else {
            self.bindings.push(entry);
        }
    }

    /// Render the live global bindings (`:dump bindings`).
    pub(crate) fn render_bindings(&self) -> String {
        if self.bindings.is_empty() {
            return "no global bindings".to_string();
        }
        let mut out = String::from("=== Global Bindings ===");
        for b in &self.bindings {
            let box_ty = self
                .tcx
                .mk_record(self.repl_box, &[b.ty], Flexivity::Irrelevant);
            // SAFETY: the session owns one reference to each binding's box;
            // it is live until `register_binding`/`Drop` release it.
            let value = unsafe {
                crate::reflect::shared_box_payload(b.boxed.as_ptr(), box_ty, &self.shapes).and_then(
                    |payload| {
                        crate::reflect::render_value(payload, b.ty, &self.shapes, &self.printers)
                    },
                )
            }
            .unwrap_or_else(|e| format!("<{e}>"));
            out.push_str(&format!(
                "\n - {} : {} = {value}",
                b.display,
                self.render_ty(b.ty)
            ));
        }
        out
    }

    /// Check an expression's type without compiling or executing it
    /// (`:type`). Elaborator state is always rolled back.
    pub(crate) fn type_of(&mut self, source: &str) -> Outcome {
        let file = self.add_input(source);
        let parsed = parse_repl(source, self.interner.clone());
        if !parsed.parse.ok() {
            return Outcome::ParseErrors {
                file,
                errors: parsed.parse.errors,
            };
        }
        if parsed.kind != ReplInputKind::Expr {
            return Outcome::Text(":type takes an expression".to_string());
        }
        let expr = surface::repl_expr(&parsed.parse.root);
        let cp = self.elab.checkpoint();
        let name = Interner::get_or_intern(&mut self.interner.clone(), "__repl_type_probe");
        let dec_name = Interner::get_or_intern(&mut self.interner.clone(), "__repl_type_probe_dec");
        let in_scope: Vec<_> = self.bindings.iter().map(|b| (b.name, b.ty)).collect();
        let result = self.elab.try_repl_expr(
            &reussir_core::semi::repl::ReplExprRequest {
                name,
                dec_name,
                export: "__repl_type_probe",
                ascription: None,
                bindings: &in_scope,
                repl_box: self.repl_box,
            },
            &expr,
        );
        match result {
            Ok((ty, _)) => {
                let rendered = self.render_ty(ty);
                self.elab.rollback(&cp);
                Outcome::Text(format!("{} : {rendered}", source.trim()))
            }
            // try_repl_expr already rolled back.
            Err(reports) => Outcome::Reports(reports),
        }
    }

    /// Monomorphize the accumulated program, lower everything not yet in the
    /// JIT, and add the module. On any failure the elaborator rolls back to
    /// `cp` — nothing reaches the JIT from a failed input. On success the
    /// caller decides when the module is permanent (see [`Self::commit`]):
    /// definitions immediately, expressions only after the value call.
    fn compile_new(&mut self, cp: &Checkpoint) -> Result<Option<PendingModule<'a>>, Outcome> {
        let (mut program, reports) = monomorphize(&self.elab.mono_input());
        if reports.iter().any(|r| r.severity == Severity::Error) {
            self.elab.rollback(cp);
            return Err(Outcome::Reports(reports));
        }
        crate::externalize::externalize(&mut program, &self.emitted);

        // Remember every ground record shape this input introduced — the
        // value walker reflects results through them. (Harvested before the
        // empty-module early return: a record declaration compiles nothing
        // but its shape must still be known.)
        {
            // Split the borrow: `render_ty` reads the elaborator only.
            let elab = &self.elab;
            crate::reflect::harvest(&mut self.shapes, &program, |ty| {
                Self::render_ty_with(elab, ty)
            });
        }

        // Nothing new to compile (e.g. a record declaration, or a generic
        // function with no instantiations yet).
        if program.functions.iter().all(|f| f.body.is_none()) && program.trampolines.is_empty() {
            return Ok(None);
        }

        let mut module = match lower_program(self.context, self.tcx, &program, None, None) {
            Ok(module) => module,
            Err(error) => {
                self.elab.rollback(cp);
                return Err(Outcome::Backend(format!("lowering failed: {error}")));
            }
        };
        // Verify before the pipeline (and implicitly after: the pipeline
        // fails loudly on invalid IR) so a codegen bug cannot poison the
        // persistent JIT session.
        if !module.as_operation().verify() {
            self.elab.rollback(cp);
            return Err(Outcome::Backend(
                "lowering produced an invalid MLIR module (this is a bug)".to_string(),
            ));
        }
        if let Err(error) = pipeline::attach_target_spec(&module, &self.data_layout, &self.triple) {
            self.elab.rollback(cp);
            return Err(Outcome::Backend(format!(
                "attaching the target spec failed: {error}"
            )));
        }
        let options = LoweringOptions {
            opt: self.opt,
            ..LoweringOptions::default()
        };
        if let Err(error) = pipeline::run_lowering_pipeline(self.context, &mut module, &options) {
            self.elab.rollback(cp);
            return Err(Outcome::Backend(format!(
                "lowering pipeline failed: {error:?}"
            )));
        }
        let handle = match self.jit.add_module_tracked(&module, self.opt) {
            Ok(handle) => handle,
            Err(error) => {
                self.elab.rollback(cp);
                return Err(Outcome::Backend(format!("JIT error: {error}")));
            }
        };
        // Record only what this module *defines*: externalized (body-less)
        // declarations are already in `emitted`, and marking any other
        // body-less function as emitted would externalize it forever without
        // its body ever being compiled.
        let mut symbols: Vec<String> = program
            .functions
            .iter()
            .filter(|f| f.body.is_some())
            .map(|f| program.symbol(f.symbol).to_string())
            .collect();
        symbols.extend(
            program
                .trampolines
                .iter()
                .map(|t| program.symbol(t.export).to_string()),
        );
        Ok(Some(PendingModule { handle, symbols }))
    }

    /// Make a compiled module permanent: keep its tracker and record its
    /// symbols so later modules declare rather than redefine them.
    fn commit(&mut self, pending: Option<PendingModule<'a>>) {
        if let Some(pending) = pending {
            self.modules.push(pending.handle);
            self.emitted.extend(pending.symbols);
        }
    }

    /// Number of accepted user definitions — functions and records; the
    /// synthetic `__repl_expr_N` wrappers, their dec companions, and the
    /// `__ReplBox` prelude are excluded (`__` names are unspellable, so the
    /// prefix test cannot hide a user definition). Shown in the TUI status
    /// line.
    pub fn definition_count(&self) -> usize {
        let functions = self
            .elab
            .elaborated
            .iter()
            .filter(|f| !self.elab.sym(f.name).starts_with("__"))
            .count();
        let records = self
            .elab
            .records
            .values()
            .filter(|r| !self.elab.sym(r.name).starts_with("__"))
            .count();
        functions + records
    }

    /// Render a ground Semi type for display (`i64`, `f64`, `List::<i64>`).
    pub(crate) fn render_ty(&self, ty: Ty<'tcx>) -> String {
        Self::render_ty_with(&self.elab, ty)
    }

    fn render_ty_with(elab: &Elaborator<'a, 'tcx>, ty: Ty<'tcx>) -> String {
        match *ty.kind() {
            TyKind::Int(reussir_core::semi::ty::IntTy::Signed(w)) => format!("i{w}"),
            TyKind::Int(reussir_core::semi::ty::IntTy::Unsigned(w)) => format!("u{w}"),
            TyKind::Fp(reussir_core::semi::ty::FpTy::Ieee(w)) => format!("f{w}"),
            TyKind::Fp(reussir_core::semi::ty::FpTy::BFloat16) => "bf16".to_string(),
            TyKind::Fp(reussir_core::semi::ty::FpTy::Float8) => "f8".to_string(),
            TyKind::Bool => "bool".to_string(),
            TyKind::Str => "str".to_string(),
            TyKind::Char => "char".to_string(),
            TyKind::Unit => "()".to_string(),
            TyKind::Bottom => "!".to_string(),
            TyKind::Nullable(inner) => {
                format!("Nullable<{}>", Self::render_ty_with(elab, inner))
            }
            TyKind::Record { def, args, .. } => {
                let mut out = elab.defs.path(def).display(elab.resolver);
                if !args.is_empty() {
                    let args: Vec<String> = args
                        .iter()
                        .map(|&a| Self::render_ty_with(elab, a))
                        .collect();
                    out.push_str(&format!("<{}>", args.join(", ")));
                }
                out
            }
            TyKind::Closure { params, ret } => {
                let params: Vec<String> = params
                    .iter()
                    .map(|&p| Self::render_ty_with(elab, p))
                    .collect();
                format!(
                    "({}) -> {}",
                    params.join(", "),
                    Self::render_ty_with(elab, ret)
                )
            }
            TyKind::Generic(_) | TyKind::Hole(_) => "_".to_string(),
        }
    }
}
