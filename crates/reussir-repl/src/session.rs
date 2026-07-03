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
use reussir_core::semi::ty::{Ty, TyCtxt, TyKind};
use reussir_core::semi::{Checkpoint, Elaborator, Report, Severity};
use reussir_core::{in_arena, surface};
use reussir_jit::{ModuleHandle, OptLevel, OrcJit};
use reussir_syntax::diagnostics::ParseError;
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
    /// Syntax errors in the input (render against the input source).
    ParseErrors(Vec<ParseError>),
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
    /// The `__repl_expr_N` counter (advanced only on success).
    counter: usize,
    pub(crate) opt: OptLevel,
}

impl<'a, 'tcx> ReplSession<'a, 'tcx> {
    pub fn new(
        tcx: &'a TyCtxt<'tcx>,
        interner: &'a Arc<MultiThreadedTokenInterner>,
        context: &'a Context,
        jit: &'a OrcJit,
        config: Config,
    ) -> Self {
        ReplSession {
            tcx,
            interner: interner.clone(),
            elab: Elaborator::new(tcx, interner),
            context,
            jit,
            emitted: FxHashSet::default(),
            modules: Vec::new(),
            counter: 0,
            opt: config.opt,
        }
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
        let parsed = parse_repl(source, self.interner.clone());
        if !parsed.parse.ok() {
            return Outcome::ParseErrors(parsed.parse.errors);
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
                let export = format!("__repl_expr_{}", self.counter);
                let name = Interner::get_or_intern(&mut self.interner.clone(), &export);
                match self.elab.try_repl_expr(name, &export, &expr) {
                    Err(reports) => Outcome::Reports(reports),
                    Ok((ty, warnings)) => match self.compile_new(&cp) {
                        Err(outcome) => outcome,
                        // The input becomes permanent only after the
                        // trampoline materialized and ran: a lookup failure
                        // evicts the module and rolls the elaborator back,
                        // leaving the session exactly as before the input.
                        Ok(pending) => {
                            match crate::value::call_and_render(self.jit, &export, ty) {
                                Ok(value) => {
                                    self.commit(pending);
                                    self.counter += 1;
                                    Outcome::Value {
                                        value,
                                        ty: self.render_ty(ty),
                                        warnings,
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

    /// Check an expression's type without compiling or executing it
    /// (`:type`). Elaborator state is always rolled back.
    pub(crate) fn type_of(&mut self, source: &str) -> Outcome {
        let parsed = parse_repl(source, self.interner.clone());
        if !parsed.parse.ok() {
            return Outcome::ParseErrors(parsed.parse.errors);
        }
        if parsed.kind != ReplInputKind::Expr {
            return Outcome::Text(":type takes an expression".to_string());
        }
        let expr = surface::repl_expr(&parsed.parse.root);
        let cp = self.elab.checkpoint();
        let name = Interner::get_or_intern(&mut self.interner.clone(), "__repl_type_probe");
        let result = self.elab.try_repl_expr(name, "__repl_type_probe", &expr);
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
        let mut symbols: Vec<String> = program
            .functions
            .iter()
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

    /// Render a ground Semi type for display (`i64`, `f64`, `List::<i64>`).
    pub(crate) fn render_ty(&self, ty: Ty<'tcx>) -> String {
        match *ty.kind() {
            TyKind::Int(reussir_core::semi::ty::IntTy::Signed(w)) => format!("i{w}"),
            TyKind::Int(reussir_core::semi::ty::IntTy::Unsigned(w)) => format!("u{w}"),
            TyKind::Fp(reussir_core::semi::ty::FpTy::Ieee(w)) => format!("f{w}"),
            TyKind::Fp(reussir_core::semi::ty::FpTy::BFloat16) => "bf16".to_string(),
            TyKind::Fp(reussir_core::semi::ty::FpTy::Float8) => "f8".to_string(),
            TyKind::Bool => "bool".to_string(),
            TyKind::Str => "str".to_string(),
            TyKind::Unit => "()".to_string(),
            TyKind::Bottom => "!".to_string(),
            TyKind::Nullable(inner) => format!("Nullable<{}>", self.render_ty(inner)),
            TyKind::Record { def, args, .. } => {
                let mut out = self.elab.defs.path(def).display(self.elab.resolver);
                if !args.is_empty() {
                    let args: Vec<String> = args.iter().map(|&a| self.render_ty(a)).collect();
                    out.push_str(&format!("<{}>", args.join(", ")));
                }
                out
            }
            TyKind::Closure { params, ret } => {
                let params: Vec<String> = params.iter().map(|&p| self.render_ty(p)).collect();
                format!("({}) -> {}", params.join(", "), self.render_ty(ret))
            }
            TyKind::Generic(_) | TyKind::Hole(_) => "_".to_string(),
        }
    }
}
