//! Calling a REPL expression's trampoline and rendering the result.
//!
//! Every non-unit expression result crosses the C ABI as **one pointer** to
//! a shared `__ReplBox<T>` rc box (see `reussir_core::semi::repl`): the
//! wrapper export returns the box, and the `_dec` companion export consumes
//! it — the ownership-pass-synthesized recursive release of the whole
//! result. The host protocol is therefore uniform: call the wrapper, walk
//! the payload with the layout mirror ([`crate::reflect`]), then call the
//! dec — held in a guard so the box is released on *every* exit path,
//! walker failures included. Unit results skip the box (a plain `void()`
//! export; nothing to show or free).
//!
//! **Global bindings** ride the same protocol in the other direction: the
//! wrapper takes one `__ReplBox` pointer per live binding (per the
//! trampoline C ABI: passed directly below four parameters, packed behind
//! one struct pointer at four or more), and *consumes* each — so the host
//! retains every passed box first, exactly what an in-language caller
//! holding its own reference would do. A `let` input keeps the returned box
//! alive instead of dec'ing it ([`call_and_bind`]).

use reussir_core::semi::ty::{Ty, TyKind};
use reussir_jit::OrcJit;

use crate::reflect::{self, PrinterRegistry, ShapeTable};

/// An owned reference to an evaluated `__ReplBox` — a host-side smart
/// pointer over the shared rc box. `Clone` retains, `Drop` releases through
/// the dec companion; otherwise move-only, so every reference has exactly
/// one releasing owner (a binding's box lives in the session's
/// `GlobalBinding` and dies with it — rebinding included — with no manual
/// bookkeeping).
///
/// The `'jit` borrow ties the reference to the engine whose materialized
/// dec-companion code `Drop` calls (mirroring
/// [`ModuleHandle`](reussir_jit::ModuleHandle)): releasing after engine
/// disposal would be a use-after-free, and the borrow makes it a compile
/// error instead of a discipline.
pub struct BoundBox<'jit> {
    boxed: *const u8,
    dec: extern "C" fn(*const u8),
    jit: std::marker::PhantomData<&'jit OrcJit>,
}

impl BoundBox<'_> {
    /// The raw box pointer, for passing to wrapper calls (which retain per
    /// reference they consume — see [`call_boxed`]) and for walking.
    pub fn as_ptr(&self) -> *const u8 {
        self.boxed
    }
}

impl Clone for BoundBox<'_> {
    fn clone(&self) -> Self {
        // SAFETY: `self` holds a live reference; the retain mirrors the
        // generated `rc.inc` (see `retain_box`).
        unsafe { retain_box(self.boxed) };
        BoundBox {
            boxed: self.boxed,
            dec: self.dec,
            jit: std::marker::PhantomData,
        }
    }
}

impl Drop for BoundBox<'_> {
    fn drop(&mut self) {
        (self.dec)(self.boxed);
    }
}

/// Bump a shared rc box's refcount from the host, mirroring what generated
/// code does before a consuming call.
///
/// # Safety
///
/// `boxed` must point at a live shared rc box (count word at offset 0).
/// Shared counts are `AtomicKind::normal` today — a plain read-modify-write,
/// exactly the generated `rc.inc` — and the REPL is single-threaded; if
/// atomic rc ever lands, this must switch to an atomic add with it.
unsafe fn retain_box(boxed: *const u8) {
    let count = boxed.cast_mut().cast::<usize>();
    unsafe { *count += 1 };
}

/// Invoke a wrapper export with one `__ReplBox` pointer per binding, per the
/// trampoline C ABI: pointer parameters pass directly below four, and pack
/// behind a single struct pointer at four or more
/// (`CABISignatureConversion.cpp`). Retains every argument first — the
/// callee consumes them.
///
/// # Safety
///
/// `address` must be the wrapper's materialized entry, whose Reussir
/// signature takes exactly `args.len()` boxes and returns a box pointer.
unsafe fn call_boxed(address: usize, args: &[*const u8]) -> *const u8 {
    unsafe {
        for &arg in args {
            retain_box(arg);
        }
        type P = *const u8;
        match args {
            [] => std::mem::transmute::<usize, extern "C" fn() -> P>(address)(),
            [a] => std::mem::transmute::<usize, extern "C" fn(P) -> P>(address)(*a),
            [a, b] => std::mem::transmute::<usize, extern "C" fn(P, P) -> P>(address)(*a, *b),
            [a, b, c] => {
                std::mem::transmute::<usize, extern "C" fn(P, P, P) -> P>(address)(*a, *b, *c)
            }
            packed => std::mem::transmute::<usize, extern "C" fn(P) -> P>(address)(
                packed.as_ptr().cast::<u8>(),
            ),
        }
    }
}

/// [`call_boxed`] for a unit-returning wrapper (`void` return in every ABI
/// shape).
unsafe fn call_unit(address: usize, args: &[*const u8]) {
    unsafe {
        for &arg in args {
            retain_box(arg);
        }
        type P = *const u8;
        match args {
            [] => std::mem::transmute::<usize, extern "C" fn()>(address)(),
            [a] => std::mem::transmute::<usize, extern "C" fn(P)>(address)(*a),
            [a, b] => std::mem::transmute::<usize, extern "C" fn(P, P)>(address)(*a, *b),
            [a, b, c] => std::mem::transmute::<usize, extern "C" fn(P, P, P)>(address)(*a, *b, *c),
            packed => std::mem::transmute::<usize, extern "C" fn(P)>(address)(
                packed.as_ptr().cast::<u8>(),
            ),
        }
    }
}

/// Call the wrapper (passing the live bindings' boxes) and return the
/// evaluated box plus its rendering, WITHOUT releasing it — the `let` path.
fn call_and_walk<'jit, 'tcx>(
    jit: &'jit OrcJit,
    export: &str,
    ty: Ty<'tcx>,
    box_ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
    args: &[*const u8],
) -> Result<(String, BoundBox<'jit>), String> {
    let address = jit.lookup(export)? as usize;
    // Resolve the dec companion *before* calling the wrapper, so a missing
    // symbol cannot strand an already-allocated box.
    let dec: extern "C" fn(*const u8) =
        unsafe { std::mem::transmute(jit.lookup(&format!("{export}_dec"))? as usize) };

    // SAFETY: the wrapper was materialized with exactly this signature (see
    // the module docs), and every passed box is live and retained inside
    // `call_boxed`.
    let boxed = unsafe { call_boxed(address, args) };
    if boxed.is_null() {
        return Err("the REPL wrapper returned a null box (this is a bug)".to_string());
    }
    // Owning from here on: any early return below releases through Drop.
    let bound = BoundBox {
        boxed,
        dec,
        jit: std::marker::PhantomData,
    };

    // The box payload is the `{ value: T }` compound; its sole member's
    // inline cell sits at offset 0, so the payload address *is* the inline
    // representation of `T` the walker expects.
    // SAFETY: `bound` owns a live box of `box_ty`'s layout.
    let rendered = unsafe {
        reflect::shared_box_payload(bound.as_ptr(), box_ty, shapes)
            .and_then(|payload| reflect::render_value(payload, ty, shapes, printers))?
    };
    Ok((rendered, bound))
}

/// Look up `export` in the JIT, call it (passing the live bindings' boxes),
/// render the boxed result via the layout walker, and release it. `ty` is
/// the expression's ground type (what the user sees); `box_ty` the wrapper's
/// actual `__ReplBox<T>` return type.
pub fn call_and_render<'tcx>(
    jit: &OrcJit,
    export: &str,
    ty: Ty<'tcx>,
    box_ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
    args: &[*const u8],
) -> Result<String, String> {
    if matches!(*ty.kind(), TyKind::Unit) {
        let address = jit.lookup(export)? as usize;
        // SAFETY: unit wrappers export `void(...)` (module docs).
        unsafe { call_unit(address, args) };
        return Ok("()".to_string());
    }
    // The evaluated box drops (releases) here; only the rendering survives.
    let (rendered, _bound) = call_and_walk(jit, export, ty, box_ty, shapes, printers, args)?;
    Ok(rendered)
}

/// [`call_and_render`], but keep the evaluated box alive and hand it to the
/// caller — the global-`let` path. The caller owns the release (rebinding
/// or session drop). Unit values are rejected upstream.
pub fn call_and_bind<'jit, 'tcx>(
    jit: &'jit OrcJit,
    export: &str,
    ty: Ty<'tcx>,
    box_ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
    args: &[*const u8],
) -> Result<(String, BoundBox<'jit>), String> {
    call_and_walk(jit, export, ty, box_ty, shapes, printers, args)
}

/// Match the old REPL's float rendering: always show a decimal point
/// (`1.0`, not `1`).
pub(crate) fn render_float(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}
