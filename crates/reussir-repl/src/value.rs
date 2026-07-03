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

use reussir_core::semi::ty::{Ty, TyKind};
use reussir_jit::OrcJit;

use crate::reflect::{self, PrinterRegistry, ShapeTable};

/// Calls the `_dec` companion on drop, releasing the displayed box.
struct DecGuard {
    dec: extern "C" fn(*const u8),
    boxed: *const u8,
}

impl Drop for DecGuard {
    fn drop(&mut self) {
        (self.dec)(self.boxed);
    }
}

/// Look up `export` in the JIT, call it, and render the boxed result via the
/// layout walker. `ty` is the expression's ground type (what the user sees);
/// `box_ty` is the wrapper's actual `__ReplBox<T>` return type, used to
/// locate the payload inside the rc box.
pub fn call_and_render<'tcx>(
    jit: &OrcJit,
    export: &str,
    ty: Ty<'tcx>,
    box_ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
) -> Result<String, String> {
    let address = jit.lookup(export)? as usize;
    // SAFETY: the wrapper trampoline was just materialized with the C ABI
    // shape its return type implies — `void()` for unit, pointer-returning
    // for the shared `__ReplBox` — and the dec companion consumes exactly
    // that pointer (see the module docs).
    unsafe {
        if matches!(*ty.kind(), TyKind::Unit) {
            let f: extern "C" fn() = std::mem::transmute(address);
            f();
            return Ok("()".to_string());
        }

        // Resolve the dec companion *before* calling the wrapper, so a
        // missing symbol cannot strand an already-allocated box.
        let dec: extern "C" fn(*const u8) =
            std::mem::transmute(jit.lookup(&format!("{export}_dec"))? as usize);

        let f: extern "C" fn() -> *const u8 = std::mem::transmute(address);
        let boxed = f();
        if boxed.is_null() {
            return Err("the REPL wrapper returned a null box (this is a bug)".to_string());
        }
        let _guard = DecGuard { dec, boxed };

        // The box payload is the `{ value: T }` compound; its sole member's
        // inline cell sits at offset 0, so the payload address *is* the
        // inline representation of `T` the walker expects.
        let payload = reflect::shared_box_payload(boxed, box_ty, shapes)?;
        reflect::render_value(payload, ty, shapes, printers)
    }
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
