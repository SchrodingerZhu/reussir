//! Calling a REPL expression's trampoline and rendering the result.
//!
//! The `reussir.trampoline` lowering *is* the universal shim
//! (`lib/Conversion/BasicOpsLowering/CABISignatureConversion.cpp`): a return
//! type is passed directly only when it is void, integer, or pointer;
//! everything else — floats included — goes through an sret-style
//! `void(ptr)` export. For the nullary `__repl_expr_N` wrappers this yields
//! exactly three host call shapes, classified here from the expression's
//! ground Semi type.

use reussir_core::semi::ctxt::DefaultCap;
use reussir_core::semi::ty::{FpTy, IntTy, Ty, TyKind};
use reussir_jit::OrcJit;

use crate::reflect::{self, PrinterRegistry, ShapeTable};

/// Look up `export` in the JIT, call it according to `ty`'s C ABI class, and
/// render the value — scalars directly, records/enums through the layout
/// walker ([`crate::reflect`]).
///
/// Displayed aggregate values are currently *leaked*: the host owns the
/// returned value but has no drop glue for it yet (a synthesized
/// `__repl_drop_N` thunk is the planned follow-up). Acceptable for a REPL;
/// `:clear` releases the whole session.
pub fn call_and_render<'tcx>(
    jit: &OrcJit,
    export: &str,
    ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
) -> Result<String, String> {
    let address = jit.lookup(export)? as usize;
    // SAFETY: the trampoline was just materialized with the C ABI signature
    // implied by `ty` (see the module docs); each arm transmutes to exactly
    // that signature.
    unsafe {
        match *ty.kind() {
            TyKind::Unit => {
                let f: extern "C" fn() = std::mem::transmute(address);
                f();
                Ok("()".to_string())
            }
            // An i1 return leaves the upper register bits unspecified.
            TyKind::Bool => {
                let f: extern "C" fn() -> u8 = std::mem::transmute(address);
                Ok(((f() & 1) != 0).to_string())
            }
            TyKind::Int(IntTy::Signed(width)) => {
                let value: i64 = match width {
                    8 => {
                        let f: extern "C" fn() -> i8 = std::mem::transmute(address);
                        f() as i64
                    }
                    16 => {
                        let f: extern "C" fn() -> i16 = std::mem::transmute(address);
                        f() as i64
                    }
                    32 => {
                        let f: extern "C" fn() -> i32 = std::mem::transmute(address);
                        f() as i64
                    }
                    64 => {
                        let f: extern "C" fn() -> i64 = std::mem::transmute(address);
                        f()
                    }
                    other => return Err(format!("cannot display an i{other} value yet")),
                };
                Ok(value.to_string())
            }
            TyKind::Int(IntTy::Unsigned(width)) => {
                let value: u64 = match width {
                    8 => {
                        let f: extern "C" fn() -> u8 = std::mem::transmute(address);
                        f() as u64
                    }
                    16 => {
                        let f: extern "C" fn() -> u16 = std::mem::transmute(address);
                        f() as u64
                    }
                    32 => {
                        let f: extern "C" fn() -> u32 = std::mem::transmute(address);
                        f() as u64
                    }
                    64 => {
                        let f: extern "C" fn() -> u64 = std::mem::transmute(address);
                        f()
                    }
                    other => return Err(format!("cannot display a u{other} value yet")),
                };
                Ok(value.to_string())
            }
            // Floats are not "trivial" in the trampoline's C ABI evaluation:
            // the export is `void(ptr)` writing the value through the pointer.
            TyKind::Fp(FpTy::Ieee(64)) => {
                let f: extern "C" fn(*mut f64) = std::mem::transmute(address);
                let mut out = 0.0f64;
                f(&mut out);
                Ok(render_float(out))
            }
            TyKind::Fp(FpTy::Ieee(32)) => {
                let f: extern "C" fn(*mut f32) = std::mem::transmute(address);
                let mut out = 0.0f32;
                f(&mut out);
                Ok(render_float(out as f64))
            }
            // A `[value]` record (or value enum) is non-trivial in the C ABI:
            // the trampoline writes the aggregate through an sret pointer.
            TyKind::Record { def, args, .. }
                if shapes
                    .get(&(def, args))
                    .is_some_and(|s| s.default_cap == DefaultCap::Value) =>
            {
                let sa = reflect::inline_size_align(ty, shapes)?;
                // Over-aligned backing store (u128 units) for the buffer.
                let mut buffer = vec![0u128; (sa.size as usize).div_ceil(16).max(1)];
                let out = buffer.as_mut_ptr().cast::<u8>();
                let f: extern "C" fn(*mut u8) = std::mem::transmute(address);
                f(out);
                reflect::render_value(out, ty, shapes, printers)
            }
            // Boxed values (shared/regional records and enums, nullables,
            // closures) come back as a pointer; hand the walker a cell
            // holding it (the inline member representation).
            TyKind::Record { .. } | TyKind::Nullable(_) | TyKind::Closure { .. } => {
                let f: extern "C" fn() -> *const u8 = std::mem::transmute(address);
                let cell: *const u8 = f();
                reflect::render_value(std::ptr::from_ref(&cell).cast::<u8>(), ty, shapes, printers)
            }
            ref other => Err(format!(
                "cannot display a value of type `{other:?}` yet: structural printing \
                 for this type is not implemented"
            )),
        }
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
