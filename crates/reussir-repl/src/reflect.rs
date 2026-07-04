//! Structural value inspection: a host-side mirror of the backend's layout
//! rules, a memory walker, and the printer-hook registry.
//!
//! The layout algorithm mirrors `deriveCompoundSizeAndAlignment`
//! (`lib/IR/ReussirTypes.cpp`) and the box conversions
//! (`lib/Conversion/TypeConverter/TypeConverter.cpp`), which are frozen by
//! ABI compatibility with already-JIT'd code:
//!
//! - C-style sequential struct layout; a member is stored as a *pointer*
//!   when it is a `[field]` link, a shared/regional record, or a closure;
//!   everything else is inline.
//! - An enum (variant record) is `{ tag: usize @ 0, payload }` with the
//!   payload at `align_to(tag_size, payload_align)` and the total aligned to
//!   `max(tag_align, payload_align)`; the payload area is the largest
//!   variant. No niche optimizations exist.
//! - A shared rc box is `{ count: usize, T }` (payload at
//!   `align_to(usize, align(T))`); a regional box has a three-pointer
//!   header instead.
//!
//! Ground record shapes are harvested from each input's monomorphized
//! `mir::Program` ([`RecordShape`]), so the walker needs no access to the
//! elaborator. Cycles (constructible through `[field]` links) are cut by a
//! seen-set of box addresses; depth is capped.

use rustc_hash::{FxHashMap, FxHashSet};

use reussir_core::full::mir;
use reussir_core::semi::ctxt::DefaultCap;
use reussir_core::semi::ty::{DefId, FpTy, IntTy, Ty, TyKind};

/// The pointer/`usize` width of the JIT host.
const WORD: u64 = std::mem::size_of::<usize>() as u64;

/// Ground record shapes keyed by the nominal identity (`def` + ground args).
pub type ShapeTable<'tcx> = FxHashMap<(DefId, &'tcx [Ty<'tcx>]), RecordShape<'tcx>>;

/// One ground record instance's display name, capability, and field/variant
/// structure (names resolved to owned strings — the mono program's interner
/// does not outlive the input that produced it).
pub struct RecordShape<'tcx> {
    pub name: String,
    pub default_cap: DefaultCap,
    pub layout: ShapeLayout<'tcx>,
}

pub enum ShapeLayout<'tcx> {
    Compound(Vec<FieldShape<'tcx>>),
    Variants(Vec<VariantShape<'tcx>>),
}

pub struct FieldShape<'tcx> {
    pub name: Option<String>,
    pub ty: Ty<'tcx>,
    pub is_field: bool,
}

pub struct VariantShape<'tcx> {
    pub name: String,
    pub fields: Vec<Ty<'tcx>>,
}

/// Harvest the ground record shapes of one monomorphized program into the
/// session's table. `name_of` renders the nominal type for display.
pub fn harvest<'tcx>(
    shapes: &mut ShapeTable<'tcx>,
    program: &mir::Program<'tcx>,
    name_of: impl Fn(Ty<'tcx>) -> String,
) {
    for instance in &program.records {
        let TyKind::Record { def, args, .. } = *instance.ty.kind() else {
            continue;
        };
        if shapes.contains_key(&(def, args)) {
            continue;
        }
        let layout = match instance.layout {
            mir::RecordLayout::Compound(members) => ShapeLayout::Compound(
                members
                    .iter()
                    .map(|m| FieldShape {
                        name: m.name.map(|s| program.symbol(s).to_string()),
                        ty: m.ty,
                        is_field: m.is_field,
                    })
                    .collect(),
            ),
            mir::RecordLayout::Variant(variants) => ShapeLayout::Variants(
                variants
                    .iter()
                    .map(|v| VariantShape {
                        name: program.symbol(v.name).to_string(),
                        fields: v.fields.to_vec(),
                    })
                    .collect(),
            ),
        };
        shapes.insert(
            (def, args),
            RecordShape {
                name: name_of(instance.ty),
                default_cap: instance.default_cap,
                layout,
            },
        );
    }
}

// ----- layout -----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAlign {
    pub size: u64,
    pub align: u64,
}

fn align_to(offset: u64, align: u64) -> u64 {
    offset.div_ceil(align.max(1)) * align.max(1)
}

/// Is this member type stored as a pointer inside an aggregate?
fn member_is_pointer<'tcx>(ty: Ty<'tcx>, is_field: bool, shapes: &ShapeTable<'tcx>) -> bool {
    if is_field {
        return true;
    }
    match *ty.kind() {
        TyKind::Closure { .. } | TyKind::Nullable(_) => true,
        TyKind::Record { def, args, .. } => shapes
            .get(&(def, args))
            .is_none_or(|s| s.default_cap != DefaultCap::Value),
        _ => false,
    }
}

/// The inline (in-memory) size/alignment of a type in *member position*
/// (pointer-stored members are a word).
fn member_size_align<'tcx>(
    ty: Ty<'tcx>,
    is_field: bool,
    shapes: &ShapeTable<'tcx>,
) -> Result<SizeAlign, String> {
    if member_is_pointer(ty, is_field, shapes) {
        return Ok(SizeAlign {
            size: WORD,
            align: WORD,
        });
    }
    inline_size_align(ty, shapes)
}

/// The size/alignment of a type's inline representation (a `[value]` record
/// expanded in place, a scalar as itself).
pub fn inline_size_align<'tcx>(
    ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
) -> Result<SizeAlign, String> {
    let scalar = |size| {
        Ok(SizeAlign {
            size,
            align: size.max(1),
        })
    };
    match *ty.kind() {
        TyKind::Bool => scalar(1),
        TyKind::Char => scalar(4),
        TyKind::Int(int) => {
            let (prefix, width) = match int {
                IntTy::Signed(w) => ("i", w),
                IntTy::Unsigned(w) => ("u", w),
            };
            match width {
                8 => scalar(1),
                16 => scalar(2),
                32 => scalar(4),
                64 => scalar(8),
                other => Err(format!("unsupported integer width {prefix}{other}")),
            }
        }
        TyKind::Fp(FpTy::Ieee(16) | FpTy::BFloat16) => scalar(2),
        TyKind::Fp(FpTy::Ieee(32)) => scalar(4),
        TyKind::Fp(FpTy::Ieee(64)) => scalar(8),
        TyKind::Fp(FpTy::Float8) => scalar(1),
        TyKind::Unit => Ok(SizeAlign { size: 0, align: 1 }),
        TyKind::Nullable(_) | TyKind::Closure { .. } => scalar(WORD),
        TyKind::Record { def, args, .. } => {
            let shape = shapes
                .get(&(def, args))
                .ok_or_else(|| "unknown record instance".to_string())?;
            match &shape.layout {
                ShapeLayout::Compound(fields) => compound_size_align(fields, shapes),
                ShapeLayout::Variants(variants) => {
                    let payload = variants_payload_size_align(variants, shapes)?;
                    let payload_offset = align_to(WORD, payload.align);
                    let align = payload.align.max(WORD);
                    Ok(SizeAlign {
                        size: align_to(payload_offset + payload.size, align),
                        align,
                    })
                }
            }
        }
        ref other => Err(format!("cannot compute the layout of `{other:?}`")),
    }
}

/// The payload address inside a **shared** rc box (`{ count: usize, T }`,
/// payload at `align_to(usize, align(T))`). Used by the value caller to
/// reach the `__ReplBox` payload without walking the box as a record.
///
/// # Safety
///
/// `boxed` must point at a live shared rc box of `box_ty`'s layout.
pub unsafe fn shared_box_payload<'tcx>(
    boxed: *const u8,
    box_ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
) -> Result<*const u8, String> {
    let inner = inline_size_align(box_ty, shapes)?;
    Ok(unsafe { boxed.add(align_to(WORD, inner.align) as usize) })
}

/// Sequential (C-style) layout of a member list: total size/align.
fn compound_size_align<'tcx>(
    fields: &[FieldShape<'tcx>],
    shapes: &ShapeTable<'tcx>,
) -> Result<SizeAlign, String> {
    let mut size = 0u64;
    let mut align = 1u64;
    for field in fields {
        let member = member_size_align(field.ty, field.is_field, shapes)?;
        size = align_to(size, member.align) + member.size;
        align = align.max(member.align);
    }
    Ok(SizeAlign {
        size: align_to(size, align),
        align,
    })
}

/// Byte offset of each member in a sequential layout.
fn member_offsets<'tcx>(
    fields: &[FieldShape<'tcx>],
    shapes: &ShapeTable<'tcx>,
) -> Result<Vec<u64>, String> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut size = 0u64;
    for field in fields {
        let member = member_size_align(field.ty, field.is_field, shapes)?;
        let offset = align_to(size, member.align);
        offsets.push(offset);
        size = offset + member.size;
    }
    Ok(offsets)
}

/// The payload area of an enum: max variant size, max variant alignment.
fn variants_payload_size_align<'tcx>(
    variants: &[VariantShape<'tcx>],
    shapes: &ShapeTable<'tcx>,
) -> Result<SizeAlign, String> {
    let mut size = 0u64;
    let mut align = 1u64;
    for variant in variants {
        let fields: Vec<FieldShape<'tcx>> = variant
            .fields
            .iter()
            .map(|&ty| FieldShape {
                name: None,
                ty,
                is_field: false,
            })
            .collect();
        let sa = compound_size_align(&fields, shapes)?;
        size = size.max(sa.size);
        align = align.max(sa.align);
    }
    Ok(SizeAlign { size, align })
}

// ----- printer hooks -----

/// A cursor over a reflected in-memory value, handed to printer hooks.
pub struct ReflectedValue<'v, 'tcx> {
    pub addr: *const u8,
    pub ty: Ty<'tcx>,
    pub shapes: &'v ShapeTable<'tcx>,
}

/// A user-installable printer for a named type. Returning `None` falls back
/// to the structural printer (registry consultation happens per node while
/// walking, so hooks compose with structural rendering of the surroundings).
pub trait ValuePrinter: Send + Sync {
    fn print(&self, value: &ReflectedValue<'_, '_>) -> Option<String>;
}

/// Printer hooks keyed by the record's display name (e.g. `List<i64>` or a
/// future unparameterized `List`). Consulted before structural rendering;
/// the structural walker is the fallback. (Installing hooks from inside the
/// language — utop's `#install_printer` — is a planned follow-up; the
/// registry is the seam it will plug into.)
#[derive(Default)]
pub struct PrinterRegistry {
    by_type: FxHashMap<String, Box<dyn ValuePrinter>>,
}

impl PrinterRegistry {
    pub fn install(&mut self, type_name: impl Into<String>, printer: Box<dyn ValuePrinter>) {
        self.by_type.insert(type_name.into(), printer);
    }

    fn lookup(&self, type_name: &str) -> Option<&dyn ValuePrinter> {
        self.by_type.get(type_name).map(|b| &**b)
    }
}

// ----- the walker -----

const MAX_DEPTH: usize = 24;

/// Render the value at `addr` (the *inline* representation of `ty`, i.e. a
/// pointer cell for boxed types) into a display string.
///
/// # Safety
///
/// `addr` must point at a live, initialized value of `ty`'s JIT layout.
pub unsafe fn render_value<'tcx>(
    addr: *const u8,
    ty: Ty<'tcx>,
    shapes: &ShapeTable<'tcx>,
    printers: &PrinterRegistry,
) -> Result<String, String> {
    let mut walker = Walker {
        shapes,
        printers,
        seen: FxHashSet::default(),
    };
    let mut out = String::new();
    unsafe { walker.render(addr, ty, false, 0, &mut out)? };
    Ok(out)
}

struct Walker<'v, 'tcx> {
    shapes: &'v ShapeTable<'tcx>,
    printers: &'v PrinterRegistry,
    /// Box addresses on the current path (cycle guard).
    seen: FxHashSet<usize>,
}

impl<'tcx> Walker<'_, 'tcx> {
    /// Render the value whose inline representation sits at `addr`.
    /// `is_field` marks `[field]`-link member cells (always pointers).
    unsafe fn render(
        &mut self,
        addr: *const u8,
        ty: Ty<'tcx>,
        is_field: bool,
        depth: usize,
        out: &mut String,
    ) -> Result<(), String> {
        if depth > MAX_DEPTH {
            out.push('…');
            return Ok(());
        }
        unsafe {
            match *ty.kind() {
                TyKind::Bool => out.push_str(if addr.read() & 1 != 0 {
                    "true"
                } else {
                    "false"
                }),
                TyKind::Char => {
                    let value = addr.cast::<u32>().read_unaligned();
                    let c = char::from_u32(value)
                        .ok_or_else(|| format!("invalid stored char code point {value}"))?;
                    out.push('\'');
                    out.extend(c.escape_debug());
                    out.push('\'');
                }
                TyKind::Int(IntTy::Signed(w)) => {
                    let v: i64 = match w {
                        8 => addr.cast::<i8>().read_unaligned() as i64,
                        16 => addr.cast::<i16>().read_unaligned() as i64,
                        32 => addr.cast::<i32>().read_unaligned() as i64,
                        64 => addr.cast::<i64>().read_unaligned(),
                        other => return Err(format!("unsupported width i{other}")),
                    };
                    out.push_str(&v.to_string());
                }
                TyKind::Int(IntTy::Unsigned(w)) => {
                    let v: u64 = match w {
                        8 => addr.read() as u64,
                        16 => addr.cast::<u16>().read_unaligned() as u64,
                        32 => addr.cast::<u32>().read_unaligned() as u64,
                        64 => addr.cast::<u64>().read_unaligned(),
                        other => return Err(format!("unsupported width u{other}")),
                    };
                    out.push_str(&v.to_string());
                }
                TyKind::Fp(FpTy::Ieee(32)) => {
                    out.push_str(&crate::value::render_float(
                        addr.cast::<f32>().read_unaligned() as f64,
                    ));
                }
                TyKind::Fp(FpTy::Ieee(64)) => {
                    out.push_str(&crate::value::render_float(
                        addr.cast::<f64>().read_unaligned(),
                    ));
                }
                // 16-bit floats widen to f32 exactly (bf16 is a truncated
                // f32; f16 widening is value-exact), so the printed decimal
                // is the true stored value.
                TyKind::Fp(FpTy::Ieee(16)) => {
                    let bits = addr.cast::<u16>().read_unaligned();
                    out.push_str(&crate::value::render_float(
                        half::f16::from_bits(bits).to_f32() as f64,
                    ));
                }
                TyKind::Fp(FpTy::BFloat16) => {
                    let bits = addr.cast::<u16>().read_unaligned();
                    out.push_str(&crate::value::render_float(
                        f32::from_bits((bits as u32) << 16) as f64,
                    ));
                }
                TyKind::Unit => out.push_str("()"),
                TyKind::Closure { .. } => out.push_str("<closure>"),
                TyKind::Nullable(inner) => {
                    let ptr = addr.cast::<*const u8>().read_unaligned();
                    if ptr.is_null() {
                        out.push_str("Null");
                    } else {
                        // The cell holds the inner (boxed) value's pointer.
                        self.render(addr, inner, is_field, depth + 1, out)?;
                    }
                }
                TyKind::Record { def, args, .. } => {
                    let shape = self
                        .shapes
                        .get(&(def, args))
                        .ok_or("unknown record instance")?;
                    if let Some(hook) = self.printers.lookup(&shape.name)
                        && let Some(text) = hook.print(&ReflectedValue {
                            addr,
                            ty,
                            shapes: self.shapes,
                        })
                    {
                        out.push_str(&text);
                        return Ok(());
                    }
                    if shape.default_cap == DefaultCap::Value && !is_field {
                        self.render_record(addr, shape, depth, out)?;
                        return Ok(());
                    }
                    // A boxed member cell holds the box pointer; the payload
                    // sits past the box header.
                    let boxed = addr.cast::<*const u8>().read_unaligned();
                    if boxed.is_null() {
                        out.push_str("<null>");
                        return Ok(());
                    }
                    // Cycle guard over the *current path* only: a box revisited
                    // while still being rendered is a cycle (constructible via
                    // `[field]` links); mere sharing (a DAG) prints normally.
                    if !self.seen.insert(boxed as usize) {
                        out.push_str("...");
                        return Ok(());
                    }
                    let inner = inline_size_align(ty, self.shapes)?;
                    let header = match shape.default_cap {
                        DefaultCap::Value => WORD, // unreachable in practice
                        DefaultCap::Shared => WORD,
                        DefaultCap::Regional => 3 * WORD,
                    };
                    let payload = boxed.add(align_to(header, inner.align) as usize);
                    let result = self.render_record(payload, shape, depth, out);
                    self.seen.remove(&(boxed as usize));
                    result?;
                }
                ref other => return Err(format!("cannot display `{other:?}` yet")),
            }
        }
        Ok(())
    }

    unsafe fn render_record(
        &mut self,
        payload: *const u8,
        shape: &RecordShape<'tcx>,
        depth: usize,
        out: &mut String,
    ) -> Result<(), String> {
        match &shape.layout {
            ShapeLayout::Compound(fields) => {
                out.push_str(&shape.name);
                unsafe { self.render_fields(payload, fields, depth, out) }
            }
            ShapeLayout::Variants(variants) => {
                let tag = unsafe { payload.cast::<usize>().read_unaligned() };
                let Some(variant) = variants.get(tag) else {
                    return Err(format!("corrupt enum tag {tag} for `{}`", shape.name));
                };
                out.push_str(&shape.name);
                out.push_str("::");
                out.push_str(&variant.name);
                if variant.fields.is_empty() {
                    return Ok(());
                }
                let fields: Vec<FieldShape<'tcx>> = variant
                    .fields
                    .iter()
                    .map(|&ty| FieldShape {
                        name: None,
                        ty,
                        is_field: false,
                    })
                    .collect();
                let payload_area = variants_payload_size_align(variants, self.shapes)?;
                let base = unsafe { payload.add(align_to(WORD, payload_area.align) as usize) };
                unsafe { self.render_fields(base, &fields, depth, out) }
            }
        }
    }

    unsafe fn render_fields(
        &mut self,
        base: *const u8,
        fields: &[FieldShape<'tcx>],
        depth: usize,
        out: &mut String,
    ) -> Result<(), String> {
        let named = fields.iter().any(|f| f.name.is_some());
        out.push_str(if named { " { " } else { "(" });
        let offsets = member_offsets(fields, self.shapes)?;
        for (i, (field, offset)) in fields.iter().zip(offsets).enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            if let Some(name) = &field.name {
                out.push_str(name);
                out.push_str(": ");
            }
            unsafe {
                self.render(
                    base.add(offset as usize),
                    field.ty,
                    field.is_field,
                    depth + 1,
                    out,
                )?;
            }
        }
        out.push_str(if named { " }" } else { ")" });
        Ok(())
    }
}
