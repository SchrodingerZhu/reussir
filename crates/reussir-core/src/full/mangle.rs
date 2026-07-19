//! Rust v0 symbol mangling for the *Full* (monomorphized) phase.
//!
//! Mangling runs on ground [`Ty`]s — every generic has been substituted and
//! every hole solved — so a function instance or type has a single, stable
//! linker symbol. The scheme is a subset of Rust's v0 format; it is the contract
//! shared with the backend (the C++ runtime and the trampoline ABI both name
//! symbols this way), so the encoding is fixed, not an implementation detail.
//!
//! # Grammar
//!
//! ```text
//! mangled-name  → "_R" (path | type)
//! path          → "C" identifier                     -- crate-root segment
//!               | "N" "v" path identifier            -- nested (value namespace)
//! identifier    → decimal [_] body                   -- ASCII; `_` if body leads with a digit/`_`
//!               | "u" decimal [_] punycode           -- non-ASCII (RFC 3492, `-`→`_`)
//! path-w/-args  → path                               -- no type arguments
//!               | "I" path {type}+ "E"               -- applied to type arguments
//! type          → "b" | "e" | "u" | "z"              -- bool | str | unit | bottom(never)
//!               | int-code | float-code
//!               | path-w/-args                       -- nominal record / Nullable
//!               | "F" "K" identifier {type}* "E" type-- closure
//! int-code      → a s l x (i8 i16 i32 i64) | h t m y (u8 u16 u32 u64)
//!               | "C4i128" | "C4u128"
//! float-code    → f d (f32 f64) | "C3f16" | "C4f128" | "C4bf16" | "C2f8"
//! ```
//!
//! A nominal record mangles as its **fully-qualified path** (so two same-named
//! records in different modules get distinct symbols) applied to its type
//! arguments; the per-use capability is *irrelevant* to the ABI and is dropped.
//! A monomorphized function instance shares the same path-with-arguments form
//! (its [`DefId`]'s path applied to the instance's type arguments).

use reussir_syntax::kind::{Resolver, TokenKey};

use crate::semi::resolve::DefTable;
use crate::semi::ty::{DefId, FpTy, IntTy, Ty, TyKind};
use crate::utils::punycode;

/// Mangles ground types and item instances into v0 linker symbols.
///
/// Holds the resolution table (for [`DefId`] → fully-qualified path) and the
/// token resolver (for path segment → text); construct one per program and reuse
/// it for every symbol.
pub struct Mangler<'a> {
    defs: &'a DefTable,
    resolver: &'a dyn Resolver<TokenKey>,
}

impl<'a> Mangler<'a> {
    pub fn new(defs: &'a DefTable, resolver: &'a dyn Resolver<TokenKey>) -> Self {
        Mangler { defs, resolver }
    }

    /// The symbol for a ground type: `_R` followed by the type encoding.
    pub fn mangle_ty(&self, ty: Ty<'_>) -> String {
        let mut out = String::from("_R");
        self.ty(&mut out, ty);
        out
    }

    /// The symbol for a monomorphized item instance: its fully-qualified path
    /// applied to `ty_args`. Functions and records share this form.
    pub fn mangle_instance(&self, def: DefId, ty_args: &[Ty<'_>]) -> String {
        let mut out = String::from("_R");
        self.path_with_args(&mut out, def, ty_args);
        out
    }

    /// The symbol for an enum's variant *payload* record: the variant name nested
    /// as a path component *under the enum applied to its type arguments*, so a
    /// generic enum mangles as `Record<Params>::Variant` — the arguments stay on
    /// the record, not the variant (`_RNvI…E<Variant>`). `List::<i32>::Cons` and
    /// `List::<u8>::Cons` thus get distinct symbols, and a variant payload reads
    /// back as a proper path component rather than a synthetic name.
    pub fn mangle_variant(&self, def: DefId, variant: &str, ty_args: &[Ty<'_>]) -> String {
        let mut out = String::from("_R");
        out.push('N');
        out.push('v');
        self.path_with_args(&mut out, def, ty_args);
        ident(&mut out, variant);
        out
    }

    // ----- paths -----

    /// A path applied to (possibly zero) type arguments. Bare path when empty,
    /// else wrapped in `I … E`.
    fn path_with_args(&self, out: &mut String, def: DefId, args: &[Ty<'_>]) {
        let segments: Vec<&str> = self
            .defs
            .path(def)
            .0
            .iter()
            .map(|&seg| self.resolver.resolve(seg))
            .collect();
        self.path_with_args_segs(out, &segments, args);
    }

    /// As [`Mangler::path_with_args`] but over already-resolved segment text, so
    /// synthetic paths (e.g. `Nullable`) reuse the same nesting logic.
    fn path_with_args_segs(&self, out: &mut String, segments: &[&str], args: &[Ty<'_>]) {
        if args.is_empty() {
            self.path(out, segments);
        } else {
            out.push('I');
            self.path(out, segments);
            for &arg in args {
                self.ty(out, arg);
            }
            out.push('E');
        }
    }

    /// The nested-path encoding: a single segment is a crate root (`C`); each
    /// further segment wraps the prefix in `Nv … <ident>`.
    fn path(&self, out: &mut String, segments: &[&str]) {
        let (last, prefix) = segments
            .split_last()
            .expect("a qualified path is never empty");
        if prefix.is_empty() {
            out.push('C');
            ident(out, last);
        } else {
            out.push('N');
            out.push('v');
            self.path(out, prefix);
            ident(out, last);
        }
    }

    // ----- types -----

    fn ty(&self, out: &mut String, ty: Ty<'_>) {
        match *ty.kind() {
            TyKind::Bool => out.push('b'),
            TyKind::Str => out.push('e'),
            TyKind::Char => out.push('c'),
            TyKind::Unit => out.push('u'),
            TyKind::Bottom => out.push('z'),
            TyKind::Int(int) => out.push_str(int_code(int)),
            TyKind::Fp(fp) => out.push_str(fp_code(fp)),
            TyKind::Nullable(inner) => {
                // `Nullable<T>` mangles as a synthetic one-segment record applied
                // to its inner type.
                self.path_with_args_segs(out, &["Nullable"], &[inner]);
            }
            TyKind::Cell { elem, kind } => {
                // The cell surface constructors (`Cell<T>`, `RefCell<T>`,
                // `Mutex<T>`, …) are root built-in type constructors,
                // represented as synthetic one-segment records in the ABI
                // spelling.
                self.path_with_args_segs(out, &[kind.surface_name()], &[elem]);
            }
            TyKind::Arc(inner) => {
                // `Arc<X>` is a root built-in type constructor too: a synthetic
                // one-segment record applied to the colored record.
                self.path_with_args_segs(out, &["Arc"], &[inner]);
            }
            TyKind::Array { elem, dims } => {
                // An array mangles as a synthetic record whose identifier
                // carries the extents (`Array512x512`), applied to the element.
                let mut name = String::from("Array");
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        name.push('x');
                    }
                    name.push_str(&d.to_string());
                }
                self.path_with_args_segs(out, &[&name], &[elem]);
            }
            TyKind::Record { def, args, .. } => {
                // The per-use capability (`flex`) is irrelevant to the ABI.
                self.path_with_args(out, def, args);
            }
            TyKind::Closure { params, ret } => {
                out.push('F');
                out.push('K');
                ident(out, "ReussirClosure");
                for &param in params {
                    self.ty(out, param);
                }
                out.push('E');
                self.ty(out, ret);
            }
            TyKind::Generic(_) => {
                panic!("v0 mangle: a generic survived into the Full phase (type is not ground)")
            }
            TyKind::Hole(_) => {
                panic!(
                    "v0 mangle: an inference hole survived into the Full phase (type is not ground)"
                )
            }
        }
    }
}

/// Mangles one identifier segment: ASCII names use a `length + body` prefix with
/// a `_` separator when the body would otherwise start with a digit or `_`;
/// non-ASCII names are Punycode-encoded (with `-`→`_`) behind a `u` tag.
fn ident(out: &mut String, name: &str) {
    if name.is_ascii() {
        // ASCII ⇒ one byte per char, so the byte length is the character count
        // without a second O(n) scan.
        push_ident_body(out, name, name.len());
    } else {
        let punycode = punycode::encode(name).replace('-', "_");
        out.push('u');
        push_ident_body(out, &punycode, punycode.len());
    }
}

/// Emits `length [_] body`, inserting the disambiguating `_` when `body` leads
/// with a digit or underscore (the grammar forbids those right after the length).
fn push_ident_body(out: &mut String, body: &str, len: usize) {
    out.push_str(&len.to_string());
    if matches!(body.bytes().next(), Some(b) if b.is_ascii_digit() || b == b'_') {
        out.push('_');
    }
    out.push_str(body);
}

/// The integral-type code; panics on widths the ABI does not define.
fn int_code(int: IntTy) -> &'static str {
    match int {
        IntTy::Signed(8) => "a",
        IntTy::Signed(16) => "s",
        IntTy::Signed(32) => "l",
        IntTy::Signed(64) => "x",
        IntTy::Signed(128) => "C4i128",
        IntTy::Unsigned(8) => "h",
        IntTy::Unsigned(16) => "t",
        IntTy::Unsigned(32) => "m",
        IntTy::Unsigned(64) => "y",
        IntTy::Unsigned(128) => "C4u128",
        IntTy::Signed(w) => panic!("v0 mangle: unsupported signed integer width {w}"),
        IntTy::Unsigned(w) => panic!("v0 mangle: unsupported unsigned integer width {w}"),
    }
}

/// The floating-point-type code; panics on widths the ABI does not define.
fn fp_code(fp: FpTy) -> &'static str {
    match fp {
        FpTy::Ieee(16) => "C3f16",
        FpTy::Ieee(32) => "f",
        FpTy::Ieee(64) => "d",
        FpTy::Ieee(128) => "C4f128",
        FpTy::Ieee(w) => panic!("v0 mangle: unsupported IEEE float width {w}"),
        FpTy::BFloat16 => "C4bf16",
        FpTy::Float8 => "C2f8",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semi::ty::{Flexivity, TyCtxt};
    use reussir_syntax::kind::InternKey;

    /// A test resolver: key `i` (1-based) resolves to segment `i - 1`.
    struct VecResolver(Vec<&'static str>);
    impl Resolver<TokenKey> for VecResolver {
        fn try_resolve(&self, key: TokenKey) -> Option<&str> {
            self.0.get(key.into_u32() as usize - 1).copied()
        }
    }

    fn key(n: u32) -> TokenKey {
        TokenKey::try_from_u32(n).expect("nonzero key")
    }

    fn ident_of(name: &str) -> String {
        let mut s = String::new();
        ident(&mut s, name);
        s
    }

    #[test]
    fn ascii_identifiers() {
        // Golden vectors ported from the Haskell `Mangle` test suite.
        assert_eq!(ident_of("foo"), "3foo");
        assert_eq!(ident_of("bach"), "4bach");
        assert_eq!(ident_of("123abc"), "6_123abc");
    }

    #[test]
    fn unicode_identifier_uses_punycode() {
        // `gödel` → punycode `gdel-5qa` → `-`→`_` → `gdel_5qa`, length 8.
        assert_eq!(ident_of("gödel"), "u8gdel_5qa");
    }

    /// Mangles a bare segment path (no type arguments) for the path tests.
    /// `path`/`path_with_args_segs` read only their `segments` argument, so a
    /// mangler over empty tables suffices.
    fn mangle_segments(segments: &[&str]) -> String {
        let defs = DefTable::new();
        let resolver = VecResolver(vec![]);
        let m = Mangler::new(&defs, &resolver);
        let mut out = String::from("_R");
        m.path_with_args_segs(&mut out, segments, &[]);
        out
    }

    #[test]
    fn root_path() {
        // Golden vectors ported from the Haskell `Mangle` test suite.
        assert_eq!(mangle_segments(&["example"]), "_RC7example");
    }

    #[test]
    fn nested_path() {
        assert_eq!(
            mangle_segments(&["mycrate", "example"]),
            "_RNvC7mycrate7example"
        );
    }

    #[test]
    fn deeply_nested_path() {
        assert_eq!(
            mangle_segments(&["a", "example", "exampla"]),
            "_RNvNvC1a7example7exampla"
        );
    }

    #[test]
    fn scalar_and_compound_type_codes() {
        crate::with_tcx(|tcx: &TyCtxt<'_>| {
            let defs = DefTable::new();
            let resolver = VecResolver(vec![]);
            let m = Mangler::new(&defs, &resolver);
            assert_eq!(m.mangle_ty(tcx.mk_bool()), "_Rb");
            assert_eq!(m.mangle_ty(tcx.mk_str()), "_Re");
            assert_eq!(m.mangle_ty(tcx.mk_char()), "_Rc");
            assert_eq!(m.mangle_ty(tcx.mk_unit()), "_Ru");
            assert_eq!(m.mangle_ty(tcx.mk_int(IntTy::Signed(32))), "_Rl");
            assert_eq!(m.mangle_ty(tcx.mk_int(IntTy::Unsigned(64))), "_Ry");
            assert_eq!(m.mangle_ty(tcx.mk_int(IntTy::Signed(128))), "_RC4i128");
            assert_eq!(m.mangle_ty(tcx.mk_fp(FpTy::Ieee(32))), "_Rf");
            assert_eq!(m.mangle_ty(tcx.mk_fp(FpTy::Ieee(64))), "_Rd");
            // Closure `(i32) -> bool` → `FK14ReussirClosurelEb`.
            let clos = tcx.mk_closure(&[tcx.mk_int(IntTy::Signed(32))], tcx.mk_bool());
            assert_eq!(m.mangle_ty(clos), "_RFK14ReussirClosurelEb");
            // `Nullable<i32>` → `IC8NullablelE`.
            let nul = tcx.mk_nullable(tcx.mk_int(IntTy::Signed(32)));
            assert_eq!(m.mangle_ty(nul), "_RIC8NullablelE");
            // `Cell<i32>` → `IC4CelllE`; `RefCell<i32>` → `IC7RefCelllE`.
            let cell = tcx.mk_cell(
                tcx.mk_int(IntTy::Signed(32)),
                crate::semi::ty::CellKind::Plain,
            );
            assert_eq!(m.mangle_ty(cell), "_RIC4CelllE");
            let refcell = tcx.mk_cell(
                tcx.mk_int(IntTy::Signed(32)),
                crate::semi::ty::CellKind::Exclusive,
            );
            assert_eq!(m.mangle_ty(refcell), "_RIC7RefCelllE");
            // `Arc<i32>` → `IC3ArclE` (a synthetic root record, like the
            // others; a real inner is always a `[shared]` record).
            let arc = tcx.mk_arc(tcx.mk_int(IntTy::Signed(32)));
            assert_eq!(m.mangle_ty(arc), "_RIC3ArclE");
        });
    }

    #[test]
    fn generic_record_application() {
        crate::with_tcx(|tcx: &TyCtxt<'_>| {
            let mut defs = DefTable::new();
            // A flat-module record `Foo` gets the single-segment path `[Foo]`.
            let foo = defs.declare_record(key(1)).expect("fresh");
            let resolver = VecResolver(vec!["Foo"]);
            let m = Mangler::new(&defs, &resolver);
            // `Foo<i32>` → `IC3FoolE`.
            let foo_i32 =
                tcx.mk_record(foo, &[tcx.mk_int(IntTy::Signed(32))], Flexivity::Irrelevant);
            assert_eq!(m.mangle_ty(foo_i32), "_RIC3FoolE");
            // `Foo` (no args) → `C3Foo`; mangling a function instance is the same.
            let foo0 = tcx.mk_record(foo, &[], Flexivity::Irrelevant);
            assert_eq!(m.mangle_ty(foo0), "_RC3Foo");
            assert_eq!(m.mangle_instance(foo, &[]), "_RC3Foo");
        });
    }

    #[test]
    fn variant_payload_nests_under_the_enum() {
        crate::with_tcx(|tcx: &TyCtxt<'_>| {
            let mut defs = DefTable::new();
            let list = defs.declare_record(key(1)).expect("fresh");
            let resolver = VecResolver(vec!["List"]);
            let m = Mangler::new(&defs, &resolver);
            // `List::Cons` (no args) nests the variant as a path component:
            // `Nv C4List 4Cons`.
            assert_eq!(m.mangle_variant(list, "Cons", &[]), "_RNvC4List4Cons");
            // For a generic enum the arguments stay on the *record*, with the
            // variant nested outside: `List<i32>::Cons` → `Nv (IC4ListlE) 4Cons`,
            // not `List::Cons<i32>`.
            let i32_ty = tcx.mk_int(IntTy::Signed(32));
            assert_eq!(
                m.mangle_variant(list, "Cons", &[i32_ty]),
                "_RNvIC4ListlE4Cons"
            );
        });
    }
}
