//! The IR construction context: a bump arena plus interning tables for types
//! and attributes.
//!
//! All IR nodes — types, attributes, values, operations, blocks — are allocated
//! in a single [`stumpalo::Arena`] bump arena and handed out as cheap `Copy`
//! handles that borrow from it. Types and attributes are additionally *interned* by
//! their printed MLIR text, so structurally equal entities share one node (and
//! later, one bytecode table entry). This mirrors how MLIR uniquing works and
//! keeps the emitted bytecode compact.
//!
//! Every type and attribute caches its printed textual form at construction.
//! That text is exactly what the bytecode writer stores in the attribute/type
//! section using MLIR's textual fallback encoding, so no separate printing pass
//! is needed at emit time.

use std::cell::RefCell;

use rustc_hash::FxHashMap;
use stumpalo::Arena;

/// Atomicity of a reference-counted or referenced value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Atomicity {
    Atomic,
    NonAtomic,
}

/// Ownership/aliasing capability carried by `!reussir.rc`/`!reussir.ref` and by
/// record fields. `Unspecified` prints nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    Unspecified,
    Shared,
    Value,
    Flex,
    Rigid,
    Field,
    Regional,
}

/// Lifetime scope of a `!reussir.str`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LifeScope {
    Local,
    Global,
}

/// The structural kind of a [`Type`].
///
/// Builtin scalar types are spelled the MLIR way (`i32`, `index`, `f64`); the
/// Reussir dialect types print with their `!reussir.*` custom syntax.
#[derive(Clone, Copy)]
pub enum TypeKind<'a> {
    /// A signless integer of the given bit width (`i1`, `i8`, ..., `i128`).
    Int(u32),
    /// The platform index type (`index`).
    Index,
    /// A floating point type, identified by its MLIR spelling (`f32`, `bf16`).
    Float(FloatKind),
    /// The unit type. Not a real MLIR type; functions returning unit emit no
    /// result and unit values never reach the bytecode.
    Unit,
    /// `!reussir.rc<inner cap atomicity>`.
    Rc {
        inner: Type<'a>,
        cap: Capability,
        atom: Atomicity,
    },
    /// `!reussir.ref<inner cap atomicity>`.
    Ref {
        inner: Type<'a>,
        cap: Capability,
        atom: Atomicity,
    },
    /// `!reussir.closure<(args) -> ret>`.
    Closure { args: &'a [Type<'a>], ret: Type<'a> },
    /// `!reussir.nullable<inner>`.
    Nullable(Type<'a>),
    /// `!reussir.region`.
    RegionHandle,
    /// `!reussir.str<local|global>`.
    Str(LifeScope),
    /// A reference to a named record instance, printed as the alias `!name`.
    Named(&'a str),
    /// `!reussir.token<align : A, size : S>` — a raw allocation token.
    Token { align: u64, size: u64 },
    /// A record type (`!reussir.record<...>`); see [`crate::records`].
    Record(&'a str),
    /// `!reussir.raw_ptr<elem>` — an unmanaged pointer to `elem`.
    RawPtr(Type<'a>),
    /// `!reussir.hole<elem>` — a write-once slot used by region builders.
    Hole(Type<'a>),
    /// `!reussir.closure_box<payload...>` — the payload box behind a closure.
    ClosureBox(&'a [Type<'a>]),
    /// `!reussir.array<d0 x d1 x ... x elem>` — a fixed-shape array.
    Array { shape: &'a [u64], elem: Type<'a> },
    /// `!reussir.ffi_object<"name", @cleanup>` — an opaque foreign object.
    FfiObject { name: &'a str, cleanup: &'a str },
}

/// MLIR floating point type spellings supported by the emitter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    F64,
    F128,
}

impl FloatKind {
    fn spelling(self) -> &'static str {
        match self {
            FloatKind::F16 => "f16",
            FloatKind::BF16 => "bf16",
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::F128 => "f128",
        }
    }
}

/// Interned type node: its structure, cached MLIR text, and owning dialect.
pub struct TypeData<'a> {
    kind: TypeKind<'a>,
    text: &'a str,
    dialect: &'static str,
}

/// A cheap, copyable handle to an interned [`TypeData`].
///
/// Equality and hashing are by identity (pointer), which is sound because the
/// context interns structurally-equal types to the same node.
#[derive(Clone, Copy)]
pub struct Type<'a>(&'a TypeData<'a>);

impl<'a> Type<'a> {
    /// The structural kind of this type.
    pub fn kind(self) -> TypeKind<'a> {
        self.0.kind
    }

    /// The cached MLIR textual spelling, e.g. `!reussir.rc<i32>`.
    pub fn text(self) -> &'a str {
        self.0.text
    }

    /// The namespace of the dialect that owns this type (`builtin`, `reussir`).
    pub fn dialect(self) -> &'static str {
        self.0.dialect
    }

    /// Whether this is the unit type, which is never emitted as a real type.
    pub fn is_unit(self) -> bool {
        matches!(self.0.kind, TypeKind::Unit)
    }
}

impl PartialEq for Type<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}
impl Eq for Type<'_> {}
impl std::hash::Hash for Type<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const TypeData).hash(state);
    }
}

/// The structural kind of an [`Attr`].
#[derive(Clone, Copy)]
pub enum AttrKind<'a> {
    /// The unit attribute (`unit`); typically used as a present-but-valueless
    /// dictionary entry.
    Unit,
    /// A boolean attribute (`true`/`false`).
    Bool(bool),
    /// A typed integer attribute, printed `value : type`.
    Integer(i64, Type<'a>),
    /// A string attribute, printed with quotes and escaping.
    Str(&'a str),
    /// A type attribute, printed as the wrapped type's text.
    TypeAttr(Type<'a>),
    /// A flat symbol reference, printed `@symbol`.
    SymbolRef(&'a str),
    /// An array attribute, printed `[a, b, ...]`.
    Array(&'a [Attr<'a>]),
    /// A dictionary attribute, printed `{k = v, ...}` (used for op attributes).
    Dict(&'a [(&'a str, Attr<'a>)]),
    /// An attribute whose textual form is supplied directly. Used for entities
    /// not yet structurally modeled (locations, `#llvm.linkage`, debug info).
    Raw,
}

/// Interned attribute node: its structure, cached MLIR text, and owning dialect.
pub struct AttrData<'a> {
    kind: AttrKind<'a>,
    text: &'a str,
    dialect: &'static str,
}

/// A cheap, copyable handle to an interned [`AttrData`].
#[derive(Clone, Copy)]
pub struct Attr<'a>(&'a AttrData<'a>);

impl<'a> Attr<'a> {
    /// The structural kind of this attribute.
    pub fn kind(self) -> AttrKind<'a> {
        self.0.kind
    }

    /// The cached MLIR textual spelling.
    pub fn text(self) -> &'a str {
        self.0.text
    }

    /// The namespace of the dialect that owns this attribute.
    pub fn dialect(self) -> &'static str {
        self.0.dialect
    }

    /// Whether this dictionary attribute is empty (so the op can omit it).
    pub fn is_empty_dict(self) -> bool {
        matches!(self.0.kind, AttrKind::Dict(entries) if entries.is_empty())
    }

    /// Whether this attribute is the unknown source location, which the bytecode
    /// format elides on block arguments.
    pub fn is_unknown_loc(self) -> bool {
        self.0.text == "loc(unknown)"
    }
}

impl PartialEq for Attr<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}
impl Eq for Attr<'_> {}
impl std::hash::Hash for Attr<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const AttrData).hash(state);
    }
}

/// Owns the bump arena and interning tables used to build a module.
///
/// A `Context` is created around a borrowed [`Arena`]; every handle it returns
/// borrows from that same arena, tying the IR's lifetime to the arena's.
pub struct Context<'a> {
    bump: &'a Arena,
    types: RefCell<FxHashMap<&'a str, Type<'a>>>,
    attrs: RefCell<FxHashMap<&'a str, Attr<'a>>>,
    pub(crate) records: RefCell<FxHashMap<&'a str, crate::records::RecordDef<'a>>>,
}

impl<'a> Context<'a> {
    /// Create a context backed by the given arena.
    pub fn new(bump: &'a Arena) -> Self {
        Context {
            bump,
            types: RefCell::new(FxHashMap::default()),
            attrs: RefCell::new(FxHashMap::default()),
            records: RefCell::new(FxHashMap::default()),
        }
    }

    /// The underlying bump arena, for code that needs to allocate directly.
    pub fn bump(&self) -> &'a Arena {
        self.bump
    }

    /// Allocate a string slice in the arena.
    pub fn alloc_str(&self, s: &str) -> &'a str {
        self.bump.alloc_str(s)
    }

    /// Allocate a slice in the arena from an iterator of items.
    pub fn alloc_slice<T: Copy>(&self, items: &[T]) -> &'a [T] {
        self.bump.alloc_slice_copy(items)
    }

    // ----- Type constructors -------------------------------------------------

    pub(crate) fn intern_type(
        &self,
        text: String,
        kind: TypeKind<'a>,
        dialect: &'static str,
    ) -> Type<'a> {
        if let Some(&t) = self.types.borrow().get(text.as_str()) {
            return t;
        }
        let text_ref: &'a str = self.bump.alloc_str(&text);
        let data: &'a TypeData<'a> = self.bump.alloc(TypeData {
            kind,
            text: text_ref,
            dialect,
        });
        let ty = Type(data);
        self.types.borrow_mut().insert(text_ref, ty);
        ty
    }

    /// A signless integer type of `width` bits (`i1`, `i32`, `i128`, ...).
    pub fn int(&self, width: u32) -> Type<'a> {
        self.intern_type(format!("i{width}"), TypeKind::Int(width), "builtin")
    }

    /// The boolean type, modeled as `i1`.
    pub fn bool_ty(&self) -> Type<'a> {
        self.int(1)
    }

    /// The platform index type (`index`).
    pub fn index(&self) -> Type<'a> {
        self.intern_type("index".to_string(), TypeKind::Index, "builtin")
    }

    /// A floating point type.
    pub fn float(&self, kind: FloatKind) -> Type<'a> {
        self.intern_type(
            kind.spelling().to_string(),
            TypeKind::Float(kind),
            "builtin",
        )
    }

    /// The unit type. It is never emitted; it signals "no result".
    pub fn unit(&self) -> Type<'a> {
        self.intern_type("()".to_string(), TypeKind::Unit, "builtin")
    }

    /// `!reussir.rc<inner cap atomicity>`.
    pub fn rc(&self, inner: Type<'a>, cap: Capability, atom: Atomicity) -> Type<'a> {
        let text = format!(
            "!reussir.rc<{}{}{}>",
            inner.text(),
            cap_suffix(cap),
            atom_suffix(atom)
        );
        self.intern_type(text, TypeKind::Rc { inner, cap, atom }, "reussir")
    }

    /// `!reussir.ref<inner cap atomicity>`.
    pub fn ref_ty(&self, inner: Type<'a>, cap: Capability, atom: Atomicity) -> Type<'a> {
        let text = format!(
            "!reussir.ref<{}{}{}>",
            inner.text(),
            cap_suffix(cap),
            atom_suffix(atom)
        );
        self.intern_type(text, TypeKind::Ref { inner, cap, atom }, "reussir")
    }

    /// `!reussir.closure<(args) -> ret>`; a unit return prints `(args)`.
    pub fn closure(&self, args: &[Type<'a>], ret: Type<'a>) -> Type<'a> {
        let args_slice = self.bump.alloc_slice_copy(args);
        let joined = join_texts(args);
        let text = if ret.is_unit() {
            format!("!reussir.closure<({joined})>")
        } else {
            format!("!reussir.closure<({joined}) -> {}>", ret.text())
        };
        self.intern_type(
            text,
            TypeKind::Closure {
                args: args_slice,
                ret,
            },
            "reussir",
        )
    }

    /// `!reussir.nullable<inner>`.
    pub fn nullable(&self, inner: Type<'a>) -> Type<'a> {
        let text = format!("!reussir.nullable<{}>", inner.text());
        self.intern_type(text, TypeKind::Nullable(inner), "reussir")
    }

    /// `!reussir.region`.
    pub fn region_handle(&self) -> Type<'a> {
        self.intern_type(
            "!reussir.region".to_string(),
            TypeKind::RegionHandle,
            "reussir",
        )
    }

    /// `!reussir.str<local>` or `!reussir.str<global>`.
    pub fn str_ty(&self, scope: LifeScope) -> Type<'a> {
        let s = match scope {
            LifeScope::Local => "local",
            LifeScope::Global => "global",
        };
        self.intern_type(
            format!("!reussir.str<{s}>"),
            TypeKind::Str(scope),
            "reussir",
        )
    }

    /// A reference to a named record instance, printed as the alias `!name`.
    pub fn named(&self, name: &str) -> Type<'a> {
        let name_ref = self.bump.alloc_str(name);
        self.intern_type(format!("!{name}"), TypeKind::Named(name_ref), "reussir")
    }

    /// `!reussir.token<align : A, size : S>` — a raw allocation token of the
    /// given alignment and size in bytes, consumed by `reussir.rc.create`.
    pub fn token(&self, align: u64, size: u64) -> Type<'a> {
        self.intern_type(
            format!("!reussir.token<align : {align}, size : {size}>"),
            TypeKind::Token { align, size },
            "reussir",
        )
    }

    /// `!reussir.raw_ptr<elem>` — an unmanaged pointer to `elem`.
    pub fn raw_ptr(&self, elem: Type<'a>) -> Type<'a> {
        self.intern_type(
            format!("!reussir.raw_ptr<{}>", elem.text()),
            TypeKind::RawPtr(elem),
            "reussir",
        )
    }

    /// `!reussir.hole<elem>` — a write-once slot threaded through region builders.
    pub fn hole(&self, elem: Type<'a>) -> Type<'a> {
        self.intern_type(
            format!("!reussir.hole<{}>", elem.text()),
            TypeKind::Hole(elem),
            "reussir",
        )
    }

    /// `!reussir.closure_box<payload...>` — the payload box wrapped by a closure.
    pub fn closure_box(&self, payload: &[Type<'a>]) -> Type<'a> {
        let payload_slice = self.bump.alloc_slice_copy(payload);
        self.intern_type(
            format!("!reussir.closure_box<{}>", join_texts(payload)),
            TypeKind::ClosureBox(payload_slice),
            "reussir",
        )
    }

    /// `!reussir.array<d0 x d1 x ... x elem>` — a fixed-shape array of `elem`.
    pub fn array(&self, shape: &[u64], elem: Type<'a>) -> Type<'a> {
        let shape_slice = self.bump.alloc_slice_copy(shape);
        let dims = shape.iter().map(|d| format!("{d} x ")).collect::<String>();
        self.intern_type(
            format!("!reussir.array<{dims}{}>", elem.text()),
            TypeKind::Array {
                shape: shape_slice,
                elem,
            },
            "reussir",
        )
    }

    /// `!reussir.ffi_object<"name", @cleanup>` — an opaque foreign object with a
    /// cleanup hook symbol.
    pub fn ffi_object(&self, name: &str, cleanup: &str) -> Type<'a> {
        let name_ref = self.bump.alloc_str(name);
        let cleanup_ref = self.bump.alloc_str(cleanup);
        self.intern_type(
            format!(
                "!reussir.ffi_object<{}, @{}>",
                quote_string(name),
                quote_symbol(cleanup)
            ),
            TypeKind::FfiObject {
                name: name_ref,
                cleanup: cleanup_ref,
            },
            "reussir",
        )
    }

    /// A builtin function type `(args) -> result`, used inside function
    /// signature attributes. A unit result is rendered as `()`; multiple results
    /// are parenthesized. This type is only ever referenced from within an
    /// attribute's text, so it is not added to the bytecode type table itself.
    pub fn func_type(&self, args: &[Type<'a>], results: &[Type<'a>]) -> Type<'a> {
        let args_text = join_texts(args);
        let results_text = match results {
            [] => "()".to_string(),
            [single] => single.text().to_string(),
            many => format!("({})", join_texts(many)),
        };
        let text = format!("({args_text}) -> {results_text}");
        // Reuse the named-type slot purely as an opaque cached-text carrier; its
        // kind is irrelevant because function types only appear inside attribute
        // text and are never emitted as standalone type-table entries.
        let name_ref = self.bump.alloc_str(&text);
        self.intern_type(text, TypeKind::Named(name_ref), "builtin")
    }

    // ----- Attribute constructors -------------------------------------------

    fn intern_attr(&self, text: String, kind: AttrKind<'a>, dialect: &'static str) -> Attr<'a> {
        if let Some(&a) = self.attrs.borrow().get(text.as_str()) {
            return a;
        }
        let text_ref: &'a str = self.bump.alloc_str(&text);
        let data: &'a AttrData<'a> = self.bump.alloc(AttrData {
            kind,
            text: text_ref,
            dialect,
        });
        let attr = Attr(data);
        self.attrs.borrow_mut().insert(text_ref, attr);
        attr
    }

    /// The unit attribute (`unit`).
    pub fn attr_unit(&self) -> Attr<'a> {
        self.intern_attr("unit".to_string(), AttrKind::Unit, "builtin")
    }

    /// A boolean attribute.
    pub fn attr_bool(&self, b: bool) -> Attr<'a> {
        self.intern_attr(b.to_string(), AttrKind::Bool(b), "builtin")
    }

    /// A typed integer attribute, printed `value : type`.
    pub fn attr_int(&self, value: i64, ty: Type<'a>) -> Attr<'a> {
        let text = format!("{value} : {}", ty.text());
        self.intern_attr(text, AttrKind::Integer(value, ty), "builtin")
    }

    /// A string attribute. The text is quoted and escaped per MLIR rules.
    pub fn attr_str(&self, s: &str) -> Attr<'a> {
        let s_ref = self.bump.alloc_str(s);
        let text = quote_string(s);
        self.intern_attr(text, AttrKind::Str(s_ref), "builtin")
    }

    /// A type attribute, printed as the wrapped type's text.
    pub fn attr_type(&self, ty: Type<'a>) -> Attr<'a> {
        self.intern_attr(ty.text().to_string(), AttrKind::TypeAttr(ty), "builtin")
    }

    /// A flat symbol reference attribute, printed `@symbol`.
    pub fn attr_symbol(&self, sym: &str) -> Attr<'a> {
        let sym_ref = self.bump.alloc_str(sym);
        let text = format!("@{}", quote_symbol(sym));
        self.intern_attr(text, AttrKind::SymbolRef(sym_ref), "builtin")
    }

    /// An array attribute, printed `[a, b, ...]`.
    pub fn attr_array(&self, items: &[Attr<'a>]) -> Attr<'a> {
        let items_ref = self.bump.alloc_slice_copy(items);
        let inner = items
            .iter()
            .map(|a| a.text())
            .collect::<Vec<_>>()
            .join(", ");
        self.intern_attr(format!("[{inner}]"), AttrKind::Array(items_ref), "builtin")
    }

    /// A dictionary attribute, printed `{k = v, ...}`. Entries are emitted in
    /// the order given; callers should sort by key when MLIR ordering matters.
    pub fn attr_dict(&self, entries: &[(&str, Attr<'a>)]) -> Attr<'a> {
        let owned: Vec<(&'a str, Attr<'a>)> = entries
            .iter()
            .map(|(k, v)| (&*self.bump.alloc_str(k), *v))
            .collect();
        let inner = owned
            .iter()
            .map(|(k, v)| format!("{} = {}", quote_symbol(k), v.text()))
            .collect::<Vec<_>>()
            .join(", ");
        let entries_ref = self.bump.alloc_slice_copy(&owned);
        self.intern_attr(
            format!("{{{inner}}}"),
            AttrKind::Dict(entries_ref),
            "builtin",
        )
    }

    /// An empty dictionary attribute (the common "no attributes" case).
    pub fn empty_dict(&self) -> Attr<'a> {
        self.attr_dict(&[])
    }

    /// A dense `i32` array attribute, printed `array<i32: a, b, ...>`. Used for
    /// `operandSegmentSizes` on variadic-segment operations.
    pub fn attr_dense_i32(&self, values: &[i32]) -> Attr<'a> {
        let inner = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.intern_attr(format!("array<i32: {inner}>"), AttrKind::Raw, "builtin")
    }

    /// A dense `i64` array attribute, printed `array<i64: a, b, ...>`. Used for
    /// `scf.index_switch` cases and `reussir.record.dispatch` tag sets.
    pub fn attr_dense_i64(&self, values: &[i64]) -> Attr<'a> {
        let inner = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.intern_attr(format!("array<i64: {inner}>"), AttrKind::Raw, "builtin")
    }

    /// An attribute supplied as raw MLIR text, attributed to `dialect`. Use for
    /// locations (`loc(...)`), `#llvm.linkage<...>`, and debug-info attributes.
    pub fn attr_raw(&self, text: &str, dialect: &'static str) -> Attr<'a> {
        self.intern_attr(text.to_string(), AttrKind::Raw, dialect)
    }

    /// The unknown source location attribute, `loc(unknown)`.
    pub fn loc_unknown(&self) -> Attr<'a> {
        self.attr_raw("loc(unknown)", "builtin")
    }

    /// A file/line/column source location, `loc("file":line:col)`.
    pub fn loc_file(&self, file: &str, line: u32, col: u32) -> Attr<'a> {
        let text = format!("loc({}:{line}:{col})", quote_string(file));
        self.attr_raw(&text, "builtin")
    }
}

pub(crate) fn cap_suffix(cap: Capability) -> &'static str {
    match cap {
        Capability::Unspecified => "",
        Capability::Shared => " shared",
        Capability::Value => " value",
        Capability::Flex => " flex",
        Capability::Rigid => " rigid",
        Capability::Field => " field",
        Capability::Regional => " regional",
    }
}

pub(crate) fn atom_suffix(atom: Atomicity) -> &'static str {
    match atom {
        Atomicity::NonAtomic => " normal",
        Atomicity::Atomic => " atomic",
    }
}

fn join_texts(types: &[Type<'_>]) -> String {
    types
        .iter()
        .map(|t| t.text())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Quote and escape a string as an MLIR string literal.
fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render a symbol/identifier, quoting it only when it is not a bare MLIR
/// identifier (which would otherwise need escaping).
fn quote_symbol(s: &str) -> String {
    let bare = !s.is_empty()
        && s.chars().enumerate().all(|(i, c)| {
            c == '_'
                || c == '$'
                || c == '.'
                || c.is_ascii_alphabetic()
                || (i > 0 && c.is_ascii_digit())
        });
    if bare { s.to_string() } else { quote_string(s) }
}
