//! Record types: the (mutually) recursive compound and variant types of the
//! Reussir dialect.
//!
//! A record is registered once by name with its kind, default capability, and
//! fields, then materialized as a [`Type`] with [`Context::record`]. Fields may
//! reference other records — including the record being defined — by name, which
//! is how recursive shapes like a cons-list are expressed.
//!
//! The printer emits the canonical self-contained spelling MLIR uses: each
//! record is expanded in full, except that a reference back to a record already
//! being expanded (an ancestor in the recursion) is written in the name-only
//! form `!reussir.record<compound "Name">`. That single string parses back to
//! the same recursive type, which is exactly what the bytecode's textual
//! fallback needs.

use crate::context::{Atomicity, Capability, Context, Type, TypeKind, atom_suffix, cap_suffix};

/// Whether a record is a product (`compound`) or a sum (`variant`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Compound,
    Variant,
}

impl RecordKind {
    fn keyword(self) -> &'static str {
        match self {
            RecordKind::Compound => "compound",
            RecordKind::Variant => "variant",
        }
    }
}

/// A field type expression. It can be a fully-formed type, a reference to a
/// (possibly not-yet-defined) record, or one of those wrapped in a pointer-like
/// Reussir type. Build these with the `field*` constructors on [`Context`].
#[derive(Clone, Copy)]
pub struct Field<'a>(&'a FieldKind<'a>);

enum FieldKind<'a> {
    Leaf(Type<'a>),
    Record(&'a str),
    Rc(Field<'a>, Capability, Atomicity),
    Ref(Field<'a>, Capability, Atomicity),
    Nullable(Field<'a>),
}

/// One field of a compound record: its type and whether it is mutable (prints
/// with the `[field]` capability marker).
#[derive(Clone, Copy)]
pub struct RecordField<'a> {
    ty: Field<'a>,
    mutable: bool,
}

/// A registered record definition. Stored in the [`Context`] record registry.
pub struct RecordDef<'a> {
    pub kind: RecordKind,
    cap: Capability,
    fields: &'a [RecordField<'a>],
}

impl<'a> Context<'a> {
    /// A field whose type is a concrete, non-record type.
    pub fn field(&self, ty: Type<'a>) -> Field<'a> {
        Field(self.bump().alloc(FieldKind::Leaf(ty)))
    }

    /// A field that references a named record (which may be defined later).
    pub fn field_record(&self, name: &str) -> Field<'a> {
        let name = self.alloc_str(name);
        Field(self.bump().alloc(FieldKind::Record(name)))
    }

    /// A field wrapping another field in `!reussir.rc<...>`.
    pub fn field_rc(&self, inner: Field<'a>, cap: Capability, atom: Atomicity) -> Field<'a> {
        Field(self.bump().alloc(FieldKind::Rc(inner, cap, atom)))
    }

    /// A field wrapping another field in `!reussir.ref<...>`.
    pub fn field_ref(&self, inner: Field<'a>, cap: Capability, atom: Atomicity) -> Field<'a> {
        Field(self.bump().alloc(FieldKind::Ref(inner, cap, atom)))
    }

    /// A field wrapping another field in `!reussir.nullable<...>`.
    pub fn field_nullable(&self, inner: Field<'a>) -> Field<'a> {
        Field(self.bump().alloc(FieldKind::Nullable(inner)))
    }

    /// Register a compound (product) record. Each field is paired with a
    /// mutability flag.
    pub fn define_compound(&self, name: &str, cap: Capability, fields: &[(Field<'a>, bool)]) {
        let owned: Vec<RecordField<'a>> = fields
            .iter()
            .map(|&(ty, mutable)| RecordField { ty, mutable })
            .collect();
        self.register(name, RecordKind::Compound, cap, &owned);
    }

    /// Register a variant (sum) record from its member field types.
    pub fn define_variant(&self, name: &str, members: &[Field<'a>]) {
        let owned: Vec<RecordField<'a>> = members
            .iter()
            .map(|&ty| RecordField { ty, mutable: false })
            .collect();
        self.register(name, RecordKind::Variant, Capability::Unspecified, &owned);
    }

    fn register(&self, name: &str, kind: RecordKind, cap: Capability, fields: &[RecordField<'a>]) {
        let name = self.alloc_str(name);
        let fields = self.bump().alloc_slice_copy(fields);
        self.records
            .borrow_mut()
            .insert(name, RecordDef { kind, cap, fields });
    }

    /// Materialize a registered record as a [`Type`]. The record and everything
    /// it transitively references must already be registered.
    pub fn record(&self, name: &str) -> Type<'a> {
        let mut visiting: Vec<&'a str> = Vec::new();
        let text = self.expand_record(name, &mut visiting);
        let name_ref = self.alloc_str(name);
        self.intern_type(text, TypeKind::Record(name_ref), "reussir")
    }

    fn expand_record(&self, name: &str, visiting: &mut Vec<&'a str>) -> String {
        // Pull the needed data out before recursing so the registry borrow is
        // not held across the recursive calls.
        let (kind, cap, fields, name_ref) = {
            let registry = self.records.borrow();
            let (&name_ref, def) = registry
                .get_key_value(name)
                .unwrap_or_else(|| panic!("record '{name}' is not defined"));
            (def.kind, def.cap, def.fields, name_ref)
        };

        // A back-reference to an ancestor currently being expanded prints in the
        // name-only form, breaking the recursion.
        if visiting.contains(&name_ref) {
            return format!("!reussir.record<{} \"{name}\">", kind.keyword());
        }

        visiting.push(name_ref);
        let body = match kind {
            RecordKind::Compound => {
                let fields_str = fields
                    .iter()
                    .map(|f| {
                        let prefix = if f.mutable { "[field] " } else { "" };
                        format!("{prefix}{}", self.expand_field(f.ty, visiting))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "!reussir.record<compound \"{name}\"{} {{{fields_str}}}>",
                    cap_bracket(cap)
                )
            }
            RecordKind::Variant => {
                let members_str = fields
                    .iter()
                    .map(|f| self.expand_field(f.ty, visiting))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("!reussir.record<variant \"{name}\" {{{members_str}}}>")
            }
        };
        visiting.pop();
        body
    }

    fn expand_field(&self, field: Field<'a>, visiting: &mut Vec<&'a str>) -> String {
        match field.0 {
            FieldKind::Leaf(ty) => ty.text().to_string(),
            FieldKind::Record(name) => self.expand_record(name, visiting),
            FieldKind::Rc(inner, cap, atom) => format!(
                "!reussir.rc<{}{}{}>",
                self.expand_field(*inner, visiting),
                cap_suffix(*cap),
                atom_suffix(*atom)
            ),
            FieldKind::Ref(inner, cap, atom) => format!(
                "!reussir.ref<{}{}{}>",
                self.expand_field(*inner, visiting),
                cap_suffix(*cap),
                atom_suffix(*atom)
            ),
            FieldKind::Nullable(inner) => {
                format!("!reussir.nullable<{}>", self.expand_field(*inner, visiting))
            }
        }
    }
}

/// The bracketed capability marker printed after a compound record's name, e.g.
/// ` [value]`. Unspecified prints nothing.
fn cap_bracket(cap: Capability) -> &'static str {
    match cap {
        Capability::Unspecified => "",
        Capability::Shared => " [shared]",
        Capability::Value => " [value]",
        Capability::Flex => " [flex]",
        Capability::Rigid => " [rigid]",
        Capability::Field => " [field]",
        Capability::Regional => " [regional]",
    }
}
